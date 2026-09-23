"""
Агент для кейса Beeline Tariff Marketing Campaigns.

Стратегия — двухфазная разведка с историческим приоритетом:
  1. Из data/change_tariff.csv строим априорную оценку эффекта перехода
     (from, to, arpu_segment): среднее arpu_change_pct × доля таких переходов.
  2. Скрещиваем с аудиторией: кандидаты (current_tariff, arpu_segment, target)
     с аудиторией >= 100, historical_score = lift × средний predicted_arpu.
  3. Скрининг: пилоты в самом дешёвом канале. Размер адаптивный: чем ближе
     априорный эффект к нулю относительно шума пилота, тем больше выборка.
  4. Уточнение: пилоты на 150–200 абонентов в том канале, который планируем
     использовать в финале, для лучших выживших после скрининга. Сколько
     пилотов отдать уточнению, решаем по ходу скрининга: если выживших ячеек
     заметно больше 10 — уточняем меньше и скринингуем больше.
  5. Оценка = вес_пилота × пилот + (1 − вес_пилота) × история,
     вес_пилота = n / (n + 50), n — все абоненты пилотов кандидата.
  6–7. Выбираем канал и жадно собираем до 10 кампаний в пределах бюджета и охвата.
     В финальном плане ячейки с одним сегментом, целевым тарифом и похожей
     оценкой объединяем в одну кампанию (filter_current_tariff =
     "tariff_X;tariff_Y"), если это повышает ожидаемый результат.
  8. Любой сбой -> возвращаем лучшее из уже найденного.

Все оценки ведутся в «базовых» единицах: относительный эффект до множителя
канала. Пилот в канале с множителем m наблюдает m × эффект + шум, поэтому
его результат делим на m, прежде чем смешивать с историей.

LLM — дополнительный сигнал, а не основа решений. Смысловые профили всех
тарифов (интернет, звонки, цена) даёт OpenAI по описаниям из
tariff_dictionary.csv. В финальном плане, если у ячейки несколько целевых
тарифов с оценкой в пределах 10% от лучшей, выбираем тот, чей профиль ближе
к поведению абонентов ячейки.

Профили берутся из tariff_profiles.json: make_tariff_profiles.py заранее
спрашивает модель несколько раз и сохраняет самый частый ответ, поэтому
решения воспроизводимы. Если файла нет — один батчевый запрос в начале act().
Без ключа, при ошибке или тайм-ауте агент работает по той же логике без LLM.
"""

import hashlib
import json
import os
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]

PRIOR_STRENGTH = 50.0         # вес пилота = n / (n + PRIOR_STRENGTH)
HISTORY_SHRINK = 5.0          # малые группы истории стягиваем к нулевому эффекту

MIN_AUDIENCE = 100
N_CANDIDATES = 20

PILOT_NOISE_STD = 0.80        # шум эффекта на одного абонента, документирован в environment.py
SCREEN_Z = 1.5                # ожидаемый результат скрининга должен отстоять от нуля на 1.5 ст. ошибки
SCREEN_MIN_SIZE, SCREEN_MAX_SIZE = 10, 200

N_REFINE = 6                  # уточнений по умолчанию, пока доля выживших не прояснится
MIN_REFINE = 3
REFINE_FORECAST_AFTER = 6     # прогноз числа выживших — после стольких скринингов
REFINE_DEADZONE = 2           # «заметно» больше или меньше 10 выживших ячеек
MAX_REFINES_PER_CELL = 2
REFINE_MIN_SIZE, REFINE_MAX_SIZE = 150, 200
REFINE_BUDGET_SHARE = 0.25    # доля бюджета, которую готовы отдать на уточняющие пилоты

MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
MERGE_SIMILARITY = 0.5        # объединяем ячейки, чьи эффекты на абонента различаются не более чем в 2 раза

LLM_MODEL = "gpt-5.4-mini"
LLM_URL = "https://api.openai.com/v1/chat/completions"
PROFILES_FILE = "tariff_profiles.json"   # профили от LLM, заранее собранные make_tariff_profiles.py
LLM_TIMEOUT = 60              # сек на единственный запрос — с большим запасом до лимита в 10 минут
TIE_TOLERANCE = 0.10          # LLM выбирает только среди целевых тарифов с оценкой не ниже 90% от лучшей
PROFILE_LEVELS = {"low": 0.0, "medium": 0.5, "high": 1.0}
SEGMENT_LEVELS = {"data_segment": {"NON_USER": 0.0, "LITE": 0.5, "HEAVY": 1.0},
                  "call_segment": {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}}


class Agent:
    def __init__(self):
        self.candidates = []

    def act(self, env):
        self.candidates = []
        llm = self._start_llm()
        try:
            return self._act(env, llm)
        except Exception as exc:
            print(f"[agent] сбой {type(exc).__name__}: {exc} — возвращаю найденные кандидаты")
            return self._fallback(env)

    # ------------------------------------------------------------------ план

    def _act(self, env, llm=None):
        self.candidates = self._build_candidates(env)
        if not self.candidates:
            return []
        channels = env.channels

        # Фаза 1: скрининг. Проверяем кандидатов по порядку, пока на уточнение
        # остаётся столько пилотов, сколько ему нужно при текущей доле выживших.
        screen_channel = min(channels, key=lambda ch: channels[ch]["cost_per_contact"])
        total_pilots = env.pilots_left
        for cand in self.candidates:
            if env.pilots_left <= self._refine_target(total_pilots):
                break
            self._screen(env, cand, screen_channel)

        # Фаза 2: уточнение на лучших выживших, в канале будущей кампании.
        n_refine = env.pilots_left
        per_pilot_cap = env.remaining_budget * REFINE_BUDGET_SHARE / max(n_refine, 1)
        plan = self._plan(self._alive(), env,
                          budget=env.remaining_budget * (1 - REFINE_BUDGET_SHARE),
                          contacts=env.remaining_contacts - n_refine * REFINE_MAX_SIZE)
        planned = {id(c): p["channel"] for p in plan for c in p["members"]}
        queue = [c for p in plan for c in p["members"]] + sorted(
            (c for c in self._alive() if id(c) not in planned),
            key=lambda c: self._estimate(c) * c["avg_arpu"] * c["audience"], reverse=True)

        # Первый круг — по пилоту на ячейку; если пилоты остались (выживших мало),
        # второй круг по тем же ячейкам в порядке ценности.
        refined = Counter()
        for round_ in range(MAX_REFINES_PER_CELL):
            for cand in queue:
                if env.pilots_left <= 0:
                    break
                if refined[cand["cell"]] > round_:
                    continue
                n = int(np.clip(cand["audience"] // 2, REFINE_MIN_SIZE, REFINE_MAX_SIZE))
                channel = self._refine_channel(cand, planned.get(id(cand)), per_pilot_cap, env)
                cost = channels[channel]["cost_per_contact"]
                if cost > 0:
                    n = min(n, int(per_pilot_cap // cost))
                if self._pilot(env, cand, channel, n):
                    refined[cand["cell"]] += 1

        # Если уточнять больше нечего, оставшиеся пилоты не простаивают — скринингуем дальше.
        for cand in self.candidates:
            if env.pilots_left <= 0:
                break
            if cand["screen_ratio"] is None and not cand["pilots"]:
                self._screen(env, cand, screen_channel)

        # Финал: каналы и кампании по итоговым оценкам; LLM — только тай-брейкер.
        plan = self._plan(self._alive(), env, env.remaining_budget, env.remaining_contacts,
                          merge=True, profiles=self._collect_llm(llm))
        return [self._campaign(p) for p in plan]

    def _alive(self):
        """Кандидаты, прошедшие скрининг (observed_lift_ratio > 0) с положительной оценкой."""
        return [c for c in self.candidates
                if c["screen_ratio"] is not None and c["screen_ratio"] > 0
                and self._estimate(c) > 0]

    # -------------------------------------------------------------- история

    @staticmethod
    def _load_history():
        for path in (Path(__file__).resolve().parent / "data" / "change_tariff.csv",
                     Path("data") / "change_tariff.csv"):
            if path.exists():
                return pd.read_csv(path)
        return None

    def _historical_lift(self):
        """(from, to, arpu_segment) -> ожидаемый относительный эффект до множителя канала."""
        change = self._load_history()
        if change is None or change.empty:
            return None
        df = change[change["AVG_ARPU_PREV_3M"] >= 100].copy()
        df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS,
                                    labels=ARPU_LABELS).astype(str)
        df["arpu_change_pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"])
                                 / df["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        keys = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
        g = (df.groupby(keys, observed=True)
             .agg(pct=("arpu_change_pct", "mean"), count=("arpu_change_pct", "size"))
             .reset_index())
        # группа из 3 переходов с +200% — скорее шум, чем закономерность
        g["pct"] = g["pct"] * g["count"] / (g["count"] + HISTORY_SHRINK)
        total = g.groupby(["tariff_plan_code_from", "arpu_segment"])["count"].transform("sum")
        g["conversion"] = g["count"] / total
        g["lift"] = g["pct"] * g["conversion"]
        return g

    def _build_candidates(self, env):
        profile = env.customer_profile
        # поведение ячейки (доля интернета и звонков, 0..1) — нужно только тай-брейкеру LLM
        levels = profile.assign(**{
            f"{dim}_level": profile[dim].map(mapping) for dim, mapping in SEGMENT_LEVELS.items()})
        cells = (levels.groupby(["current_tariff", "arpu_segment"], observed=True)
                 .agg(audience=("ID_NUMBER", "size"), avg_arpu=("predicted_arpu", "mean"),
                      data_level=("data_segment_level", "mean"),
                      call_level=("call_segment_level", "mean"))
                 .reset_index())
        cells = cells[cells["audience"] >= MIN_AUDIENCE]
        known = set(env.tariffs["tariff_plan_code"])

        history = self._historical_lift()
        if history is not None:
            cand = cells.merge(history, left_on=["current_tariff", "arpu_segment"],
                               right_on=["tariff_plan_code_from", "arpu_segment"])
            cand = cand.rename(columns={"tariff_plan_code_to": "target_tariff", "lift": "prior"})
        else:
            # без истории: апселл на более дорогие тарифы с нулевым приором — всё решат пилоты
            price = env.tariffs.set_index("tariff_plan_code")["price_tariff"]
            cand = cells.merge(pd.DataFrame({"target_tariff": sorted(known)}), how="cross")
            cand = cand[cand["target_tariff"].map(price) > cand["current_tariff"].map(price)]
            cand["prior"] = 0.0

        cand = cand[(cand["target_tariff"] != cand["current_tariff"])
                    & cand["target_tariff"].isin(known)].copy()
        cand["historical_score"] = cand["prior"] * cand["avg_arpu"]
        # без истории все score = 0, тогда первыми идут крупные ячейки
        cand = cand.sort_values(["historical_score", "audience"], ascending=False)
        if history is not None:
            cand = cand[cand["historical_score"] > 0]
        cand = cand.head(N_CANDIDATES)

        return [{
            "cell": (row.current_tariff, row.arpu_segment),
            "current_tariff": row.current_tariff,
            "arpu_segment": row.arpu_segment,
            "target_tariff": row.target_tariff,
            "audience": int(row.audience),
            "avg_arpu": float(row.avg_arpu),
            "prior": float(row.prior),
            "behavior": {"data": float(row.data_level), "calls": float(row.call_level)},
            "pilots": [],
            "screen_ratio": None,
        } for row in cand.itertuples()]

    # --------------------------------------------------------------- пилоты

    @staticmethod
    def _screen_size(cand, multiplier):
        """
        Пилот наблюдает эффект × множитель канала + шум σ/√n. Размер берём таким,
        чтобы ожидаемый результат отстоял от нуля на SCREEN_Z стандартных ошибок:
        около нуля (граница прибыли и убытка) выборка растёт, при явно высоком
        или явно отрицательном эффекте — уменьшается до подтверждающей.
        """
        signal = abs(cand["prior"]) * multiplier
        if signal <= 0:
            return SCREEN_MAX_SIZE
        n = (SCREEN_Z * PILOT_NOISE_STD / signal) ** 2
        return int(np.clip(round(n), SCREEN_MIN_SIZE, SCREEN_MAX_SIZE))

    def _screen(self, env, cand, channel):
        n = self._screen_size(cand, env.channels[channel]["conversion_multiplier"])
        if self._pilot(env, cand, channel, n):
            cand["screen_ratio"] = cand["pilots"][-1]["ratio"]

    def _refine_target(self, total_pilots):
        """
        Сколько пилотов оставить на уточнение. По доле выживших на скрининге
        прогнозируем, сколько ячеек выживет при скрининге по умолчанию. Если
        заметно больше 10 — уточнений меньше, а скринингов больше. Если выживших
        мало, скрининг не сокращаем: мало выживших чаще означает шум маленьких
        пилотов, чем плохих кандидатов, и ранняя остановка стоит охвата.
        """
        screened = [c for c in self.candidates if c["screen_ratio"] is not None]
        if len(screened) < REFINE_FORECAST_AFTER:
            return N_REFINE
        alive_cells = {c["cell"] for c in screened if c["screen_ratio"] > 0}
        projected = len(alive_cells) * (total_pilots - N_REFINE) / len(screened)
        surplus = projected - MAX_CAMPAIGNS
        if surplus < REFINE_DEADZONE:
            return N_REFINE
        return int(max(round(N_REFINE - surplus), MIN_REFINE))

    def _pilot(self, env, cand, channel, n):
        if env.pilots_left <= 0:
            return False
        try:
            res = env.run_pilot(target_tariff=cand["target_tariff"], channel=channel,
                                n_customers=n, filter_arpu_segment=cand["arpu_segment"],
                                filter_current_tariff=cand["current_tariff"])
        except (RuntimeError, ValueError):
            return False
        ratio = res.get("observed_lift_ratio")
        if ratio is None or not np.isfinite(ratio):
            return False
        cand["pilots"].append({
            "n": int(res.get("n_customers", n)),
            "mult": env.channels[channel]["conversion_multiplier"],
            "ratio": float(ratio),
        })
        return True

    def _refine_channel(self, cand, planned, cap, env):
        """Канал будущей кампании, если пилот в нём укладывается в лимит расходов."""
        channels = env.channels
        limit = min(cap, env.remaining_budget)
        est = self._estimate(cand)
        by_value = sorted(channels, key=lambda ch: est * channels[ch]["conversion_multiplier"]
                          * cand["avg_arpu"] - channels[ch]["cost_per_contact"], reverse=True)
        options = ([planned] if planned else []) + by_value
        for ch in options:
            if REFINE_MIN_SIZE * channels[ch]["cost_per_contact"] <= limit:
                return ch
        return min(channels, key=lambda ch: channels[ch]["cost_per_contact"])

    # --------------------------------------------------------------- оценка

    @staticmethod
    def _estimate(cand):
        """Шаг 5: история и пилоты, вес пилотов растёт с числом абонентов в них."""
        n = sum(p["n"] for p in cand["pilots"])
        if n <= 0:
            return cand["prior"]
        pilot_ratio = sum(p["n"] * p["ratio"] / p["mult"] for p in cand["pilots"]) / n
        w = n / (n + PRIOR_STRENGTH)
        return w * pilot_ratio + (1 - w) * cand["prior"]

    # ------------------------------------------------------------------ LLM

    @staticmethod
    def _load_tariff_descriptions():
        for path in (Path(__file__).resolve().parent / "tariff_dictionary.csv",
                     Path("tariff_dictionary.csv")):
            if path.exists():
                df = pd.read_csv(path)
                if "description" in df.columns:
                    return dict(zip(df["tariff_plan_code"], df["description"].astype(str)))
        return {}

    @staticmethod
    def _load_cached_profiles():
        """Профили из tariff_profiles.json: ответ LLM зафиксирован, решения воспроизводимы."""
        for path in (Path(__file__).resolve().parent / PROFILES_FILE, Path(PROFILES_FILE)):
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                return Agent._parse_profiles(raw.get("profiles", {}))
        return {}

    def _start_llm(self):
        """
        Профили тарифов для тай-брейкера. Сначала — из tariff_profiles.json. Если
        файла нет — один батчевый запрос к OpenAI в фоне, пока идут пилоты.
        """
        try:
            cached = self._load_cached_profiles()
            if cached:
                return {"profiles": cached, "source": PROFILES_FILE}
            key = os.environ.get("OPENAI_API_KEY")
            descriptions = self._load_tariff_descriptions() if key else {}
            if not descriptions:
                return None
            result = {}

            def worker():
                try:
                    result["profiles"] = self._request_profiles(key, descriptions)
                except Exception as exc:
                    result["error"] = exc

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            return {"thread": thread, "result": result, "started": time.monotonic()}
        except Exception:
            return None

    def _collect_llm(self, llm):
        """Живой запрос ждём не дольше LLM_TIMEOUT от старта; при любой проблеме — без LLM."""
        if llm is None:
            return {}
        try:
            if "profiles" in llm:
                profiles, source = llm["profiles"], llm["source"]
            else:
                thread, result = llm["thread"], llm["result"]
                thread.join(max(LLM_TIMEOUT - (time.monotonic() - llm["started"]), 0))
                if thread.is_alive():
                    print(f"[agent] LLM не ответил за {LLM_TIMEOUT} с — продолжаю без него")
                    return {}
                if "error" in result:
                    err = result["error"]
                    print(f"[agent] LLM недоступен ({type(err).__name__}: {err}) — продолжаю без него")
                    return {}
                profiles, source = result.get("profiles") or {}, f"живой запрос к {LLM_MODEL}"
            # отпечаток полей, влияющих на решение, — чтобы сравнивать ответы между запусками
            decisive = {code: (p["data"], p["calls"]) for code, p in sorted(profiles.items())}
            digest = hashlib.sha256(json.dumps(decisive).encode()).hexdigest()[:10]
            print(f"[agent] LLM-профили {len(profiles)} тарифов: {source} (отпечаток {digest})")
            return profiles
        except Exception:
            return {}

    @staticmethod
    def _parse_profiles(raw, codes=None):
        """{код: {"data": "low|medium|high", "calls": ...}} -> числовые уровни 0 / 0.5 / 1."""
        profiles = {}
        for code, item in raw.items():
            if (codes is not None and code not in codes) or not isinstance(item, dict):
                continue
            data = PROFILE_LEVELS.get(str(item.get("data", "")).lower())
            calls = PROFILE_LEVELS.get(str(item.get("calls", "")).lower())
            if data is not None and calls is not None:
                profiles[code] = {"data": data, "calls": calls,
                                  "summary": str(item.get("summary", ""))}
        return profiles

    @staticmethod
    def _request_profiles(key, descriptions):
        return Agent._parse_profiles(Agent._ask_llm(key, descriptions), descriptions)

    @staticmethod
    def _ask_llm(key, descriptions):
        """Один батчевый запрос: описания всех тарифов -> сырой JSON с профилями."""
        listing = "\n".join(f"{code}: {text}" for code, text in descriptions.items())
        prompt = (
            "Ниже тарифы мобильного оператора. Для каждого определи, на что он ориентирован. "
            "Верни JSON-объект, где ключ — код тарифа, а значение — объект с полями "
            '"data" и "calls" (насколько тариф рассчитан на интернет и на звонки: "low", '
            '"medium" или "high"), "price" ("budget", "mid" или "premium") и "summary" '
            "(профиль в 3–6 словах).\n\n" + listing)
        body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Ты аналитик тарифов оператора связи. Отвечай только JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            # уменьшает разброс, но не устраняет его: метки отдельных тарифов между
            # вызовами всё равно расходятся, поэтому решения берутся из tariff_profiles.json
            "temperature": 0,
        }
        request = urllib.request.Request(
            LLM_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw = json.loads(payload["choices"][0]["message"]["content"])
        if isinstance(raw.get("tariffs"), dict):
            raw = raw["tariffs"]
        return raw

    # --------------------------------------------------------- сборка плана

    def _plan(self, pool, env, budget, contacts, merge=False, profiles=None):
        """
        Шаги 6–7 плюс объединение ячеек. Юнит — будущая кампания: одна ячейка
        или несколько с общим сегментом и целевым тарифом. Слияние принимаем,
        только если оно повышает ожидаемый результат плана (например, освобождает
        слот под ещё одну ячейку сверх лимита в 10 кампаний). Предварительный план
        перед уточнением строим без слияний, чтобы не менять выбор уточняемых ячеек.
        """
        cost = {ch: spec["cost_per_contact"] for ch, spec in env.channels.items()}
        units = self._cell_units(pool, env, profiles)
        plan, value = self._assign(units, cost, budget, contacts)
        while merge:
            best = None
            for a, b in combinations(units, 2):
                merged = self._merge(a, b)
                if merged is None:
                    continue
                trial = [u for u in units if u is not a and u is not b] + [merged]
                trial_plan, trial_value = self._assign(trial, cost, budget, contacts)
                if trial_value > value and (best is None or trial_value > best[0]):
                    best = (trial_value, trial, trial_plan)
            if best is None:
                break
            value, units, plan = best
        return plan

    def _cell_units(self, pool, env, profiles=None):
        """По одному юниту на ячейку — абонент засчитывается один раз, лучший целевой тариф."""
        options = defaultdict(list)
        for cand in pool:
            est = self._estimate(cand)
            if est <= 0:
                continue
            size = min(cand["audience"], MAX_CUSTOMERS_PER_CAMPAIGN)
            lift = est * cand["avg_arpu"]
            net = {ch: size * (lift * spec["conversion_multiplier"] - spec["cost_per_contact"])
                   for ch, spec in env.channels.items()}
            if max(net.values()) <= 0:
                continue
            options[cand["cell"]].append({"members": [cand], "segment": cand["arpu_segment"],
                                          "target": cand["target_tariff"], "size": size,
                                          "net": net, "lifts": [lift]})
        units = []
        for cell_options in options.values():
            best = max(cell_options, key=lambda u: max(u["net"].values()))
            if profiles:
                best = self._tie_break(best, cell_options, profiles)
            units.append(best)
        return units

    def _tie_break(self, best, options, profiles):
        """LLM-профиль решает только среди целевых тарифов с оценкой не ниже 90% от лучшей."""
        top = self._estimate(best["members"][0])
        close = [u for u in options
                 if self._estimate(u["members"][0]) >= (1 - TIE_TOLERANCE) * top]
        if len(close) < 2:
            return best
        scores = [self._profile_match(u["members"][0], profiles) for u in close]
        if any(score is None for score in scores):
            return best
        chosen = max(zip(scores, close), key=lambda sc: (sc[0], max(sc[1]["net"].values())))[1]
        if chosen is not best:
            cand = chosen["members"][0]
            print(f"[agent] LLM: {cand['current_tariff']} {cand['arpu_segment']} -> "
                  f"{cand['target_tariff']} вместо {best['target']} (оценки близки, профиль ближе)")
        return chosen

    @staticmethod
    def _profile_match(cand, profiles):
        """Насколько профиль целевого тарифа похож на поведение ячейки: 0 — идеально, −2 — хуже всего."""
        profile = profiles.get(cand["target_tariff"])
        behavior = cand.get("behavior")
        if not profile or not behavior or any(pd.isna(v) for v in behavior.values()):
            return None
        return -(abs(profile["data"] - behavior["data"]) + abs(profile["calls"] - behavior["calls"]))

    @staticmethod
    def _merge(a, b):
        """Один сегмент и целевой тариф, похожий эффект, не больше 5000 абонентов."""
        if (a["segment"], a["target"]) != (b["segment"], b["target"]):
            return None
        if a["size"] + b["size"] > MAX_CUSTOMERS_PER_CAMPAIGN:
            return None
        lifts = a["lifts"] + b["lifts"]
        if min(lifts) < MERGE_SIMILARITY * max(lifts):
            return None
        return {"members": a["members"] + b["members"], "segment": a["segment"],
                "target": a["target"], "size": a["size"] + b["size"],
                "net": {ch: a["net"][ch] + b["net"][ch] for ch in a["net"]}, "lifts": lifts}

    @staticmethod
    def _assign(units, cost, budget, contacts):
        """
        Отбираем до 10 юнитов, стартуя с самого дешёвого канала, затем повышаем
        каналы в порядке «прирост на единицу затрат», пока позволяет бюджет. Без
        ограничения бюджета это ровно максимум чистого результата на контакт.
        """
        chosen, contacts_left, budget_left = [], contacts, budget
        for unit in sorted(units, key=lambda u: max(u["net"].values()), reverse=True):
            if len(chosen) >= MAX_CAMPAIGNS or contacts_left <= 0:
                break
            size = min(unit["size"], contacts_left)
            net = {ch: v * size / unit["size"] for ch, v in unit["net"].items()}
            affordable = [ch for ch in cost if net[ch] > 0 and size * cost[ch] <= budget_left]
            if not affordable:
                continue
            base = min(affordable, key=lambda ch: (cost[ch], -net[ch]))
            chosen.append({"members": unit["members"], "size": size, "net": net, "channel": base})
            contacts_left -= size
            budget_left -= size * cost[base]

        while True:
            best = None
            for e in chosen:
                cur = e["channel"]
                for ch in cost:
                    gain = e["net"][ch] - e["net"][cur]
                    extra = e["size"] * (cost[ch] - cost[cur])
                    if gain <= 0 or extra > budget_left:
                        continue
                    ratio = gain / extra if extra > 0 else np.inf
                    if best is None or ratio > best[0]:
                        best = (ratio, e, ch, extra)
            if best is None:
                break
            _, e, ch, extra = best
            e["channel"] = ch
            budget_left -= extra

        for e in chosen:
            e["value"] = e["net"][e["channel"]]
        chosen.sort(key=lambda e: e["value"], reverse=True)
        return chosen, sum(e["value"] for e in chosen)

    @staticmethod
    def _campaign(entry):
        members = entry["members"]
        tariffs = [c["current_tariff"] for c in members]
        head = members[0]
        return {
            "campaign_name": f"{'+'.join(tariffs)}_{head['arpu_segment']}_to_{head['target_tariff']}",
            "filter_arpu_segment": head["arpu_segment"],
            "filter_current_tariff": ";".join(tariffs),
            "target_tariff": head["target_tariff"],
            "channel": entry["channel"],
        }

    # ------------------------------------------------------------- страховка

    def _fallback(self, env):
        """Шаг 8: лучшее из найденного до сбоя, вместо пустого списка."""
        piloted = [c for c in self.candidates if c["pilots"]
                   and (c["screen_ratio"] is None or c["screen_ratio"] > 0)]
        pool = piloted or self.candidates
        try:
            plan = self._plan(pool, env, env.remaining_budget, env.remaining_contacts, merge=True)
            return [self._campaign(p) for p in plan]
        except Exception:
            pass
        try:
            free = min(env.channels, key=lambda ch: env.channels[ch]["cost_per_contact"])
            campaigns, used = [], set()
            for cand in sorted(pool, key=self._estimate, reverse=True):
                if self._estimate(cand) > 0 and cand["cell"] not in used:
                    used.add(cand["cell"])
                    campaigns.append(self._campaign({"members": [cand], "channel": free}))
            return campaigns[:MAX_CAMPAIGNS]
        except Exception:
            return []
