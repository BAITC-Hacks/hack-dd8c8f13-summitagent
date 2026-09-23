"""
Агент для кейса Beeline Tariff Marketing Campaigns.

Стратегия — двухфазная разведка с историческим приоритетом:
  1. Из data/change_tariff.csv строим априорную оценку эффекта перехода
     (from, to, arpu_segment): среднее arpu_change_pct × доля таких переходов.
  2. Скрещиваем с аудиторией: кандидаты (current_tariff, arpu_segment, target)
     с аудиторией >= 100, historical_score = lift × средний predicted_arpu.
  3. Скрининг: дешёвые пилоты на 30 абонентов в самом дешёвом канале.
  4. Уточнение: пилоты на 150–200 абонентов в том канале, который планируем
     использовать в финале, для лучших выживших после скрининга.
  5. Оценка = вес_пилота × пилот + (1 − вес_пилота) × история,
     вес_пилота = n / (n + 50), n — все абоненты пилотов кандидата.
  6–7. Выбираем канал и жадно собираем до 10 кампаний в пределах бюджета и охвата.
  8. Любой сбой -> возвращаем лучшее из уже найденного.

Все оценки ведутся в «базовых» единицах: относительный эффект до множителя
канала. Пилот в канале с множителем m наблюдает m × эффект + шум, поэтому
его результат делим на m, прежде чем смешивать с историей.

LLM не используется: решения принимаются по данным и пилотам.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]

PRIOR_STRENGTH = 50.0         # вес пилота = n / (n + PRIOR_STRENGTH)
HISTORY_SHRINK = 5.0          # малые группы истории стягиваем к нулевому эффекту

MIN_AUDIENCE = 100
N_CANDIDATES = 20
N_REFINE = 6
SCREEN_SIZE = 30
REFINE_MIN_SIZE, REFINE_MAX_SIZE = 150, 200
REFINE_BUDGET_SHARE = 0.25    # доля бюджета, которую готовы отдать на уточняющие пилоты

MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000


class Agent:
    def __init__(self):
        self.candidates = []

    def act(self, env):
        self.candidates = []
        try:
            return self._act(env)
        except Exception as exc:
            print(f"[agent] сбой {type(exc).__name__}: {exc} — возвращаю найденные кандидаты")
            return self._fallback(env)

    # ------------------------------------------------------------------ план

    def _act(self, env):
        self.candidates = self._build_candidates(env)
        if not self.candidates:
            return []
        channels = env.channels

        # Фаза 1: скрининг. Пилотов 20, поэтому под скрининг идёт всё, кроме уточнения.
        screen_channel = min(channels, key=lambda ch: channels[ch]["cost_per_contact"])
        n_screen = max(env.pilots_left - N_REFINE, 0)
        for cand in self.candidates[:n_screen]:
            if self._pilot(env, cand, screen_channel, SCREEN_SIZE):
                cand["screen_ratio"] = cand["pilots"][-1]["ratio"]

        # Фаза 2: уточнение на лучших выживших, в канале будущей кампании.
        per_pilot_cap = (env.remaining_budget * REFINE_BUDGET_SHARE
                         / max(min(N_REFINE, env.pilots_left), 1))
        plan = self._plan(self._alive(), env,
                          budget=env.remaining_budget * (1 - REFINE_BUDGET_SHARE),
                          contacts=env.remaining_contacts - N_REFINE * REFINE_MAX_SIZE)
        planned = {id(p["cand"]): p["channel"] for p in plan}
        queue = [p["cand"] for p in plan] + sorted(
            (c for c in self._alive() if id(c) not in planned),
            key=lambda c: self._estimate(c) * c["avg_arpu"] * c["audience"], reverse=True)

        refined_cells = set()
        for cand in queue:
            if env.pilots_left <= 0 or len(refined_cells) >= N_REFINE:
                break
            if cand["cell"] in refined_cells:
                continue
            n = int(np.clip(cand["audience"] // 2, REFINE_MIN_SIZE, REFINE_MAX_SIZE))
            channel = self._refine_channel(cand, planned.get(id(cand)), per_pilot_cap, env)
            cost = channels[channel]["cost_per_contact"]
            if cost > 0:
                n = min(n, int(per_pilot_cap // cost))
            if self._pilot(env, cand, channel, n):
                refined_cells.add(cand["cell"])

        # Финал: каналы и кампании по итоговым оценкам.
        plan = self._plan(self._alive(), env, env.remaining_budget, env.remaining_contacts)
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
        cells = (profile.groupby(["current_tariff", "arpu_segment"], observed=True)
                 .agg(audience=("ID_NUMBER", "size"), avg_arpu=("predicted_arpu", "mean"))
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
            "pilots": [],
            "screen_ratio": None,
        } for row in cand.itertuples()]

    # --------------------------------------------------------------- пилоты

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

    # --------------------------------------------------------- сборка плана

    def _plan(self, pool, env, budget, contacts):
        """
        Шаги 6–7. Каждой кампании — лучший канал на контакт, пока хватает бюджета.
        Сначала отбираем кампании (по одной на ячейку — абонент засчитывается
        один раз), стартуя с самого дешёвого канала. Затем повышаем каналы
        в порядке «прирост на единицу затрат», пока позволяет бюджет. Без
        ограничения бюджета это ровно максимум чистого результата на контакт.
        """
        channels = env.channels
        cost = {ch: spec["cost_per_contact"] for ch, spec in channels.items()}

        entries = []
        for cand in pool:
            est = self._estimate(cand)
            if est <= 0:
                continue
            net = {ch: est * spec["conversion_multiplier"] * cand["avg_arpu"] - cost[ch]
                   for ch, spec in channels.items()}
            if max(net.values()) <= 0:
                continue
            entries.append({"cand": cand, "net": net})
        entries.sort(key=lambda e: max(e["net"].values())
                     * min(e["cand"]["audience"], MAX_CUSTOMERS_PER_CAMPAIGN), reverse=True)

        chosen, used_cells = [], set()
        contacts_left, budget_left = contacts, budget
        for e in entries:
            if len(chosen) >= MAX_CAMPAIGNS or contacts_left <= 0:
                break
            if e["cand"]["cell"] in used_cells:
                continue
            size = min(e["cand"]["audience"], MAX_CUSTOMERS_PER_CAMPAIGN, contacts_left)
            affordable = [ch for ch in channels if e["net"][ch] > 0 and size * cost[ch] <= budget_left]
            if not affordable:
                continue
            base = min(affordable, key=lambda ch: (cost[ch], -e["net"][ch]))
            e.update(size=size, channel=base)
            chosen.append(e)
            used_cells.add(e["cand"]["cell"])
            contacts_left -= size
            budget_left -= size * cost[base]

        while True:
            best = None
            for e in chosen:
                cur = e["channel"]
                for ch in channels:
                    gain = e["size"] * (e["net"][ch] - e["net"][cur])
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
            e["value"] = e["size"] * e["net"][e["channel"]]
        chosen.sort(key=lambda e: e["value"], reverse=True)
        return chosen

    @staticmethod
    def _campaign(entry):
        cand = entry["cand"]
        return {
            "campaign_name": f"{cand['current_tariff']}_{cand['arpu_segment']}_to_{cand['target_tariff']}",
            "filter_arpu_segment": cand["arpu_segment"],
            "filter_current_tariff": cand["current_tariff"],
            "target_tariff": cand["target_tariff"],
            "channel": entry["channel"],
        }

    # ------------------------------------------------------------- страховка

    def _fallback(self, env):
        """Шаг 8: лучшее из найденного до сбоя, вместо пустого списка."""
        piloted = [c for c in self.candidates if c["pilots"]
                   and (c["screen_ratio"] is None or c["screen_ratio"] > 0)]
        pool = piloted or self.candidates
        try:
            plan = self._plan(pool, env, env.remaining_budget, env.remaining_contacts)
            return [self._campaign(p) for p in plan]
        except Exception:
            pass
        try:
            free = min(env.channels, key=lambda ch: env.channels[ch]["cost_per_contact"])
            campaigns, used = [], set()
            for cand in sorted(pool, key=self._estimate, reverse=True):
                if self._estimate(cand) > 0 and cand["cell"] not in used:
                    used.add(cand["cell"])
                    campaigns.append(self._campaign({"cand": cand, "channel": free}))
            return campaigns[:MAX_CAMPAIGNS]
        except Exception:
            return []
