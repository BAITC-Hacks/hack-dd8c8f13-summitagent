"""
Среда для агента: Beeline Tariff Marketing Campaigns Case.

Ваш агент работает с этой средой. У вас она работает на МОК-модели эффектов,
построенной из выданного вам data/change_tariff.csv. На судействе тот же самый
класс запускается на РЕАЛЬНОЙ модели поведения целевой аудитории, которой у вас
нет — и она заметно отличается от того, что показывает историческая выборка.

Поэтому история из data/ — это ваше АПРИОРНОЕ знание ("как обычно бывает"), а
чтобы узнать, как ведёт себя ИМЕННО ЭТА аудитория, нужно запускать пилоты.

Пилот — это маленькая реальная кампания:
  * стоит денег (контакты × стоимость канала) и расходует лимит охвата;
  * контактированные в пилоте абоненты идут в общий зачёт (их реально
    обработали), поэтому разведка — не бесплатная репетиция, а часть стратегии;
  * возвращает ЗАШУМЛЁННЫЙ результат: на маленькой выборке разброс огромен
    (стандартное отклонение эффекта на одного абонента ≈ 0.80 при типичном
    эффекте 0.2-0.45), так что пилот на 30 клиентах почти ничего не говорит,
    а пилот на 200 стоит заметных денег. Это и есть ваш компромисс.

ЧЕСТНАЯ ИГРА. Агенту доступно ровно то, что перечислено в AgentEnvironment:
профиль аудитории, справочники, остатки лимитов, пилоты и их история. Модель
эффектов агенту недоступна — она специально спрятана в замыкании. Попытки
достать её обходными путями (ковыряние в `__closure__`, `gc`, чтение файлов
организатора и т.п.) считаются нарушением: результат аннулируется. Смысл кейса
именно в том, чтобы добывать знание пилотами.

Интерфейс агента (файл agent.py, который вы сдаёте):

    class Agent:
        def act(self, env) -> list[dict]:
            # env.customer_profile      — DataFrame аудитории
            # env.tariffs               — справочник тарифов
            # env.channels              — стоимость/множитель каналов
            # env.remaining_budget, env.remaining_contacts, env.pilots_left
            # env.run_pilot(...)        — запустить пилот, получить шумный результат
            # env.pilot_history         — все ваши пилоты и их результаты
            # вернуть список финальных кампаний (до 10 штук)
            return [{"campaign_name": "...", "filter_arpu_segment": "HIGH",
                     "target_tariff": "tariff_8", "channel": "sms"}]
"""

import numpy as np
import pandas as pd

MAX_PILOTS = 20
MAX_PILOT_CUSTOMERS = 200
MIN_PILOT_CUSTOMERS = 10

# Измеренный по реальным данным разброс относительного эффекта на одного абонента.
# Пилот возвращает среднее по выборке, поэтому его шум = PER_CUSTOMER_STD / sqrt(n).
PER_CUSTOMER_STD = 0.804


class AgentEnvironment:
    """
    То, что видит агент. Здесь НЕТ модели эффектов и НЕТ генератора шума —
    они живут в замыкании фабрики make_environment() и недоступны как атрибуты.
    """

    def __init__(self, customer_profile, dict_tariff, channels,
                 total_budget, max_total_contacts):
        self.customer_profile = customer_profile
        self.tariffs = dict_tariff
        self.channels = channels

        self.total_budget = total_budget
        self.max_total_contacts = max_total_contacts
        self.remaining_budget = total_budget
        self.remaining_contacts = max_total_contacts
        self.pilots_left = MAX_PILOTS
        self.pilot_history = []

    def run_pilot(self, *args, **kwargs):  # подменяется в make_environment()
        raise NotImplementedError("Среду нужно создавать через make_environment()")


def _apply_filters(profile, campaign):
    result = profile
    for col, key in [("arpu_segment", "filter_arpu_segment"),
                      ("data_segment", "filter_data_segment"),
                      ("call_segment", "filter_call_segment")]:
        value = campaign.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            result = result[result[col] == value]
    tariffs = campaign.get("filter_current_tariff")
    if tariffs is not None and not (isinstance(tariffs, float) and np.isnan(tariffs)):
        wanted = [t.strip() for t in str(tariffs).split(";") if t.strip()]
        result = result[result["current_tariff"].isin(wanted)]
    return result


def make_environment(customer_profile, impact_model, dict_tariff, channels,
                     total_budget, max_total_contacts, fallback_predict, seed=None):
    """
    Создаёт среду. Возвращает (env, internals):
      env       — то, что отдаётся агенту (без секретов);
      internals — то, что нужно организатору для скоринга (проведённые пилоты).

    Модель эффектов, fallback-правило и генератор шума захвачены замыканием и
    не лежат атрибутами на env, чтобы агент не мог прочитать ответ напрямую.
    """
    _model = impact_model
    _fallback = fallback_predict
    _fallback_conversion = impact_model["conversion_rate"].median()
    _rng = np.random.default_rng(seed)
    _pilot_campaigns = []

    env = AgentEnvironment(customer_profile.copy(), dict_tariff.copy(), dict(channels),
                           total_budget, max_total_contacts)

    def _true_lift_ratio(segment, target_tariff, channel):
        multiplier = env.channels[channel]["conversion_multiplier"]
        combos = segment[["current_tariff", "arpu_segment"]].drop_duplicates()
        im = _model[_model["tariff_plan_code_to"] == target_tariff]
        combos = combos.merge(im, left_on=["current_tariff", "arpu_segment"],
                               right_on=["tariff_plan_code_from", "arpu_segment"], how="left")
        missing = combos["arpu_change_pct"].isna()
        if missing.any():
            fb = combos.loc[missing].apply(
                lambda r: _fallback(r["current_tariff"], target_tariff, r["arpu_segment"],
                                     env.tariffs, _fallback_conversion),
                axis=1)
            combos.loc[missing, "arpu_change_pct"] = [f[0] for f in fb]
            combos.loc[missing, "conversion_rate"] = [f[1] for f in fb]
        combos["ratio"] = combos["arpu_change_pct"] * (combos["conversion_rate"] * multiplier).clip(upper=1.0)
        merged = segment.merge(combos[["current_tariff", "arpu_segment", "ratio"]],
                                on=["current_tariff", "arpu_segment"], how="left")
        return merged["ratio"].fillna(0.0)

    def run_pilot(target_tariff, channel, n_customers=100,
                  filter_arpu_segment=None, filter_data_segment=None,
                  filter_call_segment=None, filter_current_tariff=None):
        """
        Запускает пилотную кампанию на малой выборке.

        Возвращает dict с зашумлённым наблюдаемым эффектом:
            observed_lift_ratio — наблюдаемый относительный эффект (с шумом выборки)
            observed_lift_total — наблюдаемый абсолютный прирост по выборке
            n_customers, cost   — сколько реально контактировали и сколько потратили
        """
        if env.pilots_left <= 0:
            raise RuntimeError(f"Исчерпан лимит пилотов ({MAX_PILOTS})")
        if channel not in env.channels:
            raise ValueError(f"Неизвестный канал {channel!r}")
        if target_tariff not in set(env.tariffs["tariff_plan_code"]):
            raise ValueError(f"Неизвестный тариф {target_tariff!r}")
        n_customers = int(np.clip(n_customers, MIN_PILOT_CUSTOMERS, MAX_PILOT_CUSTOMERS))

        campaign = {
            "campaign_name": f"pilot_{len(env.pilot_history) + 1}",
            "filter_arpu_segment": filter_arpu_segment,
            "filter_data_segment": filter_data_segment,
            "filter_call_segment": filter_call_segment,
            "filter_current_tariff": filter_current_tariff,
            "target_tariff": target_tariff,
            "channel": channel,
        }
        segment = _apply_filters(env.customer_profile, campaign).sort_values("ID_NUMBER")
        cost_per_contact = env.channels[channel]["cost_per_contact"]

        affordable = env.remaining_contacts
        if cost_per_contact > 0:
            affordable = min(affordable, int(env.remaining_budget // cost_per_contact))
        n_actual = int(min(n_customers, len(segment), max(affordable, 0)))
        if n_actual <= 0:
            raise RuntimeError("Недостаточно бюджета/охвата или пустой сегмент для пилота")

        # случайная подвыборка сегмента — пилот не должен быть систематически смещён
        picked = segment.sample(n=n_actual, random_state=int(_rng.integers(1 << 31)))

        noise = _rng.normal(0.0, PER_CUSTOMER_STD / np.sqrt(n_actual))
        observed_ratio = float(_true_lift_ratio(picked, target_tariff, channel).mean() + noise)

        cost = n_actual * cost_per_contact
        env.remaining_budget -= cost
        env.remaining_contacts -= n_actual
        env.pilots_left -= 1

        campaign["explicit_ids"] = picked["ID_NUMBER"].tolist()
        _pilot_campaigns.append(campaign)

        result = {
            "pilot": campaign["campaign_name"],
            "target_tariff": target_tariff,
            "channel": channel,
            "n_customers": n_actual,
            "cost": cost,
            "observed_lift_ratio": observed_ratio,
            "observed_lift_total": float(observed_ratio * picked["predicted_arpu"].sum()),
            "remaining_budget": env.remaining_budget,
            "remaining_contacts": env.remaining_contacts,
        }
        env.pilot_history.append(result)
        return result

    env.run_pilot = run_pilot

    class _Internals:
        """Организаторская часть: агенту не передаётся."""

        @staticmethod
        def executed_pilot_campaigns():
            return [dict(c) for c in _pilot_campaigns]

    return env, _Internals()
