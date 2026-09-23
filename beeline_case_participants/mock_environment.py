"""
Локальная МОК-среда для отладки агента.

    from mock_environment import make_mock_env
    env = make_mock_env()
    # дальше как на судействе: env.run_pilot(...), env.customer_profile, ...

ВАЖНО, прочитайте внимательно:

Эффекты в этой среде — ГРУБАЯ ЗАГЛУШКА, посчитанная из выданного вам
data/change_tariff.csv простым усреднением. Это НЕ та модель, по которой вас
будут оценивать, и числа здесь заведомо другие. Мок нужен ровно для одного:
чтобы вы могли запустить и отладить код агента, не угадывая интерфейс.

Не калибруйте агента под конкретные числа мока — на судействе они будут иными.
Калибруйте логику: как вы распределяете пилоты, как обновляете оценки по их
результатам и как решаете, когда хватит разведки и пора вкладываться.
"""

import numpy as np
import pandas as pd

from environment import make_environment

CHANNELS = {
    "push":        {"cost_per_contact": 0,   "conversion_multiplier": 0.50},
    "sms":         {"cost_per_contact": 4,   "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22,  "conversion_multiplier": 0.85},
    "call":        {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}
TOTAL_BUDGET = 100_000
MAX_TOTAL_CONTACTS = 15_000

ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]


def _mock_impact_model(change_tariff: pd.DataFrame) -> pd.DataFrame:
    """Заглушка: простое среднее относительного изменения ARPU по группе."""
    df = change_tariff.copy()
    df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS)
    df = df[df["AVG_ARPU_PREV_3M"] >= 100].copy()
    df["arpu_change_pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"])
                             / df["AVG_ARPU_PREV_3M"]).clip(-1, 3)

    grouped = (df.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"], observed=True)
               .agg(arpu_change_pct=("arpu_change_pct", "mean"), count=("ID_NUMBER", "size"))
               .reset_index())
    totals = (grouped.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)["count"]
              .sum().rename("total").reset_index())
    grouped = grouped.merge(totals, on=["tariff_plan_code_from", "arpu_segment"])
    grouped["conversion_rate"] = grouped["count"] / grouped["total"]
    return grouped.drop(columns=["total"])


def _mock_fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion):
    price = dict_tariff.set_index("tariff_plan_code")["price_tariff"]
    if current_tariff not in price.index or target_tariff not in price.index:
        return 0.0, fallback_conversion
    scale = max(price.median(), 1.0)
    ratio = (price[target_tariff] - price[current_tariff]) / scale
    return float(np.clip(ratio * 0.4, -1.0, 3.0)), fallback_conversion


def make_mock_env(seed=None, data_dir="data", profile_path="customer_profile.csv"):
    """Возвращает (env, internals) — как и настоящая среда на судействе."""
    change_tariff = pd.read_csv(f"{data_dir}/change_tariff.csv")
    dict_tariff = pd.read_csv(f"{data_dir}/dict_tariff.csv")
    profile = pd.read_csv(profile_path)
    return make_environment(
        customer_profile=profile,
        impact_model=_mock_impact_model(change_tariff),
        dict_tariff=dict_tariff,
        channels=CHANNELS,
        total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=_mock_fallback,
        seed=seed,
    )


if __name__ == "__main__":
    env, _ = make_mock_env(seed=1)
    print(f"Аудитория: {len(env.customer_profile):,} абонентов")
    print(f"Бюджет: {env.remaining_budget:,}  контактов: {env.remaining_contacts:,}  пилотов: {env.pilots_left}")
    res = env.run_pilot(target_tariff="tariff_8", channel="sms", n_customers=150,
                        filter_arpu_segment="MID", filter_current_tariff="tariff_4")
    print("Пилот:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
