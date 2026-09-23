"""
Правила подсчёта результата — общий модуль для участников и организаторов.

Здесь нет ничего секретного: сами ЭФФЕКТЫ переходов (impact-модель) передаются
снаружи как параметр, а тут только механика — лимиты охвата, бюджет каналов,
дедупликация абонентов и формула чистого результата.

Тот же самый код считает ваш результат на судействе, поэтому локальная проверка
через local_eval.py показывает ровно ту механику, что будет в финале. Отличаться
будут только эффекты: у вас — заглушка, на судействе — настоящие.
"""

import pandas as pd
import numpy as np

MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000   # лимит на одну кампанию
MAX_TOTAL_CONTACTS = 15000          # суммарный лимит охвата, включая пилоты

# канал -> (стоимость контакта, множитель к вероятности конверсии)
CHANNELS = {
    "push":         {"cost_per_contact": 0,   "conversion_multiplier": 0.50},
    "sms":          {"cost_per_contact": 4,   "conversion_multiplier": 0.65},
    "digital_ads":  {"cost_per_contact": 22,  "conversion_multiplier": 0.85},
    "call":         {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}
TOTAL_BUDGET = 100_000  # суммарный бюджет на все кампании, включая пилотные


REQUIRED_COLUMNS = ["target_tariff", "channel"]
FILTER_VALUES = {
    "filter_arpu_segment": {"LOW", "MID", "HIGH"},
    "filter_data_segment": {"NON_USER", "LITE", "HEAVY"},
    "filter_call_segment": {"LOW", "MEDIUM", "HIGH"},
}


def sanitize_campaigns(campaigns, dict_tariff):
    """
    Отбрасывает некорректные кампании вместо падения всего прогона.

    Агент — особенно с LLM внутри — вполне может выдать несуществующий тариф или
    канал. Ронять из-за этого весь результат команды (включая уже оплаченные
    пилоты) несправедливо: считаем то, что корректно, а про остальное пишем.
    Неизвестные значения ФИЛЬТРОВ не отбрасываем — они просто дадут пустой
    сегмент и нулевой эффект, это достаточное наказание.
    """
    known_tariffs = set(dict_tariff["tariff_plan_code"])
    valid = []
    for i, campaign in enumerate(campaigns or []):
        if not isinstance(campaign, dict):
            print(f"[!] Кампания #{i} отброшена: ожидался dict, получено {type(campaign).__name__}")
            continue
        target, channel = campaign.get("target_tariff"), campaign.get("channel")
        if target not in known_tariffs:
            print(f"[!] Кампания #{i} отброшена: неизвестный тариф {target!r}")
            continue
        if channel not in CHANNELS:
            print(f"[!] Кампания #{i} отброшена: неизвестный канал {channel!r}")
            continue
        valid.append(campaign)
    return valid


def validate_strategy(strategy: pd.DataFrame, dict_tariff: pd.DataFrame):
    """Явные ошибки вместо молчаливого нуля: опечатка в тарифе/сегменте должна быть видна сразу."""
    if len(strategy) > MAX_CAMPAIGNS:
        raise ValueError(f"Стратегия содержит {len(strategy)} кампаний, максимум {MAX_CAMPAIGNS}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in strategy.columns]
    if missing_cols:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {missing_cols}")

    if strategy["channel"].isna().any():
        raise ValueError("У каждой кампании должен быть указан channel (push/sms/digital_ads/call)")
    unknown_channels = set(strategy["channel"]) - set(CHANNELS)
    if unknown_channels:
        raise ValueError(f"Неизвестные каналы: {sorted(unknown_channels)}. Допустимые: {list(CHANNELS)}")

    if strategy["target_tariff"].isna().any():
        raise ValueError("У каждой кампании должен быть указан target_tariff")
    known_tariffs = set(dict_tariff["tariff_plan_code"])
    unknown_targets = set(strategy["target_tariff"]) - known_tariffs
    if unknown_targets:
        raise ValueError(f"Неизвестные целевые тарифы: {sorted(unknown_targets)}")

    for col, allowed in FILTER_VALUES.items():
        if col not in strategy.columns:
            continue
        bad = set(strategy[col].dropna().astype(str)) - allowed
        if bad:
            raise ValueError(f"Недопустимые значения в {col}: {sorted(bad)}. Допустимые: {sorted(allowed)}")

    if "filter_current_tariff" in strategy.columns:
        listed = {
            t.strip()
            for v in strategy["filter_current_tariff"].dropna()
            for t in str(v).split(";")
            if t.strip()
        }
        unknown_filters = listed - known_tariffs
        if unknown_filters:
            raise ValueError(f"Неизвестные тарифы в filter_current_tariff: {sorted(unknown_filters)}")

    # фильтры формально необязательны, но полное их отсутствие — частый признак
    # криво сформированного файла, а не осознанной стратегии "бьём по всем"
    all_filter_cols = list(FILTER_VALUES) + ["filter_current_tariff"]
    if not any(c in strategy.columns for c in all_filter_cols):
        print("[!] ВНИМАНИЕ: в файле нет ни одной колонки-фильтра — все кампании бьют по всей базе. "
              "Проверьте, что команда прислала файл в правильном формате.")


def apply_filters(profile: pd.DataFrame, campaign: pd.Series) -> pd.DataFrame:
    result = profile
    # Пилотные кампании фиксируют конкретную контактированную выборку: её нельзя
    # пересобирать по фильтрам, иначе в зачёт попадёт весь сегмент вместо пилота.
    explicit = campaign.get("explicit_ids")
    if isinstance(explicit, (list, tuple, set, np.ndarray)) and len(explicit) > 0:
        return result[result["ID_NUMBER"].isin(list(explicit))]
    if pd.notna(campaign.get("filter_arpu_segment")):
        result = result[result["arpu_segment"] == campaign["filter_arpu_segment"]]
    if pd.notna(campaign.get("filter_data_segment")):
        result = result[result["data_segment"] == campaign["filter_data_segment"]]
    if pd.notna(campaign.get("filter_call_segment")):
        result = result[result["call_segment"] == campaign["filter_call_segment"]]
    if pd.notna(campaign.get("filter_current_tariff")):
        tariffs = [t.strip() for t in str(campaign["filter_current_tariff"]).split(";") if t.strip()]
        result = result[result["current_tariff"].isin(tariffs)]
    return result


def score_campaign(segment: pd.DataFrame, target_tariff: str, impact_model: pd.DataFrame, dict_tariff: pd.DataFrame,
                    fallback_conversion: float, channel: str, fallback_predict) -> pd.DataFrame:
    """
    Возвращает segment с колонкой expected_lift_per_customer.

    Эффект ОТНОСИТЕЛЬНЫЙ: impact-модель даёт долю изменения ARPU, которая
    умножается на персональный baseline-прогноз абонента (predicted_arpu).
    Поэтому один и тот же процентный эффект стоит дороже на дорогом абоненте —
    и прогноз baseline (Шаг 2) реально влияет на счёт.
    """
    if len(segment) == 0:
        return segment.assign(expected_lift_per_customer=pd.Series(dtype=float))

    conversion_multiplier = CHANNELS[channel]["conversion_multiplier"]

    combos = segment[["current_tariff", "arpu_segment"]].drop_duplicates()
    im_slice = impact_model[impact_model["tariff_plan_code_to"] == target_tariff]

    combos = combos.merge(
        im_slice,
        left_on=["current_tariff", "arpu_segment"],
        right_on=["tariff_plan_code_from", "arpu_segment"],
        how="left",
    )

    missing = combos["arpu_change_pct"].isna()
    if missing.any():
        fb = combos.loc[missing].apply(
            lambda r: fallback_predict(r["current_tariff"], target_tariff, r["arpu_segment"], dict_tariff, fallback_conversion),
            axis=1,
        )
        combos.loc[missing, "arpu_change_pct"] = [f[0] for f in fb]
        combos.loc[missing, "conversion_rate"] = [f[1] for f in fb]

    # эффект канала: множитель к вероятности конверсии, но не более 1.0 (это вероятность)
    combos["effective_conversion_rate"] = (combos["conversion_rate"] * conversion_multiplier).clip(upper=1.0)
    combos["lift_ratio"] = combos["arpu_change_pct"] * combos["effective_conversion_rate"]
    combos = combos[["current_tariff", "arpu_segment", "lift_ratio"]]

    merged = segment.merge(combos, on=["current_tariff", "arpu_segment"], how="left")
    merged["expected_lift_per_customer"] = (merged["lift_ratio"] * merged["predicted_arpu"]).fillna(0.0)
    return merged.drop(columns=["lift_ratio"])


def score_campaigns(strategy: pd.DataFrame, profile: pd.DataFrame, impact_model: pd.DataFrame,
                     dict_tariff: pd.DataFrame, baseline_total_arpu: float,
                     fallback_predict, team_id: str = None) -> dict:
    """
    Ядро скоринга, общее для обоих зачётов (CSV и агентского).

    Применяет лимиты охвата и бюджета по порядку кампаний, считает эффект,
    дедуплицирует абонентов (каждый учитывается один раз, по лучшей кампании)
    и вычитает затраты на коммуникацию.
    """
    fallback_conversion = impact_model["conversion_rate"].median()

    total_contacts = 0
    total_cost = 0.0
    campaigns_detail = []
    scored_parts = []
    remaining_reach_budget = MAX_TOTAL_CONTACTS
    remaining_money_budget = TOTAL_BUDGET

    for idx, campaign in strategy.iterrows():
        channel = campaign["channel"]
        cost_per_contact = CHANNELS[channel]["cost_per_contact"]
        segment = apply_filters(profile, campaign).sort_values("ID_NUMBER")

        capped_at_campaign_limit = len(segment) > MAX_CUSTOMERS_PER_CAMPAIGN
        if capped_at_campaign_limit:
            segment = segment.iloc[:MAX_CUSTOMERS_PER_CAMPAIGN]

        capped_at_reach_budget = len(segment) > remaining_reach_budget
        if capped_at_reach_budget:
            segment = segment.iloc[:max(remaining_reach_budget, 0)]

        # денежный бюджет канала: бесплатный push им не ограничен, платные каналы —
        # обрезаем сегмент до того, что помещается в остаток бюджета
        capped_at_money_budget = False
        if cost_per_contact > 0:
            affordable = int(remaining_money_budget // cost_per_contact)
            if len(segment) > affordable:
                capped_at_money_budget = True
                segment = segment.iloc[:max(affordable, 0)]

        remaining_reach_budget -= len(segment)
        campaign_cost = len(segment) * cost_per_contact
        remaining_money_budget -= campaign_cost
        total_cost += campaign_cost
        total_contacts += len(segment)

        scored = score_campaign(segment, campaign["target_tariff"], impact_model, dict_tariff,
                                 fallback_conversion, channel, fallback_predict)
        campaign_name = campaign.get("campaign_name", f"campaign_{idx}")
        if len(scored) > 0:
            scored_parts.append(scored[["ID_NUMBER", "expected_lift_per_customer"]].assign(campaign=campaign_name))

        campaigns_detail.append({
            "name": campaign_name,
            "channel": channel,
            "cost": campaign_cost,
            "n_contacts": len(segment),
            "gross_lift": float(scored["expected_lift_per_customer"].sum()) if len(scored) else 0.0,
            "n_negative": int((scored["expected_lift_per_customer"] < 0).sum()) if len(scored) else 0,
            "capped_at_campaign_limit": capped_at_campaign_limit,
            "capped_at_reach_budget": capped_at_reach_budget,
            "capped_at_money_budget": capped_at_money_budget,
        })

    # ДЕДУПЛИКАЦИЯ: абонент физически может сменить тариф только один раз, поэтому
    # каждый уникальный клиент засчитывается ОДИН раз — по лучшей для него кампании.
    # Повторные контакты по-прежнему стоят денег и съедают охват, но лифт не удваивают.
    if scored_parts:
        all_scored = pd.concat(scored_parts, ignore_index=True)
        best_per_customer = all_scored.loc[all_scored.groupby("ID_NUMBER")["expected_lift_per_customer"].idxmax()]
        gross_lift = float(best_per_customer["expected_lift_per_customer"].sum())
        unique_customers = len(best_per_customer)
        n_negative = int((best_per_customer["expected_lift_per_customer"] < 0).sum())
    else:
        gross_lift, unique_customers, n_negative = 0.0, 0, 0

    # ЧИСТЫЙ результат: прирост ARPU минус затраты на коммуникацию
    net_gain = gross_lift - total_cost
    total_arpu_after = baseline_total_arpu + net_gain
    status = "PASS" if net_gain > 0 else "FAIL"
    coverage_pct = 100 * unique_customers / len(profile)
    risk_score = 100 * n_negative / unique_customers if unique_customers > 0 else 0.0

    return {
        "team_id": team_id,
        "baseline_total_arpu": baseline_total_arpu,
        "gross_arpu_lift": gross_lift,
        "total_cost": total_cost,
        "net_arpu_gain": net_gain,
        "total_arpu_after": total_arpu_after,
        "growth_vs_baseline_pct": 100 * net_gain / baseline_total_arpu,
        "status": status,
        "n_campaigns": len(strategy),
        "total_contacts": total_contacts,
        "unique_customers_targeted": unique_customers,
        "coverage_pct": coverage_pct,
        "avg_gain_per_customer": net_gain / unique_customers if unique_customers > 0 else 0.0,
        "roi": gross_lift / total_cost if total_cost > 0 else float("inf"),
        "risk_score_pct": risk_score,
        "budget_used_pct": 100 * total_cost / TOTAL_BUDGET,
        "campaigns_detail": campaigns_detail,
    }


def print_result(result: dict):
    print(f"=== Результат команды: {result['team_id']} ===")
    print(f"Статус: {result['status']}")
    print(f"Baseline ARPU (без кампаний):   {result['baseline_total_arpu']:>14,.0f}")
    print(f"Прирост ARPU (gross):           {result['gross_arpu_lift']:>14,.0f}")
    print(f"Затраты на коммуникацию:        {result['total_cost']:>14,.0f}")
    print(f"ЧИСТЫЙ РЕЗУЛЬТАТ (net):         {result['net_arpu_gain']:>14,.0f}   "
          f"({result['growth_vs_baseline_pct']:+.3f}% к baseline)")
    print(f"ARPU после кампаний:            {result['total_arpu_after']:>14,.0f}")
    print()
    print(f"Кампаний: {result['n_campaigns']}  |  контактов: {result['total_contacts']:,}  |  "
          f"уникальных абонентов: {result['unique_customers_targeted']:,} ({result['coverage_pct']:.1f}%)")
    print(f"Средний чистый прирост на абонента: {result['avg_gain_per_customer']:,.1f}")
    print(f"ROI (gross lift / затраты): {result['roi']:.2f}")
    print(f"Risk score (% абонентов с отрицательным эффектом): {result['risk_score_pct']:.1f}%")
    print(f"Бюджет: {result['total_cost']:,.0f} из {TOTAL_BUDGET:,} у.е. ({result['budget_used_pct']:.1f}%)")
    print("\nПо кампаниям (gross, до дедупликации между кампаниями):")
    for c in result["campaigns_detail"]:
        caps = []
        if c.get("capped_at_campaign_limit"):
            caps.append(f"обрезано лимитом кампании {MAX_CUSTOMERS_PER_CAMPAIGN}")
        if c.get("capped_at_reach_budget"):
            caps.append("обрезано общим лимитом охвата")
        if c.get("capped_at_money_budget"):
            caps.append("обрезано денежным бюджетом канала")
        cap_note = f"  [{'; '.join(caps)}]" if caps else ""
        print(f"  - {c['name']} [{c['channel']}]: контактов={c['n_contacts']:,}  cost={c['cost']:,.0f}  "
              f"gross_lift={c['gross_lift']:,.0f}  negative={c['n_negative']}{cap_note}")


