"""
Локальная проверка агента — ваш основной инструмент во время работы.

    python local_eval.py              # один прогон
    python local_eval.py --runs 10    # 10 прогонов с разными seed + разброс

Запускает вашего агента (файл agent.py рядом) на мок-среде, скорит результат
ТЕМ ЖЕ кодом, что и на судействе (scoring_core.py), и печатает полный отчёт:
пилоты, затраты, чистый результат.

ЧТО ЭТО ПОКАЗЫВАЕТ, А ЧТО НЕТ
  ✓ механику: лимиты, бюджет, дедупликацию, формулу — один в один как в финале;
  ✓ не падает ли агент, укладывается ли в лимиты, не жжёт ли бюджет впустую;
  ✗ НЕ показывает ваш будущий балл: эффекты в мок-среде выдуманные, на судействе
    они другие.

Поэтому не гонитесь за конкретным числом здесь. Смотрите на поведение агента:
разумно ли он тратит пилоты, не разоряется ли на дорогих каналах, устойчив ли к
неудачной серии пилотов. Прогон с --runs как раз про устойчивость: если между
seed результат скачет от плюса к минусу, на судействе вам просто не повезёт.
"""

import sys

import pandas as pd

from mock_environment import make_mock_env, _mock_fallback, _mock_impact_model, CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS
from scoring_core import MAX_CAMPAIGNS, score_campaigns, print_result, sanitize_campaigns

CAMPAIGN_FILTER_COLUMNS = ["filter_arpu_segment", "filter_data_segment",
                            "filter_call_segment", "filter_current_tariff"]


def evaluate_agent(agent, seed=None, verbose=True):
    env, internals = make_mock_env(seed=seed)

    try:
        final_campaigns = agent.act(env)
    except Exception as e:
        print(f"[!] Агент упал: {type(e).__name__}: {e}")
        print("    На судействе это не обнулит вас — проведённые пилоты идут в зачёт,")
        print("    но результат почти наверняка будет отрицательным.")
        final_campaigns = []

    final_campaigns = sanitize_campaigns(final_campaigns, env.tariffs)[:MAX_CAMPAIGNS]
    pilots = internals.executed_pilot_campaigns()

    all_campaigns = pd.DataFrame(pilots + final_campaigns)
    if all_campaigns.empty:
        print("[!] Агент не вернул ни одной кампании и не провёл ни одного пилота.")
        return None
    for col in CAMPAIGN_FILTER_COLUMNS + ["explicit_ids"]:
        if col not in all_campaigns.columns:
            all_campaigns[col] = None

    profile = env.customer_profile
    baseline = profile["predicted_arpu"].sum()
    import pandas as _pd
    mock_model = _mock_impact_model(_pd.read_csv("data/change_tariff.csv"))
    result = score_campaigns(all_campaigns, profile, mock_model, env.tariffs,
                              baseline, _mock_fallback, team_id="local")
    result["n_pilots"] = len(pilots)

    if verbose:
        print_result(result)
        print(f"\nПилотов проведено: {len(pilots)} из 20")
        print(f"Осталось бюджета: {env.remaining_budget:,.0f} из {TOTAL_BUDGET:,}")
        print(f"Осталось охвата:  {env.remaining_contacts:,} из {MAX_TOTAL_CONTACTS:,}")
    return result


def main():
    from agent import Agent

    runs = 1
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])

    if runs == 1:
        evaluate_agent(Agent(), seed=42)
        return

    results = []
    for seed in range(runs):
        res = evaluate_agent(Agent(), seed=seed, verbose=False)
        net = res["net_arpu_gain"] if res else float("nan")
        results.append(net)
        print(f"seed {seed:2}: чистый результат {net:>14,.0f}")

    s = pd.Series(results).dropna()
    print("\n--- устойчивость по прогонам ---")
    print(f"медиана: {s.median():,.0f}   минимум: {s.min():,.0f}   максимум: {s.max():,.0f}")
    print(f"прогонов в плюс: {(s > 0).sum()} из {len(s)}")
    if (s > 0).sum() not in (0, len(s)):
        print("⚠ Результат меняет знак между прогонами — агент неустойчив к неудачным пилотам.")


if __name__ == "__main__":
    main()
