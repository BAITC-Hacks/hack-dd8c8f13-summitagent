"""
ШАБЛОН АГЕНТА — скопируйте в agent.py и перепишите логику под себя.

Это рабочий базовый агент: он делает несколько пилотов по самым перспективным
(по исторической выборке) связкам, смотрит на результаты и выбирает финальные
кампании. Он намеренно простой — его легко обыграть.

Требования к вашему agent.py:
  * класс называется Agent, метод act(self, env) -> list[dict];
  * возвращает не более 10 кампаний;
  * работает без интернета и укладывается в 5 минут;
  * любые библиотеки из requirements.txt вашей команды (приложите его).
"""

import pandas as pd


class Agent:
    def act(self, env):
        profile = env.customer_profile
        tariffs = list(env.tariffs["tariff_plan_code"])

        # 1. РАЗВЕДКА: какие ячейки вообще крупные (чтобы кампания имела смысл)
        cells = (profile.groupby(["current_tariff", "arpu_segment"], observed=True)
                 .agg(n=("ID_NUMBER", "size"), arpu=("predicted_arpu", "mean"))
                 .reset_index()
                 .sort_values("n", ascending=False))

        # 2. Пилотируем несколько гипотез. Здесь гипотезы выбраны примитивно —
        #    это первое, что стоит улучшить: используйте data/change_tariff.csv,
        #    чтобы заранее отранжировать кандидатов и не тратить пилоты впустую.
        candidates = []
        for _, cell in cells.head(4).iterrows():
            for target in ["tariff_8", "tariff_9"]:
                if target == cell.current_tariff:
                    continue
                candidates.append((cell.current_tariff, cell.arpu_segment, target))

        observed = []
        for current_tariff, segment, target in candidates:
            if env.pilots_left <= 0 or env.remaining_budget < 5000:
                break
            try:
                res = env.run_pilot(
                    target_tariff=target, channel="sms", n_customers=150,
                    filter_arpu_segment=segment, filter_current_tariff=current_tariff,
                )
            except RuntimeError:
                break
            observed.append({
                "current_tariff": current_tariff, "arpu_segment": segment,
                "target_tariff": target, "ratio": res["observed_lift_ratio"],
            })

        if not observed:
            return []

        # 3. ЭКСПЛУАТАЦИЯ: берём то, что показало себя лучше всего в пилотах.
        #    Улучшения, которые напрашиваются: учитывать неопределённость оценки
        #    (пилот на 150 клиентах имеет std ≈ 0.07), подбирать канал под
        #    ценность абонента, а не всегда брать sms.
        best = pd.DataFrame(observed).sort_values("ratio", ascending=False)
        campaigns = []
        for _, row in best.head(10).iterrows():
            if row["ratio"] <= 0:
                continue
            campaigns.append({
                "campaign_name": f"main_{row['current_tariff']}_{row['target_tariff']}",
                "filter_arpu_segment": row["arpu_segment"],
                "filter_current_tariff": row["current_tariff"],
                "target_tariff": row["target_tariff"],
                "channel": "sms",
            })
        return campaigns
