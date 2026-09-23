"""
Готовит файл submission.csv для сдачи — запускает ВАШЕГО агента на мок-среде
с фиксированным seed и записывает кампании, которые он вернул.

    python make_submission.py

Сдаёте вместе: agent.py + submission.csv (+ requirements.txt, если нужен).

Зачем это нужно. На судействе агент работает против скрытой среды и шумных
пилотов, поэтому там он примет другие решения — это нормально и ожидаемо.
submission.csv нужен для другого: мы перезапустим вашего агента на мок-среде с
тем же seed и сверим, что получилось то же самое. Так мы убеждаемся, что файл
порождён сданным кодом, а не собран руками, и что агент воспроизводим.

Если агент использует случайность — зафиксируйте её от SUBMISSION_SEED, иначе
проверка не сойдётся.
"""

import pandas as pd

from mock_environment import make_mock_env

SUBMISSION_SEED = 42
CAMPAIGN_COLUMNS = ["campaign_name", "filter_arpu_segment", "filter_data_segment",
                    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel"]


def build_submission(agent, seed=SUBMISSION_SEED, **env_kwargs) -> pd.DataFrame:
    env, _ = make_mock_env(seed=seed, **env_kwargs)
    campaigns = agent.act(env) or []
    df = pd.DataFrame(campaigns)
    for col in CAMPAIGN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[CAMPAIGN_COLUMNS]


def main():
    from agent import Agent

    df = build_submission(Agent())
    df.to_csv("submission.csv", index=False)

    print(f"Кампаний в стратегии: {len(df)}")
    print(df.to_string(index=False))
    print("\nСохранено: submission.csv  (сдавайте вместе с agent.py)")


if __name__ == "__main__":
    main()
