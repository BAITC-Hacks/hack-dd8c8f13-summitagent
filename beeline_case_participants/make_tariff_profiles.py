"""
Готовит tariff_profiles.json — смысловые профили тарифов от LLM для тай-брейкера
в agent.py (выбор целевого тарифа между близкими по оценке вариантами).

    OPENAI_API_KEY=... python make_tariff_profiles.py

Модель спрашиваем несколько раз одним и тем же батчевым запросом и по каждому
тарифу берём самый частый ответ. Даже при temperature=0 метки отдельных тарифов
между вызовами расходятся, а зафиксированный в файле ответ делает решения агента
воспроизводимыми. В файл пишем и распределение голосов — видно, где модель
колебалась.
"""

import json
import os
from collections import Counter

from agent import LLM_MODEL, PROFILES_FILE, Agent

N_SAMPLES = 9
FIELDS = ("data", "calls", "price")


def build_profiles(key, n_samples=N_SAMPLES):
    descriptions = Agent._load_tariff_descriptions()
    answers = [Agent._ask_llm(key, descriptions) for _ in range(n_samples)]
    profiles = {}
    for code in descriptions:
        items = [a[code] for a in answers if isinstance(a.get(code), dict)]
        votes = {f: Counter(str(i.get(f, "")).lower() for i in items) for f in FIELDS}
        profile = {f: votes[f].most_common(1)[0][0] for f in FIELDS}
        # описание — из ответа, чьи метки совпали с итоговыми
        profile["summary"] = next((str(i.get("summary", "")) for i in items
                                   if all(str(i.get(f, "")).lower() == profile[f] for f in FIELDS)), "")
        profile["votes"] = {f: dict(votes[f]) for f in FIELDS}
        profiles[code] = profile
    return {"model": LLM_MODEL, "temperature": 0, "samples": n_samples,
            "source": "tariff_dictionary.csv", "profiles": profiles}


def main():
    result = build_profiles(os.environ["OPENAI_API_KEY"])
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    unstable = [code for code, p in result["profiles"].items()
                if any(len(p["votes"][f]) > 1 for f in ("data", "calls"))]
    print(f"Сохранено: {PROFILES_FILE} — {len(result['profiles'])} тарифов, "
          f"{result['samples']} ответов {result['model']}")
    print(f"Модель колебалась по интернету/звонкам у {len(unstable)} тарифов: {', '.join(unstable) or '—'}")


if __name__ == "__main__":
    main()
