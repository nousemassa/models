#!/usr/bin/env python3
"""Замер паритета эмбеддингов: два адреса, ЧИСЛА на выходе.

**Зачем отдельный скрипт.** «Перенесли в контейнер» — это утверждение о качестве, а не о процессе: сервис
мог подняться с другой версией CUDA, другим `--pooling` или другой моделью, и тогда векторы (а с ними
поиск и весь RAG) тихо поменяются. В репозитории такого замера не было, а `eval/retrieval-report.json`
стоит на потолке (`recall 1.0`), то есть нулевая дельта там ничего не доказывает.

**Почему без «порога».** Соблазнительно объявить «косинус ≥ 1−1e-5», но при смене CUDA-библиотек (сборка
на хосте сделана nvcc 12.4, образ несёт рантайм 12.8) расхождение порядка 1e-3…1e-4 — обычно, и такой
порог упал бы на исправной службе. Поэтому скрипт ПЕЧАТАЕТ числа и сохраняет их в файл: судить по ним
должен человек, а машинная проверка одна — размерность (её расхождение означает смену модели, и это уже
не «паритет», а другая служба).

**Что сравнивается, кроме самих векторов.** Согласие ближайших соседей: для каждого текста берём argmax
косинуса к остальным текстам на обоих адресах. Это дешёвый аналог «найди то же самое» — он ловит смену
пространства, которую покомпонентное сравнение может пропустить.

    python3 models/infer-parity.py --a http://127.0.0.1:8082/v1 --b http://127.0.0.1:8082/v1 \\
        --key-file /путь/к/api_keys.txt --out eval/embed-parity.json

Импортов нашего кода у скрипта нет (только `httpx`) — поэтому он и уехал в `models/` вместе с сервисом,
который измеряет: паритет эмбеддингов — свойство ЭТОГО проекта, а не моста.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import httpx

#: Фиксированный набор текстов: короткие, разные по смыслу, без ПДн. Набор не меняется между прогонами —
#: иначе числа «до» и «после» будут про разные входы.
TEXTS = (
    "улица Пушкина дом десять квартира сорок два",
    "ипотека на вторичное жильё ставка",
    "метро рядом с домом пешком пять минут",
    "двухкомнатная квартира семь из девяти этажей",
    "записаться на просмотр в субботу утром",
    "агентство недвижимости комиссия",
)


def embed(client: httpx.Client, base: str, key: str, text: str) -> list[float]:
    response = client.post(f"{base.rstrip('/')}/embeddings",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": "embed", "input": [text]})
    response.raise_for_status()
    data = response.json()["data"][0]["embedding"]
    return [float(value) for value in data]


def cosine(first: list[float], second: list[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm_a = math.sqrt(sum(a * a for a in first))
    norm_b = math.sqrt(sum(b * b for b in second))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def build_info(client: httpx.Client, base: str, key: str) -> str:
    """Что за сборка отвечает — без этого «паритет» не перепроверить через неделю."""
    try:
        response = client.get(f"{base.rstrip('/')}/props",
                              headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        return str(response.json().get("build_info") or "—")
    except Exception as error:  # noqa: BLE001 — инфо о сборке не имеет права ломать замер
        return f"не прочитано ({type(error).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Паритет эмбеддингов: два адреса, числа на выходе")
    parser.add_argument("--a", required=True, help="база первого адреса, ДО /v1 (например http://127.0.0.1:8092/v1)")
    parser.add_argument("--b", required=True, help="база второго адреса, ДО /v1")
    parser.add_argument("--key-file", required=True, help="файл с ключом (одна строка), как у сервиса")
    parser.add_argument("--out", default="eval/embed-parity.json", help="куда записать числа")
    args = parser.parse_args()

    key = pathlib.Path(args.key_file).read_text(encoding="utf-8").splitlines()[0].strip()
    with httpx.Client(timeout=30.0) as client:
        vectors_a = [embed(client, args.a, key, text) for text in TEXTS]
        vectors_b = [embed(client, args.b, key, text) for text in TEXTS]
        info_a, info_b = build_info(client, args.a, key), build_info(client, args.b, key)

    dim_a = {len(vector) for vector in vectors_a}
    dim_b = {len(vector) for vector in vectors_b}
    report: dict[str, object] = {
        "texts": list(TEXTS),
        "a": {"base": args.a, "build_info": info_a, "dims": sorted(dim_a)},
        "b": {"base": args.b, "build_info": info_b, "dims": sorted(dim_b)},
        "cosine_min": min(cosine(x, y) for x, y in zip(vectors_a, vectors_b, strict=True)),
        "cosine_max": max(cosine(x, y) for x, y in zip(vectors_a, vectors_b, strict=True)),
        "per_text": [
            {"text": text, "cosine": round(cosine(x, y), 9),
             "max_abs_diff": max(abs(p - q) for p, q in zip(x, y, strict=True))}
            for text, x, y in zip(TEXTS, vectors_a, vectors_b, strict=True)
        ],
        "nearest_neighbour_agreement": _nearest_agreement(vectors_a, vectors_b),
    }

    print(f"A: {args.a}  build: {info_a}  dims: {sorted(dim_a)}")
    print(f"B: {args.b}  build: {info_b}  dims: {sorted(dim_b)}")
    print(f"косинус: min {report['cosine_min']:.9f}, max {report['cosine_max']:.9f}")
    print(f"согласие ближайших соседей: {report['nearest_neighbour_agreement']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"числа записаны: {out}")

    if dim_a != dim_b:
        print("ОТКАЗ: размерности разошлись — это не «паритет», а другая модель", file=sys.stderr)
        return 1
    return 0


def _nearest_agreement(first: list[list[float]], second: list[list[float]]) -> int:
    """Сколько текстов имеют ОДНОГО И ТОГО ЖЕ ближайшего соседа в обоих пространствах."""
    def nearest(vectors: list[list[float]], index: int) -> int:
        scores = [(cosine(vectors[index], other), other_index)
                  for other_index, other in enumerate(vectors) if other_index != index]
        return max(scores)[1]

    return sum(1 for index in range(len(first)) if nearest(first, index) == nearest(second, index))


if __name__ == "__main__":
    raise SystemExit(main())
