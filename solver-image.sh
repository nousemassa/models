#!/usr/bin/env bash
# Сборка образа локального РЕШАТЕЛЯ с GPU и проверка, что GPU внутри образа виден.
#
# Почему скрипт, а не секция `build:` в compose: исходники решателя живут ВНЕ этого репозитория (путь —
# переменная `SOLVER_SRC`), а compose требует, чтобы Dockerfile лежал внутри контекста сборки. Здесь
# контекст — тот каталог, Dockerfile — свой (`solver/Dockerfile`): образ собирается из двух репозиториев,
# потому что решатель и продукт — разные проекты.
#
#   SOLVER_SRC=/путь/к/решателю bash solver-image.sh    # собрать и проверить GPU в образе
set -euo pipefail

#: Каталог ПРОЕКТА (не воркспейса): Dockerfile ищется от него.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SOLVER_SRC:?SOLVER_SRC не задан — путь к каталогу решателя}"
TAG="${SOLVER_IMAGE:-models-solver:local}"

if [[ ! -d "$SRC" ]]; then
  echo "[solver-image] ОСТАНОВ: нет каталога $SRC — задайте SOLVER_SRC" >&2
  exit 2
fi
for need in jev_server.py jev_style_decision_gguf.py manifest.json readout_config.json \
            tokenizer/tokenizer.json _build/build/jev-score-v2 \
            Jev-Style-2B-Decision-v3-Q8_0.gguf; do
  [[ -e "$SRC/$need" ]] || { echo "[solver-image] ОСТАНОВ: в $SRC нет $need" >&2; exit 2; }
done

echo "[solver-image] контекст: $SRC"
echo "[solver-image] собираю $TAG"
docker build -f "$PROJECT_DIR/solver/Dockerfile" -t "$TAG" "$SRC"

echo "[solver-image] проверяю GPU ВНУТРИ образа (это и есть проверка тулкита на нашем случае):"
docker run --rm --gpus all "$TAG" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "[solver-image] готово: $TAG"
