#!/usr/bin/env bash
# Сборка образа СЕРВИСА ЭМБЕДДИНГОВ и проверка, что GPU внутри образа виден.
#
# Почему скрипт, а не секция `build:` в compose: исходники llama.cpp живут ВНЕ этого репозитория
# (путь — переменная `LLAMA_SRC`), а compose требует, чтобы Dockerfile лежал внутри контекста сборки.
# Здесь контекст — каталог сборки llama.cpp, Dockerfile — наш (`models/embed/Dockerfile`).
#
# **Предпроверки — не формальность.** Образ собирается из бинарников, собранных НА ХОСТЕ, поэтому
# «то же самое» держится на двух числах: ревизии исходников llama.cpp и sha256 файла модели. Без них
# через неделю никто не скажет, чем собран работающий образ, — и «переехали» окажется непроверяемым.
#: Эталоны ниже — факт проверки (ревизия и sha256), а не «примерно такие же»: см. запись provenance в
#: `embed/Dockerfile`.
#
#   LLAMA_SRC=/путь/к/llama.cpp bash embed-image.sh    # собрать и проверить GPU в образе
set -euo pipefail

#: Каталог ПРОЕКТА (не воркспейса): Dockerfile ищется от него, от воркспейса проект не зависит.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_SRC:?LLAMA_SRC не задан — путь к каталогу llama.cpp (его исходники и модель)}"
SRC="$LLAMA_DIR/src/llama.cpp"          # здесь лежит build-embed (контекст сборки)
TAG="${EMBED_IMAGE:-models-embed:local}"

#: Эталоны: ревизия исходников и файл модели — то, чем собран рабочий образ.
EXPECT_COMMIT="aa94f2086147cd4bfbcff818ce998dffce5fcb39"
MODEL="$LLAMA_DIR/models/embeddinggemma-2-Q8_0.gguf"
EXPECT_MODEL_SHA="6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"

if [[ ! -d "$SRC" ]]; then
  echo "[embed-image] ОСТАНОВ: нет каталога $SRC — задайте LLAMA_SRC" >&2
  exit 2
fi
for need in build-embed/bin/llama-server; do
  [[ -e "$SRC/$need" ]] || { echo "[embed-image] ОСТАНОВ: в $SRC нет $need (сборка build-embed не найдена)" >&2; exit 2; }
done
[[ -f "$MODEL" ]] || { echo "[embed-image] ОСТАНОВ: нет модели $MODEL" >&2; exit 2; }

echo "[embed-image] сверяю ревизию llama.cpp…"
commit="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo '—')"
if [[ "$commit" != "$EXPECT_COMMIT" ]]; then
  echo "[embed-image] ОСТАНОВ: ревизия $commit ≠ эталон $EXPECT_COMMIT" >&2
  echo "  Если сборка НАМЕРЕННО обновлена — обновите эталон в models/embed/Dockerfile и здесь," >&2
  echo "  а после сборки прогоните замер паритета (models/infer-parity.py): новая сборка может дать другой вектор." >&2
  exit 2
fi

echo "[embed-image] сверяю sha256 модели…"
model_sha="$(sha256sum "$MODEL" | awk '{print $1}')"
if [[ "$model_sha" != "$EXPECT_MODEL_SHA" ]]; then
  echo "[embed-image] ОСТАНОВ: sha256 модели $model_sha ≠ эталон $EXPECT_MODEL_SHA" >&2
  exit 2
fi

echo "[embed-image] контекст: $SRC"
echo "[embed-image] собираю $TAG"
docker build -f "$PROJECT_DIR/embed/Dockerfile" -t "$TAG" "$SRC"

echo "[embed-image] проверяю GPU ВНУТРИ образа (nvidia-smi приходит от рантайма, в образе его нет):"
docker run --rm --gpus all "$TAG" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "[embed-image] проверяю, что бинарник видит CUDA-устройство (иначе был бы тихий CPU-фолбэк):"
docker run --rm --gpus all "$TAG" /app/bin/llama-server --list-devices

echo "[embed-image] готово: $TAG"
