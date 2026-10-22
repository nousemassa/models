# Образ сервиса эмбеддингов: llama.cpp с CUDA; модель и ключи монтируются снаружи.
#
# **Почему образ, а не готовый `nvidia/cuda`.** Тот же урок, что у `solver` (`models/solver/Dockerfile`):
# бинарник и библиотеки `ggml` собраны НА ХОСТЕ из исходников, а значит зависят от glibc хоста и от
# того, чем их собрали. Проверять надо НАШ рабочий случай, а не абстрактный образ из реестра.
#
# **Почему база — `ubuntu:26.04`, а CUDA берётся второй ступенью.** Первая сборка `solver` упала на
# `GLIBC_2.43 not found`: бинарник собран на Ubuntu 26.04 (glibc 2.43), а `nvidia/cuda:*ubuntu24.04`
# несёт 2.39. Готовых образов CUDA **12.x** под ubuntu26 в реестре нет, а нам нужен именно soname
# `libcudart.so.12`. Поэтому библиотеки CUDA переносятся из официального рантайм-образа: они собраны
# против более старой glibc и на новой работают.
#
# **Что должно быть в образе — посчитано по зависимостям бинарника** (`ldd llama-server`):
#   * `libllama-server-impl.so`, `libllama-common.so.0`, `libmtmd.so.0`, `libllama.so.0`,
#     `libggml{,-base,-cpu,-cuda}.so.0` — НАШ билд `build-embed/bin` (161 МБ; каталог берём целиком,
#     потому что у библиотек есть версионные симлинки). `libggml.so.0` жёстко требует
#     `libggml-cuda.so.0` — CUDA-бэкенд грузит ЛИНКЕР, а не dlopen;
#   * `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` — из рантайма CUDA (копируем весь
#     `/usr/local/cuda/lib64`, чтобы недостающая не нашлась «на живом» уже в бою);
#   * `libgomp.so.1` — OpenMP, пакетом (в чистой Ubuntu его нет: сервис падал именно на нём);
#   * `libcuda.so.1` — драйвер, его подставляет docker при GPU-запросе, в образ класть НЕЛЬЗЯ.
#   * `curl` — только ради healthcheck (он проверяет реальный ответ эмбеддинга, см. ниже).
#
# **Provenance (чем именно собран этот образ).** Собрано из
#   * llama.cpp commit `aa94f2086147cd4bfbcff818ce998dffce5fcb39` (дерево чистое), сборка `build-embed`;
#   * CMake: `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, `GGML_CUDA_ARCHITECTURES=89`, `GGML_NATIVE=ON`;
#   * компилятор CUDA `nvcc 12.4.131` (рантайм в образе — 12.8.x, совместим по soname `.so.12`);
#   * модель НЕ в образе (монтируется снаружи), её эталон:
#     `embeddinggemma-2-Q8_0.gguf`, 309 855 520 байт, sha256
#     `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc`.
# Сверяет это `models/embed-image.sh` и падает при расхождении: «образ не воспроизводим» — это про
# то, что без такой записи через неделю никто не скажет, чем собран работающий образ.
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04 AS cuda

FROM ubuntu:26.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Библиотеки CUDA целиком (первая ступень — только источник, наружу ничего не протекает).
COPY --from=cuda /usr/local/cuda/lib64/ /usr/local/cuda/lib64/

WORKDIR /app

# Бинарник и ВСЕ библиотеки нашего билда, включая cuda-слой ggml.
COPY build-embed/bin/ /app/bin/

#: **Ловушка переноса хостового бинарника (та же, что у `solver`):** RUNPATH у бинарника и у
#: `libggml-cuda.so.0` прописан АБСОЛЮТНЫМ путём каталога сборки, которого в контейнере нет ⇒ без
#: `LD_LIBRARY_PATH` сервис падал бы с «libllama.so.0: cannot open shared object file». Переменная
#: ищется РАНЬШЕ RUNPATH, поэтому хватает одной строки; сам артефакт не правим.
#
# **Два отличия от хостового запуска, и оба обязательны:**
#   * `--host 0.0.0.0`: docker-proxy доставляет опубликованный порт на eth0, а не на loopback —
#     с `127.0.0.1` из хостового юнита сервис был бы недостижим;
#   * `--device CUDA0`: без него llama.cpp при недоступной CUDA лишь ПРЕДУПРЕЖДАЕТ и поднимается на
#     CPU с зелёным `/health` (тихий фолбэк), а с флагом падает на старте (`invalid device: CUDA0`,
#     exit 1) — «не на GPU» должно быть видно, а не замаскировано.
#: Остальные флаги — дословно те же, что у хостовой службы (`llama-embed`).
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/bin \
    PYTHONUNBUFFERED=1

#: Порт слушаем внутри контейнера; наружу публикуется `127.0.0.1:8082` (см. `compose.models.yaml`).
EXPOSE 8082

#: Пути к модели и ключу приходят ТОМАМИ в нейтральный каталог `/models-src`: те же пути, что в `CMD`.
#: Файлов в образе нет — каталог создаётся монтированием (см. `compose.models.yaml`).
CMD ["/app/bin/llama-server", \
     "-m", "/models-src/embeddinggemma-2-Q8_0.gguf", \
     "--api-key-file", "/models-src/api_keys.txt", \
     "--embeddings", "--pooling", "mean", \
     "-ngl", "99", "-fa", "on", "-c", "8192", "-t", "20", \
     "--device", "CUDA0", \
     "--host", "0.0.0.0", "--port", "8082"]
