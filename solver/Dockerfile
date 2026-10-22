# Образ локального решателя `solver` с GPU.
#
# **Зачем свой образ, а не готовый `nvidia/cuda`.** Проверка тулкита делается не чужим образом, а
# НАШИМ: решатель — это то, что нам действительно нужно на GPU, и если он поднялся, значит и рантайм,
# и драйвер, и библиотеки сошлись на нашем рабочем случае, а не на тестовом.
#
#: **Почему база — Ubuntu 26.04, а CUDA берётся второй ступенью.** Бинарник и библиотеки `ggml` собраны
#: НА ХОСТЕ (Ubuntu 26.04, glibc 2.43), а образ на 24.04 несёт glibc 2.39 — это `GLIBC_2.43 not found`
#: в `libm.so.6`, то есть несовместимость базиса, а не «не хватает пакета». Готовых образов CUDA **12.x**
#: под ubuntu26 в реестре нет (там только 13.x, а нужен именно soname `libcudart.so.12`). Поэтому база —
#: `ubuntu:26.04`, а библиотеки CUDA переносятся из официального рантайм-образа: они собраны против
#: более старой glibc, значит на новой работают.
#
#: **Что должно быть в образе — посчитано по зависимостям бинарника** (`ldd jev-score-v2`):
# #:   * `libllama.so.0`, `libggml{,-base,-cpu,-cuda}.so.0` — билд llama.cpp из контекста сборки,
#     ~138 МБ; берём каталог целиком, потому что у библиотек есть версионные симлинки;
# #:   * `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libnvJitLink.so.12` — из рантайма
#     CUDA (копируется весь `/usr/local/cuda/lib64`, чтобы не искать недостающую по одной);
# #:   * `libgomp.so.1` — OpenMP, ставится пакетом (в CUDA-образе он есть, в чистой Ubuntu — нет:
#     решатель падал именно на нём после того, как нашлись библиотеки ggml);
#   * `libcuda.so.1` — драйвер, его подставляет docker при `--gpus all`, в образ его класть НЕЛЬЗЯ.
#
# **Питон:** только stdlib, `numpy` и `tokenizers` (HuggingFace, Rust-ядро: `from tokenizers import
# Tokenizer`) — версии взяты из `requirements.txt` самого решателя. Виртуальное окружение хоста
# (`.venv`) в контейнер не переносится и не должно.
#
# **Контекст сборки — каталог проекта `solver`** (исходники живут вне нашего репозитория), поэтому
# собирается скриптом `models/solver-image.sh`, а не секцией `build:` в compose: compose требует, чтобы
# Dockerfile лежал внутри контекста, а наш Dockerfile лежит в репозитории проекта-моста.
#
# Первая ступень — только источник библиотек CUDA, наружу из неё ничего не протекает.
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04 AS cuda

# Вторая ступень — боевая база: та же версия Ubuntu, что у хоста (glibc 2.43).
FROM ubuntu:26.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Библиотеки CUDA целиком: так недостающая не обнаружится «на живом» уже в бою.
COPY --from=cuda /usr/local/cuda/lib64/ /usr/local/cuda/lib64/

RUN pip3 install --no-cache-dir --break-system-packages tokenizers==0.23.2 numpy==2.5.3

WORKDIR /app

# Код решателя и его служебные файлы. `readout_config.json` и `tokenizer/` обязательны не «на всякий
# случай»: их проверяет манифест при загрузке модели (`verify_manifest(..., only=["tokenizer",
# "readout_config.json", <gguf>])`), а `readout_config.json` ещё и задаёт калиброванную температуру.
COPY jev_server.py jev_style_decision_gguf.py manifest.json readout_config.json ./

#: **Путь к файлу ключей перекрыт при сборке.** В коде решателя он собран из `Path.home()` и жёсткой
#: строки, то есть задаёт раскладку каталогов ВНУТРИ контейнера — чужую, к тому же совпадающую с путём
#: рабочего каталога на хосте. Перекрываем одной присваивающей строкой: путь берётся из `JEV_KEY_FILE`
#: (его задаёт `compose.models.yaml`), монтирование в композе — в нейтральный `/keys`.
RUN python3 - <<'PY'
import pathlib, re
path = pathlib.Path("jev_server.py")
source = path.read_text(encoding="utf-8")
patched, count = re.subn(
    r'^KEY_FILE = .*$',
    'KEY_FILE = pathlib.Path(os.environ.get("JEV_KEY_FILE", "api_keys.txt"))',
    source,
    flags=re.M,
)
if count != 1:
    raise SystemExit(f"[solver] ОСТАНОВ: ожидалась одна строка KEY_FILE, найдено {count}")
path.write_text(patched, encoding="utf-8")
PY

COPY tokenizer/ ./tokenizer/
# Бинарник и ВСЕ библиотеки своего билда (включая cuda-слой ggml).
COPY _build/build/jev-score-v2 ./_build/build/jev-score-v2
COPY _build/llama.cpp-pinned/build/bin/ ./_build/llama.cpp-pinned/build/bin/
#: Модель решателя (квантование `Q8_0`).
COPY Jev-Style-2B-Decision-v3-Q8_0.gguf ./

#: **Ловушка переноса хостового бинарника в контейнер.** У `jev-score-v2` RUNPATH прописан АБСОЛЮТНЫМ
#: путём каталога сборки, которого в контейнере нет, — решатель падал с «libllama.so.0: cannot open
#: shared object file». `LD_LIBRARY_PATH` ищется РАНЬШЕ RUNPATH (у бинарника именно RUNPATH), поэтому
#: хватает одной строки; сам артефакт не правим — иначе образ перестал бы соответствовать проверенному.
#
#: `JEV_HOST=0.0.0.0` — второе отличие от хостового запуска: в контейнере слушать 127.0.0.1
#: бессмысленно, потому что опубликованный порт docker-proxy доставляет на eth0, а не на loopback.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/_build/llama.cpp-pinned/build/bin \
    JEV_HOST=0.0.0.0 \
    JEV_PORT=8083 \
    JEV_QUANT=Q8_0 \
    JEV_NGL=8 \
    PYTHONUNBUFFERED=1

EXPOSE 8083

CMD ["python3", "jev_server.py"]
