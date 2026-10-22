# Проект `models`: сервисы с моделями и GPU. Свой Makefile — чтобы проект можно было поднять, проверить и
# собрать БЕЗ воркспейса: это и есть смысл разреза. Цели не дублируют логику: `gpu-up.sh` — единственное
# место, где живёт «снять конкурентов и поднять сервисы».
#: Интерпретатор с окружением проекта (у воркспейса — свой venv).
PY ?= python3
.PHONY: up status stop restart images check parity ci

up:
	bash gpu-up.sh up

status:
	bash gpu-up.sh status

stop:
	bash gpu-up.sh stop

restart: stop up

# Сборка обоих образов. Скрипты сверяют эталоны (ревизия llama.cpp, sha256 модели) и падают при расхождении.
images:
	bash embed-image.sh
	bash solver-image.sh

# Офлайн-проверка проекта: композ обязан РАЗБИРАТЬСЯ настоящим парсером. Демон для этого не нужен.
# Сверку имени проекта со сторожами и с общим списком делает гейт воркспейса (`make compose-check`) —
# здесь её нет намеренно: у проекта не должно быть второй копии знания о том, какие бывают проекты.
check:
	@docker compose -f compose.models.yaml config -q && echo "[models] композ разбирается ✓"

#: Замер паритета эмбеддингов: два адреса, числа на выходе. Ключ — из окружения (`LLAMA_KEYS_FILE`).
parity:
	python3 infer-parity.py --a http://127.0.0.1:8082/v1 --b http://127.0.0.1:8082/v1 \
	  --key-file "$${LLAMA_KEYS_FILE:?LLAMA_KEYS_FILE не задан}" --out eval/embed-parity.json

# Цель `ci` проекта: композ разбирается И линтер по своему скрипту (`infer-parity.py`).
#: Своего пакета у проекта нет — правила линтера всё равно свои (`models/ruff.toml`).
ci: check
	$(PY) -m ruff check .
