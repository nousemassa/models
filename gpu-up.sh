#!/usr/bin/env bash
# Подъём сервисов с моделями и GPU: эмбеддинги и решатель.
#
# **Свой каталог, своё имя проекта compose (`models`), своё окружение и свой лончер.** Сеть — общая
# (`external`, см. `compose.models.yaml`); имена служб в ней (`embed`, `solver`) — контракт
# (`LLAMA_EMBED_URL=http://embed:8082/v1` в `.env`).
#
# **Почему отдельный композ.** Служба, которой нужна модель или GPU, живёт отдельно от процессов и баз:
# так тяжёлую часть можно поднять и остановить, не задевая продукт.
#
# **Главная работа этого скрипта — не `up`, а снятие КОНКУРИРУЮЩИХ юнитов.** Хостовый юнит решателя
# стартует на секунды раньше контейнеров и занимает порт, тогда контейнер не поднимается с
# `failed to bind port 127.0.0.1:8083: address already in use`. То же грозит эмбеддингам
# (`llama-embed.service`). Поэтому `up` сначала гасит и ВЫКЛЮЧАЕТ такие юниты, и только потом поднимает
# контейнеры.
#
# Отдельного systemd-юнита у этого скрипта НЕТ намеренно: контейнеры объявлены с `restart: unless-stopped`,
# и демон поднимает их сам после загрузки — юнит был бы вторым механизмом ровно для того же действия.
# Его задача (снять конкурентов) выполняется один раз и запоминается (`disable` переживает перезагрузку).
#
#   bash models/gpu-up.sh            # снять конкурентов и поднять контейнеры
#   bash models/gpu-up.sh status     # только проверка
#   bash models/gpu-up.sh stop       # остановить сервисы (юниты НЕ включаем — это откат, см. ниже)
#
#: **⚠️ `--remove-orphans` ЗАПРЕЩЁН.** Он удаляет контейнеры, которых нет в ЭТОМ файле, — то есть ровно
#: то, что должно пережить любой переезд.
set -euo pipefail

#: Каталог ПРОЕКТА (не воркспейса): композ, окружение и скрипты ищутся от него.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#: Воркспейс нужен РОВНО для подсказки про откат (юниты лежат там). Для запуска служб проект от него не
#: зависит.
WORKSPACE="$(cd "$PROJECT_DIR/.." && pwd)"
COMPOSE="$PROJECT_DIR/compose.models.yaml"

#: Окружение проекта — СВОЙ файл рядом с композом. Это не формальность: compose читает `.env` из папки
#: СВОЕГО файла, поэтому значения из корневого `.env` воркспейса к этому проекту больше не приходят, и
#: `EMBED_MODEL_FILE`/`LLAMA_KEYS_FILE` живут теперь здесь. Умолчания в композе совпадают с живыми
#: значениями, так что отсутствие файла ничего не ломает молча — но переопределять надо ЗДЕСЬ.
#: Два файла: общая основа воркспейса + дополнения проекта. Основа ПЕРВОЙ (проект перекрывает),
#: отсутствие любого из них — не ошибка (`docker compose --env-file` можно повторять).
ENV_FILE="${MODELS_ENV:-$PROJECT_DIR/.env}"
BASE_ENV="${MODELS_BASE_ENV:-$WORKSPACE/.env}"
ENV_ARGS=()
[[ -f "$BASE_ENV" ]] && ENV_ARGS+=(--env-file "$BASE_ENV")
[[ -f "$ENV_FILE" ]] && ENV_ARGS+=(--env-file "$ENV_FILE")

compose() { docker compose "${ENV_ARGS[@]}" -f "$COMPOSE" "$@"; }

ACTION="${1:-up}"

#: Юниты, которые занимают порты наших GPU-сервисов. Порядок важен: сначала снимаем, потом поднимаем.
COMPETING_UNITS=(llama-embed.service solver.service)
#: Что проверяем и на каких портах (для `status`).
declare -A PORTS=([embed]=8082 [solver]=8083)

status() {
  local ok=0
  for service in "${!PORTS[@]}"; do
    port="${PORTS[$service]}"
    if curl -fsS -m 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[gpu] ${service}: отвечает на 127.0.0.1:${port} ✓"
    else
      echo "[gpu] ${service}: НЕ отвечает на 127.0.0.1:${port} ✗"; ok=1
    fi
  done
  return $ok
}

case "$ACTION" in
  status)
    status
    ;;
  stop)
    compose stop embed solver
    echo "[gpu] сервисы остановлены. Вернуть хостовые юниты (откат):"
    echo "  systemctl --user link $WORKSPACE/infra/systemd/llama-embed.service && systemctl --user enable --now llama-embed.service"
    ;;
  up|*)
    for unit in "${COMPETING_UNITS[@]}"; do
      if systemctl --user is-enabled "$unit" >/dev/null 2>&1; then
        echo "[gpu] снимаю хостовый юнит $unit: он занял бы порт раньше контейнера"
        systemctl --user stop "$unit" || true
        systemctl --user disable "$unit" >/dev/null 2>&1 || true
      fi
    done
    # Порт должен быть свободен ДО старта: иначе контейнер молча ляжет (он не перезапускается бесконечно).
    # Но «занят» — это про ЧУЖОГО держателя: свой уже поднятый контейнер порт держит законно, и первая
    # редакция этой проверки на нём и спотыкалась (отказывалась поднимать решатель, пока работал `embed`).
    running="$(compose ps --status running --services 2>/dev/null || true)"
    for service in "${!PORTS[@]}"; do
      port="${PORTS[$service]}"
      if grep -qx "$service" <<<"$running"; then
        echo "[gpu] порт ${port}: держит наш контейнер ${service} — это не конфликт"
        continue
      fi
      if ss -ltn 2>/dev/null | grep -q ":${port}\b"; then
        echo "[gpu] ОСТАНОВ: порт ${port} занят — разберитесь, кто его держит (ss -ltnp | grep ${port})" >&2
        exit 2
      fi
    done
    compose up -d embed solver
    sleep 20
    status || { echo "[gpu] ОСТАНОВ: сервис не ответил после старта — смотрите docker logs" >&2; exit 1; }
    ;;
esac
