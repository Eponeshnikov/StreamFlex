#!/usr/bin/env bash
#
# Находит запущенные процессы Streamlit и завершает их.
#
# Использование:
#   ./kill_streamlit.sh          # завершить все процессы streamlit
#   ./kill_streamlit.sh -9       # принудительно (SIGKILL)
#   ./kill_streamlit.sh -p 8501  # завершить только сервер на указанном порту

set -euo pipefail

SIGNAL="TERM"
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -9|--force)
            SIGNAL="KILL"
            shift
            ;;
        -p|--port)
            PORT="${2:-}"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $1" >&2
            exit 1
            ;;
    esac
done

# Собираем PID-ы процессов streamlit.
if [[ -n "$PORT" ]]; then
    # Процессы, слушающие конкретный порт.
    PIDS=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
else
    # Все процессы streamlit (исключая сам этот скрипт и grep).
    PIDS=$(pgrep -f '[s]treamlit run' 2>/dev/null || true)
fi

if [[ -z "$PIDS" ]]; then
    echo "Процессы Streamlit не найдены."
    exit 0
fi

echo "Найдены процессы Streamlit:"
ps -o pid=,cmd= -p $PIDS 2>/dev/null || true

echo "Отправляю сигнал SIG${SIGNAL}..."
kill -s "$SIGNAL" $PIDS 2>/dev/null || true

# Ждём корректного завершения; при необходимости добиваем.
sleep 2
REMAINING=$(for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done || true)
if [[ -n "$REMAINING" ]]; then
    echo "Процессы не завершились, отправляю SIGKILL: $REMAINING"
    kill -9 $REMAINING 2>/dev/null || true
fi

echo "Готово."
