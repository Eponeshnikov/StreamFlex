#!/usr/bin/env bash
#
# Запускает Streamlit-приложение внутри tmux-сессии и подключается к ней.
# Сервер продолжает работать, даже если терминал/VSCode-таск закрыт;
# при повторном запуске просто переподключается к уже живой сессии.
#
# Использование:
#   ./run_streamlit.sh                 # запустить (или переподключиться)
#   STREAMFLEX_TMUX_SESSION=foo ./run_streamlit.sh   # своё имя сессии
#   STREAMFLEX_MEM_MAX=100G ./run_streamlit.sh       # свой лимит памяти
#
# Streamlit запускается внутри cgroup-scope с жёстким лимитом памяти и
# без свопа (MemorySwapMax=0): при превышении ядро мгновенно убивает ТОЛЬКО
# этот процесс, не давая системе уйти в zram-своп и зависнуть целиком.

set -euo pipefail

SESSION="${STREAMFLEX_TMUX_SESSION:-streamflex}"
MEM_MAX="${STREAMFLEX_MEM_MAX:-100G}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Dr.Jit (Mitsuba/Sionna) ищет неверсионный libLLVM.so и пути Debian, которых
# на Fedora нет, поэтому LLVM-бэкенд падает с
# "jitc_llvm_init(): LLVM API initialization failed". Указываем самую свежую
# версионную библиотеку явно (см. DRJIT_LIBLLVM_PATH).
if [[ -z "${DRJIT_LIBLLVM_PATH:-}" ]]; then
    _llvm="$(ls -1 /usr/lib64/libLLVM.so.* /lib64/libLLVM.so.* 2>/dev/null \
        | sort -V | tail -1 || true)"
    [[ -n "${_llvm:-}" ]] && export DRJIT_LIBLLVM_PATH="$_llvm"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux-сессия '$SESSION' уже запущена — подключаюсь..."
else
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
        "export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR'; export DRJIT_LIBLLVM_PATH='${DRJIT_LIBLLVM_PATH:-}'; source '$ROOT/.venv/bin/activate' && exec systemd-run --user --scope --collect -p MemoryMax='$MEM_MAX' -p MemorySwapMax=0 streamlit run '$ROOT/app.py'"
    echo "Streamlit запущен в tmux-сессии '$SESSION' (лимит памяти $MEM_MAX, без свопа)."
fi

exec tmux attach -t "$SESSION"
