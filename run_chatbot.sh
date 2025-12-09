#!/usr/bin/env bash
# Simple launcher for chatbot GUI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    # Start Ollama in background if not running
    if ! pgrep -x ollama > /dev/null; then
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        sleep 3
    fi
fi

# Launch chatbot GUI
# Check for virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Check if --debug flag is present
DEBUG_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
        break
    fi
done

# Launch chatbot GUI (suppress stdout unless debug mode, always show errors)
if [ "$DEBUG_MODE" = true ]; then
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@"
else
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@" > /dev/null
fi

