#!/usr/bin/env bash
# Start the Telecom Egypt Assistant locally (macOS / Linux).
#
# Starts speech (:8001), core (:8000) and the Gradio UI (:7860) as background
# jobs, then waits for core to become healthy before launching the UI — core
# loads the embedder and builds the BM25 index at startup, and opening the UI
# first just shows an error.
#
# Core does NOT depend on speech: if speech fails, text chat still works and the
# UI hides the microphone. That is deliberate.
#
#   ./scripts/run_local.sh
#   ./scripts/run_local.sh --skip-speech
#   TE_PROFILE=cpu-lite ./scripts/run_local.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/smart_assistant/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

SKIP_SPEECH=0
for arg in "$@"; do
    case "$arg" in
        --skip-speech) SKIP_SPEECH=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    esac
done

echo
echo "=== Telecom Egypt Assistant ==="

# --- preflight ------------------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no interpreter found. Create the venv:" >&2
    echo "  uv venv smart_assistant --python 3.11" >&2
    echo "  uv pip install -e '.[dev]'" >&2
    exit 1
fi

command -v ffmpeg >/dev/null 2>&1 || {
    echo "WARNING: ffmpeg not on PATH - voice input will fail (text chat is fine)."
}

# An empty index is the most common cause of "it answers nothing".
if [[ ! -d "$ROOT/data/chroma" ]] || [[ -z "$(ls -A "$ROOT/data/chroma" 2>/dev/null)" ]]; then
    echo "ERROR: no knowledge base at data/chroma" >&2
    echo "  Build it:  $PYTHON -m te_assistant.ingest.build_index" >&2
    exit 1
fi

mkdir -p "$ROOT/logs"
PIDS=()

cleanup() {
    echo
    echo "stopping services..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start() {
    local name="$1" module="$2"
    "$PYTHON" -m "$module" >"$ROOT/logs/$name.log" 2>&1 &
    PIDS+=($!)
    echo "started : $name (pid $!, logs/$name.log)"
}

wait_healthy() {
    local url="$1" name="$2" timeout="${3:-300}"
    local deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "healthy : $name"
            return 0
        fi
        sleep 2
    done
    echo "timeout : $name did not become healthy"
    return 1
}

# --- start ----------------------------------------------------------------
if (( SKIP_SPEECH == 0 )); then
    start speech te_assistant.speech.service
else
    echo "skipped : speech (text-only mode)"
fi

start core te_assistant.api

echo
echo "waiting for core to load models..."
wait_healthy "http://127.0.0.1:8000/health" core 300 || true

start ui te_assistant.ui.gradio_app

cat <<EOF

  UI      http://127.0.0.1:7860
  core    http://127.0.0.1:8000/health
  metrics http://127.0.0.1:8000/metrics
EOF
(( SKIP_SPEECH == 0 )) && echo "  speech  http://127.0.0.1:8001/health"
echo
echo "Ctrl-C to stop all services."
wait
