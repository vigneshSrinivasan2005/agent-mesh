#!/usr/bin/env bash
set -e

echo "================================================="
echo "   Agent-Mesh Autonomous Worker Container        "
echo "================================================="

# Start Ollama engine in background
ollama serve &
OLLAMA_PID=$!

echo "[*] Waiting for Ollama engine to start..."
while ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done
echo "[✓] Ollama engine active."

# Model pull and preloading
if [ -n "$MODEL" ]; then
    echo "[*] Checking model: $MODEL..."
    if ! ollama list | grep -q "${MODEL%:*}"; then
        echo "[*] Pulling model $MODEL into persistent volume..."
        ollama pull "$MODEL"
    else
        echo "[✓] Model $MODEL already available on volume."
    fi

    # Pre-warm model in memory
    echo "[*] Warming model $MODEL in memory..."
    curl -s http://127.0.0.1:11434/api/generate -d "{\"model\": \"$MODEL\", \"keep_alive\": -1}" >/dev/null 2>&1 || true
fi

# Self-registration with leader if LEADER_HOST is defined
if [ -n "$LEADER_HOST" ]; then
    echo "[*] Registering with leader at $LEADER_HOST..."
    HOSTNAME_STR=$(hostname)
    curl -s -X POST "$LEADER_HOST/api/mesh/register"          -H "Content-Type: application/json"          -d "{\"name\": \"worker-$HOSTNAME_STR\", \"base_url\": \"http://$(hostname -i 2>/dev/null || echo 127.0.0.1):11434\", \"role\": \"${ROLE:-}\", \"pinned_model\": \"${MODEL:-}\"}" >/dev/null 2>&1 || true
fi

echo "[✓] Worker container ready and running."
wait $OLLAMA_PID
