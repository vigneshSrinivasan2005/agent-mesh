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

# Leader Auto-Discovery and Registration function
register_with_leader() {
    local target_leader="$1"
    local my_ip
    my_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname -i 2>/dev/null || echo "127.0.0.1")
    local host_name
    host_name=$(hostname)

    echo "[*] Attempting to register with Leader at $target_leader (My IP: $my_ip)..."
    local resp
    resp=$(curl -s -m 3 -X POST "$target_leader/api/mesh/register" \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"worker-$host_name\", \"base_url\": \"http://$my_ip:11434\", \"role\": \"${ROLE:-}\", \"pinned_model\": \"${MODEL:-}\"}")
    if [[ "$resp" == *"registered"* ]]; then
        echo "[✓] Successfully registered with Leader at $target_leader!"
        return 0
    fi
    return 1
}

# Auto-discover Leader if not explicitly provided
auto_discover_and_register() {
    if [ -n "$LEADER_HOST" ]; then
        register_with_leader "$LEADER_HOST"
        return
    fi

    echo "[*] Auto-discovering cluster leader on LAN..."

    # 1. Check local host / Docker host bridge
    local docker_gw
    docker_gw=$(ip route 2>/dev/null | awk '/default/ {print $3}' || true)
    for host in "http://127.0.0.1:8000" "http://localhost:8000" "http://host.docker.internal:8000" ${docker_gw:+"http://$docker_gw:8000"}; do
        if register_with_leader "$host"; then
            return 0
        fi
    done

    # 2. Fast sweep local /24 subnet for Leader (:8000)
    local local_ip
    local_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
    if [ -n "$local_ip" ]; then
        local subnet="${local_ip%.*}"
        echo "[*] Scanning subnet $subnet.0/24 for leader gateway..."
        for i in {1..254}; do
            (
                if curl -s -m 1 "http://$subnet.$i:8000/api/mesh" >/dev/null 2>&1; then
                    register_with_leader "http://$subnet.$i:8000"
                fi
            ) &
            # Limit parallel jobs
            if (( i % 50 == 0 )); then
                wait
            fi
        done
        wait
    fi
}

# Run initial discovery
auto_discover_and_register

# Keep continuous heartbeat & re-registration in background every 30s
(
    while true; do
        sleep 30
        auto_discover_and_register >/dev/null 2>&1 || true
    done
) &

echo "[✓] Worker container ready and running."
wait $OLLAMA_PID
