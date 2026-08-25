# Agent-Mesh: Distributed Local LLM Gateway & Multi-Device Health Mesh

> **Turn your multi-device hardware (Macs, PCs, Linux boxes, GPU rigs, homelab servers) into a single, unified AI engine for VS Code extensions ([Continue.dev](https://continue.dev), [Cline](https://github.com/cline/cline), and [Roo Code](https://github.com/RooVetGit/Roo-Code)).**

---

## The Problem: Local LLM VRAM Contention & Thrashing

Running both a fast **1.5B–3B FIM (Fill-in-the-Middle) autocomplete model** (for sub-50ms tab completions) and a heavy **14B–32B+ reasoning model** (for chat and coding loops) on a single machine causes memory thrashing:

1. Ollama or llama.cpp constantly evicts the autocomplete model from VRAM when a chat request arrives.
2. The next tab autocomplete keystroke stalls while the autocomplete model reloads into memory.

---

## The Solution: Master Gateway + Remote gRPC Node Daemons

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             VS Code Editor                                  │
│   Continue.dev / Roo Code / Cline                                           │
│   ├── Tab Autocomplete   ──────► POST http://localhost:8000/v1/completions  │
│   └── Chat / Agent Edit  ──────► POST http://localhost:8000/v1/chat/...     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MASTER GATEWAY DAEMON (Main Machine / Host)                 │
│                                                                             │
│   • Smart Payload & Role Router (FIM vs Heavy Reasoning)                    │
│   • Persistent gRPC Multiplexing & Bidirectional Telemetry Consumer         │
│   • Dynamic Docker Auto-Scaling Engine & Priority Failover                  │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │ (HTTP/2 gRPC :50051)          │ (HTTP/2 gRPC :50051)
                       ▼                               ▼
       ┌───────────────────────────────┐ ┌───────────────────────────────┐
       │   REMOTE DAEMON: Machine A    │ │   REMOTE DAEMON: Machine B    │
       │   (e.g. MacBook M-Series)     │ │   (e.g. RTX 4090 GPU Rig)     │
       │                               │ │                               │
       │  • Agent-Mesh Node Agent      │ │  • Agent-Mesh Node Agent      │
       │  • Ollama / llama.cpp (1.5B)  │ │  • Ollama / vLLM (14B-32B)    │
       │  • Sub-50ms Tab Autocomplete  │ │  • Heavy Reasoning & Coding   │
       │  • Pinned keep_alive: -1      │ │  • Pinned keep_alive: -1      │
       └───────────────────────────────┘ └───────────────────────────────┘
```

---

## Features

- **Zero VRAM Contention**: Autocomplete requests route to your fast Mac/CPU node, while heavy reasoning calls route to your GPU rig. Models stay permanently warm in VRAM (`keep_alive: -1`).
- **Persistent HTTP/2 gRPC Mesh**: Sub-millisecond binary token chunk streaming and continuous hardware telemetry without HTTP polling.
- **Dynamic Local & Remote Container Auto-Scaling**: Automatically spins up Ollama/vLLM Docker worker replicas when agent coding queues spike, and reaps them when idle.
- **Embedded Web Dashboard**: Real-time fleet health, latency percentiles (P50/P95), and VRAM monitor served directly at `http://localhost:8000/status`.
- **1-Click IDE Sync**: Export configurations for Continue.dev, Roo Code, and Cline in seconds.

---

## Quickstart & Setup Guide

### 1. Installation (Run on all machines)

```bash
# Clone the repository
git clone https://github.com/vigneshSrinivasan2005/agent-mesh.git
cd agent-mesh

# Create virtual environment and install
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -e "mesh_gateway[dev]"
```

---

### 2. Start Remote Node Agent Daemons (Worker Machines)

Run the Node Agent on each physical machine you want to pool into the mesh:

#### On Machine A (e.g. Mac / Laptop for Tab Autocomplete):
```bash
# Ensure local Ollama is running on port 11434 with pinned FIM model:
ollama run qwen2.5-coder:1.5b-base

# Start Node Agent daemon:
agent-mesh agent run --host 0.0.0.0 --port 50051 --name macbook-m-series --engine-url http://127.0.0.1:11434
```

#### On Machine B (e.g. Dedicated GPU Rig for Heavy Reasoning):
```bash
# Option 1: Native Node Agent daemon
agent-mesh agent run --host 0.0.0.0 --port 50051 --name desktop-rtx-4090 --engine-url http://127.0.0.1:11434

# Option 2: Docker Containerized Worker with GPU passthrough
agent-mesh node-up --role reasoning --model qwen2.5-coder:14b --port 11434 --gpu
# Or with Docker Compose:
docker compose -f docker/docker-compose.node-gpu.yml up -d
```

---

### 3. Configure the Master Gateway (`mesh-config.yaml`)

On your **Main Development Machine**, generate and edit `mesh-config.yaml`:

```bash
agent-mesh init
```

Configure the IP addresses and gRPC ports of your worker devices:

```yaml
mesh:
  listen_host: "0.0.0.0"
  listen_port: 8000
  grpc_port: 50051
  health_check_interval_seconds: 5.0
  fallback_enabled: true
  degraded_latency_threshold_ms: 300.0
  auto_scaling:
    enabled: true
    scale_up_queue_threshold: 3
    idle_cooldown_seconds: 60.0

nodes:
  # Machine A: Fast Mac / CPU for sub-50ms tab autocomplete
  - name: "macbook-m-series"
    base_url: "http://192.168.1.50:11434"
    grpc_port: 50051
    engine: "ollama"
    roles: ["autocomplete"]
    priority: 1
    pinned_model: "qwen2.5-coder:1.5b-base"
    model_aliases:
      "tab-autocomplete": "qwen2.5-coder:1.5b-base"
      "mesh-autocomplete": "qwen2.5-coder:1.5b-base"

  # Machine B: NVIDIA GPU Rig for heavy agent reasoning & coding loops
  - name: "desktop-rtx-4090"
    base_url: "http://192.168.1.100:11434"
    grpc_port: 50051
    engine: "ollama"
    roles: ["chat", "edit", "reasoning"]
    priority: 1
    pinned_model: "qwen2.5-coder:14b"
    model_aliases:
      "reasoning-chat": "qwen2.5-coder:14b"
      "mesh-reasoning": "qwen2.5-coder:14b"
      "deepseek-r1": "deepseek-r1:14b"

  # Machine C: Fallback Server
  - name: "homelab-server"
    base_url: "http://192.168.1.150:11434"
    grpc_port: 50051
    engine: "ollama"
    roles: ["chat", "reasoning", "autocomplete"]
    priority: 2
    pinned_model: "deepseek-r1:14b"
```

---

### 4. Run the Master Gateway Daemon

On your **Main Machine**, start the master gateway:

```bash
agent-mesh run --config mesh-config.yaml
```

*Or run the Master Gateway inside Docker:*
```bash
docker compose -f docker/docker-compose.gateway.yml up -d
```

---

### 5. Verify Connections & Mesh Health

```bash
# 1. Terminal Fleet Status Table
agent-mesh status

# 2. One-Time Multi-Device Ping Test
agent-mesh ping

# 3. Live Web Dashboard
# Open http://localhost:8000/status in your browser
```

---

## 6. Connect to Your IDE (VS Code, Roo Code, Cline)

### A. Continue.dev (1-Click Setup)

Automatically write the configuration to your Continue.dev settings:

```bash
agent-mesh export-continue --apply
```

Or view the generated configuration:
```bash
agent-mesh export-continue
```

#### Generated `~/.continue/config.yaml`:
```yaml
models:
  - name: "Mesh-Reasoning (Multi-Device)"
    provider: "openai"
    model: "reasoning-chat"
    apiBase: "http://127.0.0.1:8000/v1"
    roles:
      - chat
      - edit

tabAutocompleteModel:
  title: "Mesh-Autocomplete (Pinned FIM)"
  provider: "openai"
  model: "tab-autocomplete"
  apiBase: "http://127.0.0.1:8000/v1"
```

---

### B. Cline / Roo Code

Generate the settings snippet for Cline or Roo Code:

```bash
agent-mesh export-cline
```

In VS Code:
1. Open **Cline / Roo Code Settings** -> **API Provider** -> Select **OpenAI-Compatible**.
2. **Base URL**: `http://127.0.0.1:8000/v1`
3. **Model ID**: `reasoning-chat` (or `fast-edit`)
4. **API Key**: `not-needed` (or any dummy string)

---

## CLI Command Reference

| Command | Usage | Description |
|---|---|---|
| `agent-mesh run` | `agent-mesh run -c mesh-config.yaml` | Start the Master Gateway & Health Prober |
| `agent-mesh agent run` | `agent-mesh agent run -p 50051 -n gpu-node` | Start Remote Node Agent Daemon on worker machine |
| `agent-mesh node up` | `agent-mesh node up -r reasoning -m qwen2.5-coder:14b --gpu` | Launch a local Docker worker container |
| `agent-mesh node down` | `agent-mesh node down <container-id>` | Stop and remove a Docker worker container |
| `agent-mesh status` | `agent-mesh status` | Query live mesh health and render terminal status table |
| `agent-mesh ping` | `agent-mesh ping` | Execute a one-time network latency check across all devices |
| `agent-mesh init` | `agent-mesh init` | Create starter `mesh-config.yaml` |
| `agent-mesh export-continue` | `agent-mesh export-continue --apply` | Sync settings to VS Code Continue.dev |
| `agent-mesh export-cline` | `agent-mesh export-cline` | Output configuration snippet for Cline & Roo Code |

---

## Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat & reasoning endpoint (supports SSE / gRPC streaming) |
| `/v1/completions` | `POST` | OpenAI-compatible FIM & tab autocomplete endpoint |
| `/v1/models` | `GET` | Lists virtual aliases (`tab-autocomplete`, `reasoning-chat`) and physical models |
| `/health` | `GET` | JSON mesh health summary |
| `/health/nodes` | `GET` | Detailed telemetry array (RTT, VRAM, gRPC connection status) |
| `/status` | `GET` | Embedded HTML/JS live monitoring dashboard |

---

## Testing & CI/CD

```powershell
# Run full unit and integration test suite
pytest

# Run linter and formatter
ruff check .
ruff format --check .
```

---

## License

MIT
