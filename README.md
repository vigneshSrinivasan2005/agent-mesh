# Agent-Mesh: Distributed Local LLM Gateway & Health Monitor

> **Turn your multi-device hardware (Macs, PCs, Linux boxes, GPU rigs, homelab servers) into a single, unified AI engine for VS Code extensions ([Continue.dev](https://continue.dev), [Cline](https://github.com/cline/cline), and [Roo Code](https://github.com/RooVetGit/Roo-Code)).**

---

## The Problem: Local LLM VRAM Contention

Running both a fast **1.5B–3B FIM (Fill-in-the-Middle) autocomplete model** (for sub-50ms tab completions) and a heavy **14B–32B+ reasoning model** (for chat and coding loops) on a single machine causes memory thrashing:

1. Ollama or llama.cpp constantly evicts the autocomplete model from VRAM when a chat request arrives.
2. The next tab autocomplete keystroke stalls while the autocomplete model reloads into memory.

## The Solution: Distributed Local LLM Gateway

**Agent-Mesh** acts as a high-performance local gateway between your VS Code extensions and your physical devices:

```
┌─────────────────────────────────────────────────────────────┐
│                      VS Code Editor                         │
│                                                             │
│   Continue.dev / Roo Code / Cline                           │
│   ├── Tab Autocomplete   ──────► POST /v1/completions       │
│   └── Chat / Agent Edit  ──────► POST /v1/chat/completions  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Single Gateway URL (http://localhost:8000/v1)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Agent-Mesh Gateway (FastAPI + Async Proxy)         │
│                                                             │
│   • Smart Payload & Role Router (FIM vs Heavy Reasoning)    │
│   • Continuous Health, VRAM & Latency Tracking Engine       │
│   • Sub-millisecond SSE Streaming & Auto-Failover           │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
   Fast Autocomplete Calls        Heavy Agent / Reasoning Calls
   (Sub-50ms target)              (14B–32B+ reasoning models)
                │                             │
                ▼                             ▼
  ┌───────────────────────────┐ ┌───────────────────────────┐
  │    Device A (e.g. Mac)    │ │   Device B (e.g. PC/GPU)  │
  │  • Ollama / llama.cpp     │ │  • Ollama / vLLM          │
  │  • Pinned 1.5B/3B FIM     │ │  • Pinned 14B/32B Model   │
  │  • keep_alive: -1         │ │  • keep_alive: -1         │
  └───────────────────────────┘ └───────────────────────────┘
```

---

## Features

- **Zero VRAM Contention**: Autocomplete calls go directly to your fast low-latency device, while chat/agent requests go to your heavy compute/GPU device. Models stay permanently warm in VRAM (`keep_alive: -1`).
- **Arbitrary N-Device Pooling**: Pool as many physical machines as you have on your local network.
- **Continuous Network & VRAM Health Tracking**:
  - Live Round-Trip Time (RTT) latency (P50 and P95 measurements).
  - VRAM warm state verification (queries Ollama's `/api/ps` to ensure models aren't evicted).
  - Device state tracking (`ONLINE`, `DEGRADED`, `OFFLINE`).
- **Dynamic Failover & Load Balancing**: Automatically shifts traffic to secondary fallback devices if a machine drops offline or is overloaded.
- **Zero-Latency SSE Streaming**: Token-by-token streaming proxy with zero buffering for instantaneous editor response.
- **Embedded Live Web Dashboard**: Visual monitoring page served right at `http://localhost:8000/status`.
- **60-Second VS Code Integration**: 1-command config generation for Continue.dev and Cline.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/agent-mesh.git
cd agent-mesh

# Create virtual environment and install
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -e mesh_gateway
```

### 2. Configure Your Devices (`mesh-config.yaml`)

Run the starter config generator:

```bash
agent-mesh init
```

Edit `mesh-config.yaml` with your devices' IP addresses and pinned models:

```yaml
mesh:
  listen_host: "0.0.0.0"
  listen_port: 8000
  health_check_interval_seconds: 5.0
  fallback_enabled: true

nodes:
  # Device 1: Fast Mac / Laptop for sub-50ms tab autocomplete
  - name: "macbook-m-series"
    base_url: "http://192.168.1.50:11434"
    engine: "ollama"
    roles: ["autocomplete"]
    priority: 1
    pinned_model: "qwen2.5-coder:1.5b-base"
    model_aliases:
      "tab-autocomplete": "qwen2.5-coder:1.5b-base"

  # Device 2: High VRAM GPU Rig for reasoning & coding loops
  - name: "desktop-rtx-4090"
    base_url: "http://192.168.1.100:11434"
    engine: "ollama"
    roles: ["chat", "edit", "reasoning"]
    priority: 1
    pinned_model: "qwen2.5-coder:14b"
    model_aliases:
      "reasoning-chat": "qwen2.5-coder:14b"
      "deepseek-r1": "deepseek-r1:14b"

  # Device 3: Secondary Homelab Server (Fallback)
  - name: "homelab-server"
    base_url: "http://192.168.1.150:11434"
    engine: "ollama"
    roles: ["chat", "reasoning", "autocomplete"]
    priority: 2
    pinned_model: "deepseek-r1:14b"
```

### 3. Start the Gateway

```bash
agent-mesh run
```

### 4. Connect VS Code / Continue.dev in 60 Seconds

Generate or automatically apply the configuration for Continue.dev:

```bash
# View config
agent-mesh export-continue

# Or automatically write to ~/.continue/config.yaml
agent-mesh export-continue --apply
```

#### Continue.dev `config.yaml` Example:

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

## Live Monitoring & CLI Commands

### 1. Web Status Dashboard

Open `http://localhost:8000/status` in your browser to view:
- Global mesh status (Healthy / Degraded / Critical)
- Live RTT Latencies (P50 & P95) for all $N$ physical machines
- VRAM Pinned Model warm status (verified via `/api/ps`)
- Active in-flight requests and total throughput

### 2. Rich Terminal Status Table

```bash
# Query running gateway
agent-mesh status

# Perform a one-shot ping across all nodes in config
agent-mesh ping
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat & reasoning endpoint (supports SSE streaming) |
| `/v1/completions` | `POST` | OpenAI-compatible FIM & tab autocomplete endpoint (supports SSE streaming) |
| `/v1/models` | `GET` | Lists virtual aliases (`tab-autocomplete`, `reasoning-chat`) and physical models |
| `/health` | `GET` | JSON mesh health summary |
| `/health/nodes` | `GET` | Detailed telemetry array for each physical node |
| `/status` | `GET` | Embedded HTML/JS live monitoring dashboard |

---

## License

MIT
