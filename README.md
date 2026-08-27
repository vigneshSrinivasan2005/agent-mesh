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

## Quickstart: Zero-Config Docker & LAN Auto-Discovery

No YAML configuration required. Simply run the leader on your main machine and worker nodes on any other PC across your local network:

### 1. Start the Leader Node (Main Machine / MacBook)
```bash
docker compose up leader-node -d
```
*Starts the Master Gateway on port `8000`, serves the live Web Dashboard at `http://localhost:8000/status`, and begins auto-discovering all LAN worker devices.*

---

### 2. Start Worker Nodes (Any PC / GPU Rig on your Network)
```bash
# Auto-detects GPU and automatically infers role (<4B -> autocomplete, >=7B -> reasoning):
MODEL=deepseek-r1:8b docker compose up worker-node -d

# Or with an explicit custom role:
ROLE=autocomplete MODEL=qwen2.5-coder:1.5b-base docker compose up worker-node -d
```
*The worker node pulls the model into a persistent volume, keeps it warm in memory, and announces itself to the leader. It immediately appears on the leader's dashboard.*

---

### 3. Native CLI Mode (Without Docker)

You can also run without Docker using the `agent-mesh` CLI:

```bash
# On Leader Machine:
agent-mesh leader-up

# On Worker Machine:
agent-mesh worker-up deepseek-r1:8b
# (Or with explicit role):
agent-mesh worker-up qwen2.5-coder:1.5b-base --role autocomplete
```

---

## (Optional) Advanced Static Configuration (`mesh-config.yaml`)

If you want to manually pin specific static IPs or customize fallback priorities instead of using auto-discovery, you can generate a `mesh-config.yaml`:

```bash
agent-mesh init
```

```yaml
mesh:
  listen_host: "0.0.0.0"
  listen_port: 8000
  auto_discovery: true
  health_check_interval_seconds: 5.0

nodes:
  # Machine A: Fast FIM Autocomplete
  - name: "macbook-fast-fim"
    base_url: "http://192.168.1.50:11434"
    roles: ["autocomplete"]
    pinned_model: "qwen2.5-coder:1.5b-base"

  # Machine B: NVIDIA GPU Rig for Heavy Reasoning
  - name: "desktop-gpu-rig"
    base_url: "http://192.168.1.100:11434"
    roles: ["chat", "edit", "reasoning"]
    pinned_model: "deepseek-r1:8b"
```

---

---

### 4. Verify Connections & Mesh Health

```bash
# 1. Live Web Dashboard
# Open http://localhost:8000/status in your browser

# 2. Terminal Fleet Status Table
agent-mesh status

# 3. One-Time Multi-Device Latency Ping Test
agent-mesh ping
```

---

## 5. Connect to Your IDE (VS Code, Roo Code, Cline)

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
