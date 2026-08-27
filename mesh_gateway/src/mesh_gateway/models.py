from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class NodeState(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    INITIALIZING = "INITIALIZING"


class NodeEngine(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    GENERIC = "generic"


class ContainerType(str, Enum):
    NATIVE = "native"
    DOCKER = "docker"
    PODMAN = "podman"


class ContainerConfig(BaseModel):
    enabled: bool = Field(
        default=False, description="Whether this node is running as a managed container"
    )
    container_id: Optional[str] = Field(default=None, description="Docker container ID or name")
    image: str = Field(default="ollama/ollama:latest", description="Container image")
    host_port: int = Field(default=11434, description="Mapped host port")
    gpu_enabled: bool = Field(default=False, description="Enable GPU passthrough in container")
    is_auto_scaled: bool = Field(
        default=False, description="Whether this node instance was created by the auto-scaler"
    )


class AutoScalingConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable dynamic local container auto-scaling")
    docker_host: Optional[str] = Field(
        default=None, description="Custom Docker daemon URL or socket path"
    )
    max_replicas_per_role: int = Field(
        default=3, description="Maximum number of dynamic worker container replicas"
    )
    scale_up_queue_threshold: int = Field(
        default=3, description="Active concurrent requests threshold to trigger scale-up"
    )
    idle_cooldown_seconds: float = Field(
        default=60.0, description="Seconds of zero traffic before scaling down a dynamic replica"
    )


class NodeConfig(BaseModel):
    name: str = Field(..., description="Unique human-readable identifier for the physical node")
    base_url: str = Field(
        ..., description="Base URL of the node LLM backend (e.g. http://192.168.1.50:11434)"
    )
    engine: NodeEngine = Field(
        default=NodeEngine.OLLAMA, description="LLM engine type (ollama, vllm, openai)"
    )
    roles: List[str] = Field(
        default=["chat", "edit", "reasoning"],
        description="Assigned roles: 'autocomplete', 'reasoning', 'chat', 'edit', 'embeddings'",
    )
    priority: int = Field(
        default=1,
        description="Routing priority (1 = highest priority primary, 2 = secondary fallback)",
    )
    pinned_model: Optional[str] = Field(
        default=None,
        description="Model name to pin in memory with keep_alive: -1",
    )
    model_aliases: Dict[str, str] = Field(
        default_factory=dict,
        description="Map incoming alias (e.g. 'reasoning-chat') to exact backend model name",
    )
    api_key: Optional[str] = Field(default=None, description="Optional API key / token if secured")
    timeout_seconds: float = Field(default=120.0, description="Per-request timeout in seconds")
    max_concurrent: int = Field(default=4, description="Maximum concurrent in-flight requests")
    grpc_port: Optional[int] = Field(
        default=None,
        description="Port of Agent-Mesh Node Daemon for ultra-fast gRPC streaming and telemetry",
    )
    transport: str = Field(
        default="auto",
        description="Transport protocol preference: 'grpc', 'http', or 'auto'",
    )
    container: Optional[ContainerConfig] = Field(
        default=None, description="Container execution details if containerized"
    )


class MeshSettings(BaseModel):
    listen_host: str = Field(default="0.0.0.0", description="Host to bind the Gateway server")
    listen_port: int = Field(default=8000, description="Port to bind the Gateway server")
    grpc_port: Optional[int] = Field(default=50051, description="Default gRPC port for node agents")
    health_check_interval_seconds: float = Field(
        default=5.0,
        description="Interval in seconds for background health & latency probes",
    )
    default_timeout_seconds: float = Field(default=120.0)
    fallback_enabled: bool = Field(
        default=True,
        description="Automatically reroute to lower-priority nodes when primary is offline",
    )
    degraded_latency_threshold_ms: float = Field(
        default=300.0,
        description="Latency threshold in ms above which a node is marked DEGRADED",
    )
    auto_discovery: bool = Field(
        default=True,
        description="Automatically discover and register LAN worker nodes via UDP beacon & mDNS",
    )
    auto_scaling: AutoScalingConfig = Field(
        default_factory=AutoScalingConfig, description="Container auto-scaling engine config"
    )


class NodeRegistrationRequest(BaseModel):
    name: str
    base_url: str = "http://127.0.0.1:11434"
    engine: NodeEngine = NodeEngine.OLLAMA
    role: Optional[str] = None
    roles: Optional[List[str]] = None
    pinned_model: Optional[str] = None
    grpc_port: Optional[int] = None
    priority: int = 1


class MeshConfig(BaseModel):
    mesh: MeshSettings = Field(default_factory=MeshSettings)
    nodes: List[NodeConfig] = Field(default_factory=list)


class ModelTokenMetrics(BaseModel):
    model_name: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    last_input_tok_per_sec: Optional[float] = None     # Prompt evaluation speed
    last_output_tok_per_sec: Optional[float] = None    # Generation speed
    avg_input_tok_per_sec: Optional[float] = None
    avg_output_tok_per_sec: Optional[float] = None
    last_prompt_tokens: Optional[int] = None
    last_completion_tokens: Optional[int] = None
    last_ttft_sec: Optional[float] = None              # Time to first token
    last_duration_sec: Optional[float] = None


class NodeHealthStatus(BaseModel):
    name: str
    base_url: str
    engine: NodeEngine
    roles: List[str]
    priority: int
    state: NodeState = NodeState.INITIALIZING
    latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_check: Optional[str] = None
    loaded_models: List[str] = Field(default_factory=list)
    available_models: List[str] = Field(default_factory=list)
    configured_models: List[str] = Field(default_factory=list)
    pinned_model: Optional[str] = None
    pinned_model_warm: bool = False
    total_requests: int = 0
    active_requests: int = 0
    failed_requests: int = 0
    tokens_generated: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    error_message: Optional[str] = None
    latency_history: List[float] = Field(default_factory=list)
    
    # Active Model & Memory / Offload Telemetry
    active_model_name: Optional[str] = None
    active_model_size_mb: Optional[float] = None
    active_model_vram_mb: Optional[float] = None
    active_model_gpu_pct: Optional[float] = None
    active_model_cpu_pct: Optional[float] = None
    active_model_quantization: Optional[str] = None
    active_model_parameter_size: Optional[str] = None
    active_model_context_length: Optional[int] = None
    is_swapping: bool = False

    # Generation & Input/Output Rate Telemetry
    last_tokens_per_sec: Optional[float] = None          # Backwards compatibility alias for output
    last_input_tokens_per_sec: Optional[float] = None   # Input / prompt eval tok/s
    last_output_tokens_per_sec: Optional[float] = None  # Output / generation tok/s
    last_prompt_tokens: Optional[int] = None
    last_completion_tokens: Optional[int] = None
    last_ttft_sec: Optional[float] = None
    last_duration_sec: Optional[float] = None

    # Per-Model Token & Rate Telemetry (model_name -> ModelTokenMetrics)
    model_metrics: Dict[str, ModelTokenMetrics] = Field(default_factory=dict)

    # Container Telemetry
    is_container: bool = False
    container_id: Optional[str] = None
    container_engine: str = "native"
    is_auto_scaled: bool = False
    container_mem_mb: Optional[float] = None
    # gRPC Telemetry
    grpc_connected: bool = False
    gpu_vram_used_mb: Optional[int] = None
    gpu_vram_total_mb: Optional[int] = None
    gpu_utilization_pct: Optional[float] = None


class MeshHealthSummary(BaseModel):
    status: str
    total_nodes: int
    online_nodes: int
    degraded_nodes: int
    offline_nodes: int
    active_requests: int
    total_requests: int
    nodes: List[NodeHealthStatus]
    timestamp: float = Field(default_factory=time.time)


# --- OpenAI API Compatible Request / Response Models ---


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    extra_body: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "agent-mesh"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]
