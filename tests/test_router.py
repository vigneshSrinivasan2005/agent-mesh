import pytest
from fastapi import HTTPException

from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    MeshConfig,
    MeshSettings,
    NodeConfig,
    NodeEngine,
    NodeHealthStatus,
    NodeState,
)
from mesh_gateway.router import MeshRouter


@pytest.fixture
def sample_config():
    return MeshConfig(
        mesh=MeshSettings(fallback_enabled=True),
        nodes=[
            NodeConfig(
                name="node-mac-fim",
                base_url="http://192.168.1.50:11434",
                engine=NodeEngine.OLLAMA,
                roles=["autocomplete"],
                priority=1,
                pinned_model="qwen2.5-coder:1.5b-base",
                model_aliases={"tab-autocomplete": "qwen2.5-coder:1.5b-base"},
            ),
            NodeConfig(
                name="node-gpu-reasoning",
                base_url="http://192.168.1.100:11434",
                engine=NodeEngine.OLLAMA,
                roles=["chat", "edit", "reasoning"],
                priority=1,
                pinned_model="qwen2.5-coder:14b",
                model_aliases={
                    "reasoning-chat": "qwen2.5-coder:14b",
                    "deepseek-r1": "deepseek-r1:14b",
                },
            ),
            NodeConfig(
                name="node-fallback-server",
                base_url="http://192.168.1.150:11434",
                engine=NodeEngine.OLLAMA,
                roles=["chat", "autocomplete", "reasoning"],
                priority=2,
                pinned_model="deepseek-r1:14b",
                model_aliases={"reasoning-chat": "deepseek-r1:14b"},
            ),
        ],
    )


def test_classify_role(sample_config):
    tracker = HealthTracker(sample_config)
    router = MeshRouter(sample_config, tracker)

    # Autocomplete indicators
    assert router.classify_role("/v1/completions", "tab-autocomplete", {}) == "autocomplete"
    assert router.classify_role("/v1/completions", "qwen2.5-coder:1.5b-base", {}) == "autocomplete"
    assert router.classify_role("/v1/completions", "my-fim-model", {}) == "autocomplete"

    # Chat / Reasoning indicators
    assert router.classify_role("/v1/chat/completions", "reasoning-chat", {}) == "reasoning"
    assert router.classify_role("/v1/chat/completions", "deepseek-r1", {}) == "reasoning"
    assert router.classify_role("/v1/chat/completions", "qwen2.5-coder:14b", {}) == "chat"
    assert router.classify_role("/v1/chat/completions", "fast-edit", {}) == "edit"


def test_model_alias_resolution(sample_config):
    tracker = HealthTracker(sample_config)
    router = MeshRouter(sample_config, tracker)

    node_mac = sample_config.nodes[0]
    node_gpu = sample_config.nodes[1]

    assert router.resolve_model_name(node_mac, "tab-autocomplete") == "qwen2.5-coder:1.5b-base"
    assert router.resolve_model_name(node_gpu, "reasoning-chat") == "qwen2.5-coder:14b"
    assert router.resolve_model_name(node_gpu, "deepseek-r1") == "deepseek-r1:14b"
    assert router.resolve_model_name(node_gpu, "unaliased-model") == "unaliased-model"


def test_node_selection_and_priority(sample_config):
    tracker = HealthTracker(sample_config)
    # Mark all nodes online
    for n in sample_config.nodes:
        tracker.statuses[n.name].state = NodeState.ONLINE
        tracker.statuses[n.name].latency_ms = 25.0

    router = MeshRouter(sample_config, tracker)

    # Autocomplete request should pick node-mac-fim (priority 1)
    node, model = router.select_node("autocomplete", "tab-autocomplete")
    assert node.name == "node-mac-fim"
    assert model == "qwen2.5-coder:1.5b-base"

    # Reasoning request should pick node-gpu-reasoning (priority 1)
    node, model = router.select_node("reasoning", "reasoning-chat")
    assert node.name == "node-gpu-reasoning"
    assert model == "qwen2.5-coder:14b"


def test_automatic_failover(sample_config):
    tracker = HealthTracker(sample_config)
    # Mark primary GPU node OFFLINE
    tracker.statuses["node-mac-fim"].state = NodeState.ONLINE
    tracker.statuses["node-gpu-reasoning"].state = NodeState.OFFLINE
    tracker.statuses["node-fallback-server"].state = NodeState.ONLINE

    router = MeshRouter(sample_config, tracker)

    # Reasoning request should automatically failover to node-fallback-server (priority 2)
    node, model = router.select_node("reasoning", "reasoning-chat")
    assert node.name == "node-fallback-server"
    assert model == "deepseek-r1:14b"


def test_all_nodes_offline_raises_503(sample_config):
    tracker = HealthTracker(sample_config)
    # Mark all nodes OFFLINE
    for n in sample_config.nodes:
        tracker.statuses[n.name].state = NodeState.OFFLINE

    router = MeshRouter(sample_config, tracker)

    with pytest.raises(HTTPException) as exc_info:
        router.select_node("chat", "reasoning-chat")
    assert exc_info.value.status_code == 503
