import pytest
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    MeshConfig,
    MeshSettings,
    NodeConfig,
    NodeEngine,
    NodeState,
)


@pytest.fixture
def sample_config():
    return MeshConfig(
        mesh=MeshSettings(
            health_check_interval_seconds=1.0, degraded_latency_threshold_ms=100.0
        ),
        nodes=[
            NodeConfig(
                name="test-node-1",
                base_url="http://mock-node-1:11434",
                engine=NodeEngine.OLLAMA,
                roles=["autocomplete"],
                priority=1,
                pinned_model="qwen2.5-coder:1.5b-base",
            ),
            NodeConfig(
                name="test-node-2",
                base_url="http://mock-node-2:11434",
                engine=NodeEngine.OLLAMA,
                roles=["chat"],
                priority=1,
                pinned_model="qwen2.5-coder:14b",
            ),
        ],
    )


def test_health_summary_initialization(sample_config):
    tracker = HealthTracker(sample_config)
    summary = tracker.get_summary()

    assert summary.total_nodes == 2
    assert summary.online_nodes == 0
    assert summary.status == "critical"  # all initializing/offline initially


def test_request_metrics_tracking(sample_config):
    tracker = HealthTracker(sample_config)
    tracker.statuses["test-node-1"].state = NodeState.ONLINE

    # Start a request
    tracker.record_request_start("test-node-1")
    assert tracker.statuses["test-node-1"].active_requests == 1
    assert tracker.statuses["test-node-1"].total_requests == 1

    # End request successfully with 150 tokens
    tracker.record_request_end("test-node-1", success=True, tokens=150)
    assert tracker.statuses["test-node-1"].active_requests == 0
    assert tracker.statuses["test-node-1"].tokens_generated == 150
    assert tracker.statuses["test-node-1"].failed_requests == 0

    # Record a failed request
    tracker.record_request_start("test-node-1")
    tracker.record_request_end("test-node-1", success=False)
    assert tracker.statuses["test-node-1"].failed_requests == 1


@pytest.mark.asyncio
async def test_probe_node_offline():
    config = MeshConfig(
        nodes=[
            NodeConfig(
                name="unreachable-node",
                base_url="http://127.0.0.1:59999",  # Unused port
                engine=NodeEngine.OLLAMA,
                roles=["chat"],
            )
        ]
    )
    tracker = HealthTracker(config)
    status = await tracker.probe_node(config.nodes[0])

    assert status.state == NodeState.OFFLINE
    assert status.error_message is not None
    assert not status.pinned_model_warm


def test_detailed_telemetry_metrics(sample_config):
    tracker = HealthTracker(sample_config)
    tracker.statuses["test-node-1"].state = NodeState.ONLINE

    tracker.record_request_start("test-node-1")
    tracker.record_request_end(
        "test-node-1",
        success=True,
        prompt_tokens=1600,
        completion_tokens=500,
        duration_sec=20.0,
        tokens_per_sec=25.0,
    )

    st = tracker.statuses["test-node-1"]
    assert st.last_prompt_tokens == 1600
    assert st.last_completion_tokens == 500
    assert st.last_duration_sec == 20.0
    assert st.last_tokens_per_sec == 25.0
    assert st.total_tokens_generated == 500

