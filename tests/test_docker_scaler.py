from unittest.mock import AsyncMock, patch

import pytest
from mesh_gateway.docker_scaler import DockerScaler
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    AutoScalingConfig,
    ContainerConfig,
    MeshConfig,
    MeshSettings,
    NodeConfig,
    NodeEngine,
    NodeState,
)
from mesh_gateway.router import MeshRouter


@pytest.fixture
def mesh_setup():
    config = MeshConfig(
        mesh=MeshSettings(
            auto_scaling=AutoScalingConfig(
                enabled=True,
                max_replicas_per_role=2,
                scale_up_queue_threshold=3,
                idle_cooldown_seconds=1.0,
            )
        ),
        nodes=[
            NodeConfig(
                name="gpu-reasoning-1",
                base_url="http://192.168.1.100:11434",
                engine=NodeEngine.OLLAMA,
                roles=["reasoning", "chat"],
                priority=1,
                pinned_model="qwen2.5-coder:14b",
                container=ContainerConfig(
                    enabled=True,
                    container_id="abc123456",
                    host_port=11434,
                ),
            )
        ],
    )
    health = HealthTracker(config)
    health.statuses["gpu-reasoning-1"].state = NodeState.ONLINE
    router = MeshRouter(config, health)
    scaler = DockerScaler(config.mesh.auto_scaling, router, health)
    return config, health, router, scaler


def test_container_model_parsing():
    cont = ContainerConfig(
        enabled=True,
        container_id="test-cid-999",
        image="ollama/ollama:latest",
        host_port=11435,
        is_auto_scaled=True,
    )
    assert cont.enabled is True
    assert cont.container_id == "test-cid-999"
    assert cont.host_port == 11435
    assert cont.is_auto_scaled is True


@pytest.mark.asyncio
async def test_scale_up_under_heavy_load(mesh_setup):
    config, health, router, scaler = mesh_setup

    # Simulate heavy load: active requests = 4 (threshold is 3)
    health.statuses["gpu-reasoning-1"].active_requests = 4

    with patch.object(
        scaler,
        "start_worker_container",
        new_callable=AsyncMock,
        return_value=(True, "newcontainer123", 11436),
    ):
        await scaler.check_and_autoscale()

        # Should have registered a dynamic replica
        auto_nodes = [n for n in config.nodes if n.name.startswith("auto-reasoning")]
        assert len(auto_nodes) == 1
        new_node = auto_nodes[0]
        assert new_node.base_url == "http://127.0.0.1:11436"
        assert new_node.container is not None
        assert new_node.container.is_auto_scaled is True
        assert new_node.name in health.statuses
        assert any(n.name == new_node.name for n in router.config.nodes)


@pytest.mark.asyncio
async def test_scale_down_after_idle(mesh_setup):
    config, health, _router, scaler = mesh_setup

    # First scale up
    health.statuses["gpu-reasoning-1"].active_requests = 4
    with patch.object(
        scaler,
        "start_worker_container",
        new_callable=AsyncMock,
        return_value=(True, "newcontainer123", 11436),
    ):
        await scaler.check_and_autoscale()

    node_name = "auto-reasoning-11436"
    assert node_name in health.statuses

    # Now simulate idle traffic
    health.statuses["gpu-reasoning-1"].active_requests = 0
    health.statuses[node_name].active_requests = 0

    with patch.object(
        scaler, "stop_worker_container", new_callable=AsyncMock, return_value=True
    ):
        # First check sets idle_since timestamp
        await scaler.check_and_autoscale()
        assert scaler._scaled_nodes[node_name]["idle_since"] is not None

        # Simulate elapsed time > idle_cooldown_seconds (1.0s)
        scaler._scaled_nodes[node_name]["idle_since"] -= 2.0
        await scaler.check_and_autoscale()

        # Dynamic node should now be removed from mesh
        assert node_name not in health.statuses
        assert not any(n.name == node_name for n in config.nodes)
