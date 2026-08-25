from unittest.mock import AsyncMock, patch

import grpc
import pytest
from mesh_gateway.grpc_client import (
    GrpcMeshManager,
    GrpcNodeClient,
    extract_host_from_url,
)
from mesh_gateway.grpc_proto import mesh_service_pb2, mesh_service_pb2_grpc
from mesh_gateway.models import NodeConfig, NodeEngine
from mesh_gateway.node_agent import NodeAgentServicer


@pytest.fixture
def sample_grpc_node():
    return NodeConfig(
        name="test-gpu-node",
        base_url="http://127.0.0.1:11434",
        engine=NodeEngine.OLLAMA,
        roles=["reasoning", "chat"],
        priority=1,
        pinned_model="qwen2.5-coder:14b",
        grpc_port=50099,
        transport="grpc",
    )


def test_extract_host_from_url():
    assert extract_host_from_url("http://192.168.1.50:11434") == "192.168.1.50"
    assert extract_host_from_url("http://localhost:11434") == "localhost"
    assert extract_host_from_url("http://127.0.0.1:8000/v1") == "127.0.0.1"


@pytest.mark.asyncio
async def test_grpc_node_agent_lifecycle(sample_grpc_node):
    # 1. Start real async gRPC server
    server = grpc.aio.server()
    servicer = NodeAgentServicer(
        node_name=sample_grpc_node.name,
        local_engine_url="http://127.0.0.1:11434",
    )
    mesh_service_pb2_grpc.add_NodeAgentServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    sample_grpc_node.grpc_port = port

    try:
        # 2. Connect client
        client = GrpcNodeClient(sample_grpc_node, grpc_port=port)
        connected = await client.connect()
        assert connected is True
        assert client.is_connected is True

        # 3. Test Container Control RPC
        with patch.object(
            servicer.docker_scaler,
            "start_worker_container",
            new_callable=AsyncMock,
            return_value=(True, "cid-grpc-123", 11435),
        ):
            resp = await client._stub.ControlContainer(
                mesh_service_pb2.ContainerControlRequest(
                    action="START",
                    role="reasoning",
                    model="qwen2.5-coder:14b",
                )
            )
            assert resp.success is True
            assert resp.container_id == "cid-grpc-123"
            assert resp.port == 11435

        await client.close()
    finally:
        await servicer.close()
        await server.stop(grace=0.1)


@pytest.mark.asyncio
async def test_grpc_mesh_manager_integration(sample_grpc_node):
    server = grpc.aio.server()
    servicer = NodeAgentServicer(
        node_name=sample_grpc_node.name,
        local_engine_url="http://127.0.0.1:11434",
    )
    mesh_service_pb2_grpc.add_NodeAgentServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    sample_grpc_node.grpc_port = port
    manager = GrpcMeshManager(default_grpc_port=port)

    received_updates = []

    def _telemetry_cb(update):
        received_updates.append(update)

    try:
        registered = await manager.register_node(
            sample_grpc_node, telemetry_callback=_telemetry_cb
        )
        assert registered is True

        client = manager.get_client(sample_grpc_node.name)
        assert client is not None
        assert client.is_connected is True

        await manager.unregister_node(sample_grpc_node.name)
        assert manager.get_client(sample_grpc_node.name) is None
    finally:
        await manager.close_all()
        await servicer.close()
        await server.stop(grace=0.1)
