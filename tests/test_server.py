import pytest
from httpx import ASGITransport, AsyncClient
from mesh_gateway.models import (
    MeshConfig,
    MeshSettings,
    NodeConfig,
    NodeEngine,
)
from mesh_gateway.server import create_app


@pytest.fixture
def mock_app():
    config = MeshConfig(
        mesh=MeshSettings(health_check_interval_seconds=10.0),
        nodes=[
            NodeConfig(
                name="mock-node-fim",
                base_url="http://mock-fim:11434",
                engine=NodeEngine.OLLAMA,
                roles=["autocomplete"],
                priority=1,
                pinned_model="qwen2.5-coder:1.5b-base",
                model_aliases={"tab-autocomplete": "qwen2.5-coder:1.5b-base"},
            ),
            NodeConfig(
                name="mock-node-chat",
                base_url="http://mock-chat:11434",
                engine=NodeEngine.OLLAMA,
                roles=["chat", "reasoning"],
                priority=1,
                pinned_model="qwen2.5-coder:14b",
                model_aliases={"reasoning-chat": "qwen2.5-coder:14b"},
            ),
        ],
    )
    return create_app(config)


@pytest.mark.asyncio
async def test_health_endpoints(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test /health
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 2
        assert "status" in data

        # Test /health/nodes
        resp_nodes = await client.get("/health/nodes")
        assert resp_nodes.status_code == 200
        nodes_data = resp_nodes.json()
        assert len(nodes_data) == 2
        assert nodes_data[0]["name"] == "mock-node-fim"
        assert nodes_data[1]["name"] == "mock-node-chat"


@pytest.mark.asyncio
async def test_models_endpoint(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "tab-autocomplete" in model_ids
        assert "reasoning-chat" in model_ids
        assert "qwen2.5-coder:1.5b-base" in model_ids
        assert "qwen2.5-coder:14b" in model_ids


@pytest.mark.asyncio
async def test_status_page_endpoint(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/status")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
