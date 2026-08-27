import pytest
from httpx import ASGITransport, AsyncClient

from mesh_gateway.discovery import AutoRoleClassifier, DiscoveryManager
from mesh_gateway.models import MeshConfig, MeshSettings, NodeConfig
from mesh_gateway.server import create_app


def test_auto_role_classifier():
    # 1. Autocomplete model inference
    roles_fim = AutoRoleClassifier.infer_roles("qwen2.5-coder:1.5b-base")
    assert "autocomplete" in roles_fim
    assert "chat" not in roles_fim

    # 2. Reasoning model inference
    roles_reasoning = AutoRoleClassifier.infer_roles("deepseek-r1:8b")
    assert "reasoning" in roles_reasoning
    assert "chat" in roles_reasoning

    # 3. Explicit role override
    roles_override = AutoRoleClassifier.infer_roles("deepseek-r1:8b", explicit_role="autocomplete")
    assert roles_override == ["autocomplete"]

    # 4. Multi-role explicit override
    roles_multi = AutoRoleClassifier.infer_roles("custom-model", explicit_role="chat, reasoning")
    assert roles_multi == ["chat", "reasoning"]


@pytest.mark.asyncio
async def test_api_mesh_register_endpoint():
    cfg = MeshConfig(
        mesh=MeshSettings(auto_discovery=True, listen_port=8000),
        nodes=[],
    )
    app = create_app(cfg)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Register worker
        payload = {
            "name": "worker-gpu-desktop",
            "base_url": "http://10.0.0.216:11434",
            "pinned_model": "deepseek-r1:8b",
            "role": "chat,reasoning",
        }
        resp = await client.post("/api/mesh/register", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["node"]["name"] == "worker-gpu-desktop"
        assert "chat" in data["node"]["roles"]
        assert "reasoning" in data["node"]["roles"]

        # Check health summary has newly registered node
        h_resp = await client.get("/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert any(n["name"] == "worker-gpu-desktop" for n in h_data["nodes"])
