from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from mesh_gateway.config_loader import load_config
from mesh_gateway.discovery import AutoRoleClassifier, DiscoveryManager
from mesh_gateway.docker_scaler import DockerScaler
from mesh_gateway.grpc_client import GrpcMeshManager
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    MeshConfig,
    MeshHealthSummary,
    ModelCard,
    ModelListResponse,
    NodeConfig,
    NodeHealthStatus,
    NodeRegistrationRequest,
)
from mesh_gateway.router import MeshRouter


def create_app(config: Optional[MeshConfig] = None) -> FastAPI:
    if config is None:
        config = load_config()

    grpc_manager = GrpcMeshManager(default_grpc_port=config.mesh.grpc_port or 50051)
    health_tracker = HealthTracker(config)
    router = MeshRouter(config, health_tracker, grpc_manager=grpc_manager)
    docker_scaler = DockerScaler(config.mesh.auto_scaling, router, health_tracker)

    def _on_grpc_telemetry(update):
        if update.node_name in health_tracker.statuses:
            st = health_tracker.statuses[update.node_name]
            st.grpc_connected = True
            st.gpu_vram_used_mb = update.used_vram_mb
            st.gpu_vram_total_mb = update.total_vram_mb
            st.gpu_utilization_pct = update.gpu_utilization_pct
            if update.loaded_models:
                st.loaded_models = list(update.loaded_models)

    def _on_node_discovered(discovered_node: NodeConfig):
        health_tracker.register_node(discovered_node)
        router.register_node(discovered_node)
        if discovered_node.transport != "http":
            asyncio.create_task(
                grpc_manager.register_node(discovered_node, telemetry_callback=_on_grpc_telemetry)
            )

    discovery_mgr = DiscoveryManager(
        node_name="leader-gateway",
        is_leader=True,
        on_node_discovered=_on_node_discovered,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: connect to gRPC nodes, start health prober, discovery & auto scaler
        for node in config.nodes:
            if node.transport != "http":
                asyncio.create_task(
                    grpc_manager.register_node(node, telemetry_callback=_on_grpc_telemetry)
                )
        await health_tracker.start()
        docker_scaler.start()
        if config.mesh.auto_discovery:
            await discovery_mgr.start()
        yield
        # Shutdown: stop discovery, scaler, prober and clean up connections
        if config.mesh.auto_discovery:
            await discovery_mgr.stop()
        await docker_scaler.stop()
        await health_tracker.stop()
        await router.close()

    app = FastAPI(
        title="Agent-Mesh Local LLM Gateway",
        description="Distributed multi-device routing & health monitoring gateway for VS Code, Continue.dev, and Cline",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for local web tools & extensions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references on state
    app.state.config = config
    app.state.health = health_tracker
    app.state.router = router
    app.state.discovery = discovery_mgr

    @app.get("/health", response_model=MeshHealthSummary)
    async def get_health_summary() -> MeshHealthSummary:
        return health_tracker.get_summary()

    @app.get("/health/nodes", response_model=List[NodeHealthStatus])
    async def get_node_health() -> List[NodeHealthStatus]:
        return list(health_tracker.statuses.values())

    @app.get("/api/mesh")
    async def get_mesh_info():
        return {
            "config": config.model_dump(mode="json"),
            "health": health_tracker.get_summary().model_dump(mode="json"),
        }

    @app.post("/api/mesh/register")
    async def register_node(req: NodeRegistrationRequest):
        roles = req.roles or AutoRoleClassifier.infer_roles(req.pinned_model or "", req.role)
        aliases = {}
        if req.pinned_model:
            if "autocomplete" in roles:
                aliases["tab-autocomplete"] = req.pinned_model
            if any(r in roles for r in ["chat", "reasoning"]):
                aliases["reasoning-chat"] = req.pinned_model

        node_cfg = NodeConfig(
            name=req.name,
            base_url=req.base_url,
            engine=req.engine,
            roles=roles,
            priority=req.priority,
            pinned_model=req.pinned_model,
            model_aliases=aliases,
            grpc_port=req.grpc_port,
        )
        _on_node_discovered(node_cfg)
        return {"status": "registered", "node": node_cfg.model_dump(mode="json")}

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        """Return unified list of virtual models, aliases, and physical node models."""
        models: List[ModelCard] = [
            ModelCard(id="tab-autocomplete", root="autocomplete", parent="mesh"),
            ModelCard(id="reasoning-chat", root="reasoning", parent="mesh"),
            ModelCard(id="fast-edit", root="edit", parent="mesh"),
        ]

        # Add models defined across nodes
        seen = {"tab-autocomplete", "reasoning-chat", "fast-edit"}
        for node in config.nodes:
            if node.pinned_model and node.pinned_model not in seen:
                models.append(ModelCard(id=node.pinned_model, root=node.name, parent="mesh"))
                seen.add(node.pinned_model)
            for alias in node.model_aliases.keys():
                if alias not in seen:
                    models.append(ModelCard(id=alias, root=node.name, parent="mesh"))
                    seen.add(alias)

        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        is_stream = body.get("stream", False)

        if is_stream:
            return StreamingResponse(
                router.forward_stream("/v1/chat/completions", body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            result = await router.forward_request("/v1/chat/completions", body)
            return JSONResponse(content=result)

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        is_stream = body.get("stream", False)

        if is_stream:
            return StreamingResponse(
                router.forward_stream("/v1/completions", body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            result = await router.forward_request("/v1/completions", body)
            return JSONResponse(content=result)

    @app.get("/status")
    async def status_dashboard():
        # Serve embedded web dashboard
        search_paths = [
            Path(__file__).parent.parent.parent.parent / "web_status" / "index.html",
            Path("web_status/index.html"),
        ]
        for p in search_paths:
            if p.exists():
                return FileResponse(p, media_type="text/html")

        # Fallback inline status HTML
        return HTMLResponse(
            "<html><body><h1>Agent-Mesh Gateway</h1><p>Status API available at <a href='/health'>/health</a></p></body></html>"
        )

    return app
