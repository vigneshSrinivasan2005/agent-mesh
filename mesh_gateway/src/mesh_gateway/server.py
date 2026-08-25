from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from mesh_gateway.config_loader import load_config
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    ChatCompletionRequest,
    CompletionRequest,
    MeshConfig,
    MeshHealthSummary,
    ModelCard,
    ModelListResponse,
    NodeHealthStatus,
)
from mesh_gateway.router import MeshRouter


def create_app(config: Optional[MeshConfig] = None) -> FastAPI:
    if config is None:
        config = load_config()

    health_tracker = HealthTracker(config)
    router = MeshRouter(config, health_tracker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: start background health prober
        await health_tracker.start()
        yield
        # Shutdown: stop prober and clean up connections
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
