from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console

from mesh_gateway.models import MeshConfig, MeshSettings, NodeConfig, NodeEngine

console = Console()

DEFAULT_CONFIG_LOCATIONS = [
    Path("mesh-config.yaml"),
    Path("mesh-config.yml"),
    Path.home() / ".agent-mesh" / "config.yaml",
    Path("config.example.yaml"),
]


def get_default_config() -> MeshConfig:
    return MeshConfig(
        mesh=MeshSettings(
            listen_host="0.0.0.0",
            listen_port=8000,
            health_check_interval_seconds=5.0,
            fallback_enabled=True,
        ),
        nodes=[
            NodeConfig(
                name="macbook-fast-fim",
                base_url="http://localhost:11434",
                engine=NodeEngine.OLLAMA,
                roles=["autocomplete"],
                priority=1,
                pinned_model="qwen2.5-coder:1.5b-base",
                model_aliases={
                    "tab-autocomplete": "qwen2.5-coder:1.5b-base",
                    "qwen-1.5b": "qwen2.5-coder:1.5b-base",
                },
            ),
            NodeConfig(
                name="desktop-heavy-reasoning",
                base_url="http://localhost:11434",
                engine=NodeEngine.OLLAMA,
                roles=["chat", "edit", "reasoning"],
                priority=1,
                pinned_model="qwen2.5-coder:14b",
                model_aliases={
                    "reasoning-chat": "qwen2.5-coder:14b",
                    "deepseek-r1": "deepseek-r1:14b",
                    "qwen-14b": "qwen2.5-coder:14b",
                },
            ),
        ],
    )


def load_config(config_path: Optional[Path | str] = None) -> MeshConfig:
    """Load and parse mesh config from file or defaults."""
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return _read_config_file(path)

    # Search standard paths
    for location in DEFAULT_CONFIG_LOCATIONS:
        if location.exists():
            return _read_config_file(location)

    # Return default config if no file found
    return get_default_config()


def _read_config_file(path: Path) -> MeshConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Parse using Pydantic
    mesh_data = data.get("mesh", {})
    nodes_data = data.get("nodes", [])

    mesh_settings = MeshSettings(**mesh_data) if mesh_data else MeshSettings()
    nodes = [NodeConfig(**n) for n in nodes_data]

    return MeshConfig(mesh=mesh_settings, nodes=nodes)


def save_config(config: MeshConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
