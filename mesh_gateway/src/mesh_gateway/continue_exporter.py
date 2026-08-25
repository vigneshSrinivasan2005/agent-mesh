from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console

from mesh_gateway.models import MeshConfig

console = Console()


def generate_continue_yaml(config: MeshConfig) -> str:
    """Generate Continue.dev YAML configuration (config.yaml)."""
    gateway_base = f"http://127.0.0.1:{config.mesh.listen_port}/v1"

    data: Dict[str, Any] = {
        "models": [
            {
                "name": "Mesh-Reasoning (Multi-Device)",
                "provider": "openai",
                "model": "reasoning-chat",
                "apiBase": gateway_base,
                "roles": ["chat", "edit"],
            },
            {
                "name": "Mesh-Fast-Edit",
                "provider": "openai",
                "model": "fast-edit",
                "apiBase": gateway_base,
                "roles": ["edit"],
            },
        ],
        "tabAutocompleteModel": {
            "title": "Mesh-Autocomplete (Pinned FIM)",
            "provider": "openai",
            "model": "tab-autocomplete",
            "apiBase": gateway_base,
        },
        "customCommands": [
            {
                "name": "mesh-status",
                "prompt": "Check the health and loaded models across the Agent-Mesh local network.",
                "description": "Agent-Mesh Status Check",
            }
        ],
    }

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def generate_continue_json(config: MeshConfig) -> str:
    """Generate Continue.dev JSON configuration (config.json legacy format)."""
    gateway_base = f"http://127.0.0.1:{config.mesh.listen_port}/v1"

    data = {
        "models": [
            {
                "title": "Mesh-Reasoning",
                "provider": "openai",
                "model": "reasoning-chat",
                "apiBase": gateway_base,
            }
        ],
        "tabAutocompleteModel": {
            "title": "Mesh-Autocomplete",
            "provider": "openai",
            "model": "tab-autocomplete",
            "apiBase": gateway_base,
        },
    }
    return json.dumps(data, indent=2)


def generate_cline_config(config: MeshConfig) -> str:
    """Generate Cline / Roo Code settings snippet."""
    gateway_base = f"http://127.0.0.1:{config.mesh.listen_port}/v1"
    data = {
        "apiProvider": "openai-compatible",
        "openAiBaseUrl": gateway_base,
        "openAiModelId": "reasoning-chat",
        "openAiApiKey": "not-needed-for-local-mesh",
    }
    return json.dumps(data, indent=2)


def export_to_continue_file(config: MeshConfig, target_path: Optional[Path] = None) -> Path:
    if target_path is None:
        target_path = Path.home() / ".continue" / "config.yaml"

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists, create a backup
    if target_path.exists():
        backup_path = target_path.with_suffix(".yaml.bak")
        with (
            open(target_path, "r", encoding="utf-8") as src,
            open(backup_path, "w", encoding="utf-8") as dst,
        ):
            dst.write(src.read())
        console.print(f"[dim]Backed up existing config to {backup_path}[/dim]")

    yaml_content = generate_continue_yaml(config)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return target_path
