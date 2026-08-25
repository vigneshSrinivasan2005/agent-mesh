from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
import httpx
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
import uvicorn

from mesh_gateway import __version__
from mesh_gateway.config_loader import get_default_config, load_config, save_config
from mesh_gateway.continue_exporter import (
    export_to_continue_file,
    generate_cline_config,
    generate_continue_yaml,
)
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import NodeState
from mesh_gateway.server import create_app

app = typer.Typer(
    name="agent-mesh",
    help="Agent-Mesh: Distributed Local LLM Gateway & Health Monitor",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to mesh-config.yaml"),
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Override listen host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Override listen port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the Agent-Mesh LLM Gateway and Network Health Monitor."""
    cfg = load_config(config)

    listen_host = host or cfg.mesh.listen_host
    listen_port = port or cfg.mesh.listen_port

    console.print(
        Panel.fit(
            f"[bold cyan]Agent-Mesh Gateway v{__version__}[/bold cyan]\n"
            f"[green]OpenAI-Compatible Base URL:[/green] http://{listen_host}:{listen_port}/v1\n"
            f"[yellow]Health Status Dashboard:[/yellow]    http://{listen_host}:{listen_port}/status\n"
            f"[blue]Configured Nodes:[/blue]             {len(cfg.nodes)} physical devices\n"
            f"[dim]Tab Autocomplete -> Fast FIM Nodes | Chat/Edit -> Heavy Reasoning Nodes[/dim]",
            border_style="cyan",
            title="Local Multi-Device Mesh Active",
        )
    )

    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=listen_host, port=listen_port, reload=reload)


@app.command()
def status(
    gateway_url: str = typer.Option("http://127.0.0.1:8000", "--url", "-u", help="Agent-Mesh gateway URL"),
):
    """Query live mesh health and render device status table."""
    async def _fetch():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{gateway_url.rstrip('/')}/health")
                if resp.status_code != 200:
                    console.print(f"[bold red]Gateway returned status {resp.status_code}[/bold red]")
                    return
                data = resp.json()
                _render_status_table(data)
        except Exception as e:
            console.print(f"[bold red]Failed to connect to Gateway at {gateway_url}:[/bold red] {e}")
            console.print("[yellow]Make sure the gateway is running with:[/yellow] [bold]agent-mesh run[/bold]")

    asyncio.run(_fetch())


def _render_status_table(data: dict) -> None:
    overall_status = data.get("status", "unknown")
    status_color = "green" if overall_status == "healthy" else ("yellow" if overall_status == "degraded" else "red")

    console.print(f"\n[bold]Mesh Fleet Health:[/bold] [{status_color}]{overall_status.upper()}[/{status_color}]  "
                  f"([green]{data.get('online_nodes', 0)} Online[/green] / "
                  f"[yellow]{data.get('degraded_nodes', 0)} Degraded[/yellow] / "
                  f"[red]{data.get('offline_nodes', 0)} Offline[/red])\n")

    table = Table(title="Device Fleet Status", box=ROUNDED, header_style="bold cyan")
    table.add_column("Node Name", style="bold")
    table.add_column("Base URL")
    table.add_column("Roles", style="magenta")
    table.add_column("Priority", justify="center")
    table.add_column("State", justify="center")
    table.add_column("RTT Latency", justify="right")
    table.add_column("P50 / P95", justify="right")
    table.add_column("Pinned Model", style="blue")
    table.add_column("VRAM Warm", justify="center")
    table.add_column("Reqs (Act/Tot)", justify="right")

    for n in data.get("nodes", []):
        state = n.get("state", "UNKNOWN")
        if state == "ONLINE":
            state_str = "[bold green]ONLINE[/bold green]"
        elif state == "DEGRADED":
            state_str = "[bold yellow]DEGRADED[/bold yellow]"
        else:
            state_str = "[bold red]OFFLINE[/bold red]"

        warm_str = "[bold green]YES[/bold green]" if n.get("pinned_model_warm") else "[dim]NO[/dim]"
        roles_str = ", ".join(n.get("roles", []))
        latency_str = f"{n.get('latency_ms', 0):.1f} ms" if state != "OFFLINE" else "[red]--[/red]"
        p_str = f"{n.get('p50_latency_ms', 0):.0f} / {n.get('p95_latency_ms', 0):.0f} ms" if state != "OFFLINE" else "--"
        reqs_str = f"{n.get('active_requests', 0)} / {n.get('total_requests', 0)}"

        table.add_row(
            n.get("name", ""),
            n.get("base_url", ""),
            roles_str,
            str(n.get("priority", 1)),
            state_str,
            latency_str,
            p_str,
            n.get("pinned_model", "--") or "--",
            warm_str,
            reqs_str,
        )

    console.print(table)


@app.command()
def ping(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to mesh-config.yaml"),
):
    """Perform a one-time network latency & VRAM check across all configured devices."""
    cfg = load_config(config)
    tracker = HealthTracker(cfg)

    async def _ping_all():
        with console.status("[bold cyan]Pinging all devices across the network...[/bold cyan]"):
            await tracker.probe_all_nodes()
        summary = tracker.get_summary()
        _render_status_table(summary.model_dump(mode="json"))

    asyncio.run(_ping_all())


@app.command()
def init(
    output: str = typer.Option("mesh-config.yaml", "--output", "-o", help="Destination config path"),
):
    """Generate a starter mesh-config.yaml with recommended multi-device defaults."""
    out_path = Path(output)
    if out_path.exists():
        console.print(f"[bold yellow]File '{output}' already exists. Overwrite? (y/n):[/bold yellow] ", end="")
        choice = input().strip().lower()
        if choice != "y":
            console.print("[dim]Aborted.[/dim]")
            return

    cfg = get_default_config()
    save_config(cfg, out_path)
    console.print(f"[bold green]✓ Starter configuration created at '{output}'[/bold green]")
    console.print("[dim]Edit this file with your device IPs and run: [bold]agent-mesh run[/bold][/dim]")


@app.command(name="export-continue")
def export_continue(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to mesh-config.yaml"),
    apply_to_vscode: bool = typer.Option(
        False, "--apply", "-a", help="Automatically write to ~/.continue/config.yaml"
    ),
):
    """Generate or apply Continue.dev configuration for VS Code."""
    cfg = load_config(config)
    if apply_to_vscode:
        target = export_to_continue_file(cfg)
        console.print(f"[bold green]✓ Continue.dev config written to:[/bold green] {target}")
        console.print("[dim]Reload VS Code / Continue.dev to start using your multi-device mesh.[/dim]")
    else:
        yaml_str = generate_continue_yaml(cfg)
        console.print(Panel(yaml_str, title="~/.continue/config.yaml", border_style="cyan"))
        console.print("[dim]Run [bold]agent-mesh export-continue --apply[/bold] to write this directly to your VS Code Continue config.[/dim]")


@app.command(name="export-cline")
def export_cline(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to mesh-config.yaml"),
):
    """Output configuration snippet for Cline / Roo Code extension."""
    cfg = load_config(config)
    json_str = generate_cline_config(cfg)
    console.print(Panel(json_str, title="Cline / Roo Code Settings", border_style="green"))


if __name__ == "__main__":
    app()
