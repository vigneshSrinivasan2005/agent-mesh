from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import httpx
import typer
import uvicorn
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mesh_gateway import __version__
from mesh_gateway.config_loader import get_default_config, load_config, save_config
from mesh_gateway.continue_exporter import (
    export_to_continue_file,
    generate_cline_config,
    generate_continue_yaml,
)
from mesh_gateway.health import HealthTracker
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
    gateway_url: str = typer.Option(
        "http://127.0.0.1:8000", "--url", "-u", help="Agent-Mesh gateway URL"
    ),
):
    """Query live mesh health and render device status table."""

    async def _fetch():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{gateway_url.rstrip('/')}/health")
                if resp.status_code != 200:
                    console.print(
                        f"[bold red]Gateway returned status {resp.status_code}[/bold red]"
                    )
                    return
                data = resp.json()
                _render_status_table(data)
        except Exception as e:
            console.print(
                f"[bold red]Failed to connect to Gateway at {gateway_url}:[/bold red] {e}"
            )
            console.print(
                "[yellow]Make sure the gateway is running with:[/yellow] [bold]agent-mesh run[/bold]"
            )

    asyncio.run(_fetch())


def _render_status_table(data: dict) -> None:
    overall_status = data.get("status", "unknown")
    status_color = (
        "green"
        if overall_status == "healthy"
        else ("yellow" if overall_status == "degraded" else "red")
    )

    console.print(
        f"\n[bold]Mesh Fleet Health:[/bold] [{status_color}]{overall_status.upper()}[/{status_color}]  "
        f"([green]{data.get('online_nodes', 0)} Online[/green] / "
        f"[yellow]{data.get('degraded_nodes', 0)} Degraded[/yellow] / "
        f"[red]{data.get('offline_nodes', 0)} Offline[/red])\n"
    )

    table = Table(title="Device Fleet Status", box=ROUNDED, header_style="bold cyan")
    table.add_column("Node Name", style="bold")
    table.add_column("Base URL")
    table.add_column("Roles", style="magenta")
    table.add_column("Priority", justify="center")
    table.add_column("State", justify="center")
    table.add_column("RTT Latency", justify="right")
    table.add_column("P50 / P95", justify="right")
    table.add_column("Type", justify="center")
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
        p_str = (
            f"{n.get('p50_latency_ms', 0):.0f} / {n.get('p95_latency_ms', 0):.0f} ms"
            if state != "OFFLINE"
            else "--"
        )
        reqs_str = f"{n.get('active_requests', 0)} / {n.get('total_requests', 0)}"

        is_cont = n.get("is_container", False)
        type_str = "[cyan]Docker[/cyan]" if is_cont else "[dim]Native[/dim]"
        if n.get("is_auto_scaled"):
            type_str += " [yellow](scaled)[/yellow]"

        table.add_row(
            n.get("name", ""),
            n.get("base_url", ""),
            roles_str,
            str(n.get("priority", 1)),
            state_str,
            latency_str,
            p_str,
            type_str,
            n.get("pinned_model", "--") or "--",
            warm_str,
            reqs_str,
        )

    console.print(table)


# Subcommand group for node container operations
node_app = typer.Typer(help="Manage local Docker worker containers for Agent-Mesh.")


@app.command("leader-up")
def leader_up(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Optional model for co-located local worker"),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Role for local worker (e.g. autocomplete)"),
    port: int = typer.Option(8000, "--port", "-p", help="Gateway listen port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Gateway listen host"),
):
    """Start the Master Gateway & Dashboard in Zero-Config LAN Auto-Discovery Mode."""
    from mesh_gateway.config_loader import load_config
    cfg = load_config()
    cfg.mesh.listen_host = host
    cfg.mesh.listen_port = port
    cfg.mesh.auto_discovery = True

    console.print(
        Panel.fit(
            f"[bold cyan]Agent-Mesh Leader Gateway v{__version__}[/bold cyan]\n"
            f"[green]OpenAI-Compatible Base URL:[/green] http://{host}:{port}/v1\n"
            f"[yellow]Health Status Dashboard:[/yellow]    http://{host}:{port}/status\n"
            f"[blue]LAN Auto-Discovery:[/blue]          [bold green]ENABLED (UDP Beacon & Subnet Prober)[/bold green]\n"
            f"[dim]Auto-detecting local & LAN worker devices continuously...[/dim]",
            border_style="cyan",
            title="Leader Active",
        )
    )

    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command("worker-up")
def worker_up(
    model: str = typer.Argument("qwen2.5-coder:1.5b-base", help="Model name and version (e.g. deepseek-r1:8b)"),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Explicit role: autocomplete, chat, edit, reasoning"),
    port: int = typer.Option(11434, "--port", "-p", help="Host port to bind for worker"),
    gpu: bool = typer.Option(False, "--gpu", "-g", help="Enable NVIDIA GPU acceleration"),
    image: str = typer.Option("ollama/ollama:latest", "--image", "-i", help="Worker container image"),
    leader: Optional[str] = typer.Option(None, "--leader", "-l", help="Leader host URL (optional)"),
):
    """Launch a containerized worker node and auto-announce to leader."""
    from mesh_gateway.discovery import AutoRoleClassifier, DiscoveryManager
    from mesh_gateway.docker_scaler import DockerScaler
    from mesh_gateway.models import AutoScalingConfig

    inferred_roles = AutoRoleClassifier.infer_roles(model, role)
    roles_str = ",".join(inferred_roles)

    if not DockerScaler.is_docker_available():
        console.print("[bold red]Docker is not running. Starting native announcement instead...[/bold red]")
    else:
        scaler = DockerScaler(AutoScalingConfig())
        with console.status(f"[bold cyan]Starting Docker worker for {model} on port {port}...[/bold cyan]"):
            success, cid, assigned_port = asyncio.run(
                scaler.start_worker_container(role=roles_str, model=model, port=port, gpu=gpu, image=image)
            )
            if success:
                console.print(f"[bold green]✓ Worker container {cid[:12]} running on port {assigned_port}![/bold green]")

    # Start discovery announcer
    console.print(
        Panel.fit(
            f"[bold cyan]Agent-Mesh Worker Node[/bold cyan]\n"
            f"[green]Model:[/green]       {model}\n"
            f"[magenta]Roles:[/magenta]       {roles_str}\n"
            f"[blue]Port:[/blue]        {port}\n"
            f"[dim]Announcing to cluster leader...[/dim]",
            border_style="magenta",
            title="Worker Active",
        )
    )

    async def _announce():
        mgr = DiscoveryManager(
            node_name=f"worker-{model.replace(':', '-')}",
            is_leader=False,
            base_url=f"http://127.0.0.1:{port}",
            role=roles_str,
            pinned_model=model,
            leader_host=leader,
        )
        await mgr.start()
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await mgr.stop()

    asyncio.run(_announce())


@app.command("node-up")
@node_app.command("up")
def node_up(
    role: str = typer.Option(
        "reasoning", "--role", "-r", help="Role: reasoning, autocomplete, chat, edit"
    ),
    model: str = typer.Option("qwen2.5-coder:14b", "--model", "-m", help="Model name to preload"),
    port: int = typer.Option(11434, "--port", "-p", help="Host port to bind"),
    gpu: bool = typer.Option(False, "--gpu", "-g", help="Enable NVIDIA GPU acceleration"),
    image: str = typer.Option(
        "ollama/ollama:latest", "--image", "-i", help="Ollama or vLLM container image"
    ),
):
    """Launch a local Docker worker container for the mesh."""
    from mesh_gateway.docker_scaler import DockerScaler
    from mesh_gateway.models import AutoScalingConfig

    if not DockerScaler.is_docker_available():
        console.print(
            "[bold red]Docker is not installed or the Docker daemon is not running.[/bold red]"
        )
        raise typer.Exit(1)

    scaler = DockerScaler(AutoScalingConfig())
    with console.status(
        f"[bold cyan]Starting Docker worker for {role} ({model}) on port {port}...[/bold cyan]"
    ):
        success, cid, assigned_port = asyncio.run(
            scaler.start_worker_container(role=role, model=model, port=port, gpu=gpu, image=image)
        )

    if success:
        console.print("[bold green]✓ Docker worker container running![/bold green]")
        console.print(f"  • Container ID: [cyan]{cid}[/cyan]")
        console.print(f"  • Endpoint:     [bold]http://127.0.0.1:{assigned_port}[/bold]")
        console.print(f"  • Role:         [magenta]{role}[/magenta]")
        console.print(f"  • Model:        [blue]{model}[/blue]")
    else:
        console.print("[bold red]Failed to start Docker worker container.[/bold red]")
        raise typer.Exit(1)


@app.command("node-down")
@node_app.command("down")
def node_down(
    container_id: str = typer.Argument(
        ..., help="Container ID or name (e.g. agent-mesh-gpu-worker or ID)"
    ),
):
    """Stop and remove a local Docker worker container."""
    from mesh_gateway.docker_scaler import DockerScaler
    from mesh_gateway.models import AutoScalingConfig

    scaler = DockerScaler(AutoScalingConfig())
    with console.status(f"[bold yellow]Stopping container {container_id}...[/bold yellow]"):
        success = asyncio.run(scaler.stop_worker_container(container_id))

    if success:
        console.print(f"[bold green]✓ Container {container_id} stopped and removed.[/bold green]")
    else:
        console.print(f"[bold red]Failed to stop container {container_id}.[/bold red]")


# Subcommand group for gRPC Node Agent daemon
agent_app = typer.Typer(
    help="Manage and run the Agent-Mesh gRPC Node Agent daemon on worker machines."
)


@agent_app.command("run")
def agent_run(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host for gRPC server"),
    port: int = typer.Option(50051, "--port", "-p", help="Bind port for gRPC server"),
    name: str = typer.Option("local-worker", "--name", "-n", help="Node identifier name"),
    engine_url: str = typer.Option(
        "http://127.0.0.1:11434", "--engine-url", "-e", help="Local Ollama/vLLM URL"
    ),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Optional role e.g. autocomplete"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Pinned model name"),
    leader: Optional[str] = typer.Option(None, "--leader", "-l", help="Leader host URL"),
):
    """Run the gRPC Node Agent daemon with automatic LAN discovery."""
    from mesh_gateway.node_agent import serve_node_agent

    console.print(
        Panel.fit(
            f"[bold cyan]Agent-Mesh gRPC Node Agent[/bold cyan]\n"
            f"[green]gRPC Endpoint:[/green]     {host}:{port}\n"
            f"[blue]Local Engine URL:[/blue]  {engine_url}\n"
            f"[yellow]Node Identity:[/yellow]     {name}\n"
            f"[dim]Streaming telemetry and auto-broadcasting on LAN[/dim]",
            border_style="cyan",
            title="Node Agent Active",
        )
    )

    async def _main():
        server, discovery = await serve_node_agent(
            host=host,
            port=port,
            node_name=name,
            local_engine_url=engine_url,
            role=role,
            pinned_model=model,
            leader_host=leader,
        )
        try:
            await server.wait_for_termination()
        finally:
            await discovery.stop()

    asyncio.run(_main())


app.add_typer(agent_app, name="agent")


app.add_typer(node_app, name="node")


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
    output: str = typer.Option(
        "mesh-config.yaml", "--output", "-o", help="Destination config path"
    ),
):
    """Generate a starter mesh-config.yaml with recommended multi-device defaults."""
    out_path = Path(output)
    if out_path.exists():
        console.print(
            f"[bold yellow]File '{output}' already exists. Overwrite? (y/n):[/bold yellow] ", end=""
        )
        choice = input().strip().lower()
        if choice != "y":
            console.print("[dim]Aborted.[/dim]")
            return

    cfg = get_default_config()
    save_config(cfg, out_path)
    console.print(f"[bold green]✓ Starter configuration created at '{output}'[/bold green]")
    console.print(
        "[dim]Edit this file with your device IPs and run: [bold]agent-mesh run[/bold][/dim]"
    )


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
        console.print(
            "[dim]Reload VS Code / Continue.dev to start using your multi-device mesh.[/dim]"
        )
    else:
        yaml_str = generate_continue_yaml(cfg)
        console.print(Panel(yaml_str, title="~/.continue/config.yaml", border_style="cyan"))
        console.print(
            "[dim]Run [bold]agent-mesh export-continue --apply[/bold] to write this directly to your VS Code Continue config.[/dim]"
        )


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
