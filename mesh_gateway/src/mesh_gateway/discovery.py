from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from mesh_gateway.models import NodeConfig, NodeEngine

logger = logging.getLogger("agent_mesh.discovery")

DISCOVERY_PORT = 50052
BROADCAST_INTERVAL = 3.0


class AutoRoleClassifier:
    """Classifies model roles based on name, parameter size, or user overrides."""

    @staticmethod
    def infer_roles(model_name: str, explicit_role: Optional[str] = None) -> List[str]:
        if explicit_role:
            roles = [r.strip().lower() for r in explicit_role.split(",") if r.strip()]
            if roles:
                return roles

        model_lower = model_name.lower()

        # FIM / Autocomplete models (< 4B, coder, fim, base)
        if any(x in model_lower for x in ["fim", "autocomplete", "1.5b", "0.5b", "3b"]) and "reasoning" not in model_lower:
            return ["autocomplete"]
        if "base" in model_lower and ("1.5b" in model_lower or "3b" in model_lower):
            return ["autocomplete"]

        # Heavy reasoning / chat models (>= 7B, r1, deepseek, llama, qwen-chat)
        if any(x in model_lower for x in ["r1", "reasoning", "deepseek", "thinking"]):
            return ["chat", "edit", "reasoning"]

        return ["chat", "edit", "reasoning"]


class BeaconProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler for receiving node discovery announcements."""

    def __init__(self, on_message: Callable[[dict, Tuple[str, int]], None]):
        self.on_message = on_message
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        try:
            msg = json.loads(data.decode("utf-8"))
            self.on_message(msg, addr)
        except Exception:
            pass


class DiscoveryManager:
    """Manages LAN broadcast beaconing, listening, and auto-registration of nodes."""

    def __init__(
        self,
        node_name: str,
        is_leader: bool = False,
        base_url: Optional[str] = None,
        grpc_port: Optional[int] = None,
        role: Optional[str] = None,
        pinned_model: Optional[str] = None,
        leader_host: Optional[str] = None,
        on_node_discovered: Optional[Callable[[NodeConfig], None]] = None,
    ):
        self.node_name = node_name
        self.is_leader = is_leader
        self.base_url = base_url
        self.grpc_port = grpc_port
        self.role = role
        self.pinned_model = pinned_model
        self.leader_host = leader_host
        self.on_node_discovered = on_node_discovered

        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None
        self._probe_task: Optional[asyncio.Task] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._known_nodes: Dict[str, float] = {}

    async def start(self):
        if self._running:
            return
        self._running = True

        loop = asyncio.get_running_loop()

        # 1. Setup UDP listener socket with SO_REUSEPORT/SO_REUSEADDR
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            sock.bind(("", DISCOVERY_PORT))
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: BeaconProtocol(self._on_beacon_received),
                sock=sock,
            )
        except Exception as e:
            logger.warning(f"Could not bind discovery UDP socket on port {DISCOVERY_PORT}: {e}")

        # 2. Start broadcast task
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        # 3. If leader, start periodic local subnet & localhost prober
        if self.is_leader:
            self._probe_task = asyncio.create_task(self._leader_probe_loop())

        # 4. If worker with explicit leader_host, self-register immediately via HTTP
        if not self.is_leader and self.leader_host:
            asyncio.create_task(self._self_register_http())

    async def stop(self):
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._probe_task:
            self._probe_task.cancel()
        if self._transport:
            self._transport.close()

    def _on_beacon_received(self, msg: dict, addr: Tuple[str, int]):
        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type")
        sender_name = msg.get("name")
        if not sender_name or sender_name == self.node_name:
            return

        sender_ip = addr[0]

        if self.is_leader and msg_type == "agent_mesh_worker":
            # Worker discovered by leader
            worker_port = msg.get("http_port", 11434)
            base_url = msg.get("base_url") or f"http://{sender_ip}:{worker_port}"
            model = msg.get("model")
            explicit_role = msg.get("role")
            roles = AutoRoleClassifier.infer_roles(model or "", explicit_role)

            node_cfg = NodeConfig(
                name=sender_name,
                base_url=base_url,
                engine=NodeEngine.OLLAMA,
                roles=roles,
                priority=1,
                pinned_model=model,
                grpc_port=msg.get("grpc_port"),
            )

            now = time.time()
            if sender_name not in self._known_nodes or (now - self._known_nodes[sender_name]) > 30:
                self._known_nodes[sender_name] = now
                if self.on_node_discovered:
                    self.on_node_discovered(node_cfg)

        elif not self.is_leader and msg_type == "agent_mesh_leader":
            leader_port = msg.get("port", 8000)
            self.leader_host = f"http://{sender_ip}:{leader_port}"
            asyncio.create_task(self._self_register_http())

    async def _broadcast_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        while self._running:
            try:
                if self.is_leader:
                    payload = {
                        "type": "agent_mesh_leader",
                        "name": self.node_name,
                        "port": 8000,
                        "timestamp": time.time(),
                    }
                else:
                    payload = {
                        "type": "agent_mesh_worker",
                        "name": self.node_name,
                        "http_port": 11434,
                        "grpc_port": self.grpc_port,
                        "model": self.pinned_model,
                        "role": self.role,
                        "base_url": self.base_url,
                        "timestamp": time.time(),
                    }

                data = json.dumps(payload).encode("utf-8")
                sock.sendto(data, ("<broadcast>", DISCOVERY_PORT))
                sock.sendto(data, ("127.0.0.1", DISCOVERY_PORT))
            except Exception:
                pass
            await asyncio.sleep(BROADCAST_INTERVAL)

    async def _self_register_http(self):
        """Send HTTP registration request to leader."""
        if not self.leader_host:
            return

        url = f"{self.leader_host.rstrip("/")}/api/mesh/register"
        payload = {
            "name": self.node_name,
            "base_url": self.base_url or "http://127.0.0.1:11434",
            "grpc_port": self.grpc_port,
            "role": self.role,
            "pinned_model": self.pinned_model,
        }

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.post(url, json=payload)
        except Exception:
            pass

    async def _leader_probe_loop(self):
        """Periodically probe localhost:11434 to auto-detect local Ollama if not configured."""
        while self._running:
            try:
                await self.probe_and_register_url("http://127.0.0.1:11434", "local-ollama")
                await self.probe_and_register_url("http://localhost:11434", "local-ollama")
            except Exception:
                pass
            await asyncio.sleep(15.0)

    async def probe_and_register_url(self, base_url: str, default_name: str) -> Optional[NodeConfig]:
        """Query an Ollama endpoint, infer its installed models and roles, and register."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{base_url.rstrip("/")}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name") or m.get("model") for m in data.get("models", []) if m]
                    if not models:
                        return None

                    primary_model = models[0]
                    roles = AutoRoleClassifier.infer_roles(primary_model, self.role)

                    aliases = {}
                    if "autocomplete" in roles:
                        aliases["tab-autocomplete"] = primary_model
                    if any(r in roles for r in ["chat", "reasoning"]):
                        aliases["reasoning-chat"] = primary_model

                    node_cfg = NodeConfig(
                        name=default_name,
                        base_url=base_url,
                        engine=NodeEngine.OLLAMA,
                        roles=roles,
                        priority=1,
                        pinned_model=primary_model,
                        model_aliases=aliases,
                    )

                    if self.on_node_discovered:
                        self.on_node_discovered(node_cfg)
                    return node_cfg
        except Exception:
            pass
        return None
