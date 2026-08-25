from __future__ import annotations

import asyncio
import datetime
import statistics
import time
from typing import Dict, List, Optional
import httpx
from rich.console import Console

from mesh_gateway.models import (
    MeshConfig,
    MeshHealthSummary,
    NodeConfig,
    NodeEngine,
    NodeHealthStatus,
    NodeState,
)

console = Console()


class HealthTracker:
    """Continuous background health, latency, and VRAM state monitor for multi-device fleet."""

    def __init__(self, config: MeshConfig):
        self.config = config
        self.statuses: Dict[str, NodeHealthStatus] = {}
        self.nodes_by_name: Dict[str, NodeConfig] = {n.name: n for n in config.nodes}
        self._running = False
        self._probe_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

        # Initialize node status models
        for node in config.nodes:
            self.statuses[node.name] = NodeHealthStatus(
                name=node.name,
                base_url=node.base_url,
                engine=node.engine,
                roles=node.roles,
                priority=node.priority,
                state=NodeState.INITIALIZING,
                pinned_model=node.pinned_model,
            )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        # Run initial probe immediately
        await self.probe_all_nodes()
        self._probe_task = asyncio.create_task(self._probe_loop())

    async def stop(self) -> None:
        self._running = False
        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def _probe_loop(self) -> None:
        interval = self.config.mesh.health_check_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.probe_all_nodes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[bold red]Health check loop error:[/bold red] {e}")

    async def probe_all_nodes(self) -> None:
        """Asynchronously probe all N devices in parallel."""
        tasks = [self.probe_node(node) for node in self.config.nodes]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def probe_node(self, node: NodeConfig) -> NodeHealthStatus:
        status = self.statuses[node.name]
        start_time = time.perf_counter()
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        headers = {}
        if node.api_key:
            headers["Authorization"] = f"Bearer {node.api_key}"

        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        close_client_after = self._client is None

        try:
            # Determine health probe endpoint based on engine
            if node.engine == NodeEngine.OLLAMA:
                probe_url = f"{node.base_url.rstrip('/')}/api/tags"
                ps_url = f"{node.base_url.rstrip('/')}/api/ps"
            elif node.engine == NodeEngine.VLLM or node.engine == NodeEngine.OPENAI:
                probe_url = f"{node.base_url.rstrip('/')}/v1/models"
                ps_url = None
            else:
                probe_url = node.base_url
                ps_url = None

            resp = await client.get(probe_url, headers=headers)
            rtt_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            if resp.status_code == 200:
                # Node responded successfully
                status.latency_ms = rtt_ms
                status.last_check = now_str
                status.error_message = None

                # Record latency history (keep last 30 samples)
                status.latency_history.append(rtt_ms)
                if len(status.latency_history) > 30:
                    status.latency_history.pop(0)

                # Compute P50 and P95 latency
                if status.latency_history:
                    status.p50_latency_ms = round(statistics.median(status.latency_history), 2)
                    sorted_latencies = sorted(status.latency_history)
                    p95_idx = int(len(sorted_latencies) * 0.95)
                    status.p95_latency_ms = round(sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)], 2)

                # Check VRAM active loaded models if Ollama
                loaded_models: List[str] = []
                pinned_warm = False

                if ps_url:
                    try:
                        ps_resp = await client.get(ps_url, headers=headers)
                        if ps_resp.status_code == 200:
                            ps_data = ps_resp.json()
                            for m in ps_data.get("models", []):
                                m_name = m.get("name", "")
                                if m_name:
                                    loaded_models.append(m_name)
                    except Exception:
                        pass

                # If pinned model is set, check if it's currently warm in VRAM
                if node.pinned_model:
                    pinned_warm = any(node.pinned_model in m for m in loaded_models)
                
                status.loaded_models = loaded_models
                status.pinned_model_warm = pinned_warm

                # Update State: DEGRADED if latency exceeds threshold, else ONLINE
                if rtt_ms > self.config.mesh.degraded_latency_threshold_ms:
                    status.state = NodeState.DEGRADED
                else:
                    status.state = NodeState.ONLINE
            else:
                status.state = NodeState.DEGRADED
                status.error_message = f"HTTP {resp.status_code}"
                status.last_check = now_str

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            status.state = NodeState.OFFLINE
            status.error_message = f"Connection failed: {type(e).__name__}"
            status.last_check = now_str
            status.pinned_model_warm = False
        except Exception as e:
            status.state = NodeState.OFFLINE
            status.error_message = str(e)
            status.last_check = now_str
            status.pinned_model_warm = False
        finally:
            if close_client_after:
                await client.aclose()

        return status

    def record_request_start(self, node_name: str) -> None:
        if node_name in self.statuses:
            self.statuses[node_name].active_requests += 1
            self.statuses[node_name].total_requests += 1

    def record_request_end(self, node_name: str, success: bool, tokens: int = 0) -> None:
        if node_name in self.statuses:
            st = self.statuses[node_name]
            st.active_requests = max(0, st.active_requests - 1)
            if not success:
                st.failed_requests += 1
            if tokens > 0:
                st.tokens_generated += tokens

    def get_summary(self) -> MeshHealthSummary:
        node_list = list(self.statuses.values())
        online = sum(1 for n in node_list if n.state == NodeState.ONLINE)
        degraded = sum(1 for n in node_list if n.state == NodeState.DEGRADED)
        offline = sum(1 for n in node_list if n.state == NodeState.OFFLINE)
        active = sum(n.active_requests for n in node_list)
        total = sum(n.total_requests for n in node_list)

        overall_status = "healthy"
        if online == 0 and degraded == 0:
            overall_status = "critical"
        elif offline > 0 or degraded > 0:
            overall_status = "degraded"

        return MeshHealthSummary(
            status=overall_status,
            total_nodes=len(node_list),
            online_nodes=online,
            degraded_nodes=degraded,
            offline_nodes=offline,
            active_requests=active,
            total_requests=total,
            nodes=node_list,
        )

    def get_healthy_nodes(self, role: Optional[str] = None) -> List[NodeConfig]:
        """Return healthy node configs matching role, sorted by priority (1 is highest) and latency."""
        candidates: List[tuple[NodeConfig, NodeHealthStatus]] = []
        for node in self.config.nodes:
            st = self.statuses.get(node.name)
            if not st or st.state == NodeState.OFFLINE:
                continue
            if role and role not in node.roles:
                continue
            candidates.append((node, st))

        # Sort: priority ascending (1 before 2), then active requests, then latency
        candidates.sort(key=lambda item: (item[0].priority, item[1].active_requests, item[1].latency_ms))
        return [c[0] for c in candidates]
