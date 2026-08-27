from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import grpc
import httpx

from mesh_gateway.docker_scaler import DockerScaler
from mesh_gateway.grpc_proto import mesh_service_pb2, mesh_service_pb2_grpc
from mesh_gateway.models import AutoScalingConfig

logger = logging.getLogger("mesh_gateway.node_agent")


def get_system_hardware_stats() -> tuple[int, int, float, float]:
    """
    Query system memory / VRAM and CPU utilization.
    Returns (used_vram_mb, total_vram_mb, gpu_util_pct, cpu_util_pct).
    """
    total_vram = 0
    used_vram = 0
    gpu_util = 0.0
    cpu_util = 0.0

    # Try NVML / torch if available
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_vram = int(mem_info.total / (1024 * 1024))
        used_vram = int(mem_info.used / (1024 * 1024))
        gpu_util = float(util_info.gpu)
    except Exception:
        pass

    return used_vram, total_vram, gpu_util, cpu_util


class NodeAgentServicer(mesh_service_pb2_grpc.NodeAgentServiceServicer):
    """gRPC service implementation running on physical worker nodes."""

    def __init__(
        self,
        node_name: str = "node-agent",
        local_engine_url: str = "http://127.0.0.1:11434",
        engine_type: str = "ollama",
    ):
        self.node_name = node_name
        self.local_engine_url = local_engine_url.rstrip("/")
        self.engine_type = engine_type
        self.total_requests = 0
        self.active_requests = 0
        self.tokens_generated = 0
        self.docker_scaler = DockerScaler(AutoScalingConfig())
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=3.0))

    async def close(self):
        await self._http_client.aclose()

    async def Ping(
        self, request: mesh_service_pb2.PingRequest, context: grpc.aio.ServicerContext
    ) -> mesh_service_pb2.PingResponse:
        now_ms = int(time.time() * 1000)
        return mesh_service_pb2.PingResponse(
            client_timestamp_ms=request.client_timestamp_ms,
            server_timestamp_ms=now_ms,
            node_name=self.node_name,
            engine=self.engine_type,
        )

    async def StreamTelemetry(
        self, request: mesh_service_pb2.TelemetryRequest, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[mesh_service_pb2.TelemetryUpdate, None]:
        """Continuously push hardware and VRAM telemetry to the main gateway."""
        interval_sec = max(0.5, request.interval_ms / 1000.0) if request.interval_ms > 0 else 2.0
        logger.info(
            f"Gateway '{request.gateway_id}' subscribed to telemetry stream ({interval_sec}s interval)"
        )

        while not context.cancelled():
            try:
                start_p = time.perf_counter()
                loaded_models = []
                pinned_warm = False
                state = "ONLINE"

                # Check local engine health & loaded VRAM models
                try:
                    ps_resp = await self._http_client.get(
                        f"{self.local_engine_url}/api/ps", timeout=2.0
                    )
                    if ps_resp.status_code == 200:
                        ps_data = ps_resp.json()
                        for m in ps_data.get("models", []):
                            m_name = m.get("name", "")
                            if m_name:
                                loaded_models.append(m_name)
                except Exception:
                    state = "DEGRADED"

                rtt = round((time.perf_counter() - start_p) * 1000.0, 2)
                used_vram, total_vram, gpu_util, cpu_util = get_system_hardware_stats()

                update = mesh_service_pb2.TelemetryUpdate(
                    node_name=self.node_name,
                    state=state,
                    rtt_ms=rtt,
                    p50_ms=rtt,
                    p95_ms=rtt,
                    total_vram_mb=total_vram,
                    used_vram_mb=used_vram,
                    gpu_utilization_pct=gpu_util,
                    cpu_utilization_pct=cpu_util,
                    loaded_models=loaded_models,
                    pinned_model="",
                    pinned_model_warm=pinned_warm,
                    active_requests=self.active_requests,
                    total_requests=self.total_requests,
                    tokens_generated=self.tokens_generated,
                    timestamp_ms=int(time.time() * 1000),
                )
                yield update
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telemetry push: {e}")

            await asyncio.sleep(interval_sec)

    async def StreamCompletion(
        self, request: mesh_service_pb2.CompletionRequestProto, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[mesh_service_pb2.CompletionChunkProto, None]:
        """Proxy streaming tokens from local engine to gRPC client."""
        target_url = f"{self.local_engine_url}{request.endpoint}"
        self.active_requests += 1
        self.total_requests += 1

        try:
            payload = json.loads(request.payload_json) if request.payload_json else {}
            payload["stream"] = True

            req = self._http_client.build_request(
                "POST",
                target_url,
                json=payload,
                timeout=httpx.Timeout(request.timeout_seconds or 120.0),
            )
            resp = await self._http_client.send(req, stream=True)

            if resp.status_code >= 400:
                err_text = await resp.aread()
                yield mesh_service_pb2.CompletionChunkProto(
                    raw_chunk="",
                    is_final=True,
                    error_message=f"HTTP {resp.status_code}: {err_text.decode('utf-8', errors='replace')}",
                )
                return

            async for chunk in resp.aiter_text():
                if chunk:
                    self.tokens_generated += 1
                    yield mesh_service_pb2.CompletionChunkProto(
                        raw_chunk=chunk,
                        is_final=False,
                    )

            yield mesh_service_pb2.CompletionChunkProto(raw_chunk="", is_final=True)
        except Exception as e:
            yield mesh_service_pb2.CompletionChunkProto(
                raw_chunk="",
                is_final=True,
                error_message=str(e),
            )
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    async def ControlContainer(
        self, request: mesh_service_pb2.ContainerControlRequest, context: grpc.aio.ServicerContext
    ) -> mesh_service_pb2.ContainerControlResponse:
        action = request.action.upper()
        if action == "START":
            success, cid, port = await self.docker_scaler.start_worker_container(
                role=request.role or "reasoning",
                model=request.model,
                port=request.port or None,
                gpu=request.gpu_enabled,
            )
            return mesh_service_pb2.ContainerControlResponse(
                success=success,
                container_id=cid,
                port=port,
                message=f"Container started on port {port}"
                if success
                else "Failed to start container",
            )
        elif action == "STOP":
            success = await self.docker_scaler.stop_worker_container(request.container_id)
            return mesh_service_pb2.ContainerControlResponse(
                success=success,
                container_id=request.container_id,
                message=f"Container {request.container_id} stopped"
                if success
                else "Failed to stop",
            )
        else:
            return mesh_service_pb2.ContainerControlResponse(
                success=False,
                message=f"Unknown action: {action}",
            )


async def serve_node_agent(
    host: str = "0.0.0.0",
    port: int = 50051,
    node_name: str = "local-node",
    local_engine_url: str = "http://127.0.0.1:11434",
    role: Optional[str] = None,
    pinned_model: Optional[str] = None,
    leader_host: Optional[str] = None,
) -> tuple[grpc.aio.Server, Any]:
    """Start the gRPC Node Agent server and discovery beacon."""
    from mesh_gateway.discovery import DiscoveryManager

    server = grpc.aio.server()
    servicer = NodeAgentServicer(
        node_name=node_name,
        local_engine_url=local_engine_url,
    )
    mesh_service_pb2_grpc.add_NodeAgentServiceServicer_to_server(servicer, server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Agent-Mesh Node Agent listening on gRPC {listen_addr}")
    await server.start()

    discovery = DiscoveryManager(
        node_name=node_name,
        is_leader=False,
        base_url=local_engine_url,
        grpc_port=port,
        role=role,
        pinned_model=pinned_model,
        leader_host=leader_host,
    )
    await discovery.start()

    return server, discovery
