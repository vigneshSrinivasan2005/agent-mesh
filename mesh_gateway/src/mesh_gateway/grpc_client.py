from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Callable, Dict, Optional
from urllib.parse import urlparse

import grpc

from mesh_gateway.grpc_proto import mesh_service_pb2, mesh_service_pb2_grpc
from mesh_gateway.models import NodeConfig

logger = logging.getLogger("mesh_gateway.grpc_client")


def extract_host_from_url(base_url: str) -> str:
    """Extract hostname/IP from HTTP URL (e.g. http://192.168.1.50:11434 -> 192.168.1.50)."""
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1"


class GrpcNodeClient:
    """gRPC connection manager for a single remote worker machine."""

    def __init__(self, node: NodeConfig, grpc_port: int = 50051):
        self.node = node
        self.host = extract_host_from_url(node.base_url)
        self.grpc_port = node.grpc_port or grpc_port
        self.target = f"{self.host}:{self.grpc_port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[mesh_service_pb2_grpc.NodeAgentServiceStub] = None
        self._telemetry_task: Optional[asyncio.Task] = None
        self._is_connected = False

    async def connect(self) -> bool:
        """Establish gRPC connection to node agent."""
        try:
            self._channel = grpc.aio.insecure_channel(
                self.target,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )
            self._stub = mesh_service_pb2_grpc.NodeAgentServiceStub(self._channel)

            # Test connection with Ping
            start = time.perf_counter()
            req = mesh_service_pb2.PingRequest(client_timestamp_ms=int(time.time() * 1000))
            await asyncio.wait_for(self._stub.Ping(req), timeout=2.0)
            rtt = round((time.perf_counter() - start) * 1000.0, 2)
            self._is_connected = True
            logger.info(
                f"gRPC connection to '{self.node.name}' ({self.target}) active! (RTT: {rtt}ms)"
            )
            return True
        except Exception as e:
            self._is_connected = False
            logger.debug(f"gRPC not available on '{self.node.name}' ({self.target}): {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def stream_completion(
        self,
        endpoint: str,
        payload: dict,
        timeout_seconds: int = 120,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens over gRPC channel."""
        if not self._stub:
            raise RuntimeError(f"gRPC not connected to {self.node.name}")

        req = mesh_service_pb2.CompletionRequestProto(
            model=payload.get("model", ""),
            endpoint=endpoint,
            payload_json=json.dumps(payload),
            timeout_seconds=timeout_seconds,
        )

        try:
            async for chunk in self._stub.StreamCompletion(req):
                if chunk.error_message:
                    yield f"data: {json.dumps({'error': chunk.error_message})}\n\n"
                    return
                if chunk.raw_chunk:
                    yield chunk.raw_chunk
        except grpc.RpcError as e:
            yield f"data: {json.dumps({'error': f'gRPC RPC error from {self.node.name}: {e.details()}'})}\n\n"

    async def start_telemetry(self, callback: Callable[[mesh_service_pb2.TelemetryUpdate], None]):
        """Subscribe to streaming hardware & model telemetry from node agent."""
        if not self._stub:
            return

        async def _sub():
            req = mesh_service_pb2.TelemetryRequest(
                gateway_id="agent-mesh-gateway", interval_ms=2000
            )
            while self._is_connected:
                try:
                    async for update in self._stub.StreamTelemetry(req):
                        callback(update)
                except (grpc.RpcError, asyncio.CancelledError):
                    break
                except Exception as e:
                    logger.debug(f"Telemetry stream interrupted for {self.node.name}: {e}")
                    await asyncio.sleep(3.0)

        self._telemetry_task = asyncio.create_task(_sub())

    async def close(self):
        self._is_connected = False
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self._channel:
            await self._channel.close()


class GrpcMeshManager:
    """Manages pool of gRPC clients across all nodes in the mesh."""

    def __init__(self, default_grpc_port: int = 50051):
        self.default_grpc_port = default_grpc_port
        self.clients: Dict[str, GrpcNodeClient] = {}

    async def register_node(
        self,
        node: NodeConfig,
        telemetry_callback: Optional[Callable[[mesh_service_pb2.TelemetryUpdate], None]] = None,
    ) -> bool:
        """Attempt to establish gRPC connection to node agent."""
        client = GrpcNodeClient(node, grpc_port=node.grpc_port or self.default_grpc_port)
        connected = await client.connect()
        if connected:
            self.clients[node.name] = client
            if telemetry_callback:
                await client.start_telemetry(telemetry_callback)
            return True
        return False

    def get_client(self, node_name: str) -> Optional[GrpcNodeClient]:
        client = self.clients.get(node_name)
        if client and client.is_connected:
            return client
        return None

    async def unregister_node(self, node_name: str):
        client = self.clients.pop(node_name, None)
        if client:
            await client.close()

    async def close_all(self):
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
