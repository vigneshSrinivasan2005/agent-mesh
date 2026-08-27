from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException
from rich.console import Console

from mesh_gateway.grpc_client import GrpcMeshManager
from mesh_gateway.health import HealthTracker
from mesh_gateway.models import (
    MeshConfig,
    NodeConfig,
)

console = Console()


class MeshRouter:
    """Smart request routing, role classification, model alias translation, and proxying."""

    def __init__(
        self,
        config: MeshConfig,
        health_tracker: HealthTracker,
        grpc_manager: Optional[GrpcMeshManager] = None,
    ):
        self.config = config
        self.health = health_tracker
        self.grpc_manager = grpc_manager
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0))

    async def close(self) -> None:
        await self._http_client.aclose()
        if self.grpc_manager:
            await self.grpc_manager.close_all()

    def register_node(self, node: NodeConfig) -> None:
        """Register a dynamically discovered or scaled node with the router."""
        norm_url = node.base_url.replace("://localhost", "://127.0.0.1").rstrip("/")
        for idx, existing in enumerate(self.config.nodes):
            if existing.name == node.name:
                self.config.nodes[idx] = node
                return
            existing_url = existing.base_url.replace("://localhost", "://127.0.0.1").rstrip("/")
            if existing_url == norm_url:
                return
        self.config.nodes.append(node)

    def unregister_node(self, node_name: str) -> None:
        """Unregister a scaled node when scaled down."""
        self.config.nodes = [n for n in self.config.nodes if n.name != node_name]

    def get_nodes_for_role(self, role: str) -> List[NodeConfig]:
        """Return all configured nodes assigned to a specific role."""
        return [n for n in self.config.nodes if role in n.roles]

    def classify_role(self, path: str, requested_model: str, body: Dict[str, Any]) -> str:
        """Classify incoming request into a mesh role: 'autocomplete', 'reasoning', 'chat', 'edit'."""
        model_lower = requested_model.lower()

        # FIM / Tab autocomplete indicators
        if "autocomplete" in model_lower or "fim" in model_lower or "tab" in model_lower:
            return "autocomplete"
        if path.endswith("/completions") and not path.endswith("/chat/completions"):
            # Standard /v1/completions usually represents FIM or raw completion
            return "autocomplete"

        # Reasoning / Heavy Agent indicators
        if "reasoning" in model_lower or "r1" in model_lower or "deepseek" in model_lower:
            return "reasoning"
        if "edit" in model_lower:
            return "edit"
        if "chat" in model_lower or "coder" in model_lower:
            return "chat"

        # Default to chat/reasoning for chat completions, autocomplete for plain completions
        if path.endswith("/chat/completions"):
            return "chat"
        return "autocomplete"

    def select_node(self, role: str, requested_model: str) -> Tuple[NodeConfig, str]:
        """
        Select best healthy node for the role with automatic failover.
        Returns (selected_node, resolved_model_name).
        """
        # First attempt: find healthy nodes explicitly offering this role
        healthy_nodes = self.health.get_healthy_nodes(role=role)

        # Fallback 1: if no healthy node for specific role, try related roles
        if not healthy_nodes and self.config.mesh.fallback_enabled:
            if role in ["reasoning", "edit"]:
                healthy_nodes = self.health.get_healthy_nodes(role="chat")
            elif role == "chat":
                healthy_nodes = self.health.get_healthy_nodes(role="reasoning")

        # Fallback 2: try any healthy node in the entire mesh
        if not healthy_nodes and self.config.mesh.fallback_enabled:
            healthy_nodes = self.health.get_healthy_nodes()

        if not healthy_nodes:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "No healthy nodes available in the mesh to handle this request",
                    "role_requested": role,
                    "model_requested": requested_model,
                    "mesh_summary": self.health.get_summary().model_dump(mode="json"),
                },
            )

        selected_node = healthy_nodes[0]
        resolved_model = self.resolve_model_name(selected_node, requested_model)
        return selected_node, resolved_model

    def resolve_model_name(self, node: NodeConfig, requested_model: str) -> str:
        """Translate client model alias to the physical backend model name."""
        # 1. Check exact alias match in node config
        if requested_model in node.model_aliases:
            return node.model_aliases[requested_model]

        # 2. Check lower-case alias
        req_lower = requested_model.lower()
        for alias, target in node.model_aliases.items():
            if alias.lower() == req_lower:
                return target

        # 3. If generic alias, fallback to pinned model if configured
        if req_lower in ["tab-autocomplete", "autocomplete", "fim", "mesh-autocomplete"]:
            if node.pinned_model:
                return node.pinned_model
        if req_lower in ["reasoning-chat", "mesh-reasoning", "reasoning", "chat"]:
            if node.pinned_model:
                return node.pinned_model

        # 4. Otherwise use requested model name directly
        return requested_model

    async def forward_request(
        self,
        path: str,
        body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Forward a non-streaming JSON request to the selected node."""
        requested_model = body.get("model", "")
        role = self.classify_role(path, requested_model, body)
        node, resolved_model = self.select_node(role, requested_model)

        # Mutate model in payload
        forward_body = dict(body)
        forward_body["model"] = resolved_model

        # Build target URL (Ollama supports /v1/chat/completions and /v1/completions natively)
    @staticmethod
    def _estimate_prompt_tokens(body: Dict[str, Any]) -> int:
        """Estimate prompt tokens from request messages or prompt text."""
        text = ""
        if "prompt" in body:
            p = body["prompt"]
            text = p if isinstance(p, str) else " ".join(str(x) for x in p)
        elif "messages" in body:
            for m in body.get("messages", []):
                c = m.get("content", "")
                text += c if isinstance(c, str) else str(c)
        return max(1, len(text) // 4)

    async def forward_request(
        self,
        path: str,
        body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Forward a non-streaming completion request to the optimal node."""
        requested_model = body.get("model", "")
        role = self.classify_role(path, requested_model, body)
        node, resolved_model = self.select_node(role, requested_model)

        forward_body = dict(body)
        forward_body["model"] = resolved_model
        resolved_base = node.base_url.replace("://localhost", "://127.0.0.1")
        target_url = f"{resolved_base.rstrip('/')}{path}"
        req_headers = {"Content-Type": "application/json"}
        if node.api_key:
            req_headers["Authorization"] = f"Bearer {node.api_key}"

        estimated_p_tokens = self._estimate_prompt_tokens(forward_body)
        start_time = time.perf_counter()
        self.health.record_request_start(node.name)
        try:
            resp = await self._http_client.post(
                target_url,
                json=forward_body,
                headers=req_headers,
                timeout=httpx.Timeout(node.timeout_seconds),
            )
            duration = round(time.perf_counter() - start_time, 3)
            if resp.status_code >= 400:
                self.health.record_request_end(
                    node.name,
                    success=False,
                    model_name=resolved_model,
                    duration_sec=duration,
                )
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Node '{node.name}' returned error: {resp.text}",
                )

            data = resp.json()
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            p_tokens = usage.get("prompt_tokens") or estimated_p_tokens
            c_tokens = usage.get("completion_tokens") or usage.get("total_tokens", 0)

            if not c_tokens and isinstance(data, dict):
                for choice in data.get("choices", []):
                    c_text = choice.get("message", {}).get("content", "") or choice.get("text", "")
                    if c_text:
                        c_tokens += max(1, len(c_text) // 4)

            in_tps = None
            out_tps = None
            if isinstance(data, dict):
                if data.get("prompt_eval_duration") and data["prompt_eval_duration"] > 0:
                    p_eval_s = data["prompt_eval_duration"] / 1e9
                    in_tps = round(p_tokens / p_eval_s, 1)
                if data.get("eval_duration") and data["eval_duration"] > 0:
                    eval_s = data["eval_duration"] / 1e9
                    out_tps = round(c_tokens / eval_s, 1)

            if out_tps is None and duration > 0 and c_tokens > 0:
                out_tps = round(c_tokens / duration, 1)
            if in_tps is None and duration > 0 and p_tokens > 0:
                in_tps = round(p_tokens / (duration * 0.3), 1)

            self.health.record_request_end(
                node.name,
                success=True,
                model_name=resolved_model,
                tokens=c_tokens,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                duration_sec=duration,
                input_tokens_per_sec=in_tps,
                output_tokens_per_sec=out_tps,
            )
            return data
        except httpx.RequestError as e:
            duration = round(time.perf_counter() - start_time, 3)
            self.health.record_request_end(
                node.name,
                success=False,
                model_name=resolved_model,
                duration_sec=duration,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to communicate with node '{node.name}' at {node.base_url}: {e}",
            )

    async def forward_stream(
        self,
        path: str,
        body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Forward a streaming SSE request directly with zero buffering."""
        requested_model = body.get("model", "")
        role = self.classify_role(path, requested_model, body)
        node, resolved_model = self.select_node(role, requested_model)

        forward_body = dict(body)
        forward_body["model"] = resolved_model
        resolved_base = node.base_url.replace("://localhost", "://127.0.0.1")
        target_url = f"{resolved_base.rstrip('/')}{path}"
        req_headers = {"Content-Type": "application/json"}
        if node.api_key:
            req_headers["Authorization"] = f"Bearer {node.api_key}"

        estimated_p_tokens = self._estimate_prompt_tokens(forward_body)
        start_time = time.perf_counter()
        first_token_time = None
        self.health.record_request_start(node.name)
        token_count = 0
        parsed_p_tokens = None
        parsed_c_tokens = None
        success = False

        # 1. Try gRPC streaming if node is connected via gRPC
        if self.grpc_manager:
            grpc_client = self.grpc_manager.get_client(node.name)
            if grpc_client and node.transport != "http":
                try:
                    async for chunk in grpc_client.stream_completion(
                        endpoint=path,
                        payload=forward_body,
                        timeout_seconds=int(node.timeout_seconds),
                    ):
                        if chunk:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            yield chunk
                    success = True
                    return
                except Exception as e:
                    console.print(
                        f"[yellow]gRPC streaming failed for {node.name}, falling back to HTTP:[/yellow] {e}"
                    )

        # 2. HTTP/SSE streaming fallback
        try:
            req = self._http_client.build_request(
                "POST",
                target_url,
                json=forward_body,
                headers=req_headers,
                timeout=httpx.Timeout(node.timeout_seconds),
            )
            resp = await self._http_client.send(req, stream=True)

            if resp.status_code >= 400:
                error_bytes = await resp.aread()
                duration = round(time.perf_counter() - start_time, 3)
                self.health.record_request_end(
                    node.name,
                    success=False,
                    model_name=resolved_model,
                    duration_sec=duration,
                )
                yield f"data: {json.dumps({'error': error_bytes.decode('utf-8', errors='replace')})}\n\n"
                return

            async for chunk in resp.aiter_text():
                if chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1

                    # Extract usage metadata if present in stream
                    if "data: " in chunk:
                        for line in chunk.splitlines():
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                try:
                                    parsed = json.loads(line[6:])
                                    if "usage" in parsed and parsed["usage"]:
                                        parsed_p_tokens = parsed["usage"].get("prompt_tokens")
                                        parsed_c_tokens = parsed["usage"].get("completion_tokens")
                                except Exception:
                                    pass
                    yield chunk

            success = True
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Streaming error from node {node.name}: {e}'})}\n\n"
        finally:
            duration = round(time.perf_counter() - start_time, 3)
            ttft = round(first_token_time - start_time, 3) if first_token_time else round(duration * 0.3, 3)
            gen_time = max(0.001, duration - ttft)

            final_p_tokens = parsed_p_tokens or estimated_p_tokens
            final_c_tokens = parsed_c_tokens or token_count

            in_tps = round(final_p_tokens / ttft, 1) if (ttft > 0 and final_p_tokens > 0) else None
            out_tps = round(final_c_tokens / gen_time, 1) if (gen_time > 0 and final_c_tokens > 0) else None

            self.health.record_request_end(
                node.name,
                success=success,
                model_name=resolved_model,
                tokens=final_c_tokens,
                prompt_tokens=final_p_tokens,
                completion_tokens=final_c_tokens,
                duration_sec=duration,
                ttft_sec=ttft,
                input_tokens_per_sec=in_tps,
                output_tokens_per_sec=out_tps,
            )
