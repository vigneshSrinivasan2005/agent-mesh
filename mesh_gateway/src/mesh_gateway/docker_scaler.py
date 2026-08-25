from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from mesh_gateway.models import AutoScalingConfig, ContainerConfig, NodeConfig, NodeEngine

if TYPE_CHECKING:
    from mesh_gateway.health import HealthTracker
    from mesh_gateway.router import MeshRouter

logger = logging.getLogger("mesh_gateway.docker_scaler")


def find_free_port(start_port: int = 11435, max_port: int = 11500) -> int:
    """Find an unused TCP port on localhost."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{max_port}")


class DockerScaler:
    """Manages local Dockerized LLM worker containers and handles dynamic auto-scaling."""

    def __init__(
        self,
        config: AutoScalingConfig,
        router: Optional["MeshRouter"] = None,
        health_tracker: Optional["HealthTracker"] = None,
    ):
        self.config = config
        self.router = router
        self.health_tracker = health_tracker
        self._scaled_nodes: Dict[
            str, Dict
        ] = {}  # node_name -> {container_id, port, role, idle_since, model}
        self._lock = asyncio.Lock()
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False

    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker CLI / Daemon is accessible on the host machine."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        try:
            res = subprocess.run(
                [docker_bin, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    async def start_worker_container(
        self,
        role: str,
        model: Optional[str] = None,
        port: Optional[int] = None,
        gpu: bool = False,
        image: str = "ollama/ollama:latest",
    ) -> Tuple[bool, str, int]:
        """
        Launch a new Ollama worker container on the local machine.
        Returns (success, container_id, host_port).
        """
        if not self.is_docker_available():
            logger.warning(
                "Cannot start container: Docker is not available or daemon is not running."
            )
            return False, "", 0

        port = port or find_free_port()
        container_name = f"agent-mesh-worker-{role}-{port}"

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:11434",
            "-e",
            "OLLAMA_KEEP_ALIVE=-1",
            "-e",
            "OLLAMA_NUM_PARALLEL=4",
        ]

        if gpu:
            cmd.extend(["--gpus", "all"])

        # Mount volume for shared cache
        cmd.extend(["-v", "agent_mesh_ollama_models:/root/.ollama", image])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                err_msg = stderr.decode().strip()
                logger.error(f"Failed to start container {container_name}: {err_msg}")
                return False, "", 0

            container_id = stdout.decode().strip()[:12]
            logger.info(f"Started Docker worker {container_name} ({container_id}) on port {port}")

            # Preload pinned model if specified
            if model:
                asyncio.create_task(self._preload_model_in_container(port, model))

            return True, container_id, port
        except Exception as e:
            logger.error(f"Error launching container: {e}")
            return False, "", 0

    async def _preload_model_in_container(self, port: int, model: str):
        """Asynchronously triggers Ollama model pull / warmup in container."""
        import httpx

        url = f"http://127.0.0.1:{port}/api/generate"
        logger.info(f"Warming model '{model}' in container on port {port}...")
        # Wait up to 10s for container ollama server to accept connections
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json={"model": model, "keep_alive": -1})
                    if resp.status_code == 200:
                        logger.info(f"Model '{model}' is warm in container on port {port}")
                        return
            except Exception:
                continue

    async def stop_worker_container(self, container_id_or_name: str) -> bool:
        """Stop and remove a worker container."""
        if not self.is_docker_available():
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_id_or_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            logger.info(f"Stopped and removed container: {container_id_or_name}")
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to stop container {container_id_or_name}: {e}")
            return False

    async def check_and_autoscale(self):
        """
        Periodically evaluate active traffic per role and scale out / in containers.
        """
        if not self.config.enabled or not self.router or not self.health_tracker:
            return

        async with self._lock:
            # 1. Scale Up Check: inspect queue depth / active requests per role
            for role in ["reasoning", "chat", "autocomplete", "edit"]:
                nodes = self.router.get_nodes_for_role(role)
                active_reqs = sum(
                    self.health_tracker.statuses[n.name].active_requests
                    for n in nodes
                    if n.name in self.health_tracker.statuses
                )

                current_dynamic_count = sum(
                    1 for info in self._scaled_nodes.values() if info["role"] == role
                )

                # If queue exceeds threshold and we have room to scale up
                if (
                    active_reqs >= self.config.scale_up_queue_threshold
                    and current_dynamic_count < self.config.max_replicas_per_role
                ):
                    logger.info(
                        f"Auto-scaling UP for role '{role}': active_requests={active_reqs}, threshold={self.config.scale_up_queue_threshold}"
                    )
                    # Find model from primary node
                    pinned_model = next((n.pinned_model for n in nodes if n.pinned_model), None)
                    success, cid, port = await self.start_worker_container(
                        role=role,
                        model=pinned_model,
                        gpu=False,
                    )

                    if success:
                        node_name = f"auto-{role}-{port}"
                        new_node = NodeConfig(
                            name=node_name,
                            base_url=f"http://127.0.0.1:{port}",
                            engine=NodeEngine.OLLAMA,
                            roles=[role],
                            priority=1,
                            pinned_model=pinned_model,
                            container=ContainerConfig(
                                enabled=True,
                                container_id=cid,
                                host_port=port,
                                is_auto_scaled=True,
                            ),
                        )
                        self._scaled_nodes[node_name] = {
                            "container_id": cid,
                            "port": port,
                            "role": role,
                            "idle_since": None,
                            "model": pinned_model,
                            "node": new_node,
                        }
                        self.router.register_node(new_node)
                        self.health_tracker.register_node(new_node)

            # 2. Scale Down Check: inspect idle dynamically scaled nodes
            now = time.time()
            to_remove = []
            for node_name, info in list(self._scaled_nodes.items()):
                status = self.health_tracker.statuses.get(node_name)
                active = status.active_requests if status else 0

                if active == 0:
                    if info["idle_since"] is None:
                        info["idle_since"] = now
                    elif now - info["idle_since"] >= self.config.idle_cooldown_seconds:
                        logger.info(
                            f"Auto-scaling DOWN for idle node '{node_name}' (idle {now - info['idle_since']:.1f}s)"
                        )
                        to_remove.append(node_name)
                else:
                    info["idle_since"] = None

            for node_name in to_remove:
                info = self._scaled_nodes.pop(node_name)
                await self.stop_worker_container(info["container_id"])
                self.router.unregister_node(node_name)
                self.health_tracker.unregister_node(node_name)

    async def _run_loop(self):
        while self._running:
            try:
                await self.check_and_autoscale()
            except Exception as e:
                logger.error(f"Error during auto-scaling check: {e}")
            await asyncio.sleep(5.0)

    def start(self):
        """Start background auto-scaler task."""
        if not self.config.enabled:
            return
        self._running = True
        self._scaling_task = asyncio.create_task(self._run_loop())
        logger.info("Docker auto-scaling engine started.")

    async def stop(self):
        """Stop background scaler and clean up dynamic containers."""
        self._running = False
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass

        # Cleanup dynamic replicas on exit
        for node_name, info in list(self._scaled_nodes.items()):
            await self.stop_worker_container(info["container_id"])
        self._scaled_nodes.clear()
        logger.info("Docker auto-scaling engine stopped.")
