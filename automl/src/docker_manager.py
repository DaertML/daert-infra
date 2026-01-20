"""Docker management module for AutoML experiments."""

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container
from typing import Dict, List, Optional, Any
from pathlib import Path


class DockerManager:
    """Manager class for Docker container operations."""

    def __init__(self, registry: str = "localhost:5000", network: str = "automl_network"):
        self.registry = registry
        self.network = network
        self.client = docker.from_env()
        self._ensure_network()

    def _ensure_network(self) -> None:
        """Ensure Docker network exists."""
        try:
            self.client.networks.get(self.network)
        except NotFound:
            self.client.networks.create(
                self.network,
                driver="bridge",
                attachable=True
            )

    def pull_image(self, image: str) -> None:
        """Pull a Docker image."""
        self.client.images.pull(image)

    def build_image(self, dockerfile: str, tag: str,
                    build_args: Optional[Dict[str, str]] = None) -> None:
        """Build a Docker image from Dockerfile."""
        path = str(Path(dockerfile).parent)

        self.client.images.build(
            path=path,
            dockerfile=Path(dockerfile).name,
            tag=tag,
            buildargs=build_args
        )

    def run_container(self, image: str, name: Optional[str] = None,
                      command: Optional[str] = None, detach: bool = True,
                      ports: Optional[Dict[str, str]] = None,
                      volumes: Optional[Dict[str, Dict[str, str]]] = None,
                      environment: Optional[Dict[str, str]] = None,
                      network: Optional[str] = None,
                      remove: bool = False) -> Container:
        """Run a Docker container."""
        network = network or self.network

        container = self.client.containers.run(
            image=image,
            name=name,
            command=command,
            detach=detach,
            ports=ports or {},
            volumes=volumes or {},
            environment=environment or {},
            network=network,
            remove=remove
        )

        return container

    def start_container(self, container_id: str) -> Container:
        """Start a stopped container."""
        container = self.client.containers.get(container_id)
        container.start()
        return container

    def stop_container(self, container_id: str, timeout: int = 10) -> Container:
        """Stop a running container."""
        container = self.client.containers.get(container_id)
        container.stop(timeout=timeout)
        return container

    def restart_container(self, container_id: str) -> Container:
        """Restart a container."""
        container = self.client.containers.get(container_id)
        container.restart()
        return container

    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a container."""
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
            return True
        except NotFound:
            return False

    def get_container(self, container_id: str) -> Optional[Container]:
        """Get a container by ID."""
        try:
            return self.client.containers.get(container_id)
        except NotFound:
            return None

    def get_container_by_name(self, name: str) -> Optional[Container]:
        """Get a container by name."""
        for container in self.client.containers.list(all=True):
            if container.name == name:
                return container
        return None

    def list_containers(self, all: bool = True) -> List[Container]:
        """List all containers."""
        return self.client.containers.list(all=all)

    def get_container_logs(self, container_id: str, stream: bool = False) -> str:
        """Get container logs."""
        container = self.get_container(container_id)
        if container:
            return container.logs(stream=stream).decode('utf-8')
        return ""

    def get_container_status(self, container_id: str) -> str:
        """Get container status."""
        container = self.get_container(container_id)
        if container:
            return container.status
        return "not_found"

    def exec_command(self, container_id: str, command: str) -> tuple:
        """Execute a command in a container."""
        container = self.get_container(container_id)
        if container:
            exit_code, output = container.exec_run(command)
            return exit_code, output.decode('utf-8')
        return -1, ""

    def get_container_ip(self, container_id: str) -> str:
        """Get container IP address."""
        container = self.get_container(container_id)
        if container:
            networks = container.attrs['NetworkSettings']['Networks']
            if networks:
                return list(networks.values())[0]['IPAddress']
        return ""

    def build_mlflow_container(self, name: str = "automl-mlflow",
                                port: int = 5000,
                                volume: str = "/home/vagrant/mlflow/artifacts") -> Container:
        """Build and run MLflow tracking server container."""
        self.pull_image("ghcr.io/mlflow/mlflow:v3.8.1")

        return self.run_container(
            image="ghcr.io/mlflow/mlflow:v3.8.1",
            name=name,
            command=f"mlflow server --host 0.0.0.0 --port {port}",
            detach=True,
            ports={"5000/tcp": port},
            volumes={volume: {"bind": "/mlflow", "mode": "rw"}},
            environment={"MLFLOW_TRACKING_URI": f"http://localhost:{port}"}
        )

    def build_experiment_container(self, experiment_id: str, image: str,
                                    volume: str, environment: Dict[str, str]) -> Container:
        """Build and run an experiment container."""
        container_name = f"automl-experiment-{experiment_id}"

        return self.run_container(
            image=image,
            name=container_name,
            command=f"python run_experiment.py --experiment-id {experiment_id}",
            detach=True,
            volumes={volume: {"bind": "/experiment", "mode": "rw"}},
            environment=environment,
            remove=True
        )

    def network_exists(self, network_name: str) -> bool:
        """Check if a network exists."""
        try:
            self.client.networks.get(network_name)
            return True
        except NotFound:
            return False

    def prune_networks(self) -> Dict[str, Any]:
        """Prune unused networks."""
        return self.client.networks.prune()

    def prune_images(self, dangling: bool = True) -> Dict[str, Any]:
        """Prune unused images."""
        return self.client.images.prune(dangling_only=dangling)

    def get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information."""
        info = self.client.info()
        return {
            'version': info.get('ServerVersion'),
            'containers': info.get('Containers', 0),
            'images': info.get('Images', 0),
            'driver': info.get('Driver'),
            'kernel_version': info.get('KernelVersion')
        }
