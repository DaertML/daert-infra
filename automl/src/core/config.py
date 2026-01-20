"""Configuration management module for AutoML Experiment Manager."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class VMConfig:
    """Virtual Machine configuration."""
    provider: str = "virtualbox"
    base_box: str = "ubuntu/focal64"
    memory: str = "4096"
    cpus: int = 2
    network_ip: str = "192.168.56.10"
    hostname: str = "automl-master"
    synced_folder_enabled: bool = True
    synced_folder_source: str = "./experiments"
    synced_folder_destination: str = "/home/vagrant/experiments"


@dataclass
class DockerConfig:
    """Docker configuration."""
    registry: str = "localhost:5000"
    mlflow_image: str = "ghcr.io/mlflow/mlflow:v3.8.1"
    mlflow_port: int = 5000
    mlflow_volume: str = "/home/vagrant/mlflow/artifacts"
    base_image: str = "python:3.11-slim"
    network: str = "automl_network"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "http://localhost:5000"
    artifact_root: str = "./mlflow_artifacts"
    default_experiment: str = "Default"


@dataclass
class ExperimentsConfig:
    """Experiments storage configuration."""
    storage_dir: str = "./experiments"
    results_dir: str = "./results"
    models_dir: str = "./models"


@dataclass
class CLIConfig:
    """CLI configuration."""
    prompt: str = "automl> "
    colors_enabled: bool = True
    primary_color: str = "#00ff00"
    secondary_color: str = "#00cc00"


class Config:
    """Main configuration manager that loads and manages all configuration."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                'AUTOML_CONFIG',
                str(Path(__file__).parent.parent / 'config.yaml')
            )

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'automl': {
                'name': 'AutoML Experiment Manager',
                'version': '1.0.0',
                'log_level': 'INFO'
            },
            'vm': {
                'provider': 'virtualbox',
                'base_box': 'ubuntu/focal64',
                'memory': '4096',
                'cpus': 2,
                'network': {
                    'ip': '192.168.56.10',
                    'hostname': 'automl-master'
                },
                'synced_folder': {
                    'enabled': True,
                    'source': './experiments',
                    'destination': '/home/vagrant/experiments'
                }
            },
            'docker': {
                'registry': 'localhost:5000',
                'mlflow': {
                    'image': 'ghcr.io/mlflow/mlflow:v3.8.1',
                    'port': 5000,
                    'volume': '/home/vagrant/mlflow/artifacts'
                },
                'experiment': {
                    'base_image': 'python:3.11-slim',
                    'network': 'automl_network'
                }
            },
            'mlflow': {
                'tracking_uri': 'http://localhost:5000',
                'artifact_root': './mlflow_artifacts',
                'experiment': {
                    'default_name': 'Default'
                }
            },
            'experiments': {
                'storage_dir': './experiments',
                'results_dir': './results',
                'models_dir': './models'
            },
            'cli': {
                'prompt': 'automl> ',
                'colors': {
                    'enabled': True,
                    'primary': '#00ff00',
                    'secondary': '#00cc00'
                }
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_vm_config(self) -> VMConfig:
        """Get VM configuration."""
        vm_data = self.get('vm', {})
        network_data = vm_data.get('network', {})
        synced_data = vm_data.get('synced_folder', {})

        return VMConfig(
            provider=vm_data.get('provider', 'virtualbox'),
            base_box=vm_data.get('base_box', 'ubuntu/focal64'),
            memory=vm_data.get('memory', '4096'),
            cpus=vm_data.get('cpus', 2),
            network_ip=network_data.get('ip', '192.168.56.10'),
            hostname=network_data.get('hostname', 'automl-master'),
            synced_folder_enabled=synced_data.get('enabled', True),
            synced_folder_source=synced_data.get('source', './experiments'),
            synced_folder_destination=synced_data.get('destination', '/home/vagrant/experiments')
        )

    def get_docker_config(self) -> DockerConfig:
        """Get Docker configuration."""
        docker_data = self.get('docker', {})
        mlflow_data = docker_data.get('mlflow', {})
        experiment_data = docker_data.get('experiment', {})

        return DockerConfig(
            registry=docker_data.get('registry', 'localhost:5000'),
            mlflow_image=mlflow_data.get('image', 'ghcr.io/mlflow/mlflow:v3.8.1'),
            mlflow_port=mlflow_data.get('port', 5000),
            mlflow_volume=mlflow_data.get('volume', '/home/vagrant/mlflow/artifacts'),
            base_image=experiment_data.get('base_image', 'python:3.11-slim'),
            network=experiment_data.get('network', 'automl_network')
        )

    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration."""
        mlflow_data = self.get('mlflow', {})
        experiment_data = mlflow_data.get('experiment', {})

        return MLflowConfig(
            tracking_uri=mlflow_data.get('tracking_uri', 'http://localhost:5000'),
            artifact_root=mlflow_data.get('artifact_root', './mlflow_artifacts'),
            default_experiment=experiment_data.get('default_name', 'Default')
        )

    def get_experiments_config(self) -> ExperimentsConfig:
        """Get experiments configuration."""
        exp_data = self.get('experiments', {})

        return ExperimentsConfig(
            storage_dir=exp_data.get('storage_dir', './experiments'),
            results_dir=exp_data.get('results_dir', './results'),
            models_dir=exp_data.get('models_dir', './models')
        )

    def get_cli_config(self) -> CLIConfig:
        """Get CLI configuration."""
        cli_data = self.get('cli', {})
        colors_data = cli_data.get('colors', {})

        return CLIConfig(
            prompt=cli_data.get('prompt', 'automl> '),
            colors_enabled=colors_data.get('enabled', True),
            primary_color=colors_data.get('primary', '#00ff00'),
            secondary_color=colors_data.get('secondary', '#00cc00')
        )

    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file."""
        self._config = {}
        self.load(config_path)

    @property
    def config(self) -> Dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._config.copy()
