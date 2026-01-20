"""AutoML Experiment Manager Package."""

__version__ = "1.0.0"
__author__ = "AutoML Team"

from src.core.config import Config
from src.core.experiment import ExperimentManager, Experiment, ExperimentStatus
from src.mlflow_manager import MLflowManager
from src.docker_manager import DockerManager
from src.vm_manager import VMManager
from src.cli.cli import cli

__all__ = [
    'Config',
    'ExperimentManager',
    'Experiment',
    'ExperimentStatus',
    'MLflowManager',
    'DockerManager',
    'VMManager',
    'cli',
]
