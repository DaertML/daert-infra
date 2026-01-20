"""Experiment management module for AutoML."""

import os
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    DEPLOYED = "deployed"


@dataclass
class Experiment:
    """Experiment data class."""
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_type: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    container_id: Optional[str] = None
    deployed_endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        data['status'] = ExperimentStatus(data.get('status', 'pending'))
        return cls(**data)


class ExperimentManager:
    """Manager class for AutoML experiments."""

    def __init__(self, storage_dir: str = "./experiments"):
        self.storage_dir = Path(storage_dir)
        self.experiments_file = self.storage_dir / "experiments.json"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if not self.experiments_file.exists():
            self._save_experiments({})

    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load experiments from file."""
        try:
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_experiments(self, experiments: Dict[str, Dict[str, Any]]) -> None:
        """Save experiments to file."""
        with open(self.experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2)

    def create(self, name: str, model_type: Optional[str] = None,
               parameters: Optional[Dict[str, Any]] = None) -> Experiment:
        """Create a new experiment."""
        experiments = self._load_experiments()

        if name in experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        experiment = Experiment(
            name=name,
            model_type=model_type,
            parameters=parameters or {}
        )

        experiments[experiment.id] = experiment.to_dict()
        self._save_experiments(experiments)

        return experiment

    def get(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID (supports full or partial ID)."""
        experiments = self._load_experiments()

        # Try exact match first
        data = experiments.get(experiment_id)
        if data:
            return Experiment.from_dict(data)

        # Try partial match (prefix)
        for exp_id, exp_data in experiments.items():
            if exp_id.startswith(experiment_id):
                return Experiment.from_dict(exp_data)

        return None

    def get_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        experiments = self._load_experiments()

        for data in experiments.values():
            if data['name'] == name:
                return Experiment.from_dict(data)
        return None

    def list(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        experiments = self._load_experiments()
        result = []

        for data in experiments.values():
            exp = Experiment.from_dict(data)
            if status is None or exp.status == status:
                result.append(exp)

        return sorted(result, key=lambda x: x.created_at, reverse=True)

    def update(self, experiment_id: str, **kwargs) -> Optional[Experiment]:
        """Update an experiment."""
        experiments = self._load_experiments()

        if experiment_id not in experiments:
            return None

        data = experiments[experiment_id]
        current = Experiment.from_dict(data)

        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)

        current.updated_at = datetime.now().isoformat()
        experiments[experiment_id] = current.to_dict()
        self._save_experiments(experiments)

        return current

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        experiments = self._load_experiments()

        if experiment_id not in experiments:
            return False

        experiment = Experiment.from_dict(experiments[experiment_id])

        if experiment.artifacts_path:
            artifacts_path = Path(experiment.artifacts_path)
            if artifacts_path.exists():
                shutil.rmtree(artifacts_path)

        del experiments[experiment_id]
        self._save_experiments(experiments)

        return True

    def start(self, experiment_id: str) -> Optional[Experiment]:
        """Start an experiment."""
        return self.update(experiment_id, status=ExperimentStatus.RUNNING)

    def stop(self, experiment_id: str) -> Optional[Experiment]:
        """Stop a running experiment."""
        return self.update(experiment_id, status=ExperimentStatus.STOPPED)

    def complete(self, experiment_id: str, metrics: Dict[str, float],
                 mlflow_run_id: Optional[str] = None) -> Optional[Experiment]:
        """Mark an experiment as completed with metrics."""
        return self.update(
            experiment_id,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            mlflow_run_id=mlflow_run_id
        )

    def fail(self, experiment_id: str, error: str) -> Optional[Experiment]:
        """Mark an experiment as failed."""
        return self.update(
            experiment_id,
            status=ExperimentStatus.FAILED,
            metrics={'error': error}
        )

    def deploy(self, experiment_id: str, endpoint: str) -> Optional[Experiment]:
        """Mark an experiment as deployed."""
        return self.update(
            experiment_id,
            status=ExperimentStatus.DEPLOYED,
            deployed_endpoint=endpoint
        )

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory for an experiment."""
        return self.storage_dir / experiment_id

    def get_artifacts_dir(self, experiment_id: str) -> Path:
        """Get the artifacts directory for an experiment."""
        artifacts_dir = self.get_experiment_dir(experiment_id) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir
