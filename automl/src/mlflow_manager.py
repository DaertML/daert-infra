"""MLflow management module for AutoML experiments."""

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any
from pathlib import Path


class MLflowManager:
    """Manager class for MLflow operations."""

    def __init__(self, tracking_uri: str = "http://localhost:5000",
                 artifact_root: str = "./mlflow_artifacts"):
        self.tracking_uri = tracking_uri
        self.artifact_root = Path(artifact_root)
        self._ensure_artifact_root()

    def _ensure_artifact_root(self) -> None:
        """Ensure artifact root directory exists."""
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def set_tracking_uri(self, uri: Optional[str] = None) -> None:
        """Set the MLflow tracking URI."""
        if uri:
            self.tracking_uri = uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def create_experiment(self, name: str, artifact_location: Optional[str] = None) -> str:
        """Create a new MLflow experiment."""
        self.set_tracking_uri()

        artifact_path = str(self.artifact_root / name) if artifact_location is None else artifact_location

        exp_id = mlflow.create_experiment(name, artifact_location=artifact_path)
        mlflow.set_experiment(name)

        return exp_id

    def get_or_create_experiment(self, name: str) -> str:
        """Get existing experiment or create a new one."""
        self.set_tracking_uri()

        try:
            exp = mlflow.get_experiment_by_name(name)
            if exp:
                mlflow.set_experiment(name)
                return exp.experiment_id
        except Exception:
            pass

        return self.create_experiment(name)

    def start_run(self, experiment_name: str, run_name: Optional[str] = None,
                  nested: bool = False) -> Any:
        """Start an MLflow run."""
        self.set_tracking_uri()
        self.get_or_create_experiment(experiment_name)

        return mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str, flavor: str = "sklearn") -> None:
        """Log a model to the current run."""
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            mlflow.pyfunc.log_model(artifact_path, python_model=model)

    def log_artifact(self, local_path: str) -> None:
        """Log an artifact to the current run."""
        mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str) -> None:
        """Log all artifacts from a directory to the current run."""
        mlflow.log_artifacts(local_dir)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)

    def get_run(self, run_id: str) -> Optional[Any]:
        """Get a run by ID."""
        self.set_tracking_uri()
        client = MlflowClient()

        try:
            return client.get_run(run_id)
        except Exception:
            return None

    def search_runs(self, experiment_name: str, filter_string: Optional[str] = None,
                    max_results: int = 100) -> List[Any]:
        """Search for runs in an experiment."""
        self.set_tracking_uri()
        client = MlflowClient()

        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return []

        return client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string or "",
            max_results=max_results,
            order_by=["start_time DESC"]
        )

    def get_experiment_metrics(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get all runs and their metrics for an experiment."""
        runs = self.search_runs(experiment_name)

        results = []
        for run in runs:
            results.append({
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': {k: v for k, v in run.data.metrics.items()},
                'params': {k: v for k, v in run.data.params.items()}
            })

        return results

    def get_best_run(self, experiment_name: str, metric: str = "accuracy",
                     mode: str = "max") -> Optional[Any]:
        """Get the best run based on a metric."""
        runs = self.search_runs(experiment_name)

        if not runs:
            return None

        best_run = None
        best_value = float('-inf') if mode == "max" else float('inf')

        for run in runs:
            if metric in run.data.metrics:
                value = run.data.metrics[metric]
                if (mode == "max" and value > best_value) or \
                   (mode == "min" and value < best_value):
                    best_value = value
                    best_run = run

        return best_run

    def download_artifacts(self, run_id: str, artifact_path: str,
                           dst_path: Optional[str] = None) -> str:
        """Download artifacts from a run."""
        self.set_tracking_uri()
        client = MlflowClient()

        dst = dst_path or str(self.artifact_root / "downloads")
        client.download_artifacts(run_id, artifact_path, dst_path=dst)

        return dst

    def register_model(self, run_id: str, model_path: str, model_name: str) -> None:
        """Register a model to the model registry."""
        self.set_tracking_uri()
        client = MlflowClient()

        client.create_registered_model(model_name)
        client.create_model_version(
            name=model_name,
            source=f"{self.tracking_uri}/artifacts/{run_id}/{model_path}",
            run_id=run_id
        )
