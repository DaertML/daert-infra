"""Experiment runner script for Docker containers."""

import os
import sys
import json
import argparse
import mlflow

sys.path.insert(0, '/experiment')

from src.experiments.base import create_experiment


def run_experiment(experiment_id: str, tracking_uri: str = "http://localhost:5000"):
    """Run an experiment and log to MLflow."""
    experiments_file = "/experiment/experiments.json"

    with open(experiments_file, 'r') as f:
        experiments = json.load(f)

    if experiment_id not in experiments:
        raise ValueError(f"Experiment not found: {experiment_id}")

    experiment_data = experiments[experiment_id]
    params = experiment_data.get('parameters', {})

    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment in MLflow if it doesn't exist
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        exp = client.get_experiment_by_name(experiment_data['name'])
        if exp is None:
            print(f"Creating MLflow experiment: {experiment_data['name']}")
            client.create_experiment(experiment_data['name'])
    except Exception as e:
        print(f"Warning: Could not create MLflow experiment: {e}")
    
    mlflow.set_experiment(experiment_data['name'])

    experiment = create_experiment(experiment_id, params)
    metrics = experiment.run()

    experiments[experiment_id]['status'] = 'completed'
    experiments[experiment_id]['metrics'] = metrics
    experiments[experiment_id]['mlflow_run_id'] = mlflow.active_run().info.run_id
    experiments[experiment_id]['artifacts_path'] = f"/mlflow/artifacts/{experiment_id}"

    with open(experiments_file, 'w') as f:
        json.dump(experiments, f, indent=2)

    print(f"Experiment completed with metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run an AutoML experiment')
    parser.add_argument('--experiment-id', required=True, help='Experiment ID')
    parser.add_argument('--tracking-uri', default='http://localhost:5000',
                        help='MLflow tracking URI')

    args = parser.parse_args()

    try:
        run_experiment(args.experiment_id, args.tracking_uri)
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
