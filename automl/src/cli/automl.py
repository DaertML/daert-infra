"""AutoML experiment runner that tries multiple configurations automatically."""

import click
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from itertools import product

from src.core.config import Config
from src.core.experiment import ExperimentManager, ExperimentStatus
from src.experiments.base import create_experiment, CSVExperiment
from src.experiments.data_loader import load_data
import mlflow


SCRIPT_DIR = Path(__file__).parent.parent.parent


# Model configurations to try
MODEL_CONFIGS = [
    {
        "name": "RandomForest",
        "model_type": "random_forest",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    {
        "name": "GradientBoosting",
        "model_type": "gradient_boosting",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_samples_split": [2, 5]
        }
    },
    {
        "name": "XGBoost",
        "model_type": "xgboost",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    },
    {
        "name": "LightGBM",
        "model_type": "lightgbm",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [20, 31, 50],
            "subsample": [0.8, 1.0]
        }
    },
    {
        "name": "LogisticRegression",
        "model_type": "logistic_regression",
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [100, 200, 500]
        }
    },
    {
        "name": "SVM",
        "model_type": "svm",
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    },
    {
        "name": "KNN",
        "model_type": "knn",
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }
]


def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combo_dict = {}
        for i, key in enumerate(keys):
            combo_dict[key] = combo[i]
        combinations.append(combo_dict)
    
    return combinations


def get_trial_id() -> str:
    """Generate a unique trial ID."""
    import uuid
    return f"trial_{uuid.uuid4().hex[:8]}"


def run_trial(data_path: str, target_column: str, model_name: str, model_type: str,
              model_params: Dict[str, Any], task: str, test_size: float,
              trial_seed: int,
              progress_callback=None) -> Dict[str, Any]:
    """Run a single trial with given configuration."""
    trial_id = get_trial_id()
    
    params = {
        'data_path': data_path,
        'target_column': target_column,
        'task': task,
        'test_size': test_size,
        'model_type': model_type,
        'model_params': model_params,
        'random_state': trial_seed
    }
    
    experiment_id = f"automl_{trial_id}"
    
    print(f"DEBUG: trial_seed={trial_seed}, params random_state={params.get('random_state', 'NOT SET')}", file=sys.stderr)
    
    try:
        experiment = create_experiment(experiment_id, params)
        metrics = experiment.run()
        
        return {
            'trial_id': trial_id,
            'model_name': model_name,
            'model_type': model_type,
            'params': model_params,
            'metrics': metrics,
            'status': 'completed',
            'mlflow_run_id': experiment.mlflow_run_id
        }
    except Exception as e:
        return {
            'trial_id': trial_id,
            'model_name': model_name,
            'model_type': model_type,
            'params': model_params,
            'metrics': {},
            'status': 'failed',
            'error': str(e)
        }


def rank_experiments(results: List[Dict[str, Any]], metric: str = 'accuracy') -> List[Dict[str, Any]]:
    """Rank experiments by metric."""
    completed = [r for r in results if r['status'] == 'completed' and metric in r['metrics']]
    sorted_results = sorted(completed, key=lambda x: x['metrics'][metric], reverse=True)
    return sorted_results


@click.command()
@click.argument('data')
@click.option('--target', '-t', help='Target column name')
@click.option('--task', '-T', type=click.Choice(['auto', 'classification', 'regression']),
              default='auto', help='Task type')
@click.option('--test-size', '-s', type=float, default=0.2, help='Test split ratio')
@click.option('--n-trials', '-n', type=int, default=10, help='Max number of trials')
@click.option('--timeout', '-x', type=int, default=600, help='Timeout in seconds')
@click.option('--model', '-m', multiple=True, help='Models to try (random_forest, gradient_boosting, etc.)')
def automl_run(data, target, task, test_size, n_trials, timeout, model):
    """Run automatic ML experiment with multiple models and hyperparameters."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Resolve data path
    if data.endswith('.csv'):
        data_path = str(SCRIPT_DIR / "data" / data)
    else:
        data_path = str(SCRIPT_DIR / "data" / f"{data}.csv")
    
    if not Path(data_path).exists():
        click.echo(f"Error: Data file not found: {data_path}", err=True)
        sys.exit(1)
    
    click.echo(f"AutoML Experiment")
    click.echo(f"=" * 50)
    click.echo(f"Data: {data_path}")
    click.echo(f"Target: {target or 'auto-detect'}")
    click.echo(f"Task: {task}")
    click.echo(f"Max trials: {n_trials}")
    click.echo(f"Timeout: {timeout}s")
    
    # Auto-detect target if not provided
    if not target:
        import pandas as pd
        df = pd.read_csv(data_path)
        # Simple heuristic: look for common target names
        for col in ['target', 'label', 'class', 'y', 'survived', 'price', 'outcome']:
            if col.lower() in [c.lower() for c in df.columns]:
                target = col
                break
        if not target:
            target = df.columns[-1]
        click.echo(f"Auto-detected target: {target}")
    
    # Setup MLflow
    mlflow_tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create or get experiment
    experiment_name = f"AutoML_{data}_{target}"
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        click.echo(f"Warning: Could not setup MLflow experiment: {e}")
    
    # Filter models to try
    if model:
        model_filter = list(model)
    else:
        model_filter = [m['model_type'] for m in MODEL_CONFIGS]
    
    # Generate trials
    trials = []
    for model_config in MODEL_CONFIGS:
        if model_config['model_type'] not in model_filter:
            continue
        
        combinations = generate_param_combinations(model_config['params'])
        for combo in combinations[:3]:  # Limit to 3 param combos per model
            if len(trials) >= n_trials:
                break
            trials.append({
                'model_name': model_config['name'],
                'model_type': model_config['model_type'],
                'params': combo,
                'seed': len(trials) + 1  # Unique seed for each trial
            })
    
    click.echo(f"\nWill run {len(trials)} trials across {len([m for m in MODEL_CONFIGS if m['model_type'] in model_filter])} models")
    click.echo()
    
    # Run trials
    results = []
    start_time = time.time()
    
    for idx, trial in enumerate(trials):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            click.echo(f"\nTimeout reached ({timeout}s). Stopping trials.")
            break
        
        click.echo(f"[{idx+1}/{len(trials)}] {trial['model_name']}: {trial['params']}")
        
        result = run_trial(
            data_path=data_path,
            target_column=target,
            model_name=trial['model_name'],
            model_type=trial['model_type'],
            model_params=trial['params'],
            task=task,
            test_size=test_size,
            trial_seed=trial.get('seed', idx + 1)
        )
        
        results.append(result)
        
        if result['status'] == 'completed':
            click.echo(f"  ✓ {result['metrics']}")
        else:
            click.echo(f"  ✗ {result.get('error', 'Unknown error')}")
    
    # Results summary
    click.echo(f"\n{'=' * 50}")
    click.echo("AutoML Results Summary")
    click.echo(f"{'=' * 50}")
    
    # Rank by accuracy (or first metric)
    ranked = rank_experiments(results)
    
    if ranked:
        table = Table(title="Top Experiments")
        table.add_column("Rank")
        table.add_column("Model")
        table.add_column("Accuracy")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1")
        
        for i, r in enumerate(ranked[:10], 1):
            m = r['metrics']
            table.add_row(
                str(i),
                r['model_name'],
                f"{m.get('accuracy', m.get('r2', 0)):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}"
            )
        
        console.print(table)
        
        best = ranked[0]
        click.echo(f"\nBest Model: {best['model_name']}")
        click.echo(f"Best Parameters: {best['params']}")
        click.echo(f"Best Metrics: {best['metrics']}")
        if best.get('mlflow_run_id'):
            click.echo(f"MLflow Run: {mlflow_tracking_uri}/#/experiments/0/runs/{best['mlflow_run_id']}")
    else:
        click.echo("\nNo successful experiments completed.")
    
    # Save results
    results_file = SCRIPT_DIR / "results" / f"automl_{data}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'data': data,
            'target': target,
            'task': task,
            'total_trials': len(results),
            'successful': len([r for r in results if r['status'] == 'completed']),
            'results': results
        }, f, indent=2)
    
    click.echo(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    automl_run()
