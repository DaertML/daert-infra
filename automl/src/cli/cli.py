"""CLI module for AutoML Experiment Manager."""

import click
import sys
from pathlib import Path
from typing import Optional

from src.core.config import Config
from src.core.experiment import ExperimentManager, ExperimentStatus
from src.mlflow_manager import MLflowManager
from src.docker_manager import DockerManager
from src.vm_manager import VMManager
from src.cli.automl import automl_run


SCRIPT_DIR = Path(__file__).parent.parent.parent


def get_experiment_manager() -> ExperimentManager:
    """Get an ExperimentManager with the correct storage path."""
    return ExperimentManager(str(SCRIPT_DIR / "experiments"))


@click.group()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """AutoML Experiment Manager CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.group()
def data():
    """Manage data files."""
    pass


@data.command('upload')
@click.argument('csv_path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Name for the dataset')
@click.pass_context
def upload_data(ctx, csv_path, name):
    """Upload and analyze a CSV data file."""
    try:
        import pandas as pd
        import json

        df = pd.read_csv(csv_path)
        click.echo(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        if not name:
            name = Path(csv_path).stem

        data_dir = SCRIPT_DIR / "data"
        data_dir.mkdir(exist_ok=True)

        dest_path = data_dir / f"{name}.csv"
        df.to_csv(dest_path, index=False)

        from src.experiments.data_loader import DataAnalyzer
        analyzer = DataAnalyzer(df)
        analyzer.infer_target()

        summary = analyzer.get_column_summary()

        click.echo(f"\nData saved to: {dest_path}")
        click.echo(f"\nColumn Analysis:")
        click.echo(f"  Numerical: {len(summary['numerical'])} columns")
        click.echo(f"  Categorical: {len(summary['categorical'])} columns")
        click.echo(f"  Target column: {summary['target']}")

        analysis_file = data_dir / f"{name}.analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(summary, f, indent=2)

        click.echo(f"\nAnalysis saved to: {analysis_file}")

    except Exception as e:
        click.echo(f"Error uploading data: {e}", err=True)
        sys.exit(1)


@data.command('list')
@click.pass_context
def list_data(ctx):
    """List uploaded data files."""
    try:
        data_dir = SCRIPT_DIR / "data"
        if not data_dir.exists():
            click.echo("No data files uploaded yet.")
            return

        from rich.console import Console
        from rich.table import Table
        import pandas as pd
        import json

        console = Console()
        table = Table(title="Uploaded Data Files")
        table.add_column("Name")
        table.add_column("Rows")
        table.add_column("Columns")
        table.add_column("Numerical")
        table.add_column("Categorical")

        for f in data_dir.glob("*.csv"):
            df = pd.read_csv(f)
            analysis_file = f.with_suffix('.analysis.json')
            if analysis_file.exists():
                with open(analysis_file) as a:
                    analysis = json.load(a)
                numerical = len(analysis.get('numerical', []))
                categorical = len(analysis.get('categorical', []))
            else:
                numerical = len(df.select_dtypes(include=['number']).columns)
                categorical = len(df.select_dtypes(include=['object']).columns)

            table.add_row(f.stem, str(df.shape[0]), str(df.shape[1]),
                         str(numerical), str(categorical))

        console.print(table)

    except Exception as e:
        click.echo(f"Error listing data: {e}", err=True)
        sys.exit(1)


@cli.group()
def experiment():
    """Manage AutoML experiments."""
    pass


@experiment.command('create')
@click.argument('name')
@click.option('--data', '-d', help='Data file name (without extension)')
@click.option('--target', '-t', help='Target column name')
@click.option('--task', '-T', type=click.Choice(['auto', 'classification', 'regression']),
              default='auto', help='Task type')
@click.option('--model-type', '-m', type=str, help='Model type (auto, random_forest, etc.)')
@click.option('--test-size', '-s', type=float, default=0.2, help='Test split ratio')
@click.option('--select-features', '-f', is_flag=True, help='Auto-select best features')
@click.pass_context
def create_experiment_cmd(ctx, name, data, target, task, model_type, test_size, select_features):
    """Create a new experiment with data."""
    try:
        exp_manager = get_experiment_manager()

        parameters = {
            'task': task,
            'test_size': test_size,
            'select_features': select_features
        }

        if data:
            parameters['data_path'] = str(SCRIPT_DIR / "data" / f"{data}.csv")

        if target:
            parameters['target_column'] = target

        if model_type:
            parameters['model_type'] = model_type

        experiment = exp_manager.create(name, model_type, parameters)
        click.echo(f"Created experiment: {experiment.name} (ID: {experiment.id})")

        if data:
            click.echo(f"  Data: {data}.csv")
            click.echo(f"  Target: {target or 'auto-detected'}")
            click.echo(f"  Task: {task}")

    except Exception as e:
        click.echo(f"Error creating experiment: {e}", err=True)
        sys.exit(1)


@experiment.command('list')
@click.option('--status', '-s', type=str, help='Filter by status')
@click.pass_context
def list_experiments(ctx, status):
    """List all experiments."""
    try:
        exp_manager = get_experiment_manager()
        status_enum = ExperimentStatus(status) if status else None
        experiments = exp_manager.list(status=status_enum)

        if not experiments:
            click.echo("No experiments found.")
            return

        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title="Experiments")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Model Type")
        table.add_column("Created")
        table.add_column("ID")

        for exp in experiments:
            table.add_row(exp.name, exp.status.value, exp.model_type or "N/A",
                         exp.created_at[:10] if exp.created_at else "N/A", exp.id)

        console.print(table)
    except Exception as e:
        click.echo(f"Error listing experiments: {e}", err=True)
        sys.exit(1)


@experiment.command('run')
@click.argument('experiment_id')
@click.pass_context
def run_experiment(ctx, experiment_id):
    """Run an experiment locally."""
    from src.experiments.base import create_experiment
    import mlflow
    import os

    exp_manager = get_experiment_manager()
    # Try to get by name first, then by ID
    experiment = exp_manager.get_by_name(experiment_id) or exp_manager.get(experiment_id)

    if not experiment:
        click.echo(f"Experiment not found: {experiment_id}")
        sys.exit(1)

    def log_progress(percent, message):
        sys.stdout.write(f"\r  [{percent:3d}%] {message}")
        sys.stdout.flush()

    click.echo(f"Running experiment: {experiment.name}...")
    exp_manager.update(experiment_id, status=ExperimentStatus.RUNNING)

    try:
        # Use environment variable or default to container name
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://automl-mlflow:5000')
        click.echo(f"  → MLflow URI: {mlflow_tracking_uri}")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Create experiment in MLflow if it doesn't exist
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        try:
            exp = client.get_experiment_by_name(experiment.name)
            if exp is None:
                click.echo(f"  → Creating MLflow experiment: {experiment.name}")
                client.create_experiment(experiment.name)
        except Exception:
            pass
        mlflow.set_experiment(experiment.name)

        experiment.parameters['progress_callback'] = log_progress
        automl_exp = create_experiment(experiment_id, experiment.parameters)

        metrics = automl_exp.run()

        print()  # New line after progress
        # Get the run_id from the experiment's MLflow run
        run_id = automl_exp.mlflow_run_id if hasattr(automl_exp, 'mlflow_run_id') and automl_exp.mlflow_run_id else None
        if not run_id:
            # Fallback: try to get the most recent run for this experiment
            try:
                exp = client.get_experiment_by_name(experiment.name)
                if exp:
                    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
                    if runs:
                        run_id = runs[0].info.run_id
            except Exception:
                pass
        
        exp_manager.complete(experiment_id, metrics, run_id)

        click.echo(f"\n✓ Experiment completed!")
        click.echo(f"\nMetrics:")
        for k, v in metrics.items():
            click.echo(f"  {k}: {v:.4f}")
        if run_id:
            click.echo(f"\nView in MLflow: {mlflow_tracking_uri}/#/experiments/0/runs/{run_id}")

    except Exception as e:
        print()  # New line after progress
        exp_manager.fail(experiment_id, str(e))
        click.echo(f"\n✗ Experiment failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@experiment.command('logs')
@click.argument('experiment_id')
@click.option('--watch', '-w', is_flag=True, help='Watch for changes')
@click.pass_context
def experiment_logs(ctx, experiment_id, watch):
    """Show experiment logs and metrics."""
    import time

    def show_logs():
        exp_manager = get_experiment_manager()
        experiment = exp_manager.get(experiment_id)

        if not experiment:
            click.echo(f"Experiment not found: {experiment_id}")
            return None

        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        icons = {
            'pending': '○',
            'running': '⏳',
            'completed': '✓',
            'failed': '✗',
            'stopped': '⏸',
            'deployed': '🚀'
        }
        icon = icons.get(experiment.status.value, '?')

        metrics_str = '\n'.join([f"  {k}: {v:.4f}" for k, v in experiment.metrics.items()]) if experiment.metrics else '  (pending)'
        params_str = '\n'.join([f"  {k}: {v}" for k, v in experiment.parameters.items()]) if experiment.parameters else '  No parameters'

        logs = f"""Experiment: {experiment.name} {icon}
ID: {experiment.id}
Status: {experiment.status.value.upper()}
Model Type: {experiment.model_type or 'N/A'}
Created: {experiment.created_at}
Updated: {experiment.updated_at}

Metrics:
{metrics_str}

MLflow: {'http://localhost:5000/#/experiments/0/runs/' + experiment.mlflow_run_id if experiment.mlflow_run_id else 'N/A'}

Parameters:
{params_str}
"""
        console.print(Panel(logs, title=f"Experiment {experiment.name}", expand=False))
        return experiment.status

    try:
        if watch:
            click.echo(f"Watching experiment {experiment_id}... (Ctrl+C to stop)\n")
            try:
                while True:
                    status = show_logs()
                    if status and status.value in ['completed', 'failed', 'stopped']:
                        click.echo(f"\n→ Experiment {status.value}")
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                click.echo("\nStopped watching.")
        else:
            show_logs()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@experiment.command('delete')
@click.argument('experiment_id')
@click.pass_context
def delete_experiment(ctx, experiment_id):
    """Delete an experiment."""
    try:
        exp_manager = get_experiment_manager()
        success = exp_manager.delete(experiment_id)

        if success:
            click.echo(f"Deleted experiment: {experiment_id}")
        else:
            click.echo(f"Experiment not found: {experiment_id}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error deleting experiment: {e}", err=True)
        sys.exit(1)


@experiment.command('start')
@click.argument('experiment_id')
@click.pass_context
def start_experiment(ctx, experiment_id):
    """Start an experiment."""
    try:
        exp_manager = get_experiment_manager()
        experiment = exp_manager.start(experiment_id)

        if experiment:
            click.echo(f"Started experiment: {experiment.name}")
        else:
            click.echo(f"Experiment not found: {experiment_id}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting experiment: {e}", err=True)
        sys.exit(1)


@experiment.command('stop')
@click.argument('experiment_id')
@click.pass_context
def stop_experiment(ctx, experiment_id):
    """Stop a running experiment."""
    try:
        exp_manager = get_experiment_manager()
        experiment = exp_manager.stop(experiment_id)

        if experiment:
            click.echo(f"Stopped experiment: {experiment.name}")
        else:
            click.echo(f"Experiment not found: {experiment_id}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error stopping experiment: {e}", err=True)
        sys.exit(1)


@experiment.command('metrics')
@click.argument('experiment_id')
@click.pass_context
def experiment_metrics(ctx, experiment_id):
    """Show experiment metrics in detail."""
    try:
        exp_manager = get_experiment_manager()
        experiment = exp_manager.get(experiment_id)

        if not experiment:
            click.echo(f"Experiment not found: {experiment_id}")
            sys.exit(1)

        if not experiment.metrics:
            click.echo("No metrics available yet. Experiment may still be running.")
            sys.exit(0)

        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title=f"Metrics for {experiment.name}")
        table.add_column("Metric")
        table.add_column("Value")

        for key, value in experiment.metrics.items():
            table.add_row(key, str(value))

        console.print(table)

        if experiment.mlflow_run_id:
            click.echo(f"\nView in MLflow: http://localhost:5000/#/experiments/0/runs/{experiment.mlflow_run_id}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('automl')
@click.argument('data')
@click.option('--target', '-t', help='Target column name')
@click.option('--task', '-T', type=click.Choice(['auto', 'classification', 'regression']),
              default='auto', help='Task type')
@click.option('--test-size', '-s', type=float, default=0.2, help='Test split ratio')
@click.option('--n-trials', '-n', type=int, default=10, help='Max number of trials')
@click.option('--timeout', '-x', type=int, default=600, help='Timeout in seconds')
@click.pass_context
def automl(ctx, data, target, task, test_size, n_trials, timeout):
    """Run automatic ML with multiple models and hyperparameters."""
    from src.cli.automl import automl_run as run_automl
    from click.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(run_automl, [
        data,
        '--target', target or '',
        '--task', task,
        '--test-size', str(test_size),
        '--n-trials', str(n_trials),
        '--timeout', str(timeout)
    ])
    
    if result.exit_code != 0:
        click.echo(f"Error: {result.output}", err=True)
        sys.exit(result.exit_code)


def main():
    """Main entry point."""
    cli()
