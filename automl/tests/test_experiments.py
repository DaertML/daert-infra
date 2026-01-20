"""Tests for experiment management."""

import pytest
from src.core.experiment import ExperimentManager, ExperimentStatus

def test_create_experiment(experiment_manager):
    """Test creating a new experiment."""
    experiment = experiment_manager.create(
        name="test_exp",
        model_type="sklearn",
        parameters={"n_estimators": 100}
    )

    assert experiment is not None
    assert experiment.name == "test_exp"
    assert experiment.model_type == "sklearn"
    assert experiment.status == ExperimentStatus.PENDING
    assert experiment.id is not None

def test_get_experiment(experiment_manager, sample_experiment):
    """Test getting an experiment by ID."""
    retrieved = experiment_manager.get(sample_experiment.id)

    assert retrieved is not None
    assert retrieved.name == sample_experiment.name

def test_list_experiments(experiment_manager):
    """Test listing experiments."""
    experiment_manager.create("exp1", "sklearn")
    experiment_manager.create("exp2", "xgboost")

    experiments = experiment_manager.list()

    assert len(experiments) == 2

def test_update_experiment(experiment_manager, sample_experiment):
    """Test updating an experiment."""
    updated = experiment_manager.update(
        sample_experiment.id,
        status=ExperimentStatus.RUNNING
    )

    assert updated is not None
    assert updated.status == ExperimentStatus.RUNNING

def test_delete_experiment(experiment_manager, sample_experiment):
    """Test deleting an experiment."""
    success = experiment_manager.delete(sample_experiment.id)

    assert success is True

    retrieved = experiment_manager.get(sample_experiment.id)
    assert retrieved is None

def test_start_stop_experiment(experiment_manager, sample_experiment):
    """Test starting and stopping an experiment."""
    started = experiment_manager.start(sample_experiment.id)
    assert started.status == ExperimentStatus.RUNNING

    stopped = experiment_manager.stop(sample_experiment.id)
    assert stopped.status == ExperimentStatus.STOPPED

def test_complete_experiment(experiment_manager, sample_experiment):
    """Test completing an experiment with metrics."""
    completed = experiment_manager.complete(
        sample_experiment.id,
        metrics={"accuracy": 0.95},
        mlflow_run_id="test-run-id"
    )

    assert completed.status == ExperimentStatus.COMPLETED
    assert completed.metrics["accuracy"] == 0.95
    assert completed.mlflow_run_id == "test-run-id"

def test_deploy_experiment(experiment_manager, sample_experiment):
    """Test deploying an experiment."""
    deployed = experiment_manager.deploy(
        sample_experiment.id,
        endpoint="http://localhost:8080/invocations"
    )

    assert deployed.status == ExperimentStatus.DEPLOYED
    assert deployed.deployed_endpoint == "http://localhost:8080/invocations"
