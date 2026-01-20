"""Test configuration for AutoML Experiment Manager."""

import pytest
import tempfile
import os
import json
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def experiment_manager(temp_dir):
    """Create an experiment manager with temporary storage."""
    from src.core.experiment import ExperimentManager
    return ExperimentManager(storage_dir=temp_dir)

@pytest.fixture
def sample_experiment(experiment_manager):
    """Create a sample experiment."""
    return experiment_manager.create(
        name="test_experiment",
        model_type="sklearn",
        parameters={"test_param": "value"}
    )

@pytest.fixture
def config():
    """Create a config instance."""
    from src.core.config import Config
    return Config()
