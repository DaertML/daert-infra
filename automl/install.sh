#!/bin/bash

set -e

echo "========================================="
echo "AutoML Experiment Manager - Installation"
echo "========================================="

# Check for prerequisites
echo ""
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi
echo "✓ Python 3 found: $(python3 --version)"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi
echo "✓ pip3 found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Warning: Docker is not installed (required for container management)"
else
    echo "✓ Docker found: $(docker --version)"
fi

# Check Vagrant
if ! command -v vagrant &> /dev/null; then
    echo "Warning: Vagrant is not installed (required for VM management)"
else
    echo "✓ Vagrant found: $(vagrant --version)"
fi

# Check VirtualBox
if ! command -v VBoxManage &> /dev/null; then
    echo "Warning: VirtualBox is not installed (required for VM management)"
else
    echo "✓ VirtualBox found: $(VBoxManage --version)"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p experiments
mkdir -p mlflow_artifacts
mkdir -p models
mkdir -p results
mkdir -p vms/master/experiments
mkdir -p vms/master/mlflow/artifacts
echo "✓ Directories created"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt
echo "✓ Python dependencies installed"

# Create experiments.json
echo ""
echo "Initializing experiments database..."
echo "{}" > experiments/experiments.json
echo "✓ Experiments database initialized"

# Create initial config if not exists
if [ ! -f config.yaml ]; then
    echo "Using default config.yaml"
fi

echo ""
echo "========================================="
echo "Installation completed successfully!"
echo "========================================="
echo ""
echo "Quick Start:"
echo "  1. Initialize VM: automl vm init"
echo "  2. Start VM:      automl vm up"
echo "  3. Create experiment: automl experiment create my_exp"
echo "  4. Start experiment:  automl experiment start my_exp"
echo ""
echo "For more information, see README.md"
