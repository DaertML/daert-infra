# AutoML Experiment Manager

A comprehensive CLI tool for managing AutoML experiments with Docker containers, MLflow tracking, and VirtualBox VMs managed by Vagrant.

## Features

- **Experiment Management**: Create, start, stop, delete, and monitor ML experiments
- **MLflow Integration**: Track experiments, metrics, and models with MLflow
- **Docker Support**: Run experiments in isolated Docker containers
- **VM Management**: Spin up VirtualBox VMs with Vagrant for isolated environments
- **Model Deployment**: Deploy trained models for inference
- **CLI Interface**: Rich command-line interface with interactive prompts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Host Machine                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AutoML CLI (this tool)                 │    │
│  └─────────────────────────────────────────────────────┘    │
│           │                    │                           │
│           ▼                    ▼                           │
│  ┌────────────────┐    ┌─────────────────────┐             │
│  │  Vagrant/      │    │  Local Docker       │             │
│  │  VirtualBox    │    │  (optional)         │             │
│  └────────────────┘    └─────────────────────┘             │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              VM: automl-master                       │  │
│  │                                                      │  │
│  │   ┌─────────────┐    ┌─────────────────────────┐    │  │
│  │   │   MLflow    │    │  Experiment Containers  │    │  │
│  │   │   Server    │    │  (Docker)               │    │  │
│  │   │  :5000      │    │                         │    │  │
│  │   └─────────────┘    └─────────────────────────┘    │  │
│  │                                                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Vagrant (optional, for VM-based setup)
- VirtualBox (optional, for VM-based setup)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
cd /path/to/automl

# Enter the CLI (builds and runs container automatically)
./run.sh

# To exit, type 'exit' or Ctrl+D
# To re-enter, just run ./run.sh again
```

**Quick Commands:**

```bash
./run.sh              # Enter interactive CLI
./run.sh status       # Check container status
./run.sh mlflow       # Start MLflow server
./run.sh exec <cmd>   # Run command in container
./run.sh stop         # Stop all containers
```

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker usage.

### Option 2: Local Installation

```bash
# Clone the repository
cd /path/to/automl

# Install dependencies
pip install -r requirements.txt

# Run installation script
chmod +x install.sh
./install.sh

# Or install the package directly
pip install -e .
```

## Usage

### Initialize VM (Optional, for VM-based setup)

```bash
# Initialize Vagrant VM with VirtualBox
./run.sh exec automl vm init

# Start the VM
./run.sh exec automl vm up

# Check VM status
./run.sh exec automl vm status

# SSH into VM
./run.sh exec automl vm ssh "ls -la /home/vagrant/"
```

### Start MLflow

```bash
# Start MLflow tracking server
./run.sh mlflow

# Or from inside the container
automl@container:~$ automl mlflow start --port 5000
```

### Manage Experiments

```bash
# Create a new experiment
./run.sh exec automl experiment create my_experiment --model-type sklearn

# List all experiments
./run.sh exec automl experiment list

# Start an experiment
./run.sh exec automl experiment start <experiment-id>

# Stop a running experiment
./run.sh exec automl experiment stop <experiment-id>

# Show experiment details
./run.sh exec automl experiment info <experiment-id>

# Delete an experiment
./run.sh exec automl experiment delete <experiment-id>
```

### Deploy Models

```bash
# Deploy a trained model for inference
./run.sh exec automl deploy model <experiment-id> --port 8080

# Stop a deployment
./run.sh exec automl deploy stop <experiment-id>
```

### Docker Management

```bash
# List running containers
./run.sh exec automl docker ps

# Show Docker system info
./run.sh exec automl docker info

# View container logs
./run.sh exec automl docker logs <container-id>
```

### System Status

```bash
# Show overall system status
./run.sh exec automl status
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
vm:
  provider: "virtualbox"
  memory: "4096"
  cpus: 2
  network_ip: "192.168.56.10"

docker:
  registry: "localhost:5000"
  mlflow:
    image: "mlflow/mlflow:2.9.2"
    port: 5000

mlflow:
  tracking_uri: "http://localhost:5000"
```

## Directory Structure

```
automl/
├── src/
│   ├── cli/              # CLI commands
│   ├── core/             # Core configuration
│   ├── experiments/      # AutoML experiment classes
│   ├── mlflow_manager.py # MLflow integration
│   ├── docker_manager.py # Docker container management
│   └── vm_manager.py     # VM management
├── vms/
│   └── master/           # Vagrant VM configuration
├── experiments/          # Experiment storage
├── mlflow_artifacts/     # MLflow artifacts
├── models/               # Deployed models
├── results/              # Experiment results
├── config.yaml           # Main configuration
└── requirements.txt      # Python dependencies
```

## Creating Custom Experiments

Create custom experiment classes by extending `AutoMLExperiment`:

```python
from src.experiments.base import AutoMLExperiment

class MyCustomExperiment(AutoMLExperiment):
    def load_data(self):
        # Your data loading logic
        return X, y
    
    def build_model(self):
        # Your model building logic
        self.model = ...
    
    def train(self, X_train, y_train):
        # Your training logic
        self.model.fit(X_train, y_train)
```

## Supported Model Types

- `sklearn`: Scikit-learn models (Random Forest, Gradient Boosting, SVM, etc.)
- `xgboost`: XGBoost classifier
- `lightgbm`: LightGBM classifier

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AUTOML_CONFIG` | Path to custom config file |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
