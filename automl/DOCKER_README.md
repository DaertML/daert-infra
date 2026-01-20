# Running AutoML CLI in Docker

This guide explains how to run the AutoML CLI inside a Docker container with all dependencies pre-installed.

## Quick Start

```bash
# 1. Build and enter the container
./run.sh

# 2. You now have an interactive shell with automl available
automl@container:~$ automl --help

# 3. To exit, just type 'exit' or press Ctrl+D
automl@container:~$ exit

# 4. To re-enter, run the same command again
./run.sh
```

## Available Commands

| Command | Description |
|---------|-------------|
| `./run.sh` | Enter interactive CLI shell |
| `./run.sh status` | Check container status |
| `./run.sh mlflow` | Start MLflow tracking server |
| `./run.sh exec <cmd>` | Run a command in the container |
| `./run.sh build` | Rebuild the Docker image |
| `./run.sh stop` | Stop all containers |
| `./run.sh logs` | View container logs |
| `./run.sh help` | Show help |

## Examples

### Enter Interactive Shell

```bash
./run.sh
```

You'll see a prompt like:
```
automl@container:/home/pc/Desktop/open-ml-labs/automl$
```

### Run a Single Command

```bash
./run.sh exec automl experiment list
./run.sh exec automl status
./run.sh exec automl vm status
```

### Start MLflow Server

```bash
./run.sh mlflow
```

MLflow will be available at `http://localhost:5000`

### View Logs

```bash
./run.sh logs          # Follow automl container logs
./run.sh logs mlflow   # Follow mlflow server logs
```

### Stop Everything

```bash
./run.sh stop
```

## Volume Mounts

The container mounts the following directories:

| Host | Container | Purpose |
|------|-----------|---------|
| `./experiments` | `/home/pc/Desktop/open-ml-labs/automl/experiments` | Experiment storage |
| `./mlflow_artifacts` | `/home/pc/Desktop/open-ml-labs/automl/mlflow_artifacts` | MLflow artifacts |
| `./models` | `/home/pc/Desktop/open-ml-labs/automl/models` | Deployed models |
| `./results` | `/home/pc/Desktop/open-ml-labs/automl/results` | Experiment results |
| `./config.yaml` | `/home/pc/Desktop/open-ml-labs/automl/config.yaml` | Configuration file |

## Persistence

All your experiments, models, and results are stored on your host machine in the `./experiments`, `./models`, and `./results` directories. They persist between container runs.

## Docker Compose Services

The `docker-compose.yml` defines two services:

1. **automl** - The CLI container with all dependencies
2. **mlflow** - MLflow tracking server (port 5000)

### Starting MLflow

```bash
# Start only MLflow
docker compose up -d mlflow

# View MLflow logs
docker compose logs -f mlflow

# Stop MLflow
docker compose stop mlflow
```

## Custom Commands

To run custom Python scripts in the container:

```bash
./run.sh exec python my_script.py
./run.sh exec python -c "from src.core.experiment import ExperimentManager; print('Hello!')"
```

## Rebuilding the Image

If you modify dependencies or add new packages:

```bash
./run.sh build
```

## Shell Alias (Optional)

Add this to your `~/.bashrc` or `~/.zshrc` for shorter commands:

```bash
alias automl-docker='cd /path/to/automl && ./run.sh'
alias automl-status='/path/to/automl/run.sh status'
```

## Troubleshooting

### Container won't start

```bash
# Check Docker is running
docker ps

# Check for errors
docker compose logs automl
```

### Port already in use

```bash
# Find what's using port 5000
lsof -i :5000

# Stop the conflicting process or change the port in docker-compose.yml
```

### Permission issues

```bash
# Fix permissions on mounted directories
sudo chown -R $(id -u):$(id -g) experiments models results mlflow_artifacts
```
