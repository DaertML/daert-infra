#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_COMPOSE="docker-compose"
if docker compose version &>/dev/null; then
    DOCKER_COMPOSE="docker compose"
fi

echo "=========================================="
echo "  AutoML CLI - Docker Environment"
echo "=========================================="
echo ""
echo "Commands:"
echo "  ./run.sh              Enter interactive CLI"
echo "  ./run.sh status       Check container status"
echo "  ./run.sh mlflow       Start MLflow server"
echo "  ./run.sh exec <cmd>   Run command in container"
echo "  ./run.sh build        Rebuild Docker image"
echo "  ./run.sh stop         Stop all containers"
echo "  ./run.sh help         Show this help"
echo ""

case "${1:-interactive}" in
    interactive|i)
        echo "Starting AutoML CLI container..."
        $DOCKER_COMPOSE up -d automl

        echo ""
        echo "Container started. Attaching..."
        echo "Type 'exit' to leave the shell"
        echo ""
        $DOCKER_COMPOSE exec --index=1 automl /bin/bash --login
        ;;
    status)
        echo "Container Status:"
        docker ps -a --filter "name=automl-cli" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "MLflow Status:"
        docker ps -a --filter "name=automl-mlflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ;;
    mlflow)
        echo "Starting MLflow server..."
        $DOCKER_COMPOSE up -d mlflow
        echo "MLflow available at: http://localhost:5000"
        ;;
    exec)
        shift
        if [ -z "$*" ]; then
            echo "Usage: $0 exec <command>"
            exit 1
        fi
        $DOCKER_COMPOSE exec automl "$@"
        ;;
    build)
        echo "Building Docker image..."
        $DOCKER_COMPOSE build automl
        echo "Build complete!"
        ;;
    stop)
        echo "Stopping all AutoML containers..."
        $DOCKER_COMPOSE down
        echo "All containers stopped."
        ;;
    logs)
        shift
        $DOCKER_COMPOSE logs -f "${@:-automl}"
        ;;
    up)
        echo "Starting all containers..."
        $DOCKER_COMPOSE up -d
        echo "Containers started. Run './run.sh' to enter CLI"
        ;;
    down)
        echo "Stopping all AutoML containers..."
        $DOCKER_COMPOSE down
        echo "All containers stopped."
        ;;
    help|-h|--help)
        echo "Usage: $0 [command]"
        echo ""
        echo "AutoML Docker Environment Manager"
        echo ""
        echo "Commands:"
        echo "  (none)           Enter interactive CLI shell"
        echo "  i, interactive   Enter interactive CLI shell"
        echo "  status           Check container status"
        echo "  mlflow           Start MLflow server"
        echo "  exec <cmd>       Run command in container"
        echo "  build            Rebuild Docker image"
        echo "  up               Start all containers"
        echo "  down             Stop all containers"
        echo "  stop             Stop all containers"
        echo "  logs [service]   View container logs"
        echo "  help             Show this help"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                    # Enter interactive CLI"
        echo "  ./run.sh status             # Check container status"
        echo "  ./run.sh exec automl status # Run automl status command"
        echo "  ./run.sh mlflow             # Start MLflow server"
        echo ""
        ;;
esac
