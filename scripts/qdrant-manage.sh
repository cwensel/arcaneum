#!/bin/bash
# Qdrant management script (RDR-002)

set -e

COMPOSE_FILE="deploy/docker-compose.yml"

case "$1" in
    start)
        echo "🚀 Starting Qdrant..."
        docker compose -f "$COMPOSE_FILE" up -d
        sleep 3
        if curl -sf http://localhost:6333/healthz > /dev/null; then
            echo "✅ Qdrant started successfully"
            echo "📊 REST API: http://localhost:6333"
            echo "🔗 Dashboard: http://localhost:6333/dashboard"
        else
            echo "❌ Qdrant failed to start"
            exit 1
        fi
        ;;
    stop)
        echo "🛑 Stopping Qdrant..."
        docker compose -f "$COMPOSE_FILE" down
        echo "✅ Qdrant stopped"
        ;;
    restart)
        echo "🔄 Restarting Qdrant..."
        docker compose -f "$COMPOSE_FILE" restart
        sleep 2
        curl -sf http://localhost:6333/healthz && echo "✅ Restarted"
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f qdrant
        ;;
    status)
        echo "📊 Qdrant Status:"
        docker compose -f "$COMPOSE_FILE" ps
        echo ""
        curl -s http://localhost:6333/healthz && echo "✅ Healthy" || echo "❌ Unhealthy"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
