#!/bin/bash
set -e

case "$1" in
    start)
        echo "🚀 Starting Qdrant..."
        docker compose up -d
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
        docker compose down
        echo "✅ Qdrant stopped"
        ;;
    restart)
        echo "🔄 Restarting Qdrant..."
        docker compose restart
        sleep 2
        curl -sf http://localhost:6333/healthz && echo "✅ Restarted"
        ;;
    logs)
        docker compose logs -f qdrant
        ;;
    status)
        echo "📊 Qdrant Status:"
        docker compose ps
        echo ""
        curl -s http://localhost:6333/healthz && echo "✅ Healthy" || echo "❌ Unhealthy"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
