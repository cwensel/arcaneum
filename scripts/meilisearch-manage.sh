#!/bin/bash
# MeiliSearch management script (RDR-008)

set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

COMPOSE_FILE="deploy/docker-compose.yml"
MEILISEARCH_URL="${MEILISEARCH_URL:-http://localhost:7700}"

case "$1" in
    start)
        echo "Starting MeiliSearch..."
        docker compose -f "$COMPOSE_FILE" up -d meilisearch
        sleep 3
        if curl -sf "${MEILISEARCH_URL}/health" > /dev/null; then
            echo "MeiliSearch started successfully"
            echo "HTTP API: ${MEILISEARCH_URL}"
        else
            echo "MeiliSearch failed to start"
            exit 1
        fi
        ;;
    stop)
        echo "Stopping MeiliSearch..."
        docker compose -f "$COMPOSE_FILE" stop meilisearch
        echo "MeiliSearch stopped"
        ;;
    restart)
        echo "Restarting MeiliSearch..."
        docker compose -f "$COMPOSE_FILE" restart meilisearch
        sleep 2
        curl -sf "${MEILISEARCH_URL}/health" && echo "Restarted"
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f meilisearch
        ;;
    status)
        echo "MeiliSearch Status:"
        docker compose -f "$COMPOSE_FILE" ps meilisearch
        echo ""
        if curl -sf "${MEILISEARCH_URL}/health" > /dev/null; then
            echo "Healthy"
            if [ -n "${MEILISEARCH_API_KEY}" ]; then
                curl -sf -H "Authorization: Bearer ${MEILISEARCH_API_KEY}" \
                    "${MEILISEARCH_URL}/stats" | jq '.' 2>/dev/null || echo "Stats unavailable (jq not installed or API error)"
            else
                echo "Stats unavailable (MEILISEARCH_API_KEY not set)"
            fi
        else
            echo "Unhealthy"
        fi
        ;;
    create-dump)
        echo "Creating dump..."
        if [ -z "${MEILISEARCH_API_KEY}" ]; then
            echo "Error: MEILISEARCH_API_KEY required"
            exit 1
        fi
        curl -X POST "${MEILISEARCH_URL}/dumps" \
            -H "Authorization: Bearer ${MEILISEARCH_API_KEY}"
        echo ""
        echo "Dump creation initiated"
        ;;
    list-indexes)
        echo "MeiliSearch Indexes:"
        if [ -z "${MEILISEARCH_API_KEY}" ]; then
            echo "Error: MEILISEARCH_API_KEY required"
            exit 1
        fi
        curl -sf -H "Authorization: Bearer ${MEILISEARCH_API_KEY}" \
            "${MEILISEARCH_URL}/indexes" | jq '.results[] | {uid: .uid, primaryKey: .primaryKey}' 2>/dev/null || echo "No indexes found or jq not installed"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|create-dump|list-indexes}"
        exit 1
        ;;
esac
