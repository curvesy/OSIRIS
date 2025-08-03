#!/bin/bash

# AURA Intelligence - Complete Service Startup Script
# Starts all required services for end-to-end testing and development

set -e

echo "🚀 Starting AURA Intelligence Services..."
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start all services
echo "📦 Starting all services with Docker Compose..."
docker compose -f docker-compose.dev.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."

# Function to check if a port is open
check_port() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            echo "✅ $service is ready (port $port)"
            return 0
        fi
        echo "⏳ Waiting for $service (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service failed to start on port $port"
    return 1
}

# Check all services
echo ""
echo "🔍 Checking service health..."

check_port localhost 5432 "PostgreSQL"
check_port localhost 7687 "Neo4j"
check_port localhost 6379 "Redis"
check_port localhost 9092 "Kafka"
check_port localhost 7233 "Temporal"
check_port localhost 16686 "Jaeger"
check_port localhost 9090 "Prometheus"
check_port localhost 3000 "Grafana"

echo ""
echo "🎉 All services are running!"
echo "========================================"
echo ""
echo "📊 Service URLs:"
echo "  • Neo4j Browser:    http://localhost:7474 (neo4j/dev_password)"
echo "  • Grafana:          http://localhost:3000 (admin/admin)"
echo "  • Prometheus:       http://localhost:9090"
echo "  • Jaeger UI:        http://localhost:16686"
echo "  • Temporal UI:      http://localhost:8088"
echo ""
echo "🧪 Ready for testing:"
echo "  python core/test_end_to_end_gpu_allocation.py"
echo ""
echo "🛑 To stop all services:"
echo "  docker compose -f core/docker-compose.dev.yml down"
echo ""