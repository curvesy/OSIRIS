#!/bin/bash
# 🧪 AURA Intelligence Development Test Runner

set -e

echo "🧪 AURA Intelligence Development Test"
echo "===================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "📋 Step 1: Starting development services..."
echo "   This will start PostgreSQL, Neo4j, Redis, and Grafana locally"

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

echo "✅ Services started"
echo ""

echo "⏳ Step 2: Waiting for services to be ready..."
sleep 15

# Check if services are healthy
echo "🏥 Checking service health..."

# Check PostgreSQL
if docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U aura_dev > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Ready"
else
    echo "⚠️  PostgreSQL: Still starting up"
fi

# Check Neo4j
if curl -f http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j: Ready"
else
    echo "⚠️  Neo4j: Still starting up"
fi

# Check Redis
if docker-compose -f docker-compose.dev.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Ready"
else
    echo "⚠️  Redis: Still starting up"
fi

echo ""
echo "📊 Step 3: Services are available at:"
echo "   • Neo4j Browser: http://localhost:7474 (neo4j/dev_password)"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo ""

echo "🧪 Step 4: Running development test..."
echo "   This will test our shadow mode logging system"
echo ""

# Make sure the test script is executable
chmod +x run_dev_test.py

# Run the development test
python run_dev_test.py

echo ""
echo "🎉 Development test complete!"
echo ""
echo "📋 What you can do next:"
echo "   1. Check the test results above"
echo "   2. Open Neo4j Browser: http://localhost:7474"
echo "   3. Open Grafana: http://localhost:3000"
echo "   4. Run the test again: python run_dev_test.py"
echo "   5. Stop services: docker-compose -f docker-compose.dev.yml down"
echo ""
echo "🔍 Debugging tips:"
echo "   • View service logs: docker-compose -f docker-compose.dev.yml logs [service]"
echo "   • Check service status: docker-compose -f docker-compose.dev.yml ps"
echo "   • Restart services: docker-compose -f docker-compose.dev.yml restart"
echo ""
echo "🌟 Your AURA Intelligence development environment is ready!"
