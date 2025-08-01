# 🚀 AURA Intelligence Platform

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready AI orchestration platform with advanced agent collaboration, memory systems, and enterprise-grade observability.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Development](#-development)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### One-Liner Container Launch

```bash
# Clone and run with Docker Compose
git clone https://github.com/your-org/aura-intelligence.git
cd aura-intelligence/core
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Verify Installation

```bash
# Check health
curl http://localhost:8080/health

# View metrics
curl http://localhost:9090/metrics

# Check logs
docker-compose logs -f aura-core
```

## ✨ Features

- **🤖 Advanced Agent System**: Multi-agent collaboration with consciousness-level enhancements
- **🧠 Memory Management**: Vector-based memory with semantic search and consolidation
- **📊 Enterprise Observability**: Prometheus metrics, Jaeger tracing, structured logging
- **🔒 Security First**: JWT auth, encryption at rest, rate limiting, RBAC
- **🚦 Deployment Modes**: Shadow, Canary, and Production deployment strategies
- **⚡ High Performance**: Circuit breakers, retries, async processing
- **🎯 Type Safety**: Pydantic v2 models, Python 3.13+ features

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AURA Intelligence                      │
├─────────────────────────────────────────────────────────┤
│  API Layer          │  Agent Council    │  Observability │
│  ├─ FastAPI        │  ├─ 7 Agents     │  ├─ Prometheus │
│  ├─ GraphQL        │  ├─ LangGraph    │  ├─ Grafana    │
│  └─ WebSocket      │  └─ Collective   │  └─ Jaeger     │
├─────────────────────────────────────────────────────────┤
│  Memory Layer       │  Integration      │  Security      │
│  ├─ Vector Store   │  ├─ PostgreSQL   │  ├─ JWT Auth   │
│  ├─ Redis Cache    │  ├─ RabbitMQ     │  ├─ Encryption │
│  └─ Qdrant         │  └─ External APIs│  └─ RBAC       │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.13+ (3.11+ minimum)
- Docker & Docker Compose
- 2GB+ RAM available
- 10GB+ disk space

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-production.txt

# Run validation
python -m aura_intelligence.utils.validation
```

### Docker Installation

```bash
# Build image
docker build -t aura-intelligence/core:latest .

# Run container
docker run -d \
  --name aura-core \
  -p 8080:8080 \
  -p 9090:9090 \
  -e AURA_ENVIRONMENT=production \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  aura-intelligence/core:latest
```

## ⚙️ Configuration

### Environment Variables

All configuration is done via environment variables with the `AURA_` prefix:

```bash
# Core Settings
AURA_ENVIRONMENT=production          # development, staging, production, enterprise
AURA_LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL

# API Keys (required)
AURA_API__OPENAI_API_KEY=sk-...     # OpenAI API key
AURA_SECURITY__JWT_SECRET_KEY=...   # JWT signing key
AURA_SECURITY__ENCRYPTION_KEY=...   # 32-byte encryption key

# Agent Configuration
AURA_AGENT__AGENT_COUNT=7           # Number of agents
AURA_AGENT__ENHANCEMENT_LEVEL=ultimate  # basic, advanced, ultimate, consciousness

# Deployment Mode
AURA_DEPLOYMENT__DEPLOYMENT_MODE=shadow  # shadow, canary, production
```

See [.env.example](.env.example) for all available options.

### Configuration Validation

```python
from aura_intelligence.config import AURASettings

# Load and validate configuration
settings = AURASettings.from_env()
settings.print_configuration_summary()

# Check for issues
warnings = settings.validate_configuration()
if warnings:
    print("Configuration warnings:", warnings)
```

## 🚢 Deployment

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Scale agents
docker-compose up -d --scale aura-core=3

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace aura-system

# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl -n aura-system get pods
kubectl -n aura-system get svc

# View logs
kubectl -n aura-system logs -l app=aura-intelligence -f
```

### Shadow Mode Deployment

Shadow mode allows testing in production without affecting real traffic:

```bash
# Enable shadow mode
export AURA_DEPLOYMENT__DEPLOYMENT_MODE=shadow
export AURA_DEPLOYMENT__SHADOW_TRAFFIC_PERCENTAGE=100

# Deploy
docker-compose up -d

# Monitor shadow logs
docker-compose logs -f aura-core | grep SHADOW
```

## 📊 Monitoring

### Health Checks

```bash
# Liveness check
curl http://localhost:8080/health/live

# Readiness check
curl http://localhost:8080/health/ready

# Detailed health status
curl http://localhost:8080/health
```

### Metrics

Access Prometheus metrics at `http://localhost:9090/metrics`:

```promql
# Agent performance
rate(aura_agent_cycles_total[5m])
histogram_quantile(0.95, aura_agent_cycle_duration_seconds)

# Memory usage
aura_memory_operations_total{operation="store"}
aura_memory_size_bytes

# API latency
histogram_quantile(0.99, aura_api_request_duration_seconds)
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin):

1. **Agent Council Dashboard**: Real-time agent performance
2. **Memory Systems**: Vector store operations and cache hits
3. **API Performance**: Request rates, latencies, errors

### Tracing

Access Jaeger UI at `http://localhost:16686` to view distributed traces.

### Checking Agent Council Events

To verify the agent council is receiving events:

```bash
# Check agent logs
docker-compose logs aura-core | grep -E "(AGENT|EVENT|COUNCIL)"

# Monitor metrics
curl -s http://localhost:9090/metrics | grep aura_agent

# Example log output:
# 2024-01-15T10:30:45.123Z INFO  [agent_council] Event received: user_query
# 2024-01-15T10:30:45.234Z INFO  [agent_1] Processing event in cycle 42
# 2024-01-15T10:30:45.345Z INFO  [agent_council] Consensus reached: action=respond
```

## 🛠️ Development

### Project Structure

```
core/
├── src/
│   └── aura_intelligence/
│       ├── config/          # Modular configuration (Pydantic v2)
│       ├── agents/          # Agent implementations
│       ├── memory/          # Memory systems
│       ├── orchestration/   # Workflow orchestration
│       ├── api/            # API endpoints
│       └── utils/          # Utilities and decorators
├── tests/                  # Test suite
├── k8s/                   # Kubernetes manifests
├── monitoring/            # Monitoring configs
└── deployment/           # Deployment scripts
```

### Code Style

- Python 3.13+ features (match/case, typing improvements)
- Type hints everywhere
- Pydantic v2 for data validation
- Structured logging with context
- Comprehensive docstrings

### Adding a New Feature

1. Create feature branch
2. Implement with type hints and tests
3. Update configuration if needed
4. Add monitoring metrics
5. Document in README
6. Submit PR with tests passing

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aura_intelligence --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Test with real services
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Shadow Mode Validation

```python
# demo_shadow_validation.py
from aura_intelligence.deployment import ShadowValidator

validator = ShadowValidator()
results = validator.run_validation(
    test_events=[
        {"type": "user_query", "content": "Test query"},
        {"type": "system_event", "action": "health_check"}
    ]
)

print(f"Shadow validation: {'PASSED' if results.passed else 'FAILED'}")
print(f"Events processed: {results.events_processed}")
print(f"Latency p95: {results.latency_p95}ms")
```

## 🔧 Troubleshooting

### Common Issues

#### Container won't start
```bash
# Check logs
docker-compose logs aura-core

# Verify environment
docker-compose config

# Check resources
docker system df
```

#### High memory usage
```bash
# Adjust memory limits
export AURA_AGENT__MAX_MEMORY_MB=512
export AURA_MEMORY__CACHE_SIZE_MB=256

# Monitor memory
docker stats aura-core
```

#### API errors
```bash
# Enable debug logging
export AURA_LOG_LEVEL=DEBUG

# Check circuit breaker status
curl http://localhost:8080/health/circuits
```

### Debug Mode

```bash
# Enable debug mode
export AURA_DEBUG=true
export AURA_LOG_LEVEL=DEBUG
export AURA_LOG_FORMAT=text

# Run with verbose output
docker-compose up
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- Documentation: [docs.aura-intelligence.ai](https://docs.aura-intelligence.ai)
- Issues: [GitHub Issues](https://github.com/your-org/aura-intelligence/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/aura-intelligence/discussions)

---

Built with ❤️ by the AURA Intelligence Team
