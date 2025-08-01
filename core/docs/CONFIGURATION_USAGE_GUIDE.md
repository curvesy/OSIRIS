# AURA Intelligence Configuration Usage Guide

## Overview

The AURA Intelligence configuration system uses **Pydantic Settings** with full support for nested environment variable loading. All configuration classes inherit from `BaseSettings` to ensure consistent, predictable, and environment-driven configuration management.

## Architecture

### Configuration Hierarchy

```
AURASettings (main)
├── AgentSettings (AURA_AGENT__*)
├── APISettings (AURA_API__*)
├── MemorySettings (AURA_MEMORY__*)
├── ObservabilitySettings (AURA_OBSERVABILITY__*)
├── IntegrationSettings (AURA_INTEGRATION__*)
├── SecuritySettings (AURA_SECURITY__*)
└── DeploymentSettings (AURA_DEPLOYMENT__*)
```

### Environment Variable Format

All nested configuration classes support the **double-underscore format**:

```bash
# Main settings
AURA_ENVIRONMENT=production
AURA_DEBUG=false
AURA_LOG_LEVEL=INFO

# Nested settings (double underscore)
AURA_API__OPENAI_API_KEY=sk-your-key-here
AURA_AGENT__AGENT_COUNT=10
AURA_MEMORY__VECTOR_STORE_TYPE=pinecone
AURA_OBSERVABILITY__METRICS_PORT=8888
```

## Usage Examples

### 1. Basic Configuration Loading

```python
from aura_intelligence.config import AURASettings

# Load configuration from environment variables
settings = AURASettings()

# Access nested configuration
print(f"Agent count: {settings.agent.agent_count}")
print(f"OpenAI key: {settings.api.get_api_key('openai')}")
print(f"Vector store: {settings.memory.vector_store_type}")
```

### 2. Environment Variable Examples

```bash
# API Configuration
export AURA_API__OPENAI_API_KEY="sk-your-openai-key"
export AURA_API__ANTHROPIC_API_KEY="your-anthropic-key"
export AURA_API__OPENAI_MODEL="gpt-4-turbo"
export AURA_API__OPENAI_TEMPERATURE=0.7

# Agent Configuration
export AURA_AGENT__AGENT_COUNT=7
export AURA_AGENT__ENHANCEMENT_LEVEL="ultimate"
export AURA_AGENT__ENABLE_CONSCIOUSNESS=true
export AURA_AGENT__CYCLE_INTERVAL=1.0

# Memory Configuration
export AURA_MEMORY__VECTOR_STORE_TYPE="chroma"
export AURA_MEMORY__MAX_MEMORIES=10000
export AURA_MEMORY__ENABLE_CACHE=true

# Observability Configuration
export AURA_OBSERVABILITY__ENABLE_METRICS=true
export AURA_OBSERVABILITY__METRICS_PORT=9090
export AURA_OBSERVABILITY__PROMETHEUS_ENABLED=true

# Security Configuration
export AURA_SECURITY__ENABLE_AUTH=true
export AURA_SECURITY__JWT_SECRET_KEY="your-jwt-secret"
export AURA_SECURITY__AUTH_PROVIDER="jwt"

# Deployment Configuration
export AURA_DEPLOYMENT__DEPLOYMENT_MODE="production"
export AURA_DEPLOYMENT__PRODUCTION_REPLICAS=3
export AURA_DEPLOYMENT__CONTAINER_TAG="v1.0.0"
```

### 3. Configuration in Code

```python
from aura_intelligence.config import (
    AURASettings,
    EnvironmentType,
    EnhancementLevel,
    APISettings,
    AgentSettings
)

# Create settings with overrides
settings = AURASettings(
    environment=EnvironmentType.PRODUCTION,
    agent=AgentSettings(
        agent_count=10,
        enhancement_level=EnhancementLevel.ULTIMATE
    ),
    api=APISettings(
        openai_api_key="sk-your-key",
        openai_model="gpt-4-turbo"
    )
)

# Validate configuration
warnings = settings.validate_configuration()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")

# Print configuration summary
settings.print_configuration_summary()
```

### 4. Docker Environment

```dockerfile
# Dockerfile
ENV AURA_ENVIRONMENT=production
ENV AURA_API__OPENAI_API_KEY=${OPENAI_API_KEY}
ENV AURA_AGENT__AGENT_COUNT=5
ENV AURA_MEMORY__VECTOR_STORE_TYPE=pinecone
ENV AURA_OBSERVABILITY__METRICS_PORT=9090
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  aura-intelligence:
    image: aura-intelligence:latest
    environment:
      - AURA_ENVIRONMENT=production
      - AURA_API__OPENAI_API_KEY=${OPENAI_API_KEY}
      - AURA_AGENT__AGENT_COUNT=7
      - AURA_MEMORY__VECTOR_STORE_TYPE=chroma
      - AURA_OBSERVABILITY__ENABLE_METRICS=true
      - AURA_SECURITY__ENABLE_AUTH=true
```

### 5. Kubernetes Configuration

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aura-config
data:
  AURA_ENVIRONMENT: "production"
  AURA_AGENT__AGENT_COUNT: "10"
  AURA_MEMORY__VECTOR_STORE_TYPE: "pinecone"
  AURA_OBSERVABILITY__METRICS_PORT: "9090"
  AURA_DEPLOYMENT__DEPLOYMENT_MODE: "production"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: aura-secrets
type: Opaque
stringData:
  AURA_API__OPENAI_API_KEY: "sk-your-openai-key"
  AURA_SECURITY__JWT_SECRET_KEY: "your-jwt-secret"
```

## Configuration Classes

### AgentSettings (AURA_AGENT__*)

Controls agent behavior and enhancement levels.

```bash
AURA_AGENT__AGENT_COUNT=7                    # Number of agents (1-100)
AURA_AGENT__ENHANCEMENT_LEVEL=ultimate       # basic|advanced|ultimate|consciousness
AURA_AGENT__CYCLE_INTERVAL=1.0              # Cycle interval in seconds
AURA_AGENT__ENABLE_CONSCIOUSNESS=true       # Enable consciousness features
AURA_AGENT__MAX_MEMORY_MB=1024              # Memory limit per agent
```

### APISettings (AURA_API__*)

Manages external API configurations and keys.

```bash
AURA_API__OPENAI_API_KEY=sk-your-key        # OpenAI API key
AURA_API__OPENAI_MODEL=gpt-4-turbo          # OpenAI model
AURA_API__OPENAI_TEMPERATURE=0.7            # Temperature (0.0-2.0)
AURA_API__ANTHROPIC_API_KEY=your-key        # Anthropic API key
AURA_API__RATE_LIMIT_REQUESTS_PER_MINUTE=60 # Rate limiting
```

### MemorySettings (AURA_MEMORY__*)

Controls memory management and vector stores.

```bash
AURA_MEMORY__VECTOR_STORE_TYPE=chroma        # chroma|pinecone|weaviate|faiss|qdrant
AURA_MEMORY__MAX_MEMORIES=10000              # Maximum memories to store
AURA_MEMORY__ENABLE_CACHE=true               # Enable memory caching
AURA_MEMORY__CACHE_SIZE_MB=512               # Cache size in MB
```

### ObservabilitySettings (AURA_OBSERVABILITY__*)

Configures monitoring, metrics, and logging.

```bash
AURA_OBSERVABILITY__ENABLE_METRICS=true     # Enable metrics collection
AURA_OBSERVABILITY__METRICS_PORT=9090       # Metrics endpoint port
AURA_OBSERVABILITY__PROMETHEUS_ENABLED=true # Enable Prometheus
AURA_OBSERVABILITY__LOG_FORMAT=json         # json|text
AURA_OBSERVABILITY__ENABLE_TRACING=true     # Enable distributed tracing
```

### SecuritySettings (AURA_SECURITY__*)

Manages authentication and security policies.

```bash
AURA_SECURITY__ENABLE_AUTH=true              # Enable authentication
AURA_SECURITY__AUTH_PROVIDER=jwt             # jwt|oauth2|saml
AURA_SECURITY__JWT_SECRET_KEY=your-secret    # JWT signing key
AURA_SECURITY__ENABLE_RATE_LIMITING=true     # Enable rate limiting
AURA_SECURITY__ENABLE_CORS=true              # Enable CORS
```

### DeploymentSettings (AURA_DEPLOYMENT__*)

Controls deployment modes and container settings.

```bash
AURA_DEPLOYMENT__DEPLOYMENT_MODE=production  # shadow|canary|production
AURA_DEPLOYMENT__PRODUCTION_REPLICAS=3       # Number of replicas
AURA_DEPLOYMENT__CONTAINER_TAG=latest        # Container image tag
AURA_DEPLOYMENT__CPU_LIMIT=2000m             # CPU limit
AURA_DEPLOYMENT__MEMORY_LIMIT=2Gi            # Memory limit
```

## Best Practices

### 1. Environment-Specific Configuration

```bash
# Development
export AURA_ENVIRONMENT=development
export AURA_DEBUG=true
export AURA_LOG_LEVEL=DEBUG

# Production
export AURA_ENVIRONMENT=production
export AURA_DEBUG=false
export AURA_LOG_LEVEL=INFO
```

### 2. Secret Management

- Use environment variables for secrets
- Never commit secrets to version control
- Use secret management systems (Kubernetes secrets, AWS Secrets Manager, etc.)

```bash
# Good - using environment variables
export AURA_API__OPENAI_API_KEY="${OPENAI_API_KEY}"
export AURA_SECURITY__JWT_SECRET_KEY="${JWT_SECRET}"

# Bad - hardcoded secrets
export AURA_API__OPENAI_API_KEY="sk-hardcoded-key"
```

### 3. Configuration Validation

```python
from aura_intelligence.config import AURASettings

settings = AURASettings()

# Validate configuration
warnings = settings.validate_configuration()
if warnings:
    print("Configuration warnings:")
    for warning in warnings:
        print(f"  - {warning}")
    
    # Decide whether to continue or exit
    if settings.is_production and warnings:
        raise ValueError("Production deployment requires valid configuration")
```

### 4. Testing Configuration

```python
import os
import pytest
from aura_intelligence.config import AURASettings

def test_configuration_loading():
    # Set test environment variables
    os.environ['AURA_API__OPENAI_API_KEY'] = 'test-key'
    os.environ['AURA_AGENT__AGENT_COUNT'] = '5'
    
    settings = AURASettings()
    
    assert settings.api.get_api_key('openai') == 'test-key'
    assert settings.agent.agent_count == 5
```

## Migration Guide

### From Legacy Configuration

If migrating from a legacy configuration system:

1. **Identify current configuration sources**
2. **Map to new environment variable format**
3. **Update deployment scripts**
4. **Test configuration loading**

```python
# Legacy (deprecated)
config = {
    'api_keys': {'openai': 'sk-key'},
    'agent_count': 7
}

# New approach
os.environ['AURA_API__OPENAI_API_KEY'] = 'sk-key'
os.environ['AURA_AGENT__AGENT_COUNT'] = '7'
settings = AURASettings()
```

## Troubleshooting

### Common Issues

1. **Environment variables not loading**
   - Check variable names (double underscore format)
   - Verify environment variable is set: `echo $AURA_API__OPENAI_API_KEY`
   - Ensure proper inheritance from `BaseSettings`

2. **Type conversion errors**
   - Check data types (strings for secrets, integers for counts)
   - Use proper boolean values: `true`/`false` (lowercase)

3. **Validation errors**
   - Check field constraints (min/max values)
   - Verify enum values match exactly
   - Review field validators

### Debug Configuration Loading

```python
import os
from aura_intelligence.config import AURASettings

# Debug environment variables
print("Environment variables:")
for key, value in os.environ.items():
    if key.startswith('AURA_'):
        print(f"  {key}={value}")

# Load and inspect configuration
settings = AURASettings()
print(f"Loaded configuration:")
print(f"  Environment: {settings.environment}")
print(f"  Agent count: {settings.agent.agent_count}")
print(f"  API key present: {settings.api.has_openai_key}")

# Print full configuration summary
settings.print_configuration_summary()
```

## Support

For configuration issues:

1. Check this documentation
2. Verify environment variable format
3. Test with minimal configuration
4. Review validation warnings
5. Check logs for configuration errors

The configuration system is designed to be robust, predictable, and environment-driven. Following these patterns ensures consistent behavior across all deployment environments.