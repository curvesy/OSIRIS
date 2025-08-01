# 🧪 STREAMING TDA COMPREHENSIVE TESTING GUIDE
## Advanced Testing, Chaos Engineering, and CI/CD Integration

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Testing Framework Architecture](#testing-framework-architecture)
3. [Running Tests](#running-tests)
4. [Chaos Engineering](#chaos-engineering)
5. [Performance Benchmarking](#performance-benchmarking)
6. [CI/CD Integration](#cicd-integration)
7. [Monitoring & Diagnostics](#monitoring--diagnostics)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Streaming TDA testing framework provides comprehensive validation through:

- **Unit Tests**: Component-level validation
- **Integration Tests**: End-to-end pipeline testing
- **Chaos Tests**: Fault injection and resilience validation
- **Performance Benchmarks**: Reproducible performance measurement
- **Long-Running Stress Tests**: Memory leak and degradation detection

### Key Features

✅ **Fault Injection**: Network partitions, Kafka failures, resource exhaustion  
✅ **Realistic Data**: IoT sensors, financial markets, network traffic patterns  
✅ **Reproducible Benchmarks**: Seed control, environment capture  
✅ **CI/CD Ready**: Automated validation, regression detection  
✅ **Production Scenarios**: Real-world failure modes and recovery  

---

## 🏗️ Testing Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Testing Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Unit      │    │ Integration  │    │    Chaos      │  │
│  │   Tests     │    │   Tests      │    │ Engineering   │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                   │                     │          │
│         └───────────────────┴─────────────────────┘          │
│                             │                                 │
│                    ┌────────────────┐                        │
│                    │  Test Runner   │                        │
│                    └────────────────┘                        │
│                             │                                 │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                   CI/CD Pipeline                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────────┐    │  │
│  │  │  Smoke   │─▶│Regression│─▶│  Performance    │    │  │
│  │  │  Tests   │  │  Tests   │  │   Benchmarks    │    │  │
│  │  └──────────┘  └──────────┘  └─────────────────┘    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Start test infrastructure
docker-compose -f docker-compose.test.yml up -d

# Verify services
./scripts/verify-test-env.sh
```

### Test Execution Commands

#### Unit Tests
```bash
# Run all unit tests
pytest tests/unit -v

# Run specific test module
pytest tests/unit/test_streaming_window.py -v

# Run with coverage
pytest tests/unit --cov=aura_intelligence --cov-report=html
```

#### Integration Tests
```bash
# Run integration tests
pytest tests/integration -v --timeout=300

# Run specific integration scenario
pytest tests/integration/test_kafka_integration.py::test_end_to_end_pipeline -v

# Run with real Kafka (requires docker-compose)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 pytest tests/integration -v
```

#### Performance Benchmarks
```bash
# Run all benchmarks
python -m aura_intelligence.testing.benchmark_framework

# Run specific benchmark suite
python -m aura_intelligence.testing.benchmark_framework --suite streaming_tda

# Run with custom configuration
python -m aura_intelligence.testing.benchmark_framework \
    --config benchmark_config.yaml \
    --output results/
```

#### Chaos Tests
```bash
# Run chaos engineering suite
python -m aura_intelligence.testing.chaos_engineering

# Run specific chaos scenario
python -m aura_intelligence.testing.chaos_engineering \
    --scenario kafka_network_partition \
    --duration 300

# Run with monitoring
python -m aura_intelligence.testing.chaos_engineering \
    --monitor \
    --prometheus-url http://localhost:9090
```

### Test Configuration

Create `test_config.yaml`:
```yaml
test_environment:
  kafka:
    bootstrap_servers: "localhost:9092"
    schema_registry_url: "http://localhost:8081"
  
  toxiproxy:
    host: "localhost:8474"
    
  monitoring:
    prometheus_url: "http://localhost:9090"
    grafana_url: "http://localhost:3000"

chaos_scenarios:
  - name: "kafka_partition"
    fault_types: ["network_partition"]
    duration_minutes: 5
    intensity: 1.0
    
  - name: "resource_exhaustion"
    fault_types: ["memory_pressure", "cpu_spike"]
    duration_minutes: 10
    intensity: 0.8

benchmarks:
  seed: 42
  warmup_iterations: 10
  test_iterations: 100
  
performance_baselines:
  streaming_small: 0.01
  streaming_medium: 0.1
  streaming_large: 1.0
```

---

## 🔥 Chaos Engineering

### Fault Injection Scenarios

#### 1. Network Partitions
```python
# Simulate network split between Kafka and TDA processor
scenario = ChaosScenario(
    name="kafka_network_split",
    fault_types=[FaultType.NETWORK_PARTITION],
    duration=timedelta(minutes=5),
    intensity=1.0,
    targets=["kafka", "schema_registry"]
)
```

#### 2. Kafka Broker Failures
```python
# Simulate rolling broker failures
scenario = ChaosScenario(
    name="kafka_rolling_restart",
    fault_types=[FaultType.PARTIAL_FAILURE],
    duration=timedelta(minutes=15),
    intensity=0.5,
    targets=["broker-1", "broker-2", "broker-3"]
)
```

#### 3. Resource Exhaustion
```python
# Simulate memory and CPU pressure
scenario = ChaosScenario(
    name="resource_stress",
    fault_types=[
        FaultType.MEMORY_PRESSURE,
        FaultType.CPU_SPIKE
    ],
    duration=timedelta(minutes=20),
    intensity=0.9,
    targets=["system"]
)
```

#### 4. Cascading Failures
```python
# Simulate cascading component failures
scenario = ChaosScenario(
    name="cascade_failure",
    fault_types=[FaultType.CASCADE_FAILURE],
    duration=timedelta(minutes=30),
    intensity=0.8,
    targets=["kafka", "schema_registry", "tda_processor"]
)
```

### Running Chaos Tests

```bash
# Start chaos orchestrator
python -m aura_intelligence.testing.chaos_orchestrator \
    --config chaos_config.yaml \
    --report chaos_report.html

# Monitor during chaos
watch -n 1 'curl -s http://localhost:8080/metrics | grep chaos_'

# Analyze results
python -m aura_intelligence.testing.analyze_chaos_results \
    --input chaos_results/ \
    --output analysis_report.pdf
```

### Validation Criteria

✅ **Data Integrity**: No data loss during failures  
✅ **Recovery Time**: < 30 seconds after fault removal  
✅ **Performance Impact**: < 20% degradation during partial failures  
✅ **Error Handling**: Graceful degradation, no crashes  

---

## 📊 Performance Benchmarking

### Benchmark Suites

#### Streaming TDA Suite
```python
STREAMING_TDA_SUITE = BenchmarkSuite(
    name="streaming_tda",
    benchmarks=[
        BenchmarkConfig("small", data_size=1000),
        BenchmarkConfig("medium", data_size=10000),
        BenchmarkConfig("large", data_size=100000),
    ]
)
```

#### Multi-Scale Suite
```python
MULTI_SCALE_SUITE = BenchmarkSuite(
    name="multi_scale",
    benchmarks=[
        BenchmarkConfig("2_scales", scales=2),
        BenchmarkConfig("4_scales", scales=4),
        BenchmarkConfig("8_scales", scales=8),
    ]
)
```

### Running Benchmarks

```bash
# Run with default configuration
make benchmark

# Run with custom parameters
python -m aura_intelligence.testing.benchmark_runner \
    --suite multi_scale \
    --iterations 100 \
    --warmup 20 \
    --seed 42

# Compare against baseline
python -m aura_intelligence.testing.benchmark_compare \
    --current results/latest/ \
    --baseline results/baseline/ \
    --tolerance 10
```

### Benchmark Reports

Reports are generated in HTML format with:
- Performance metrics (mean, p50, p95, p99)
- Distribution plots
- Baseline comparisons
- Environment details
- Regression analysis

---

## 🔄 CI/CD Integration

### GitHub Actions Workflow

`.github/workflows/streaming-tda-tests.yml`:
```yaml
name: Streaming TDA Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
          
      - name: Run unit tests
        run: |
          pytest tests/unit -v --junitxml=test-results/junit.xml
          
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: unit-test-results
          path: test-results/

  integration-tests:
    runs-on: ubuntu-latest
    services:
      kafka:
        image: confluentinc/cp-kafka:latest
        ports:
          - 9092:9092
          
    steps:
      - uses: actions/checkout@v3
      
      - name: Run integration tests
        run: |
          pytest tests/integration -v --timeout=600
          
  performance-benchmarks:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3
      
      - name: Run benchmarks
        run: |
          python -m aura_intelligence.testing.benchmark_framework \
            --ci-mode \
            --baseline-branch main
            
      - name: Comment PR with results
        uses: actions/github-script@v6
        if: github.event_name == 'pull_request'
        with:
          script: |
            const results = require('./benchmark_results/summary.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Benchmark Results\n${results.summary}`
            });

  chaos-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Run chaos tests
        run: |
          make chaos-test-ci
          
      - name: Upload chaos report
        uses: actions/upload-artifact@v3
        with:
          name: chaos-report
          path: chaos_report.html
```

### Jenkins Pipeline

`Jenkinsfile`:
```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements-test.txt'
                sh 'docker-compose -f docker-compose.test.yml up -d'
            }
        }
        
        stage('Unit Tests') {
            steps {
                sh 'pytest tests/unit -v --junitxml=reports/unit.xml'
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh 'pytest tests/integration -v --junitxml=reports/integration.xml'
            }
        }
        
        stage('Performance Tests') {
            when {
                branch 'main'
            }
            steps {
                sh 'make benchmark-ci'
                publishHTML([
                    reportDir: 'benchmark_results',
                    reportFiles: 'report.html',
                    reportName: 'Benchmark Report'
                ])
            }
        }
        
        stage('Chaos Tests') {
            when {
                branch 'main'
            }
            steps {
                sh 'make chaos-test-ci'
            }
        }
    }
    
    post {
        always {
            junit 'reports/*.xml'
            archiveArtifacts artifacts: 'benchmark_results/**/*'
        }
    }
}
```

### Makefile Integration

```makefile
# Testing targets
.PHONY: test test-unit test-integration test-chaos benchmark

test: test-unit test-integration

test-unit:
	pytest tests/unit -v --cov=aura_intelligence

test-integration:
	docker-compose -f docker-compose.test.yml up -d
	pytest tests/integration -v
	docker-compose -f docker-compose.test.yml down

test-chaos:
	python -m aura_intelligence.testing.chaos_engineering \
		--config chaos_config.yaml \
		--duration 1800

benchmark:
	python -m aura_intelligence.testing.benchmark_framework \
		--all-suites \
		--output benchmark_results/

benchmark-ci:
	python -m aura_intelligence.testing.benchmark_framework \
		--ci-mode \
		--fail-on-regression \
		--tolerance 10

chaos-test-ci:
	python -m aura_intelligence.testing.chaos_engineering \
		--scenarios network_partition,resource_exhaustion \
		--duration 600 \
		--ci-mode
```

---

## 📈 Monitoring & Diagnostics

### Metrics Collection

```python
# Prometheus metrics exposed
streaming_tda_throughput_total
streaming_tda_latency_seconds
streaming_tda_errors_total
chaos_faults_injected_total
chaos_recovery_time_seconds
benchmark_score
```

### Grafana Dashboards

Import provided dashboards:
- `dashboards/streaming-tda-overview.json`
- `dashboards/chaos-engineering.json`
- `dashboards/benchmark-trends.json`

### Logging

```python
# Structured logging with correlation IDs
logger.info(
    "test_event",
    test_id=test_id,
    scenario=scenario_name,
    metrics={
        "throughput": throughput,
        "latency_p99": latency_p99,
        "errors": error_count
    }
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Flaky Tests
```bash
# Run with retry logic
pytest tests/integration --reruns 3 --reruns-delay 5

# Increase timeouts
pytest tests/integration --timeout=600
```

#### 2. Resource Constraints
```bash
# Limit parallel execution
pytest tests -n 2  # Run with 2 workers

# Increase memory limits
export PYTEST_XDIST_WORKER_COUNT=2
```

#### 3. Kafka Connection Issues
```bash
# Verify Kafka is running
docker-compose ps

# Check logs
docker-compose logs kafka

# Reset Kafka state
docker-compose down -v
docker-compose up -d
```

#### 4. Performance Regression
```bash
# Compare with historical data
python -m aura_intelligence.testing.analyze_trends \
    --weeks 4 \
    --metric throughput

# Profile specific test
python -m cProfile -o profile.stats \
    aura_intelligence.testing.benchmark_framework
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export PYTEST_VERBOSE=3

# Run with debugger
pytest tests/unit/test_streaming_window.py --pdb

# Capture all output
pytest tests -s -v --capture=no
```

---

## 📚 Best Practices

### 1. Test Isolation
- Each test should be independent
- Clean up resources in teardown
- Use fixtures for common setup

### 2. Realistic Data
- Use production-like data patterns
- Include edge cases and anomalies
- Vary data volumes and rates

### 3. Reproducibility
- Always set random seeds
- Capture environment details
- Version test data and configurations

### 4. CI/CD Integration
- Run smoke tests on every commit
- Full suite on merge to main
- Performance benchmarks weekly
- Chaos tests before releases

### 5. Monitoring
- Track test execution time trends
- Alert on flaky tests
- Monitor resource usage during tests

---

## 🎯 Conclusion

The comprehensive testing framework ensures the Streaming TDA platform is:
- **Reliable**: Validated against real-world failures
- **Performant**: Benchmarked and regression-tested
- **Scalable**: Tested under various loads
- **Maintainable**: Well-documented and CI/CD integrated

Regular execution of these tests provides confidence in production deployments and enables rapid development while maintaining quality.

---

*"Test early, test often, test realistically."*