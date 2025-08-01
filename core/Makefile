# AURA Intelligence Validation Makefile
# Run validation tests and deployment procedures

.PHONY: help install test validate docker clean all

# Default target
help:
	@echo "AURA Intelligence Validation Commands:"
	@echo "======================================"
	@echo "make install          - Install all dependencies"
	@echo "make test            - Run unit tests"
	@echo "make integration     - Run integration tests"
	@echo "make validate        - Run complete validation suite"
	@echo "make docker-build    - Build Docker containers"
	@echo "make docker-up       - Start Docker environment"
	@echo "make docker-test     - Run tests in Docker"
	@echo "make chaos           - Run chaos engineering tests"
	@echo "make security        - Run security audit"
	@echo "make load-test       - Run load testing"
	@echo "make dr-drill        - Run disaster recovery drill"
	@echo "make report          - Generate validation report"
	@echo "make clean           - Clean up test artifacts"
	@echo "make all             - Run complete validation pipeline"

# Python environment setup
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Create virtual environment
$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Install dependencies
install: $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	$(PIP) install -r requirements-production.txt

# Run unit tests
test: $(VENV)
	@echo "Running unit tests..."
	$(PYTEST) tests/ -v --tb=short

# Run specific test categories
test-core: $(VENV)
	$(PYTEST) test_core_*.py -v

test-phase1: $(VENV)
	$(PYTEST) test_phase_1*.py -v

test-phase2: $(VENV)
	$(PYTEST) test_phase_2*.py -v

test-phase3: $(VENV)
	$(PYTEST) test_phase_3*.py -v

# Run integration tests
integration: $(VENV)
	@echo "Running integration tests..."
	$(PYTEST) tests/integration/ -v --tb=short

# Run complete validation suite
validate: $(VENV)
	@echo "Running complete validation suite..."
	$(PYTHON) run_all_validations.py

# Docker commands
docker-build:
	@echo "Building Docker containers..."
	docker-compose -f docker-compose.production.yml build

docker-up:
	@echo "Starting Docker environment..."
	docker-compose -f docker-compose.production.yml up -d
	@echo "Waiting for services to be ready..."
	sleep 30

docker-down:
	@echo "Stopping Docker environment..."
	docker-compose -f docker-compose.production.yml down

docker-test: docker-up
	@echo "Running tests in Docker..."
	docker-compose -f docker-compose.production.yml exec -T aura-test pytest tests/ -v

# Chaos engineering tests
chaos: $(VENV)
	@echo "Running chaos engineering experiments..."
	$(PYTHON) -m src.aura_intelligence.chaos.experiments

# Security audit
security: $(VENV)
	@echo "Running security audit..."
	$(PYTHON) scripts/security_audit.py

# Load testing
load-test: $(VENV)
	@echo "Running load tests..."
	$(PYTHON) scripts/load_test.py --users 50 --duration 5m

# Disaster recovery drill
dr-drill:
	@echo "Running disaster recovery drill..."
	$(PYTHON) scripts/dr_drill.py

# Generate validation report
report: $(VENV)
	@echo "Generating validation report..."
	$(PYTHON) scripts/generate_validation_report.py

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf test-results/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run everything
all: install test integration validate report
	@echo "Complete validation pipeline finished!"

# Production deployment
deploy-staging:
	@echo "Deploying to staging..."
	./deploy_phase3d.sh staging

deploy-production:
	@echo "Deploying to production (requires confirmation)..."
	@read -p "Are you sure you want to deploy to production? [y/N] " confirm && \
	[ "$$confirm" = "y" ] && ./deploy_phase3d.sh production || echo "Deployment cancelled"

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

# Database operations
db-migrate:
	@echo "Running database migrations..."
	$(PYTHON) -m alembic upgrade head

db-backup:
	@echo "Backing up database..."
	./backup.sh

# Quick validation commands
quick-validate: $(VENV)
	@echo "Running quick validation checks..."
	$(PYTHON) scripts/quick_validation.py

validate-event-store: $(VENV)
	@echo "Validating event store..."
	$(PYTEST) test_production_hardening.py::test_event_store_idempotency -v

validate-projections: $(VENV)
	@echo "Validating projections..."
	$(PYTEST) test_production_hardening.py::test_projection_resilience -v

validate-chaos: $(VENV)
	@echo "Validating chaos resilience..."
	$(PYTEST) test_production_hardening.py::test_chaos_experiments -v

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install
	$(PIP) install -e .

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

lint:
	@echo "Linting code..."
	flake8 src/ tests/
	mypy src/

# CI/CD helpers
ci-test:
	@echo "Running CI tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

ci-validate:
	@echo "Running CI validation..."
	$(PYTHON) run_complete_validation.py integration