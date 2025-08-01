# 🚀 AURA Intelligence Validation Quick Start Guide

This guide helps you run the AURA Intelligence validation suite to achieve operational sign-off.

## 📋 Prerequisites

- Python 3.9+ installed
- Docker and Docker Compose (optional but recommended)
- Make command available
- Git installed

## 🎯 Quick Validation Steps

### 1. Basic Validation (No Dependencies)

Run the simple validation script that checks basic project health:

```bash
cd /workspace/core
python3 run_validation.py
```

This will:
- ✅ Check Python version
- ✅ Verify project structure
- ✅ Test core module imports
- ✅ Validate configuration files
- ✅ Generate a validation report

### 2. Using the Makefile

The Makefile provides all validation commands:

```bash
# Show all available commands
make help

# Install dependencies (if you have venv access)
make install

# Run unit tests
make test

# Run integration tests
make integration

# Run complete validation
make validate

# Run specific test phases
make test-phase1
make test-phase2
make test-phase3
```

### 3. Docker-Based Validation

If you have Docker installed:

```bash
# Build containers
make docker-build

# Start environment
make docker-up

# Run tests in Docker
make docker-test

# Stop environment
make docker-down
```

### 4. Interactive Validation Checklist

Use the interactive checklist to track manual validation:

```bash
python3 scripts/validation_checklist.py
```

This provides an interactive menu to:
- View validation status
- Update test results
- Generate reports
- Export results

## 📊 Manual Validation Tasks

### Security Audit

1. **Network Security**
   ```bash
   # Check open ports
   netstat -tuln | grep LISTEN
   
   # Verify TLS certificates
   openssl s_client -connect localhost:8000 -showcerts
   ```

2. **Container Security**
   ```bash
   # Scan Docker images
   docker scan aura-production:latest
   ```

3. **Secrets Management**
   ```bash
   # Verify no hardcoded secrets
   grep -r "password\|secret\|key" src/ --exclude-dir=__pycache__
   ```

### Disaster Recovery Drill

1. **Backup Test**
   ```bash
   # Create backup
   ./backup.sh
   
   # Verify backup integrity
   tar -tzf backup_*.tar.gz | head -20
   ```

2. **Failover Test**
   ```bash
   # Simulate service failure
   docker stop aura-production
   
   # Verify automatic recovery
   docker ps | grep aura
   ```

### Load Testing

1. **Using Locust (if available)**
   ```bash
   # Start Locust UI
   locust -f tests/load/locustfile.py --host=http://localhost:8000
   
   # Open browser to http://localhost:8089
   ```

2. **Simple Load Test**
   ```bash
   # Generate load with curl
   for i in {1..100}; do
     curl -X POST http://localhost:8000/api/debate \
       -H "Content-Type: application/json" \
       -d '{"topic": "Test debate '$i'"}' &
   done
   wait
   ```

## 📝 Validation Report Generation

After running tests, generate the final report:

```bash
# If you ran the simple validation
cat validation_report.json | python3 -m json.tool

# If using the checklist
python3 scripts/validation_checklist.py
# Select option 3 for summary report
# Select option 4 to export results
```

## 🎯 Sign-Off Criteria

For operational sign-off, ensure:

1. **Automated Tests**: >99% pass rate
2. **Security Audit**: No critical findings
3. **DR Drill**: RTO <5 min, RPO <1 min achieved
4. **Load Tests**: System handles target load
5. **Documentation**: All docs updated

## 🆘 Troubleshooting

### Python Dependencies Issue
```bash
# Use system Python packages
python3 -m pip install --user -r requirements.txt

# Or use pipx for isolated environments
pipx install -r requirements.txt
```

### Docker Not Available
```bash
# Run tests locally without Docker
python3 -m pytest tests/ -v

# Or use the simple validation
python3 run_validation.py
```

### Permission Denied
```bash
# Make scripts executable
chmod +x run_validation.py
chmod +x scripts/validation_checklist.py
```

## 📞 Support

For validation support:
- Check existing reports: `PRODUCTION_VALIDATION_REPORT.md`
- Review hardening plan: `PRODUCTION_HARDENING_PLAN.md`
- Contact: validation-team@aura-intelligence.ai

---

**Remember**: The DNA of AURA is proven resilience. Let's prove it! 🚀