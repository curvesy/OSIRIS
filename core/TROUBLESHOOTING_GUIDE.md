# 🔧 AURA Intelligence Troubleshooting Guide

**Version**: 1.0  
**Date**: January 2025  
**Purpose**: Comprehensive troubleshooting for AURA Intelligence platform  

---

## 🚨 Emergency Procedures

### System Unresponsive
```bash
# Check system status
systemctl status aura-api
systemctl status aura-workers
systemctl status redis
systemctl status postgresql

# Restart services if needed
sudo systemctl restart aura-api
sudo systemctl restart aura-workers
sudo systemctl restart redis
sudo systemctl restart postgresql
```

### Database Connection Issues
```bash
# Test database connectivity
psql -h localhost -U aura_user -d aura_db -c "SELECT 1;"

# Check Redis connection
redis-cli ping

# Verify connection strings in configuration
cat /etc/aura/config/database.conf
```

### Memory Issues
```bash
# Check memory usage
free -h
top -p $(pgrep -f aura)

# Restart if memory usage > 90%
sudo systemctl restart aura-workers
```

---

## 🔍 Common Issues and Solutions

### 1. Import Errors

#### Problem: ModuleNotFoundError for prometheus_client
```python
# Error: No module named 'prometheus_client'
```

**Solution**:
```bash
# Install missing dependency
pip install prometheus_client

# Or add to requirements.txt
echo "prometheus_client>=0.17.0" >> requirements.txt
pip install -r requirements.txt
```

#### Problem: AURA module import failures
```python
# Error: No module named 'aura_intelligence'
```

**Solution**:
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:/opt/aura/src"

# Or install in development mode
pip install -e .
```

### 2. Event Store Issues

#### Problem: Event processing failures
```python
# Error: Event store connection failed
```

**Solution**:
```bash
# Check event store health
curl -f http://localhost:8000/health

# Verify event store configuration
cat /etc/aura/config/event_store.conf

# Restart event store service
sudo systemctl restart aura-event-store
```

#### Problem: Projection lag
```python
# Warning: Projection lag > 60 seconds
```

**Solution**:
```bash
# Check projection status
curl -s http://localhost:8000/metrics | grep projection_lag

# Restart projection service
sudo systemctl restart aura-projections

# Check for dead letter queue
curl -s http://localhost:8000/api/v1/dead-letter-queue
```

### 3. Agent System Issues

#### Problem: Agent timeout errors
```python
# Error: Agent timeout after 30 seconds
```

**Solution**:
```bash
# Check agent health
curl -f http://localhost:8000/api/v1/agents/health

# Restart agent services
sudo systemctl restart aura-agents

# Check agent logs
journalctl -u aura-agents -f
```

#### Problem: Debate system failures
```python
# Error: Debate consensus not reached
```

**Solution**:
```bash
# Check debate system status
curl -f http://localhost:8000/api/v1/debates/status

# Restart debate orchestrator
sudo systemctl restart aura-debate-orchestrator

# Check debate logs
tail -f /var/log/aura/debate.log
```

### 4. Performance Issues

#### Problem: High latency (> 100ms)
```bash
# Check current performance
curl -s http://localhost:8000/metrics | grep response_time
```

**Solution**:
```bash
# Scale up workers
sudo systemctl restart aura-workers --scale=3

# Check resource usage
top -p $(pgrep -f aura)
free -h
df -h

# Optimize database queries
psql -h localhost -U aura_user -d aura_db -c "VACUUM ANALYZE;"
```

#### Problem: Memory leaks
```bash
# Check memory usage over time
watch -n 5 'free -h && ps aux | grep aura'
```

**Solution**:
```bash
# Restart services to clear memory
sudo systemctl restart aura-api
sudo systemctl restart aura-workers

# Check for memory leaks in logs
grep -i "memory\|leak" /var/log/aura/*.log
```

### 5. Security Issues

#### Problem: Authentication failures
```python
# Error: Invalid authentication token
```

**Solution**:
```bash
# Check authentication service
curl -f http://localhost:8000/api/v1/auth/health

# Verify token configuration
cat /etc/aura/config/auth.conf

# Restart auth service
sudo systemctl restart aura-auth
```

#### Problem: Permission denied errors
```bash
# Error: Permission denied for file operations
```

**Solution**:
```bash
# Check file permissions
ls -la /opt/aura/
chmod 755 /opt/aura/
chown -R aura:aura /opt/aura/

# Check service user permissions
id aura
```

---

## 📊 Monitoring and Diagnostics

### Health Check Commands
```bash
# Overall system health
curl -f http://localhost:8000/health

# Detailed health check
curl -f http://localhost:8000/health/detailed

# Component-specific health
curl -f http://localhost:8000/api/v1/agents/health
curl -f http://localhost:8000/api/v1/events/health
curl -f http://localhost:8000/api/v1/debates/health
```

### Log Analysis
```bash
# View real-time logs
tail -f /var/log/aura/application.log

# Search for errors
grep -i "error\|exception\|failed" /var/log/aura/*.log

# Check recent errors
journalctl -u aura-api --since "1 hour ago" | grep -i error
```

### Performance Monitoring
```bash
# Check current metrics
curl -s http://localhost:8000/metrics

# Monitor specific metrics
watch -n 5 'curl -s http://localhost:8000/metrics | grep -E "(requests_total|response_time|error_rate)"'
```

---

## 🔧 Configuration Issues

### Problem: Configuration file not found
```bash
# Error: Cannot find configuration file
```

**Solution**:
```bash
# Check configuration directory
ls -la /etc/aura/config/

# Create default configuration if missing
sudo mkdir -p /etc/aura/config/
sudo cp /opt/aura/config/default.conf /etc/aura/config/aura.conf

# Verify configuration syntax
aura-config-validate /etc/aura/config/aura.conf
```

### Problem: Environment variables not set
```bash
# Error: Required environment variable not set
```

**Solution**:
```bash
# Check current environment
env | grep AURA

# Set required environment variables
export AURA_DATABASE_URL="postgresql://aura_user:password@localhost:5432/aura_db"
export AURA_REDIS_URL="redis://localhost:6379"
export AURA_LOG_LEVEL="INFO"

# Add to system environment
echo 'export AURA_DATABASE_URL="postgresql://aura_user:password@localhost:5432/aura_db"' >> /etc/environment
```

---

## 🚨 Critical Error Recovery

### Database Corruption
```bash
# Stop all services
sudo systemctl stop aura-api aura-workers

# Create backup
pg_dump aura_db > /tmp/aura_backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
psql -h localhost -U postgres -c "DROP DATABASE aura_db;"
psql -h localhost -U postgres -c "CREATE DATABASE aura_db;"
psql -h localhost -U aura_user -d aura_db < /tmp/aura_backup_*.sql

# Restart services
sudo systemctl start aura-api aura-workers
```

### Event Store Recovery
```bash
# Check event store integrity
aura-event-store --verify-integrity

# Rebuild projections if needed
aura-projections --rebuild-all

# Restore from checkpoint
aura-event-store --restore-checkpoint /var/lib/aura/checkpoints/latest
```

### Complete System Recovery
```bash
# Full system restart
sudo systemctl stop aura-*
sudo systemctl start postgresql redis
sudo systemctl start aura-api aura-workers aura-event-store

# Verify all services
systemctl status aura-*
```

---

## 📞 Support Contacts

### Primary Contacts
- **DevOps Team**: devops@aura-intelligence.ai
- **Engineering Team**: engineering@aura-intelligence.ai
- **Security Team**: security@aura-intelligence.ai

### Escalation Path
1. Check this troubleshooting guide
2. Contact DevOps team
3. Escalate to Engineering lead
4. Contact CTO for critical issues

### Emergency Contacts
- **On-Call Engineer**: +1-555-0123
- **System Administrator**: +1-555-0124
- **CTO**: +1-555-0125

---

## 📋 Quick Reference

### Common Commands
```bash
# Service management
sudo systemctl {start|stop|restart|status} aura-*

# Log viewing
tail -f /var/log/aura/application.log
journalctl -u aura-api -f

# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8000/metrics

# Database operations
psql -h localhost -U aura_user -d aura_db
redis-cli ping
```

### Configuration Files
- **Main Config**: `/etc/aura/config/aura.conf`
- **Database Config**: `/etc/aura/config/database.conf`
- **Log Config**: `/etc/aura/config/logging.conf`
- **Security Config**: `/etc/aura/config/security.conf`

### Log Files
- **Application**: `/var/log/aura/application.log`
- **Error**: `/var/log/aura/error.log`
- **Access**: `/var/log/aura/access.log`
- **System**: `/var/log/aura/system.log`

---

**Note**: This troubleshooting guide should be updated with any new issues or solutions discovered during operation. 