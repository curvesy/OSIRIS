"""
📊 Health Monitoring & Metrics

Prometheus metrics, health checks, and autoscaling triggers for the
Phase 2C Intelligence Flywheel system.
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from aura_intelligence.utils.logger import get_logger


@dataclass
class HealthThresholds:
    """Health check thresholds for autoscaling decisions."""
    
    # Performance thresholds
    max_response_time_ms: float = 100.0
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_disk_percent: float = 90.0
    
    # Database thresholds
    max_db_size_gb: float = 10.0
    max_query_time_ms: float = 50.0
    max_concurrent_connections: int = 100
    
    # Error rate thresholds
    max_error_rate_percent: float = 5.0
    max_circuit_breaker_failures: int = 10
    
    # Autoscaling triggers
    scale_up_cpu_threshold: float = 70.0
    scale_down_cpu_threshold: float = 30.0
    scale_up_memory_threshold: float = 75.0
    scale_down_memory_threshold: float = 40.0


class PrometheusMetrics:
    """
    📈 Prometheus Metrics Collector
    
    Collects and exposes metrics for monitoring and autoscaling.
    """
    
    def __init__(self, registry: CollectorRegistry = None):
        """Initialize Prometheus metrics."""
        
        if not PROMETHEUS_AVAILABLE:
            self.logger = get_logger(__name__)
            self.logger.warning("⚠️ Prometheus client not available - metrics disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'aura_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'aura_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_query_count = Counter(
            'aura_db_queries_total',
            'Total database queries',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'aura_db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            registry=self.registry
        )
        
        self.db_size_bytes = Gauge(
            'aura_db_size_bytes',
            'Database size in bytes',
            registry=self.registry
        )
        
        self.db_connections = Gauge(
            'aura_db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage_bytes = Gauge(
            'aura_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.memory_cache_hits = Counter(
            'aura_memory_cache_hits_total',
            'Memory cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.memory_cache_misses = Counter(
            'aura_memory_cache_misses_total',
            'Memory cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'aura_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'aura_circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['service'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage_percent = Gauge(
            'aura_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'aura_disk_usage_percent',
            'Disk usage percentage',
            ['mount'],
            registry=self.registry
        )
        
        # Intelligence Flywheel metrics
        self.flywheel_cycles = Counter(
            'aura_flywheel_cycles_total',
            'Intelligence Flywheel cycles',
            ['stage', 'status'],
            registry=self.registry
        )
        
        self.flywheel_insights = Gauge(
            'aura_flywheel_insights_generated',
            'Insights generated by Intelligence Flywheel',
            registry=self.registry
        )
        
        self.logger = get_logger(__name__)
        self.logger.info("📈 Prometheus metrics initialized")
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics."""
        if not self.enabled:
            return
        
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_db_query(self, operation: str, status: str, duration: float):
        """Record database query metrics."""
        if not self.enabled:
            return
        
        self.db_query_count.labels(operation=operation, status=status).inc()
        self.db_query_duration.labels(operation=operation).observe(duration)
    
    def update_db_size(self, size_bytes: int):
        """Update database size metric."""
        if not self.enabled:
            return
        
        self.db_size_bytes.set(size_bytes)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        if not self.enabled:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage_percent.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage_bytes.labels(type='used').set(memory.used)
        self.memory_usage_bytes.labels(type='available').set(memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.disk_usage_percent.labels(mount='/').set(disk_percent)
    
    def record_circuit_breaker_state(self, service: str, state: str):
        """Record circuit breaker state."""
        if not self.enabled:
            return
        
        state_mapping = {"closed": 0, "open": 1, "half_open": 2}
        self.circuit_breaker_state.labels(service=service).set(state_mapping.get(state, 0))
    
    def record_flywheel_cycle(self, stage: str, status: str):
        """Record Intelligence Flywheel cycle."""
        if not self.enabled:
            return
        
        self.flywheel_cycles.labels(stage=stage, status=status).inc()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


class HealthChecker:
    """
    🏥 Health Checker
    
    Comprehensive health monitoring with autoscaling recommendations.
    """
    
    def __init__(self, 
                 thresholds: HealthThresholds = None,
                 metrics: PrometheusMetrics = None):
        """Initialize health checker."""
        
        self.thresholds = thresholds or HealthThresholds()
        self.metrics = metrics
        
        # Health status tracking
        self.last_health_check = None
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # Autoscaling recommendations
        self.scale_recommendations: List[Dict[str, Any]] = []
        
        self.logger = get_logger(__name__)
        self.logger.info("🏥 Health checker initialized")
    
    async def perform_health_check(self, 
                                  db_ops=None,
                                  redis_ops=None,
                                  additional_checks: List[Callable] = None) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Args:
            db_ops: DuckDB operations instance
            redis_ops: Redis operations instance
            additional_checks: Additional health check functions
            
        Returns:
            Health status with autoscaling recommendations
        """
        
        start_time = time.time()
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "metrics": {},
            "autoscaling": {
                "recommendations": [],
                "current_load": {}
            }
        }
        
        try:
            # System resource checks
            system_health = await self._check_system_resources()
            health_status["checks"]["system"] = system_health
            
            # Database health checks
            if db_ops:
                db_health = await self._check_database_health(db_ops)
                health_status["checks"]["database"] = db_health
            
            # Redis health checks
            if redis_ops:
                redis_health = await self._check_redis_health(redis_ops)
                health_status["checks"]["redis"] = redis_health
            
            # Additional custom checks
            if additional_checks:
                for i, check_func in enumerate(additional_checks):
                    try:
                        check_result = await check_func()
                        health_status["checks"][f"custom_{i}"] = check_result
                    except Exception as e:
                        health_status["checks"][f"custom_{i}"] = {
                            "status": "error",
                            "error": str(e)
                        }
            
            # Determine overall health
            overall_healthy = all(
                check.get("status") == "healthy" 
                for check in health_status["checks"].values()
            )
            
            health_status["overall_status"] = "healthy" if overall_healthy else "degraded"
            
            # Generate autoscaling recommendations
            autoscaling_recs = self._generate_autoscaling_recommendations(health_status)
            health_status["autoscaling"]["recommendations"] = autoscaling_recs
            
            # Update metrics
            if self.metrics:
                self.metrics.update_system_metrics()
            
            # Store in history
            self._update_health_history(health_status)
            
            check_duration = time.time() - start_time
            health_status["check_duration_seconds"] = check_duration
            
            self.last_health_check = datetime.now()
            
            if overall_healthy:
                self.logger.debug(f"💚 Health check passed in {check_duration:.3f}s")
            else:
                self.logger.warning(f"⚠️ Health check degraded in {check_duration:.3f}s")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            
            health_status.update({
                "overall_status": "error",
                "error": str(e),
                "check_duration_seconds": time.time() - start_time
            })
            
            return health_status
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status
            status = "healthy"
            issues = []
            
            if cpu_percent > self.thresholds.max_cpu_percent:
                status = "degraded"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.thresholds.max_memory_percent:
                status = "degraded"
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.thresholds.max_disk_percent:
                status = "degraded"
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _check_database_health(self, db_ops) -> Dict[str, Any]:
        """Check database health."""
        
        try:
            # Get database health from resilient operations
            db_health = db_ops.get_health_status()
            
            # Additional database-specific checks
            circuit_status = db_health.get("circuit_breaker", {}).get("state", "unknown")
            dlq_size = db_health.get("dead_letter_queue", {}).get("queue_size", 0)
            
            status = "healthy"
            issues = []
            
            if circuit_status != "closed":
                status = "degraded"
                issues.append(f"Circuit breaker {circuit_status}")
            
            if dlq_size > 10:
                status = "degraded"
                issues.append(f"High DLQ size: {dlq_size}")
            
            return {
                "status": status,
                "circuit_breaker_state": circuit_status,
                "dead_letter_queue_size": dlq_size,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _check_redis_health(self, redis_ops) -> Dict[str, Any]:
        """Check Redis health."""
        
        try:
            # Get Redis health from resilient operations
            redis_health = redis_ops.get_health_status()
            
            circuit_status = redis_health.get("circuit_breaker", {}).get("state", "unknown")
            dlq_size = redis_health.get("dead_letter_queue", {}).get("queue_size", 0)
            
            status = "healthy"
            issues = []
            
            if circuit_status != "closed":
                status = "degraded"
                issues.append(f"Circuit breaker {circuit_status}")
            
            if dlq_size > 5:
                status = "degraded"
                issues.append(f"High DLQ size: {dlq_size}")
            
            return {
                "status": status,
                "circuit_breaker_state": circuit_status,
                "dead_letter_queue_size": dlq_size,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_autoscaling_recommendations(self, health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate autoscaling recommendations based on health status."""
        
        recommendations = []
        
        try:
            system_check = health_status.get("checks", {}).get("system", {})
            
            cpu_percent = system_check.get("cpu_percent", 0)
            memory_percent = system_check.get("memory_percent", 0)
            
            # CPU-based scaling recommendations
            if cpu_percent > self.thresholds.scale_up_cpu_threshold:
                recommendations.append({
                    "action": "scale_up",
                    "reason": f"High CPU usage: {cpu_percent:.1f}%",
                    "metric": "cpu",
                    "current_value": cpu_percent,
                    "threshold": self.thresholds.scale_up_cpu_threshold,
                    "priority": "high" if cpu_percent > 85 else "medium"
                })
            
            elif cpu_percent < self.thresholds.scale_down_cpu_threshold:
                recommendations.append({
                    "action": "scale_down",
                    "reason": f"Low CPU usage: {cpu_percent:.1f}%",
                    "metric": "cpu",
                    "current_value": cpu_percent,
                    "threshold": self.thresholds.scale_down_cpu_threshold,
                    "priority": "low"
                })
            
            # Memory-based scaling recommendations
            if memory_percent > self.thresholds.scale_up_memory_threshold:
                recommendations.append({
                    "action": "scale_up",
                    "reason": f"High memory usage: {memory_percent:.1f}%",
                    "metric": "memory",
                    "current_value": memory_percent,
                    "threshold": self.thresholds.scale_up_memory_threshold,
                    "priority": "high" if memory_percent > 90 else "medium"
                })
            
            elif memory_percent < self.thresholds.scale_down_memory_threshold:
                recommendations.append({
                    "action": "scale_down",
                    "reason": f"Low memory usage: {memory_percent:.1f}%",
                    "metric": "memory",
                    "current_value": memory_percent,
                    "threshold": self.thresholds.scale_down_memory_threshold,
                    "priority": "low"
                })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate autoscaling recommendations: {e}")
        
        return recommendations
    
    def _update_health_history(self, health_status: Dict[str, Any]):
        """Update health history for trend analysis."""
        
        # Add to history
        self.health_history.append({
            "timestamp": health_status["timestamp"],
            "overall_status": health_status["overall_status"],
            "system_cpu": health_status.get("checks", {}).get("system", {}).get("cpu_percent", 0),
            "system_memory": health_status.get("checks", {}).get("system", {}).get("memory_percent", 0)
        })
        
        # Trim history if too large
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = [
            entry for entry in self.health_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
        
        if not recent_history:
            return {"status": "no_data", "message": "No recent health data available"}
        
        # Calculate trends
        avg_cpu = sum(entry["system_cpu"] for entry in recent_history) / len(recent_history)
        avg_memory = sum(entry["system_memory"] for entry in recent_history) / len(recent_history)
        
        healthy_count = sum(1 for entry in recent_history if entry["overall_status"] == "healthy")
        health_percentage = (healthy_count / len(recent_history)) * 100
        
        return {
            "period_hours": hours,
            "data_points": len(recent_history),
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "health_percentage": health_percentage,
            "trend_status": "improving" if health_percentage > 90 else "degrading" if health_percentage < 70 else "stable"
        }
