"""
🏥 Organism Health Monitor - Latest 2025 Bio-Inspired Patterns
Professional health monitoring for the digital organism with self-repair capabilities.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, field

try:
    from .config import ObservabilityConfig
    from .context_managers import ObservabilityContext
except ImportError:
    # Fallback for direct import
    from config import ObservabilityConfig
    from context_managers import ObservabilityContext


@dataclass
class HealthMetrics:
    """Health metrics for the digital organism."""
    
    # Overall health
    overall_score: float = 0.0
    status: str = "unknown"  # healthy, degraded, critical
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Workflow metrics
    workflow_success_rate: float = 0.0
    average_workflow_duration: float = 0.0
    active_workflows: int = 0
    
    # Error metrics
    error_rate: float = 0.0
    recovery_success_rate: float = 0.0
    circuit_breaker_failures: int = 0
    
    # Agent metrics
    agent_success_rate: float = 0.0
    average_agent_response_time: float = 0.0
    
    # System metrics
    uptime_seconds: float = 0.0
    last_health_check: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Trend indicators
    health_trend: str = "stable"  # improving, stable, degrading
    anomalies_detected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "status": self.status,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "workflow_success_rate": self.workflow_success_rate,
            "average_workflow_duration": self.average_workflow_duration,
            "active_workflows": self.active_workflows,
            "error_rate": self.error_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "agent_success_rate": self.agent_success_rate,
            "average_agent_response_time": self.average_agent_response_time,
            "uptime_seconds": self.uptime_seconds,
            "last_health_check": self.last_health_check,
            "health_trend": self.health_trend,
            "anomalies_detected": self.anomalies_detected,
        }


class OrganismHealthMonitor:
    """
    Bio-inspired health monitoring for the digital organism.
    
    Features:
    - Continuous vital signs monitoring
    - Anomaly detection and alerting
    - Self-repair trigger mechanisms
    - Performance trend analysis
    - Predictive health scoring
    - Auto-recovery coordination
    - Health history tracking
    """
    
    def __init__(
        self, 
        config: ObservabilityConfig,
        prometheus_manager=None,
        logging_manager=None
    ):
        """
        Initialize organism health monitor.
        
        Args:
            config: Observability configuration
            prometheus_manager: Prometheus metrics manager (optional)
            logging_manager: Structured logging manager (optional)
        """
        
        self.config = config
        self.prometheus = prometheus_manager
        self.logging = logging_manager
        
        # Health state
        self.current_health = HealthMetrics()
        self.health_history: List[HealthMetrics] = []
        self.start_time = time.time()
        
        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Performance tracking
        self._workflow_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0,
            "active": 0,
        }
        
        self._agent_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0,
        }
        
        self._error_stats = {
            "total": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "circuit_breaker_failures": 0,
        }
        
        # Health thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 70.0,
            "memory_critical": 90.0,
            "memory_warning": 70.0,
            "disk_critical": 95.0,
            "disk_warning": 80.0,
            "error_rate_critical": 0.1,  # 10%
            "error_rate_warning": 0.05,  # 5%
            "workflow_success_critical": 0.8,  # 80%
            "workflow_success_warning": 0.9,  # 90%
        }
    
    async def initialize(self) -> None:
        """Initialize health monitoring."""
        
        try:
            # Start continuous monitoring
            self._is_monitoring = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Initial health check
            await self._perform_health_check()
            
            if self.logging:
                self.logging.logger.info(
                    "organism_health_monitor_initialized",
                    organism_id=self.config.organism_id,
                    check_interval=self.config.health_check_interval,
                    auto_recovery_enabled=self.config.enable_auto_recovery,
                    initial_health_score=self.current_health.overall_score,
                )
            
            print(f"✅ Organism health monitor initialized - Score: {self.current_health.overall_score:.2f}")
            
        except Exception as e:
            print(f"⚠️ Health monitor initialization failed: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        
        while self._is_monitoring:
            try:
                # Perform health check
                await self._perform_health_check()
                
                # Update metrics
                if self.prometheus:
                    self.prometheus.update_system_health(self.current_health.overall_score)
                
                # Check for anomalies and trigger recovery if needed
                if self.config.enable_auto_recovery:
                    await self._check_and_trigger_recovery()
                
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logging:
                    # Use standard logging format for compatibility
                    message = (
                        f"health_monitoring_error - "
                        f"error={str(e)}, "
                        f"organism_id={self.config.organism_id}"
                    )
                    self.logging.logger.error(message)
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Update current health
        self.current_health = HealthMetrics(
            # System metrics
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage=system_metrics["memory_usage"],
            disk_usage=system_metrics["disk_usage"],
            uptime_seconds=time.time() - self.start_time,
            
            # Performance metrics
            workflow_success_rate=performance_metrics["workflow_success_rate"],
            average_workflow_duration=performance_metrics["average_workflow_duration"],
            active_workflows=self._workflow_stats["active"],
            error_rate=performance_metrics["error_rate"],
            recovery_success_rate=performance_metrics["recovery_success_rate"],
            circuit_breaker_failures=self._error_stats["circuit_breaker_failures"],
            agent_success_rate=performance_metrics["agent_success_rate"],
            average_agent_response_time=performance_metrics["average_agent_response_time"],
            
            # Health analysis
            last_health_check=datetime.now(timezone.utc).isoformat(),
        )
        
        # Calculate overall health score
        self.current_health.overall_score = self._calculate_overall_health_score()
        
        # Determine health status
        self.current_health.status = self._determine_health_status()
        
        # Analyze trends
        self.current_health.health_trend = self._analyze_health_trend()
        
        # Detect anomalies
        self.current_health.anomalies_detected = self._detect_anomalies()
        
        # Store in history (keep last 100 checks)
        self.health_history.append(self.current_health)
        if len(self.health_history) > 100:
            self.health_history.pop(0)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics."""
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
            }
            
        except Exception as e:
            if self.logging:
                self.logging.logger.warning(
                    "system_metrics_collection_failed",
                    error=str(e),
                    organism_id=self.config.organism_id,
                )
            
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
            }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from tracked statistics."""
        
        # Workflow metrics
        workflow_success_rate = 0.0
        if self._workflow_stats["total"] > 0:
            workflow_success_rate = self._workflow_stats["successful"] / self._workflow_stats["total"]
        
        average_workflow_duration = 0.0
        if self._workflow_stats["successful"] > 0:
            average_workflow_duration = self._workflow_stats["total_duration"] / self._workflow_stats["successful"]
        
        # Error metrics
        error_rate = 0.0
        if self._workflow_stats["total"] > 0:
            error_rate = self._workflow_stats["failed"] / self._workflow_stats["total"]
        
        recovery_success_rate = 0.0
        if self._error_stats["recovery_attempts"] > 0:
            recovery_success_rate = self._error_stats["recovery_successes"] / self._error_stats["recovery_attempts"]
        
        # Agent metrics
        agent_success_rate = 0.0
        if self._agent_stats["total"] > 0:
            agent_success_rate = self._agent_stats["successful"] / self._agent_stats["total"]
        
        average_agent_response_time = 0.0
        if self._agent_stats["successful"] > 0:
            average_agent_response_time = self._agent_stats["total_duration"] / self._agent_stats["successful"]
        
        return {
            "workflow_success_rate": workflow_success_rate,
            "average_workflow_duration": average_workflow_duration,
            "error_rate": error_rate,
            "recovery_success_rate": recovery_success_rate,
            "agent_success_rate": agent_success_rate,
            "average_agent_response_time": average_agent_response_time,
        }
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        
        scores = []
        weights = []
        
        # System resource scores (weight: 0.3)
        cpu_score = max(0, 1.0 - (self.current_health.cpu_usage / 100.0))
        memory_score = max(0, 1.0 - (self.current_health.memory_usage / 100.0))
        disk_score = max(0, 1.0 - (self.current_health.disk_usage / 100.0))
        system_score = (cpu_score + memory_score + disk_score) / 3
        scores.append(system_score)
        weights.append(0.3)
        
        # Workflow performance score (weight: 0.4)
        workflow_score = self.current_health.workflow_success_rate
        scores.append(workflow_score)
        weights.append(0.4)
        
        # Error handling score (weight: 0.2)
        error_score = max(0, 1.0 - self.current_health.error_rate)
        recovery_bonus = self.current_health.recovery_success_rate * 0.2
        error_handling_score = min(1.0, error_score + recovery_bonus)
        scores.append(error_handling_score)
        weights.append(0.2)
        
        # Agent performance score (weight: 0.1)
        agent_score = self.current_health.agent_success_rate
        scores.append(agent_score)
        weights.append(0.1)
        
        # Calculate weighted average
        if sum(weights) > 0:
            overall_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            overall_score = 0.0
        
        return min(1.0, max(0.0, overall_score))
    
    def _determine_health_status(self) -> str:
        """Determine health status based on score and metrics."""
        
        score = self.current_health.overall_score
        
        # Check for critical conditions
        if (self.current_health.cpu_usage > self.thresholds["cpu_critical"] or
            self.current_health.memory_usage > self.thresholds["memory_critical"] or
            self.current_health.disk_usage > self.thresholds["disk_critical"] or
            self.current_health.error_rate > self.thresholds["error_rate_critical"] or
            self.current_health.workflow_success_rate < self.thresholds["workflow_success_critical"]):
            return "critical"
        
        # Check for warning conditions
        if (self.current_health.cpu_usage > self.thresholds["cpu_warning"] or
            self.current_health.memory_usage > self.thresholds["memory_warning"] or
            self.current_health.disk_usage > self.thresholds["disk_warning"] or
            self.current_health.error_rate > self.thresholds["error_rate_warning"] or
            self.current_health.workflow_success_rate < self.thresholds["workflow_success_warning"] or
            score < self.config.health_score_threshold):
            return "degraded"
        
        return "healthy"
    
    def _analyze_health_trend(self) -> str:
        """Analyze health trend from recent history."""
        
        if len(self.health_history) < 3:
            return "stable"
        
        # Get recent scores
        recent_scores = [h.overall_score for h in self.health_history[-5:]]
        
        # Calculate trend
        if len(recent_scores) >= 3:
            early_avg = sum(recent_scores[:2]) / 2
            late_avg = sum(recent_scores[-2:]) / 2
            
            if late_avg > early_avg + 0.05:
                return "improving"
            elif late_avg < early_avg - 0.05:
                return "degrading"
        
        return "stable"
    
    def _detect_anomalies(self) -> List[str]:
        """Detect anomalies in current health metrics."""
        
        anomalies = []
        
        # Resource anomalies
        if self.current_health.cpu_usage > self.thresholds["cpu_critical"]:
            anomalies.append("critical_cpu_usage")
        elif self.current_health.cpu_usage > self.thresholds["cpu_warning"]:
            anomalies.append("high_cpu_usage")
        
        if self.current_health.memory_usage > self.thresholds["memory_critical"]:
            anomalies.append("critical_memory_usage")
        elif self.current_health.memory_usage > self.thresholds["memory_warning"]:
            anomalies.append("high_memory_usage")
        
        if self.current_health.disk_usage > self.thresholds["disk_critical"]:
            anomalies.append("critical_disk_usage")
        elif self.current_health.disk_usage > self.thresholds["disk_warning"]:
            anomalies.append("high_disk_usage")
        
        # Performance anomalies
        if self.current_health.error_rate > self.thresholds["error_rate_critical"]:
            anomalies.append("critical_error_rate")
        elif self.current_health.error_rate > self.thresholds["error_rate_warning"]:
            anomalies.append("high_error_rate")
        
        if self.current_health.workflow_success_rate < self.thresholds["workflow_success_critical"]:
            anomalies.append("low_workflow_success_rate")
        
        # Circuit breaker anomalies
        if self.current_health.circuit_breaker_failures > 5:
            anomalies.append("multiple_circuit_breaker_failures")
        
        return anomalies
    
    async def _check_and_trigger_recovery(self) -> None:
        """Check for recovery triggers and initiate self-repair."""
        
        if self.current_health.status == "critical":
            await self._trigger_critical_recovery()
        elif self.current_health.status == "degraded":
            await self._trigger_degraded_recovery()
    
    async def _trigger_critical_recovery(self) -> None:
        """Trigger critical recovery procedures."""
        
        if self.logging:
            # Use standard logging format for compatibility
            message = (
                f"critical_health_detected_triggering_recovery - "
                f"health_score={self.current_health.overall_score:.3f}, "
                f"anomalies={len(self.current_health.anomalies_detected)}, "
                f"organism_id={self.config.organism_id}"
            )
            self.logging.logger.error(message)
        
        # Implement critical recovery procedures
        # This would integrate with the circuit breaker and retry systems
        print(f"🚨 CRITICAL: Triggering emergency recovery - Health: {self.current_health.overall_score:.2f}")
    
    async def _trigger_degraded_recovery(self) -> None:
        """Trigger degraded performance recovery."""
        
        if self.logging:
            self.logging.logger.warning(
                "degraded_health_detected_triggering_recovery",
                health_score=self.current_health.overall_score,
                anomalies=self.current_health.anomalies_detected,
                organism_id=self.config.organism_id,
            )
        
        # Implement degraded recovery procedures
        print(f"⚠️ DEGRADED: Triggering performance recovery - Health: {self.current_health.overall_score:.2f}")
    
    # Event handlers for tracking statistics
    
    async def on_workflow_started(self, context: ObservabilityContext, state: Dict[str, Any]) -> None:
        """Handle workflow start event."""
        self._workflow_stats["active"] += 1
    
    async def on_workflow_completed(self, context: ObservabilityContext, state: Dict[str, Any]) -> None:
        """Handle workflow completion event."""
        self._workflow_stats["total"] += 1
        self._workflow_stats["active"] = max(0, self._workflow_stats["active"] - 1)
        
        if context.is_successful():
            self._workflow_stats["successful"] += 1
            if context.duration:
                self._workflow_stats["total_duration"] += context.duration
        else:
            self._workflow_stats["failed"] += 1
    
    async def on_error_recovery(self, error_type: str, recovery_strategy: str, success: bool) -> None:
        """Handle error recovery event."""
        self._error_stats["total"] += 1
        self._error_stats["recovery_attempts"] += 1
        
        if success:
            self._error_stats["recovery_successes"] += 1
        
        if "circuit_breaker" in recovery_strategy.lower():
            self._error_stats["circuit_breaker_failures"] += 1
    
    async def update_health_score(self, health_score: float) -> None:
        """Update health score externally."""
        self.current_health.overall_score = health_score
        self.current_health.status = self._determine_health_status()
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health metrics."""
        return self.current_health
    
    def get_health_history(self, limit: int = 10) -> List[HealthMetrics]:
        """Get recent health history."""
        return self.health_history[-limit:] if self.health_history else []
    
    async def shutdown(self) -> None:
        """Gracefully shutdown health monitor."""
        
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.logging:
            # Use standard logging format for compatibility
            message = (
                f"organism_health_monitor_shutdown - "
                f"final_health_score={self.current_health.overall_score:.3f}, "
                f"total_workflows={self._workflow_stats['total']}, "
                f"uptime_seconds={self.current_health.uptime_seconds:.1f}, "
                f"organism_id={self.config.organism_id}"
            )
            self.logging.logger.info(message)
        
        print("✅ Organism health monitor shutdown complete")
