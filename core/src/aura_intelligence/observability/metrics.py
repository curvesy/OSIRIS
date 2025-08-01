"""
Prometheus metrics for AURA Intelligence observability.

This module defines all Prometheus metrics used throughout the system
for monitoring, alerting, and performance analysis.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from prometheus_client import REGISTRY, CollectorRegistry
from typing import Optional

# Create a custom registry if needed
metrics_registry = REGISTRY

# ============================================================================
# Event Store Metrics
# ============================================================================

# Event processing metrics
event_processed_total = Counter(
    'aura_events_processed_total',
    'Total number of events processed',
    ['event_type', 'status'],
    registry=metrics_registry
)

event_processing_duration = Histogram(
    'aura_event_processing_duration_seconds',
    'Event processing duration in seconds',
    ['event_type'],
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10],
    registry=metrics_registry
)

event_size_bytes = Histogram(
    'aura_event_size_bytes',
    'Size of events in bytes',
    ['event_type'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    registry=metrics_registry
)

# Idempotency metrics
duplicate_events_detected = Counter(
    'aura_duplicate_events_detected_total',
    'Total number of duplicate events detected',
    ['event_type', 'detection_method'],
    registry=metrics_registry
)

idempotency_cache_size = Gauge(
    'aura_idempotency_cache_size',
    'Current size of idempotency cache',
    registry=metrics_registry
)

# Event store errors
event_store_errors = Counter(
    'aura_event_store_errors_total',
    'Total number of event store errors',
    ['error_type', 'operation'],
    registry=metrics_registry
)

event_store_connection_status = Gauge(
    'aura_event_store_connection_status',
    'Event store connection status (1=connected, 0=disconnected)',
    registry=metrics_registry
)

# ============================================================================
# Projection Metrics
# ============================================================================

projection_lag = Gauge(
    'aura_projection_lag_seconds',
    'Projection lag behind event stream in seconds',
    ['projection_name'],
    registry=metrics_registry
)

projection_errors = Counter(
    'aura_projection_errors_total',
    'Total projection errors',
    ['projection_name', 'error_type'],
    registry=metrics_registry
)

projection_events_processed = Counter(
    'aura_projection_events_processed_total',
    'Total events processed by projection',
    ['projection_name', 'event_type'],
    registry=metrics_registry
)

projection_processing_duration = Histogram(
    'aura_projection_processing_duration_seconds',
    'Time spent processing events in projections',
    ['projection_name'],
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5],
    registry=metrics_registry
)

projection_health = Gauge(
    'aura_projection_health',
    'Projection health status (0=unhealthy, 1=healthy)',
    ['projection_name'],
    registry=metrics_registry
)

projection_dlq_size = Gauge(
    'aura_projection_dlq_size',
    'Number of events in projection dead letter queue',
    ['projection_name'],
    registry=metrics_registry
)

# ============================================================================
# Debate System Metrics
# ============================================================================

active_debates = Gauge(
    'aura_active_debates',
    'Number of currently active debates',
    registry=metrics_registry
)

debate_duration = Histogram(
    'aura_debate_duration_seconds',
    'Duration of debates in seconds',
    ['outcome'],  # consensus, timeout, failed
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
    registry=metrics_registry
)

debate_consensus_rate = Gauge(
    'aura_debate_consensus_rate',
    'Rate of debates reaching consensus (rolling window)',
    ['time_window'],  # 1h, 24h, 7d
    registry=metrics_registry
)

debate_arguments_total = Counter(
    'aura_debate_arguments_total',
    'Total number of arguments in debates',
    ['agent_type', 'argument_type'],
    registry=metrics_registry
)

debate_participant_count = Histogram(
    'aura_debate_participant_count',
    'Number of participants per debate',
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    registry=metrics_registry
)

# ============================================================================
# Agent Metrics
# ============================================================================

agent_decisions_proposed = Counter(
    'aura_agent_decisions_proposed_total',
    'Total decisions proposed by agents',
    ['agent_id', 'decision_type'],
    registry=metrics_registry
)

agent_decisions_approved = Counter(
    'aura_agent_decisions_approved_total',
    'Total decisions approved',
    ['agent_id', 'decision_type'],
    registry=metrics_registry
)

agent_decisions_rejected = Counter(
    'aura_agent_decisions_rejected_total',
    'Total decisions rejected',
    ['agent_id', 'decision_type', 'reason'],
    registry=metrics_registry
)

agent_confidence_score = Histogram(
    'aura_agent_confidence_score',
    'Agent confidence scores',
    ['agent_id', 'context'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=metrics_registry
)

agent_response_time = Histogram(
    'aura_agent_response_time_seconds',
    'Agent response time in seconds',
    ['agent_id', 'operation'],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50],
    registry=metrics_registry
)

agent_errors = Counter(
    'aura_agent_errors_total',
    'Total agent errors',
    ['agent_id', 'error_type'],
    registry=metrics_registry
)

# ============================================================================
# System Health Metrics
# ============================================================================

system_health_score = Gauge(
    'aura_system_health_score',
    'Overall system health score (0-100)',
    registry=metrics_registry
)

system_uptime_seconds = Gauge(
    'aura_system_uptime_seconds',
    'System uptime in seconds',
    registry=metrics_registry
)

system_component_status = Gauge(
    'aura_system_component_status',
    'Component status (1=healthy, 0=unhealthy)',
    ['component'],
    registry=metrics_registry
)

# ============================================================================
# Performance Metrics
# ============================================================================

request_duration = Histogram(
    'aura_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10],
    registry=metrics_registry
)

request_size_bytes = Histogram(
    'aura_request_size_bytes',
    'Request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=metrics_registry
)

response_size_bytes = Histogram(
    'aura_response_size_bytes',
    'Response size in bytes',
    ['method', 'endpoint', 'status'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=metrics_registry
)

concurrent_requests = Gauge(
    'aura_concurrent_requests',
    'Number of concurrent requests',
    ['endpoint'],
    registry=metrics_registry
)

# ============================================================================
# Resource Usage Metrics
# ============================================================================

memory_usage_bytes = Gauge(
    'aura_memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],
    registry=metrics_registry
)

cpu_usage_percent = Gauge(
    'aura_cpu_usage_percent',
    'CPU usage percentage',
    ['component'],
    registry=metrics_registry
)

goroutines_count = Gauge(
    'aura_goroutines_count',
    'Number of goroutines',
    registry=metrics_registry
)

open_file_descriptors = Gauge(
    'aura_open_file_descriptors',
    'Number of open file descriptors',
    registry=metrics_registry
)

# ============================================================================
# Database Metrics
# ============================================================================

db_connections_active = Gauge(
    'aura_db_connections_active',
    'Number of active database connections',
    ['database', 'pool'],
    registry=metrics_registry
)

db_connections_idle = Gauge(
    'aura_db_connections_idle',
    'Number of idle database connections',
    ['database', 'pool'],
    registry=metrics_registry
)

db_query_duration = Histogram(
    'aura_db_query_duration_seconds',
    'Database query duration',
    ['database', 'operation'],
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5],
    registry=metrics_registry
)

db_errors = Counter(
    'aura_db_errors_total',
    'Total database errors',
    ['database', 'operation', 'error_type'],
    registry=metrics_registry
)

# ============================================================================
# Circuit Breaker Metrics
# ============================================================================

circuit_breaker_state = Gauge(
    'aura_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['name'],
    registry=metrics_registry
)

circuit_breaker_failures = Counter(
    'aura_circuit_breaker_failures_total',
    'Total circuit breaker failures',
    ['name'],
    registry=metrics_registry
)

circuit_breaker_successes = Counter(
    'aura_circuit_breaker_successes_total',
    'Total circuit breaker successes',
    ['name'],
    registry=metrics_registry
)

# ============================================================================
# Chaos Engineering Metrics
# ============================================================================

chaos_experiments_run = Counter(
    'aura_chaos_experiments_run_total',
    'Total chaos experiments run',
    ['experiment_type', 'target'],
    registry=metrics_registry
)

chaos_experiments_failed = Counter(
    'aura_chaos_experiments_failed_total',
    'Total chaos experiments that revealed failures',
    ['experiment_type', 'target', 'failure_type'],
    registry=metrics_registry
)

chaos_injection_active = Gauge(
    'aura_chaos_injection_active',
    'Whether chaos injection is currently active',
    ['injection_type'],
    registry=metrics_registry
)

# ============================================================================
# Business Metrics
# ============================================================================

business_decisions_made = Counter(
    'aura_business_decisions_made_total',
    'Total business decisions made',
    ['decision_type', 'outcome'],
    registry=metrics_registry
)

business_value_generated = Counter(
    'aura_business_value_generated_total',
    'Total business value generated (in arbitrary units)',
    ['value_type'],
    registry=metrics_registry
)

# ============================================================================
# Helper Functions
# ============================================================================

def record_event_processed(event_type: str, status: str = "success") -> None:
    """Record that an event was processed"""
    event_processed_total.labels(event_type=event_type, status=status).inc()

def record_event_duration(event_type: str, duration: float) -> None:
    """Record event processing duration"""
    event_processing_duration.labels(event_type=event_type).observe(duration)

def update_system_health(score: float) -> None:
    """Update system health score (0-100)"""
    system_health_score.set(max(0, min(100, score)))

def update_component_status(component: str, is_healthy: bool) -> None:
    """Update component health status"""
    system_component_status.labels(component=component).set(1 if is_healthy else 0)

def record_agent_decision(agent_id: str, decision_type: str, outcome: str) -> None:
    """Record agent decision outcome"""
    if outcome == "proposed":
        agent_decisions_proposed.labels(agent_id=agent_id, decision_type=decision_type).inc()
    elif outcome == "approved":
        agent_decisions_approved.labels(agent_id=agent_id, decision_type=decision_type).inc()
    elif outcome == "rejected":
        agent_decisions_rejected.labels(
            agent_id=agent_id, 
            decision_type=decision_type,
            reason="unknown"
        ).inc()

def update_debate_metrics(active: int, consensus_rate_1h: float) -> None:
    """Update debate-related metrics"""
    active_debates.set(active)
    debate_consensus_rate.labels(time_window="1h").set(consensus_rate_1h)

# ============================================================================
# Metric Info
# ============================================================================

system_info = Info(
    'aura_system',
    'System information',
    registry=metrics_registry
)

# Set system info
system_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'architecture': 'event-sourced',
    'platform': 'kubernetes'
})