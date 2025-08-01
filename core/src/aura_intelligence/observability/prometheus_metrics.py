"""
📈 Prometheus Metrics Manager - Latest 2025 AI Patterns
Professional Prometheus integration with AI/LLM specific metrics and cost tracking.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, multiprocess, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)

try:
    from .config import ObservabilityConfig
    from .context_managers import ObservabilityContext
except ImportError:
    # Fallback for direct import
    from config import ObservabilityConfig
    from context_managers import ObservabilityContext


class PrometheusMetricsManager:
    """
    Latest 2025 Prometheus integration with AI-specific metrics.
    
    Features:
    - AI workflow performance metrics
    - LLM usage and cost tracking
    - Agent performance monitoring
    - Error rate and recovery metrics
    - System health and organism vitals
    - Multi-process support for production
    - Custom AI-specific metric types
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize Prometheus metrics manager.
        
        Args:
            config: Observability configuration
        """
        
        self.config = config
        self.is_available = PROMETHEUS_AVAILABLE
        self.registry: Optional[CollectorRegistry] = None
        self.http_server = None
        
        # Metric instruments
        self._workflow_metrics = {}
        self._agent_metrics = {}
        self._llm_metrics = {}
        self._error_metrics = {}
        self._system_metrics = {}
        self._cost_metrics = {}
    
    async def initialize(self) -> None:
        """
        Initialize Prometheus metrics with latest 2025 AI patterns.
        """
        
        if not self.is_available:
            print("⚠️ Prometheus not available - skipping metrics initialization")
            return
        
        try:
            # Setup registry (multi-process if configured)
            if self.config.prometheus_enable_multiprocess:
                self.registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(self.registry)
            else:
                self.registry = None  # Use default registry
            
            # Create AI-specific metrics
            self._create_workflow_metrics()
            self._create_agent_metrics()
            self._create_llm_metrics()
            self._create_error_metrics()
            self._create_system_metrics()
            self._create_cost_metrics()
            
            # Start HTTP server for metrics endpoint
            if self.config.prometheus_port > 0:
                self.http_server = start_http_server(
                    self.config.prometheus_port,
                    registry=self.registry
                )
            
            print(f"✅ Prometheus metrics initialized on port {self.config.prometheus_port}")
            
        except Exception as e:
            print(f"⚠️ Prometheus initialization failed: {e}")
            self.is_available = False
    
    def _create_workflow_metrics(self) -> None:
        """Create workflow-specific metrics."""
        
        if not self.is_available:
            return
        
        # Workflow execution metrics
        self._workflow_metrics = {
            'duration': Histogram(
                'ai_workflow_duration_seconds',
                'Duration of AI workflow executions',
                ['workflow_type', 'status', 'organism_id'],
                registry=self.registry,
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
            ),
            
            'total': Counter(
                'ai_workflow_total',
                'Total number of AI workflow executions',
                ['workflow_type', 'status', 'organism_id'],
                registry=self.registry
            ),
            
            'active': Gauge(
                'ai_workflow_active',
                'Number of currently active workflows',
                ['workflow_type', 'organism_id'],
                registry=self.registry
            ),
            
            'evidence_count': Histogram(
                'ai_workflow_evidence_count',
                'Number of evidence items processed per workflow',
                ['workflow_type', 'organism_id'],
                registry=self.registry,
                buckets=[1, 5, 10, 20, 50, 100, 200, 500]
            ),
            
            'recovery_attempts': Counter(
                'ai_workflow_recovery_attempts_total',
                'Total number of error recovery attempts',
                ['workflow_type', 'recovery_strategy', 'success', 'organism_id'],
                registry=self.registry
            ),
        }
    
    def _create_agent_metrics(self) -> None:
        """Create agent-specific metrics."""
        
        if not self.is_available:
            return
        
        self._agent_metrics = {
            'calls': Counter(
                'ai_agent_calls_total',
                'Total number of agent tool calls',
                ['agent_name', 'tool_name', 'status', 'organism_id'],
                registry=self.registry
            ),
            
            'duration': Histogram(
                'ai_agent_call_duration_seconds',
                'Duration of agent tool calls',
                ['agent_name', 'tool_name', 'organism_id'],
                registry=self.registry,
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            ),
            
            'active': Gauge(
                'ai_agent_active_calls',
                'Number of currently active agent calls',
                ['agent_name', 'organism_id'],
                registry=self.registry
            ),
            
            'success_rate': Gauge(
                'ai_agent_success_rate',
                'Success rate of agent calls (0-1)',
                ['agent_name', 'tool_name', 'organism_id'],
                registry=self.registry
            ),
        }
    
    def _create_llm_metrics(self) -> None:
        """Create LLM-specific metrics with latest 2025 patterns."""
        
        if not self.is_available:
            return
        
        self._llm_metrics = {
            'tokens': Counter(
                'ai_llm_tokens_total',
                'Total number of LLM tokens processed',
                ['model_name', 'token_type', 'organism_id'],  # token_type: input/output
                registry=self.registry
            ),
            
            'requests': Counter(
                'ai_llm_requests_total',
                'Total number of LLM requests',
                ['model_name', 'status', 'organism_id'],
                registry=self.registry
            ),
            
            'latency': Histogram(
                'ai_llm_request_duration_seconds',
                'LLM request latency',
                ['model_name', 'organism_id'],
                registry=self.registry,
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
            ),
            
            'throughput': Gauge(
                'ai_llm_throughput_tokens_per_second',
                'LLM throughput in tokens per second',
                ['model_name', 'organism_id'],
                registry=self.registry
            ),
            
            'context_length': Histogram(
                'ai_llm_context_length',
                'LLM context length distribution',
                ['model_name', 'organism_id'],
                registry=self.registry,
                buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
            ),
        }
    
    def _create_error_metrics(self) -> None:
        """Create error and recovery metrics."""
        
        if not self.is_available:
            return
        
        self._error_metrics = {
            'total': Counter(
                'ai_errors_total',
                'Total number of AI system errors',
                ['error_type', 'component', 'severity', 'organism_id'],
                registry=self.registry
            ),
            
            'recovery_success': Counter(
                'ai_error_recovery_success_total',
                'Successful error recoveries',
                ['error_type', 'recovery_strategy', 'organism_id'],
                registry=self.registry
            ),
            
            'recovery_failure': Counter(
                'ai_error_recovery_failure_total',
                'Failed error recoveries',
                ['error_type', 'recovery_strategy', 'organism_id'],
                registry=self.registry
            ),
            
            'circuit_breaker_state': Enum(
                'ai_circuit_breaker_state',
                'Circuit breaker state',
                ['component', 'organism_id'],
                states=['closed', 'open', 'half_open'],
                registry=self.registry
            ),
        }
    
    def _create_system_metrics(self) -> None:
        """Create system health and organism vitals."""
        
        if not self.is_available:
            return
        
        self._system_metrics = {
            'health_score': Gauge(
                'ai_system_health_score',
                'Overall system health score (0-1)',
                ['organism_id'],
                registry=self.registry
            ),
            
            'uptime': Gauge(
                'ai_system_uptime_seconds',
                'System uptime in seconds',
                ['organism_id'],
                registry=self.registry
            ),
            
            'memory_usage': Gauge(
                'ai_system_memory_usage_bytes',
                'System memory usage',
                ['component', 'organism_id'],
                registry=self.registry
            ),
            
            'active_connections': Gauge(
                'ai_system_active_connections',
                'Number of active connections',
                ['connection_type', 'organism_id'],
                registry=self.registry
            ),
            
            'organism_generation': Info(
                'ai_organism_info',
                'Organism information',
                registry=self.registry
            ),
        }
        
        # Set organism info
        self._system_metrics['organism_generation'].info({
            'organism_id': self.config.organism_id,
            'generation': str(self.config.organism_generation),
            'environment': self.config.deployment_environment,
            'version': self.config.service_version,
            'neural_observability_version': '2025.7.27',
        })
    
    def _create_cost_metrics(self) -> None:
        """Create cost tracking metrics."""
        
        if not self.is_available:
            return
        
        self._cost_metrics = {
            'total_usd': Counter(
                'ai_cost_total_usd',
                'Total AI operation costs in USD',
                ['cost_type', 'model_name', 'organism_id'],  # cost_type: llm_usage, compute, storage
                registry=self.registry
            ),
            
            'cost_per_request': Histogram(
                'ai_cost_per_request_usd',
                'Cost per request in USD',
                ['model_name', 'organism_id'],
                registry=self.registry,
                buckets=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            ),
            
            'budget_utilization': Gauge(
                'ai_budget_utilization_ratio',
                'Budget utilization ratio (0-1)',
                ['budget_type', 'organism_id'],
                registry=self.registry
            ),
        }
    
    def record_workflow_started(self, context: ObservabilityContext) -> None:
        """Record workflow start metrics."""
        
        if not self.is_available or not self._workflow_metrics:
            return
        
        labels = {
            'workflow_type': context.workflow_type,
            'organism_id': self.config.organism_id
        }
        
        # Increment active workflows
        self._workflow_metrics['active'].labels(**labels).inc()
    
    def record_workflow_completed(self, context: ObservabilityContext) -> None:
        """Record workflow completion metrics."""
        
        if not self.is_available or not self._workflow_metrics:
            return
        
        labels = {
            'workflow_type': context.workflow_type,
            'status': context.status,
            'organism_id': self.config.organism_id
        }
        
        # Record duration
        if context.duration:
            self._workflow_metrics['duration'].labels(**labels).observe(context.duration)
        
        # Increment total
        self._workflow_metrics['total'].labels(**labels).inc()
        
        # Decrement active
        active_labels = {
            'workflow_type': context.workflow_type,
            'organism_id': self.config.organism_id
        }
        self._workflow_metrics['active'].labels(**active_labels).dec()
        
        # Record evidence count
        evidence_count = context.metadata.get('evidence_count', 0)
        if evidence_count > 0:
            evidence_labels = {
                'workflow_type': context.workflow_type,
                'organism_id': self.config.organism_id
            }
            self._workflow_metrics['evidence_count'].labels(**evidence_labels).observe(evidence_count)
    
    def record_agent_started(self, agent_context: Dict[str, Any]) -> None:
        """Record agent call start."""
        
        if not self.is_available or not self._agent_metrics:
            return
        
        labels = {
            'agent_name': agent_context['agent_name'],
            'organism_id': self.config.organism_id
        }
        
        self._agent_metrics['active'].labels(**labels).inc()
    
    def record_agent_completed(self, agent_context: Dict[str, Any]) -> None:
        """Record agent call completion."""
        
        if not self.is_available or not self._agent_metrics:
            return
        
        agent_name = agent_context['agent_name']
        tool_name = agent_context['tool_name']
        status = agent_context.get('status', 'unknown')
        duration = agent_context.get('duration', 0)
        
        # Record call
        call_labels = {
            'agent_name': agent_name,
            'tool_name': tool_name,
            'status': status,
            'organism_id': self.config.organism_id
        }
        self._agent_metrics['calls'].labels(**call_labels).inc()
        
        # Record duration
        if duration > 0:
            duration_labels = {
                'agent_name': agent_name,
                'tool_name': tool_name,
                'organism_id': self.config.organism_id
            }
            self._agent_metrics['duration'].labels(**duration_labels).observe(duration)
        
        # Decrement active
        active_labels = {
            'agent_name': agent_name,
            'organism_id': self.config.organism_id
        }
        self._agent_metrics['active'].labels(**active_labels).dec()
    
    async def track_llm_usage(
        self, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int,
        latency_seconds: float, 
        cost_usd: Optional[float] = None
    ) -> None:
        """Track LLM usage with comprehensive metrics."""
        
        if not self.is_available or not self._llm_metrics:
            return
        
        base_labels = {
            'model_name': model_name,
            'organism_id': self.config.organism_id
        }
        
        # Record tokens
        self._llm_metrics['tokens'].labels(
            token_type='input', **base_labels
        ).inc(input_tokens)
        
        self._llm_metrics['tokens'].labels(
            token_type='output', **base_labels
        ).inc(output_tokens)
        
        # Record request
        self._llm_metrics['requests'].labels(
            status='success', **base_labels
        ).inc()
        
        # Record latency
        self._llm_metrics['latency'].labels(**base_labels).observe(latency_seconds)
        
        # Calculate and record throughput
        total_tokens = input_tokens + output_tokens
        if latency_seconds > 0:
            throughput = total_tokens / latency_seconds
            self._llm_metrics['throughput'].labels(**base_labels).set(throughput)
        
        # Record context length (input tokens as proxy)
        self._llm_metrics['context_length'].labels(**base_labels).observe(input_tokens)
        
        # Record cost if available
        if cost_usd is not None and self._cost_metrics:
            self._cost_metrics['total_usd'].labels(
                cost_type='llm_usage',
                model_name=model_name,
                organism_id=self.config.organism_id
            ).inc(cost_usd)
            
            self._cost_metrics['cost_per_request'].labels(**base_labels).observe(cost_usd)
    
    def record_error_recovery(self, error_type: str, recovery_strategy: str, success: bool) -> None:
        """Record error recovery attempt."""
        
        if not self.is_available or not self._error_metrics:
            return
        
        labels = {
            'error_type': error_type,
            'recovery_strategy': recovery_strategy,
            'organism_id': self.config.organism_id
        }
        
        if success:
            self._error_metrics['recovery_success'].labels(**labels).inc()
        else:
            self._error_metrics['recovery_failure'].labels(**labels).inc()
    
    def update_system_health(self, health_score: float) -> None:
        """Update system health score."""
        
        if not self.is_available or not self._system_metrics:
            return
        
        self._system_metrics['health_score'].labels(
            organism_id=self.config.organism_id
        ).set(health_score)
    
    def update_circuit_breaker_state(self, component: str, state: str) -> None:
        """Update circuit breaker state."""
        
        if not self.is_available or not self._error_metrics:
            return
        
        self._error_metrics['circuit_breaker_state'].labels(
            component=component,
            organism_id=self.config.organism_id
        ).state(state)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown Prometheus metrics."""
        
        if self.http_server:
            self.http_server.shutdown()
        
        print("✅ Prometheus metrics shutdown complete")
