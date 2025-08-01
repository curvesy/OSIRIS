"""
📝 Structured Logging Manager - Latest 2025 Patterns
Professional structured logging with cryptographic signatures and trace correlation.
"""

import json
import time
import hashlib
import hmac
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
import sys

import structlog

try:
    from .config import ObservabilityConfig
    from .context_managers import ObservabilityContext
except ImportError:
    # Fallback for direct import
    from config import ObservabilityConfig
    from context_managers import ObservabilityContext


class StructuredLoggingManager:
    """
    Latest 2025 structured logging with cryptographic signatures.
    
    Features:
    - JSON structured logging with trace correlation
    - Cryptographic signatures for audit trails
    - AI-specific log fields and context
    - Performance-optimized logging
    - Multi-level filtering and routing
    - Integration with OpenTelemetry traces
    - Secure log tampering detection
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize structured logging manager.
        
        Args:
            config: Observability configuration
        """
        
        self.config = config
        self.logger: Optional[logging.Logger] = None
        self.is_available = True  # Always available with fallback
        
        # Cryptographic signing
        self._signing_key = self._generate_signing_key()
        self._log_sequence = 0
    
    async def initialize(self) -> None:
        """
        Initialize structured logging with latest 2025 patterns.
        """
        
        # Initialize structlog - no fallback, strict requirements
        self._initialize_structlog()

        # Test logging
        self.logger.info(
            "structured_logging_initialized",
            organism_id=self.config.organism_id,
            generation=self.config.organism_generation,
            crypto_signatures_enabled=self.config.log_enable_crypto_signatures,
            correlation_enabled=self.config.log_enable_correlation,
            log_level=self.config.log_level,
            neural_observability_version="2025.7.27"
        )

        print(f"✅ Structured logging initialized - Level: {self.config.log_level}")
    
    def _initialize_structlog(self) -> None:
        """Initialize with structlog for advanced features."""
        
        # Configure structlog processors
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            self._add_organism_context,
            self._add_trace_correlation,
        ]
        
        # Add cryptographic signature processor if enabled
        if self.config.log_enable_crypto_signatures:
            processors.append(self._add_cryptographic_signature)
        
        # Add JSON processor
        processors.append(structlog.processors.JSONRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.config.log_level.upper())
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Create logger
        self.logger = structlog.get_logger("collective_intelligence_neural_observability")
    

    
    def _generate_signing_key(self) -> bytes:
        """Generate cryptographic signing key for log integrity."""
        
        # In production, this should come from secure key management
        key_material = f"{self.config.organism_id}:{self.config.service_version}:neural_observability"
        return hashlib.sha256(key_material.encode()).digest()
    
    def _add_organism_context(self, logger, method_name, event_dict):
        """Add organism context to all log entries."""
        
        event_dict.update({
            "organism_id": self.config.organism_id,
            "organism_generation": self.config.organism_generation,
            "deployment_environment": self.config.deployment_environment,
            "service_version": self.config.service_version,
            "neural_observability_version": "2025.7.27",
        })
        
        return event_dict
    
    def _add_trace_correlation(self, logger, method_name, event_dict):
        """Add trace correlation information."""
        
        if not self.config.log_enable_correlation:
            return event_dict
        
        # Try to get current trace context from OpenTelemetry
        try:
            from opentelemetry import trace
            
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                event_dict.update({
                    "trace_id": format(span_context.trace_id, '032x'),
                    "span_id": format(span_context.span_id, '016x'),
                    "trace_flags": span_context.trace_flags,
                })
        except Exception:
            # Graceful degradation if OpenTelemetry not available
            pass
        
        return event_dict
    
    def _add_cryptographic_signature(self, logger, method_name, event_dict):
        """Add cryptographic signature for log integrity."""
        
        if not self.config.log_enable_crypto_signatures:
            return event_dict
        
        try:
            # Increment sequence number
            self._log_sequence += 1
            
            # Create signature payload
            signature_payload = {
                "sequence": self._log_sequence,
                "timestamp": event_dict.get("timestamp"),
                "level": event_dict.get("level"),
                "event": event_dict.get("event", ""),
                "organism_id": self.config.organism_id,
            }
            
            # Create HMAC signature
            payload_json = json.dumps(signature_payload, sort_keys=True)
            signature = hmac.new(
                self._signing_key,
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Add signature fields
            event_dict.update({
                "log_sequence": self._log_sequence,
                "log_signature": signature,
                "signature_algorithm": "HMAC-SHA256",
            })
            
        except Exception as e:
            # Don't fail logging if signature fails
            event_dict["signature_error"] = str(e)
        
        return event_dict
    
    def log_workflow_started(self, context: ObservabilityContext, state: Dict[str, Any]) -> None:
        """Log workflow start with comprehensive context."""

        if not self.logger:
            return

        self.logger.info(
            "workflow_started",
            workflow_id=context.workflow_id,
            workflow_type=context.workflow_type,
            evidence_count=context.metadata.get("evidence_count", 0),
            has_errors=context.metadata.get("has_errors", False),
            recovery_attempts=context.metadata.get("recovery_attempts", 0),
            system_health_status=context.metadata.get("system_health_status", "unknown"),
            agents_involved=context.metadata.get("agents_involved", []),
            trace_id=context.trace_id,
            span_id=context.span_id,
            start_time=context.start_time,
            tags=context.tags,
        )
    
    def log_workflow_completed(self, context: ObservabilityContext, state: Dict[str, Any]) -> None:
        """Log workflow completion with results."""
        
        if not self.logger:
            return
        
        self.logger.info(
            "workflow_completed",
            workflow_id=context.workflow_id,
            workflow_type=context.workflow_type,
            status=context.status,
            duration=context.duration,
            error=context.error,
            final_evidence_count=len(state.get("evidence_log", [])),
            final_error_count=len(state.get("error_log", [])),
            recovery_attempts=state.get("error_recovery_attempts", 0),
            system_health_score=state.get("system_health", {}).get("health_score", 0.0),
            trace_id=context.trace_id,
            span_id=context.span_id,
            tags=context.tags,
        )
    
    def log_agent_started(self, agent_context: Dict[str, Any]) -> None:
        """Log agent call start."""
        
        if not self.logger:
            return
        
        self.logger.info(
            "agent_call_started",
            agent_name=agent_context['agent_name'],
            tool_name=agent_context['tool_name'],
            workflow_id=agent_context.get('workflow_context', {}).get('workflow_id', 'unknown'),
            input_count=len(agent_context.get('inputs', {})),
            trace_id=agent_context.get('trace_id'),
            span_id=agent_context.get('span_id'),
            start_time=agent_context['start_time'],
        )
    
    def log_agent_completed(self, agent_context: Dict[str, Any]) -> None:
        """Log agent call completion."""
        
        if not self.logger:
            return
        
        self.logger.info(
            "agent_call_completed",
            agent_name=agent_context['agent_name'],
            tool_name=agent_context['tool_name'],
            workflow_id=agent_context.get('workflow_context', {}).get('workflow_id', 'unknown'),
            status=agent_context.get('status', 'unknown'),
            duration=agent_context.get('duration', 0),
            error=agent_context.get('error'),
            output_count=len(agent_context.get('outputs', {})) if agent_context.get('outputs') else 0,
            trace_id=agent_context.get('trace_id'),
            span_id=agent_context.get('span_id'),
        )
    
    def log_llm_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        cost_usd: Optional[float] = None
    ) -> None:
        """Log LLM usage with cost and performance metrics."""

        if not self.logger:
            return

        self.logger.info(
            "llm_usage",
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=latency_seconds,
            throughput_tokens_per_second=(input_tokens + output_tokens) / latency_seconds if latency_seconds > 0 else 0,
            cost_usd=cost_usd,
            cost_per_token=cost_usd / (input_tokens + output_tokens) if cost_usd and (input_tokens + output_tokens) > 0 else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_error_recovery(self, error_type: str, recovery_strategy: str, success: bool) -> None:
        """Log error recovery attempt."""

        if not self.logger:
            return

        log_level = "info" if success else "warning"
        getattr(self.logger, log_level)(
            "error_recovery_attempt",
            error_type=error_type,
            recovery_strategy=recovery_strategy,
            success=success,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_system_health_update(self, health_score: float) -> None:
        """Log system health score update."""
        
        if not self.logger:
            return
        
        # Determine log level based on health score
        if health_score >= 0.8:
            log_level = "info"
        elif health_score >= 0.5:
            log_level = "warning"
        else:
            log_level = "error"
        
        getattr(self.logger, log_level)(
            "system_health_update",
            health_score=health_score,
            health_status="healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "critical",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_circuit_breaker_event(self, component: str, event: str, state: str) -> None:
        """Log circuit breaker state changes."""
        
        if not self.logger:
            return
        
        self.logger.warning(
            "circuit_breaker_event",
            component=component,
            event=event,
            new_state=state,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_anomaly_detected(self, anomaly_type: str, severity: str, details: Dict[str, Any]) -> None:
        """Log anomaly detection events."""
        
        if not self.logger:
            return
        
        log_level = "error" if severity == "high" else "warning" if severity == "medium" else "info"
        
        getattr(self.logger, log_level)(
            "anomaly_detected",
            anomaly_type=anomaly_type,
            severity=severity,
            details=details,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security-related events."""
        
        if not self.logger:
            return
        
        self.logger.warning(
            "security_event",
            event_type=event_type,
            details=details,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str, tags: Dict[str, str] = None) -> None:
        """Log performance metrics."""
        
        if not self.logger:
            return
        
        self.logger.info(
            "performance_metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def verify_log_integrity(self, log_entry: Dict[str, Any]) -> bool:
        """
        Verify the cryptographic signature of a log entry.
        
        Args:
            log_entry: Log entry with signature fields
            
        Returns:
            bool: True if signature is valid
        """
        
        if not self.config.log_enable_crypto_signatures:
            return True  # No verification needed
        
        try:
            # Extract signature fields
            sequence = log_entry.get("log_sequence")
            signature = log_entry.get("log_signature")
            
            if not sequence or not signature:
                return False
            
            # Recreate signature payload
            signature_payload = {
                "sequence": sequence,
                "timestamp": log_entry.get("timestamp"),
                "level": log_entry.get("level"),
                "event": log_entry.get("event", ""),
                "organism_id": self.config.organism_id,
            }
            
            # Verify signature
            payload_json = json.dumps(signature_payload, sort_keys=True)
            expected_signature = hmac.new(
                self._signing_key,
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown structured logging."""
        
        if self.logger:
            self.logger.info(
                "structured_logging_shutdown",
                total_log_sequence=self._log_sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        print("✅ Structured logging shutdown complete")
