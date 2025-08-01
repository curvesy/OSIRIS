"""
🏭 Schema Builders & Factories - Automated Construction

Production-grade builders and factories for creating schema instances with:
- Automatic signature generation
- Trace context propagation
- Validation and error handling
- Type-safe construction
- Default value management

Automates all the hard parts of schema creation.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from opentelemetry import trace
from opentelemetry.propagate import inject

from ..schemas.state import (
    AgentState, DossierEntry, ActionRecord, DecisionPoint,
    EvidenceContent, ActionIntent, DecisionOption, DecisionCriterion,
    EvidenceType, ActionType, ActionCategory, TaskStatus,
    SignatureAlgorithm, utc_now, get_crypto_provider, get_action_category
)

tracer = trace.get_tracer(__name__)


class TraceContextManager:
    """Manages OpenTelemetry trace context propagation."""
    
    @staticmethod
    def get_current_trace_context() -> Dict[str, str]:
        """Get current W3C trace context."""
        headers = {}
        inject(headers)
        return headers
    
    @staticmethod
    def extract_traceparent(headers: Optional[Dict[str, str]] = None) -> str:
        """Extract traceparent from headers or current context."""
        if headers and 'traceparent' in headers:
            return headers['traceparent']
        
        # Get from current span
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            flags = '01' if span_context.trace_flags.sampled else '00'
            return f"00-{trace_id}-{span_id}-{flags}"
        
        # Generate new trace context
        trace_id = format(uuid.uuid4().int >> 64, '032x')
        span_id = format(uuid.uuid4().int >> 96, '016x')
        return f"00-{trace_id}-{span_id}-01"


class DossierEntryBuilder:
    """Builder for creating DossierEntry instances with automatic signing."""
    
    def __init__(self, workflow_id: str, task_id: str, correlation_id: str):
        self.workflow_id = workflow_id
        self.task_id = task_id
        self.correlation_id = correlation_id
        self.reset()
    
    def reset(self) -> 'DossierEntryBuilder':
        """Reset builder to initial state."""
        self._data = {
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'correlation_id': self.correlation_id,
            'entry_id': f"evidence_{uuid.uuid4().hex}",
            'schema_version': "2.0",
            'traceparent': TraceContextManager.extract_traceparent(),
            'collection_timestamp': utc_now(),
            'signature_algorithm': SignatureAlgorithm.HMAC_SHA256,
            'tags': [],
            'metadata': {},
            'related_entries': [],
            'contradicts': [],
            'supports': [],
            'derived_from': []
        }
        return self
    
    def with_evidence_type(self, evidence_type: EvidenceType) -> 'DossierEntryBuilder':
        """Set evidence type."""
        self._data['evidence_type'] = evidence_type
        return self
    
    def with_content(self, content: EvidenceContent) -> 'DossierEntryBuilder':
        """Set evidence content."""
        self._data['content'] = content
        return self
    
    def with_summary(self, summary: str) -> 'DossierEntryBuilder':
        """Set evidence summary."""
        self._data['summary'] = summary
        return self
    
    def with_source(self, source: str, reliability: float = 0.8) -> 'DossierEntryBuilder':
        """Set evidence source and reliability."""
        self._data['source'] = source
        self._data['source_reliability'] = reliability
        return self
    
    def with_collection_method(self, method: str) -> 'DossierEntryBuilder':
        """Set collection method."""
        self._data['collection_method'] = method
        return self
    
    def with_confidence(self, confidence: float) -> 'DossierEntryBuilder':
        """Set confidence score."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self._data['confidence'] = confidence
        return self
    
    def with_freshness(self, freshness: float) -> 'DossierEntryBuilder':
        """Set freshness score."""
        if not 0.0 <= freshness <= 1.0:
            raise ValueError("Freshness must be between 0.0 and 1.0")
        self._data['freshness'] = freshness
        return self
    
    def with_agent(self, agent_id: str, public_key: str) -> 'DossierEntryBuilder':
        """Set signing agent information."""
        self._data['signing_agent_id'] = agent_id
        self._data['agent_public_key'] = public_key
        return self
    
    def with_signature_algorithm(self, algorithm: SignatureAlgorithm) -> 'DossierEntryBuilder':
        """Set signature algorithm."""
        self._data['signature_algorithm'] = algorithm
        return self
    
    def with_tags(self, tags: List[str]) -> 'DossierEntryBuilder':
        """Set tags."""
        self._data['tags'] = tags
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'DossierEntryBuilder':
        """Set metadata."""
        self._data['metadata'] = metadata
        return self
    
    def with_trace_context(self, traceparent: str, tracestate: Optional[str] = None) -> 'DossierEntryBuilder':
        """Set trace context."""
        self._data['traceparent'] = traceparent
        if tracestate:
            self._data['tracestate'] = tracestate
        return self
    
    def build(self, private_key: str) -> DossierEntry:
        """Build and sign the DossierEntry."""
        # Validate required fields
        required_fields = [
            'evidence_type', 'content', 'summary', 'source', 'collection_method',
            'confidence', 'freshness', 'signing_agent_id', 'agent_public_key'
        ]
        
        for field in required_fields:
            if field not in self._data:
                raise ValueError(f"Required field '{field}' not set")
        
        # Create instance without signature first
        self._data['content_signature'] = 'placeholder'
        entry = DossierEntry(**self._data)
        
        # Generate signature
        signature = entry.sign_content(private_key)
        self._data['content_signature'] = signature
        
        # Create final instance
        return DossierEntry(**self._data)


class ActionRecordBuilder:
    """Builder for creating ActionRecord instances with automatic signing."""
    
    def __init__(self, workflow_id: str, task_id: str, correlation_id: str):
        self.workflow_id = workflow_id
        self.task_id = task_id
        self.correlation_id = correlation_id
        self.reset()
    
    def reset(self) -> 'ActionRecordBuilder':
        """Reset builder to initial state."""
        self._data = {
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'correlation_id': self.correlation_id,
            'action_id': f"action_{uuid.uuid4().hex}",
            'schema_version': "2.0",
            'traceparent': TraceContextManager.extract_traceparent(),
            'execution_timestamp': utc_now(),
            'signature_algorithm': SignatureAlgorithm.HMAC_SHA256,
            'tags': [],
            'metadata': {},
            'supporting_evidence': [],
            'side_effects': [],
            'affected_systems': [],
            'rollback_available': False
        }
        return self
    
    def with_action_type(self, action_type: ActionType) -> 'ActionRecordBuilder':
        """Set action type and automatically determine category."""
        self._data['action_type'] = action_type
        self._data['action_category'] = get_action_category(action_type)
        return self
    
    def with_action_name(self, name: str) -> 'ActionRecordBuilder':
        """Set action name."""
        self._data['action_name'] = name
        return self
    
    def with_description(self, description: str) -> 'ActionRecordBuilder':
        """Set action description."""
        self._data['description'] = description
        return self
    
    def with_structured_intent(self, intent: ActionIntent) -> 'ActionRecordBuilder':
        """Set structured action intent."""
        self._data['structured_intent'] = intent
        return self
    
    def with_result(self, result: str, result_data: Optional[Dict[str, Any]] = None) -> 'ActionRecordBuilder':
        """Set action result."""
        self._data['result'] = result
        if result_data:
            self._data['result_data'] = result_data
        return self
    
    def with_confidence(self, confidence: float) -> 'ActionRecordBuilder':
        """Set confidence score."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self._data['confidence'] = confidence
        return self
    
    def with_agent(self, agent_id: str, public_key: str) -> 'ActionRecordBuilder':
        """Set executing agent information."""
        self._data['executing_agent_id'] = agent_id
        self._data['agent_public_key'] = public_key
        return self
    
    def with_duration(self, duration_ms: float) -> 'ActionRecordBuilder':
        """Set execution duration."""
        self._data['duration_ms'] = duration_ms
        return self
    
    def with_decision_rationale(self, rationale: str) -> 'ActionRecordBuilder':
        """Set decision rationale."""
        self._data['decision_rationale'] = rationale
        return self
    
    def build(self, private_key: str) -> ActionRecord:
        """Build and sign the ActionRecord."""
        # Validate required fields
        required_fields = [
            'action_type', 'action_name', 'description', 'structured_intent',
            'result', 'confidence', 'executing_agent_id', 'agent_public_key',
            'decision_rationale'
        ]
        
        for field in required_fields:
            if field not in self._data:
                raise ValueError(f"Required field '{field}' not set")
        
        # Create instance without signature first
        self._data['action_signature'] = 'placeholder'
        action = ActionRecord(**self._data)
        
        # Generate signature
        signature = action.sign_action(private_key)
        self._data['action_signature'] = signature
        
        # Create final instance
        return ActionRecord(**self._data)


class AgentStateBuilder:
    """Builder for creating AgentState instances with automatic signing."""
    
    def __init__(self, task_id: str, workflow_id: str, correlation_id: str):
        self.task_id = task_id
        self.workflow_id = workflow_id
        self.correlation_id = correlation_id
        self.reset()
    
    def reset(self) -> 'AgentStateBuilder':
        """Reset builder to initial state."""
        current_time = utc_now()
        self._data = {
            'task_id': self.task_id,
            'workflow_id': self.workflow_id,
            'correlation_id': self.correlation_id,
            'state_version': 1,
            'schema_version': "2.0",
            'traceparent': TraceContextManager.extract_traceparent(),
            'created_at': current_time,
            'updated_at': current_time,
            'signature_timestamp': current_time,
            'signature_algorithm': SignatureAlgorithm.HMAC_SHA256,
            'status': TaskStatus.PENDING,
            'priority': "normal",
            'urgency': "medium",
            'overall_confidence': 0.0,
            'confidence_calculation_method': "weighted_average",
            'context_dossier': [],
            'decision_points': [],
            'action_log': [],
            'stakeholder_notifications': [],
            'communication_log': [],
            'next_agents': [],
            'errors': [],
            'error_count': 0,
            'recovery_attempts': 0,
            'tags': [],
            'metadata': {},
            'success_metrics': {},
            'workflow_phase': "investigation"
        }
        return self
    
    def with_task_type(self, task_type: str) -> 'AgentStateBuilder':
        """Set task type."""
        self._data['task_type'] = task_type
        return self
    
    def with_initial_event(self, event: Dict[str, Any]) -> 'AgentStateBuilder':
        """Set initial event."""
        self._data['initial_event'] = event
        return self
    
    def with_trigger_source(self, source: str) -> 'AgentStateBuilder':
        """Set trigger source."""
        self._data['trigger_source'] = source
        return self
    
    def with_agent(self, agent_id: str, public_key: str) -> 'AgentStateBuilder':
        """Set modifier agent information."""
        self._data['last_modifier_agent_id'] = agent_id
        self._data['agent_public_key'] = public_key
        return self
    
    def with_status(self, status: TaskStatus) -> 'AgentStateBuilder':
        """Set task status."""
        self._data['status'] = status
        return self
    
    def with_priority(self, priority: str, urgency: str = "medium") -> 'AgentStateBuilder':
        """Set priority and urgency."""
        self._data['priority'] = priority
        self._data['urgency'] = urgency
        return self
    
    def build(self, private_key: str) -> AgentState:
        """Build and sign the AgentState."""
        # Validate required fields
        required_fields = [
            'task_type', 'initial_event', 'trigger_source',
            'last_modifier_agent_id', 'agent_public_key'
        ]
        
        for field in required_fields:
            if field not in self._data:
                raise ValueError(f"Required field '{field}' not set")
        
        # Create instance without signature first
        self._data['state_signature'] = 'placeholder'
        state = AgentState(**self._data)
        
        # Generate signature
        signature = state.sign_state(private_key)
        self._data['state_signature'] = signature
        
        # Create final instance
        return AgentState(**self._data)


# Factory functions for common use cases
def create_log_evidence(
    workflow_id: str,
    task_id: str,
    correlation_id: str,
    log_level: str,
    log_text: str,
    logger_name: str,
    agent_id: str,
    public_key: str,
    private_key: str,
    confidence: float = 0.8,
    source: str = "system_logs"
) -> DossierEntry:
    """Factory function for creating log evidence."""
    from ..schemas.state import LogEvidence
    
    log_content = LogEvidence(
        log_level=log_level,
        log_text=log_text,
        logger_name=logger_name,
        log_timestamp=utc_now().isoformat(),
        structured_data={}
    )
    
    return (DossierEntryBuilder(workflow_id, task_id, correlation_id)
            .with_evidence_type(EvidenceType.LOG_ENTRY)
            .with_content(log_content)
            .with_summary(f"{log_level}: {log_text[:100]}...")
            .with_source(source)
            .with_collection_method("automated_log_collection")
            .with_confidence(confidence)
            .with_freshness(1.0)
            .with_agent(agent_id, public_key)
            .build(private_key))


def create_metric_evidence(
    workflow_id: str,
    task_id: str,
    correlation_id: str,
    metric_name: str,
    metric_value: float,
    metric_unit: str,
    agent_id: str,
    public_key: str,
    private_key: str,
    confidence: float = 0.9,
    labels: Optional[Dict[str, str]] = None
) -> DossierEntry:
    """Factory function for creating metric evidence."""
    from ..schemas.state import MetricEvidence
    
    metric_content = MetricEvidence(
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        metric_type="gauge",
        labels=labels or {},
        measurement_timestamp=utc_now().isoformat()
    )
    
    return (DossierEntryBuilder(workflow_id, task_id, correlation_id)
            .with_evidence_type(EvidenceType.METRIC)
            .with_content(metric_content)
            .with_summary(f"Metric {metric_name}: {metric_value} {metric_unit}")
            .with_source("metrics_system")
            .with_collection_method("automated_metric_collection")
            .with_confidence(confidence)
            .with_freshness(1.0)
            .with_agent(agent_id, public_key)
            .build(private_key))
