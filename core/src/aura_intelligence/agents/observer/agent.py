"""
🔍 ObserverAgent - Production-Grade System Observer

The first agent in The Collective, responsible for:
- Continuous system observation and event processing
- Evidence creation with cryptographic signatures
- Workflow initiation using immutable state management
- Integration with TDA engines for anomaly detection

Built on our world-class modular schema architecture.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..schemas.state import AgentState
from ..schemas.evidence import DossierEntry
from ..schemas.action import ActionRecord
from ..schemas.decision import DecisionPoint
from ..schemas.enums import TaskStatus, EvidenceType, ActionType, RiskLevel
from ..schemas.base import utc_now
from ..schemas.crypto import get_crypto_provider, SignatureAlgorithm


class ObserverAgent:
    """
    Production-grade Observer Agent with async support, retry logic,
    and comprehensive observability integration.
    
    The first agent in The Collective that demonstrates our world-class
    modular architecture in action.
    """
    
    def __init__(
        self,
        agent_id: str,
        private_key: str,
        public_key: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.private_key = private_key
        self.public_key = public_key
        self.config = config or {}
        
        # Initialize crypto provider
        self.crypto = get_crypto_provider(SignatureAlgorithm.HMAC_SHA256)
        
        # Initialize OpenTelemetry tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Performance metrics
        self.metrics = {
            "events_processed": 0,
            "errors": 0,
            "avg_processing_time_ms": 0.0,
            "evidence_created": 0,
            "workflows_initiated": 0
        }
        
        print(f"🔍 ObserverAgent '{self.agent_id}' initialized with world-class schemas")
    
    async def process_event(
        self,
        raw_event: Dict[str, Any],
        retry_count: int = 3
    ) -> AgentState:
        """
        Process a single event with retry logic and comprehensive error handling.
        
        This is the main entry point that demonstrates our end-to-end architecture:
        Raw Event → Evidence Creation → State Initialization → Immutable Updates
        """
        start_time = utc_now()
        
        with self.tracer.start_as_current_span("observer_agent.process_event") as span:
            try:
                span.set_attribute("agent_id", self.agent_id)
                span.set_attribute("event_type", raw_event.get("type", "unknown"))
                
                # Generate globally unique identifiers
                workflow_id = f"wf_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
                task_id = f"task_{self.agent_id}_{int(start_time.timestamp())}"
                correlation_id = raw_event.get("correlation_id", str(uuid.uuid4()))
                
                span.set_attribute("workflow_id", workflow_id)
                span.set_attribute("task_id", task_id)
                span.set_attribute("correlation_id", correlation_id)
                
                # Step 1: Create structured, cryptographically signed evidence
                evidence = await self._create_evidence_from_event(
                    raw_event, workflow_id, task_id, correlation_id
                )
                
                # Step 2: Initialize workflow state with comprehensive metadata
                initial_state = self._initialize_workflow_state(
                    raw_event, workflow_id, task_id, correlation_id
                )
                
                # Step 3: Add evidence using pure functional update (immutable)
                updated_state = initial_state.with_evidence(
                    evidence, self.agent_id, self.private_key,
                    traceparent=span.get_span_context().trace_id
                )
                
                # Step 4: Make initial decision about the event
                decision = await self._make_initial_decision(updated_state)
                final_state = updated_state.with_decision(
                    decision, self.agent_id, self.private_key,
                    traceparent=span.get_span_context().trace_id
                )
                
                # Update metrics
                processing_time = (utc_now() - start_time).total_seconds() * 1000
                self._update_metrics(processing_time)
                
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("processing_time_ms", processing_time)
                span.set_attribute("state_version", final_state.state_version)
                
                print(f"✅ Event processed successfully: {workflow_id} (v{final_state.state_version})")
                return final_state
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                if retry_count > 0:
                    print(f"⚠️ Retrying event processing (attempts left: {retry_count})")
                    await asyncio.sleep(2 ** (3 - retry_count))  # Exponential backoff
                    return await self.process_event(raw_event, retry_count - 1)
                else:
                    self.metrics["errors"] += 1
                    raise
    
    async def _create_evidence_from_event(
        self,
        event: Dict[str, Any],
        workflow_id: str,
        task_id: str,
        correlation_id: str
    ) -> DossierEntry:
        """
        Create cryptographically signed evidence from raw event.
        
        This demonstrates our typed evidence system with Union types
        and comprehensive validation.
        """
        with self.tracer.start_as_current_span("create_evidence") as span:
            # Import evidence types (avoiding circular imports)
            from ..schemas.evidence import LogEvidence
            
            # Create typed evidence content
            log_content = LogEvidence(
                log_level=event.get("level", "info"),
                log_text=event.get("message", ""),
                logger_name=event.get("source", "unknown_logger"),
                log_timestamp=event.get("timestamp", utc_now().isoformat()),
                structured_data=event.get("fields", {})
            )
            
            # Create evidence entry with our world-class schemas
            evidence = DossierEntry(
                # Identity
                entry_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                task_id=task_id,
                correlation_id=correlation_id,
                
                # Evidence content (typed Union)
                evidence_type=EvidenceType.LOG_ENTRY,
                content=log_content,
                
                # Metadata
                summary=f"Initial trigger: {log_content.log_text[:120]}...",
                source=event.get("source", "ObserverAgent"),
                collection_method="stream_ingestion",
                collection_timestamp=utc_now(),
                
                # Quality metrics
                confidence=0.95,
                reliability=0.9,
                freshness=1.0,
                completeness=0.85,
                
                # Agent information
                collecting_agent_id=self.agent_id,
                agent_public_key=self.public_key,
                
                # Cryptographic signature (placeholder - will be signed)
                signature="placeholder",
                signature_algorithm=SignatureAlgorithm.HMAC_SHA256,
                signature_timestamp=utc_now()
            )
            
            # Sign the evidence
            evidence_bytes = evidence.get_canonical_representation().encode('utf-8')
            signature = self.crypto.sign(evidence_bytes, self.private_key)
            
            # Create final signed evidence
            signed_evidence = evidence.copy(update={'signature': signature})
            
            span.set_attribute("evidence_type", evidence.evidence_type.value)
            span.set_attribute("confidence", evidence.confidence)
            
            self.metrics["evidence_created"] += 1
            return signed_evidence
    
    def _initialize_workflow_state(
        self,
        event: Dict[str, Any],
        workflow_id: str,
        task_id: str,
        correlation_id: str
    ) -> AgentState:
        """
        Initialize comprehensive workflow state using our immutable state management.
        
        This demonstrates the heart of The Collective's memory and coordination.
        """
        with self.tracer.start_as_current_span("initialize_state") as span:
            # Determine task type intelligently
            task_type = self._determine_task_type(event)
            
            # Create initial state with comprehensive metadata
            initial_state = AgentState(
                # Identity & versioning
                task_id=task_id,
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                state_version=1,
                schema_version="2.0",
                
                # Cryptographic authentication
                state_signature="placeholder",  # Will be signed
                signature_algorithm=SignatureAlgorithm.HMAC_SHA256,
                last_modifier_agent_id=self.agent_id,
                agent_public_key=self.public_key,
                signature_timestamp=utc_now(),
                
                # Task information
                task_type=task_type,
                priority=event.get("priority", "normal"),
                status=TaskStatus.PENDING,
                urgency=event.get("urgency", "medium"),
                
                # Initial context
                initial_event=event,
                initial_context={"agent_version": "1.0", "schema_version": "2.0"},
                trigger_source=event.get("source", "unknown"),
                
                # Temporal information
                created_at=utc_now(),
                updated_at=utc_now(),
                
                # Metadata
                tags=self._extract_tags(event),
                metadata={
                    "created_by": self.agent_id,
                    "event_type": event.get("type", "unknown"),
                    "processing_version": "1.0"
                }
            )
            
            # Sign the initial state
            state_bytes = initial_state._get_canonical_state().encode('utf-8')
            signature = self.crypto.sign(state_bytes, self.private_key)
            
            # Return signed state
            signed_state = initial_state.copy(update={'state_signature': signature})
            
            span.set_attribute("task_type", task_type)
            span.set_attribute("initial_priority", signed_state.priority)
            
            self.metrics["workflows_initiated"] += 1
            return signed_state
    
    async def _make_initial_decision(self, state: AgentState) -> DecisionPoint:
        """
        Make initial decision about how to handle the event.
        
        This demonstrates our enhanced explainability and decision rationale.
        """
        with self.tracer.start_as_current_span("make_decision") as span:
            from ..schemas.decision import DecisionCriterion, DecisionOption
            
            # Define decision criteria
            criteria = [
                DecisionCriterion(
                    criterion_id="urgency",
                    name="Event Urgency",
                    description="How urgent is this event?",
                    weight=0.4,
                    measurement_method="categorical_mapping"
                ),
                DecisionCriterion(
                    criterion_id="confidence",
                    name="Evidence Confidence",
                    description="How confident are we in the evidence?",
                    weight=0.3,
                    measurement_method="confidence_score"
                ),
                DecisionCriterion(
                    criterion_id="impact",
                    name="Potential Impact",
                    description="What's the potential business impact?",
                    weight=0.3,
                    measurement_method="impact_assessment"
                )
            ]
            
            # Define decision options
            options = [
                DecisionOption(
                    option_id="escalate",
                    name="Escalate to Human",
                    description="Forward to human analyst for review",
                    estimated_effort_hours=0.5,
                    estimated_cost=50.0,
                    risk_level=RiskLevel.LOW,
                    scores={"urgency": 0.8, "confidence": 0.9, "impact": 0.7}
                ),
                DecisionOption(
                    option_id="auto_investigate",
                    name="Automated Investigation",
                    description="Continue with automated analysis",
                    estimated_effort_hours=0.1,
                    estimated_cost=5.0,
                    risk_level=RiskLevel.MEDIUM,
                    scores={"urgency": 0.6, "confidence": 0.8, "impact": 0.5}
                )
            ]
            
            # Make decision based on state
            chosen_option = "auto_investigate" if state.overall_confidence > 0.8 else "escalate"
            
            # Create decision point
            decision = DecisionPoint(
                # Identity
                decision_id=str(uuid.uuid4()),
                workflow_id=state.workflow_id,
                task_id=state.task_id,
                correlation_id=state.correlation_id,
                
                # Decision details
                decision_type="workflow_routing",
                decision_method="rule_based",
                criteria=criteria,
                options=options,
                chosen_option_id=chosen_option,
                
                # Rationale
                rationale=f"Chose {chosen_option} based on confidence {state.overall_confidence}",
                confidence_in_decision=0.85,
                
                # Agent information
                deciding_agent_id=self.agent_id,
                agent_public_key=self.public_key,
                
                # Cryptographic signature
                signature="placeholder",
                signature_algorithm=SignatureAlgorithm.HMAC_SHA256,
                signature_timestamp=utc_now(),
                decision_timestamp=utc_now()
            )
            
            # Sign the decision
            decision_bytes = decision.get_canonical_representation().encode('utf-8')
            signature = self.crypto.sign(decision_bytes, self.private_key)
            
            signed_decision = decision.copy(update={'signature': signature})
            
            span.set_attribute("chosen_option", chosen_option)
            span.set_attribute("decision_confidence", 0.85)
            
            return signed_decision
    
    def _determine_task_type(self, event: Dict[str, Any]) -> str:
        """Intelligently determine task type from event characteristics."""
        event_str = str(event).lower()
        
        # Security events
        if any(indicator in event_str for indicator in ["threat", "malware", "unauthorized", "breach"]):
            return "security_investigation"
        
        # Performance events
        if "metrics" in event and any(m in event.get("metrics", {}) for m in ["cpu", "memory", "latency"]):
            return "performance_analysis"
        
        # Error events
        if event.get("level") in ["error", "critical", "fatal"]:
            return "error_investigation"
        
        # Default
        return "general_observation"
    
    def _extract_tags(self, event: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from event for categorization."""
        tags = []
        
        if "level" in event:
            tags.append(f"level:{event['level']}")
        
        if "source" in event:
            tags.append(f"source:{event['source']}")
        
        if "environment" in event:
            tags.append(f"env:{event['environment']}")
        
        return tags
    
    def _update_metrics(self, processing_time_ms: float):
        """Update internal performance metrics."""
        self.metrics["events_processed"] += 1
        
        # Calculate moving average
        current_avg = self.metrics["avg_processing_time_ms"]
        count = self.metrics["events_processed"]
        self.metrics["avg_processing_time_ms"] = (
            (current_avg * (count - 1) + processing_time_ms) / count
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Return comprehensive health and performance metrics."""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": utc_now().isoformat(),
            "schema_version": "2.0"
        }
    
    def __str__(self) -> str:
        return f"ObserverAgent[{self.agent_id}] - Events: {self.metrics['events_processed']}, Errors: {self.metrics['errors']}"
