"""
LNN Council Agent

Main agent implementation that orchestrates all modular components.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import structlog

from .contracts import (
    CouncilRequest,
    CouncilResponse,
    VoteDecision,
    VoteConfidence,
    AgentMetrics,
    AgentCapability
)
from .interfaces import (
    ICouncilAgent,
    INeuralEngine,
    IContextProvider,
    IFeatureExtractor,
    IDecisionMaker,
    IEvidenceCollector,
    IReasoningEngine,
    IStorageAdapter,
    IEventPublisher,
    IMemoryManager,
    IResourceManager
)
from .orchestrator import RequestOrchestrator
from aura_intelligence.observability import create_tracer, create_meter

logger = structlog.get_logger()
tracer = create_tracer("lnn_council_agent")
meter = create_meter("lnn_council_agent")

# Metrics
request_counter = meter.create_counter(
    name="council.requests",
    description="Number of council requests processed"
)

decision_histogram = meter.create_histogram(
    name="council.decision_time",
    description="Time to make decisions in milliseconds"
)


class LNNCouncilAgent(ICouncilAgent):
    """
    Production LNN Council Agent with clean separation of concerns.
    
    This agent orchestrates various components to process council requests
    using liquid neural networks for decision making.
    """
    
    def __init__(
        self,
        agent_id: str,
        capabilities: List[AgentCapability],
        neural_engine: INeuralEngine,
        context_provider: IContextProvider,
        feature_extractor: IFeatureExtractor,
        decision_maker: IDecisionMaker,
        evidence_collector: IEvidenceCollector,
        reasoning_engine: IReasoningEngine,
        storage_adapter: Optional[IStorageAdapter] = None,
        event_publisher: Optional[IEventPublisher] = None,
        memory_manager: Optional[IMemoryManager] = None,
        resource_manager: Optional[IResourceManager] = None
    ):
        """
        Initialize with dependency injection of all components.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            neural_engine: Neural network engine
            context_provider: Context gathering component
            feature_extractor: Feature extraction component
            decision_maker: Decision making component
            evidence_collector: Evidence collection component
            reasoning_engine: Reasoning generation component
            storage_adapter: Optional storage component
            event_publisher: Optional event publishing component
            memory_manager: Optional memory management component
            resource_manager: Optional resource management component
        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        
        # Core components (required)
        self.neural_engine = neural_engine
        self.context_provider = context_provider
        self.feature_extractor = feature_extractor
        self.decision_maker = decision_maker
        self.evidence_collector = evidence_collector
        self.reasoning_engine = reasoning_engine
        
        # Optional components
        self.storage_adapter = storage_adapter
        self.event_publisher = event_publisher
        self.memory_manager = memory_manager
        self.resource_manager = resource_manager
        
        # Create orchestrator
        self.orchestrator = RequestOrchestrator(
            neural_engine=neural_engine,
            context_provider=context_provider,
            feature_extractor=feature_extractor,
            decision_maker=decision_maker,
            evidence_collector=evidence_collector,
            reasoning_engine=reasoning_engine,
            memory_manager=memory_manager
        )
        
        # Metrics
        self._metrics = AgentMetrics()
        self._start_time = datetime.now(timezone.utc)
        
        logger.info(
            "LNN Council Agent initialized",
            agent_id=agent_id,
            capabilities=[c.value for c in capabilities]
        )
    
    async def process_request(self, request: CouncilRequest) -> CouncilResponse:
        """
        Process a council request and return a response.
        
        This is the main entry point that orchestrates all components.
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("process_request") as span:
            span.set_attribute("agent.id", self.agent_id)
            span.set_attribute("request.id", str(request.request_id))
            span.set_attribute("request.type", request.request_type)
            
            try:
                # Check if we can handle this request
                if not self._can_handle_request(request):
                    return self._create_delegate_response(request, start_time)
                
                # Orchestrate the request processing
                response = await self.orchestrator.process(request, self.agent_id)
                
                # Store decision if storage is available
                if self.storage_adapter:
                    await self._store_decision(response)
                
                # Publish event if publisher is available
                if self.event_publisher:
                    await self._publish_decision(response)
                
                # Update metrics
                self._update_metrics(response, time.time() - start_time)
                
                # Record metrics
                request_counter.add(1, {"status": "success"})
                decision_histogram.record(
                    (time.time() - start_time) * 1000,
                    {"decision": response.decision}
                )
                
                return response
                
            except Exception as e:
                logger.error(
                    "Error processing request",
                    agent_id=self.agent_id,
                    request_id=str(request.request_id),
                    error=str(e)
                )
                
                # Record error
                request_counter.add(1, {"status": "error"})
                span.record_exception(e)
                
                # Create error response
                return self._create_error_response(request, str(e), start_time)
    
    def _can_handle_request(self, request: CouncilRequest) -> bool:
        """Check if agent can handle this request based on capabilities."""
        if not request.capabilities_required:
            return True
        
        # Check if we have all required capabilities
        agent_caps = set(self.capabilities)
        required_caps = set(request.capabilities_required)
        
        return required_caps.issubset(agent_caps)
    
    def _create_delegate_response(
        self,
        request: CouncilRequest,
        start_time: float
    ) -> CouncilResponse:
        """Create a delegation response when agent can't handle request."""
        return CouncilResponse(
            request_id=request.request_id,
            agent_id=self.agent_id,
            decision=VoteDecision.DELEGATE,
            confidence=0.0,
            reasoning=f"Agent lacks required capabilities: {request.capabilities_required}",
            evidence=[],
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _create_error_response(
        self,
        request: CouncilRequest,
        error: str,
        start_time: float
    ) -> CouncilResponse:
        """Create an error response."""
        return CouncilResponse(
            request_id=request.request_id,
            agent_id=self.agent_id,
            decision=VoteDecision.ABSTAIN,
            confidence=0.0,
            reasoning=f"Error processing request: {error}",
            evidence=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={"error": error}
        )
    
    async def _store_decision(self, response: CouncilResponse):
        """Store decision using storage adapter."""
        try:
            await self.storage_adapter.store_decision(
                response,
                metadata={
                    "agent_capabilities": [c.value for c in self.capabilities],
                    "agent_metrics": self._metrics.dict()
                }
            )
        except Exception as e:
            logger.error(
                "Failed to store decision",
                agent_id=self.agent_id,
                request_id=str(response.request_id),
                error=str(e)
            )
    
    async def _publish_decision(self, response: CouncilResponse):
        """Publish decision event."""
        try:
            await self.event_publisher.publish_decision(
                response,
                topic=f"council.decisions.{self.agent_id}"
            )
        except Exception as e:
            logger.error(
                "Failed to publish decision",
                agent_id=self.agent_id,
                request_id=str(response.request_id),
                error=str(e)
            )
    
    def _update_metrics(self, response: CouncilResponse, processing_time: float):
        """Update agent metrics."""
        new_total_decisions = self._metrics.total_decisions + 1
        
        # Calculate new approval rate
        if response.decision == VoteDecision.APPROVE:
            approval_count = self._metrics.approval_rate * self._metrics.total_decisions + 1
            new_approval_rate = approval_count / new_total_decisions
        else:
            approval_count = self._metrics.approval_rate * self._metrics.total_decisions
            new_approval_rate = approval_count / new_total_decisions
        
        # Calculate new average confidence
        total_confidence = self._metrics.average_confidence * self._metrics.total_decisions
        new_average_confidence = (total_confidence + response.confidence) / new_total_decisions
        
        # Calculate new average processing time
        total_time = self._metrics.average_processing_time_ms * self._metrics.total_decisions
        new_average_processing_time_ms = (total_time + processing_time * 1000) / new_total_decisions
        
        # Create new metrics object
        self._metrics = self._metrics.update(
            total_decisions=new_total_decisions,
            approval_rate=new_approval_rate,
            average_confidence=new_average_confidence,
            average_processing_time_ms=new_average_processing_time_ms
        )
    
    async def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [cap.value for cap in self.capabilities]
    
    async def get_metrics(self) -> AgentMetrics:
        """Get agent performance metrics."""
        return self._metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "agent_id": self.agent_id,
            "status": "healthy",
            "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            "capabilities": [cap.value for cap in self.capabilities],
            "components": {}
        }
        
        # Check neural engine
        try:
            neural_metrics = self.neural_engine.get_metrics()
            health["components"]["neural_engine"] = {
                "status": "healthy",
                "metrics": neural_metrics
            }
        except Exception as e:
            health["components"]["neural_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        # Check resource manager if available
        if self.resource_manager:
            try:
                resource_health = await self.resource_manager.health_check()
                health["components"]["resources"] = resource_health
            except Exception as e:
                health["components"]["resources"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        return health
    
    async def initialize(self):
        """Initialize the agent and all components."""
        logger.info("Initializing LNN Council Agent", agent_id=self.agent_id)
        
        # Initialize neural engine
        await self.neural_engine.initialize({})
        
        # Initialize resource manager if available
        if self.resource_manager:
            await self.resource_manager.initialize()
        
        logger.info("LNN Council Agent initialized successfully", agent_id=self.agent_id)
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up LNN Council Agent", agent_id=self.agent_id)
        
        # Cleanup resource manager if available
        if self.resource_manager:
            await self.resource_manager.cleanup()
        
        # Publish final metrics if publisher available
        if self.event_publisher:
            await self.event_publisher.publish_metrics(
                self._metrics,
                topic=f"council.metrics.{self.agent_id}"
            )
        
        logger.info("LNN Council Agent cleanup completed", agent_id=self.agent_id)