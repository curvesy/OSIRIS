"""
Request Orchestrator

Orchestrates the request processing workflow.
"""

import time
from typing import Optional, List

import torch
import structlog

from .contracts import (
    CouncilRequest,
    CouncilResponse,
    ContextSnapshot,
    NeuralFeatures,
    DecisionEvidence,
    VoteDecision,
    VoteConfidence
)
from .interfaces import (
    INeuralEngine,
    IContextProvider,
    IFeatureExtractor,
    IDecisionMaker,
    IEvidenceCollector,
    IReasoningEngine,
    IMemoryManager
)
from aura_intelligence.observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("request_orchestrator")


class RequestOrchestrator:
    """
    Orchestrates the request processing workflow.
    
    Coordinates between different components to process a council request
    through the complete pipeline.
    """
    
    def __init__(
        self,
        neural_engine: INeuralEngine,
        context_provider: IContextProvider,
        feature_extractor: IFeatureExtractor,
        decision_maker: IDecisionMaker,
        evidence_collector: IEvidenceCollector,
        reasoning_engine: IReasoningEngine,
        memory_manager: Optional[IMemoryManager] = None
    ):
        """Initialize with required components."""
        self.neural_engine = neural_engine
        self.context_provider = context_provider
        self.feature_extractor = feature_extractor
        self.decision_maker = decision_maker
        self.evidence_collector = evidence_collector
        self.reasoning_engine = reasoning_engine
        self.memory_manager = memory_manager
    
    async def process(
        self,
        request: CouncilRequest,
        agent_id: str
    ) -> CouncilResponse:
        """
        Process a council request through the complete pipeline.
        
        Steps:
        1. Gather context
        2. Extract features
        3. Run neural inference
        4. Make decision
        5. Collect evidence
        6. Generate reasoning
        7. Store in memory (if available)
        8. Create response
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("orchestrate_request") as span:
            span.set_attribute("request.id", str(request.request_id))
            span.set_attribute("request.type", request.request_type)
            
            # Step 1: Gather context
            context = await self._gather_context(request)
            
            # Step 2: Extract features
            features = await self._extract_features(request, context)
            
            # Step 3: Run neural inference
            neural_output = await self._run_neural_inference(features)
            
            # Step 4: Make decision
            decision, confidence = await self._make_decision(
                neural_output,
                features,
                context
            )
            
            # Step 5: Collect evidence
            evidence = await self._collect_evidence(
                request,
                decision,
                confidence,
                context
            )
            
            # Step 6: Generate reasoning
            reasoning = await self._generate_reasoning(
                request,
                decision,
                confidence,
                evidence,
                context
            )
            
            # Step 7: Create response
            response = CouncilResponse(
                request_id=request.request_id,
                agent_id=agent_id,
                decision=decision,
                confidence=float(confidence),
                reasoning=reasoning,
                evidence=evidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "context_scope": request.context_scope.value,
                    "neural_output_shape": str(neural_output.shape),
                    "feature_count": len(features.feature_names)
                }
            )
            
            # Step 8: Store in memory if available
            if self.memory_manager:
                await self._store_in_memory(request, response)
            
            span.set_attribute("decision", decision.value)
            span.set_attribute("confidence", float(confidence))
            span.set_attribute("processing_time_ms", response.processing_time_ms)
            
            return response
    
    async def _gather_context(self, request: CouncilRequest) -> ContextSnapshot:
        """Gather context for the request."""
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("context_scope", request.context_scope.value)
            
            # Gather context based on scope
            context = await self.context_provider.gather_context(
                request,
                scope=request.context_scope.value
            )
            
            # If memory manager available, enrich with similar experiences
            if self.memory_manager:
                similar = await self.memory_manager.recall_similar(request, limit=5)
                if similar:
                    context = context.merge_with(
                        ContextSnapshot(recent_decisions=similar)
                    )
            
            span.set_attribute("patterns_count", len(context.historical_patterns))
            span.set_attribute("decisions_count", len(context.recent_decisions))
            
            return context
    
    async def _extract_features(
        self,
        request: CouncilRequest,
        context: ContextSnapshot
    ) -> NeuralFeatures:
        """Extract features from request and context."""
        with tracer.start_as_current_span("extract_features") as span:
            features = await self.feature_extractor.extract_features(
                request,
                context
            )
            
            span.set_attribute("feature_count", len(features.feature_names))
            span.set_attribute("feature_names", ",".join(features.feature_names[:5]))
            
            return features
    
    async def _run_neural_inference(self, features: NeuralFeatures) -> torch.Tensor:
        """Run neural network inference."""
        with tracer.start_as_current_span("neural_inference") as span:
            # Convert features to tensor
            feature_tensor = torch.from_numpy(features.normalized_features).float()
            
            # Run inference
            output = await self.neural_engine.forward(feature_tensor)
            
            span.set_attribute("output_shape", str(output.shape))
            span.set_attribute("output_device", str(output.device))
            
            return output
    
    async def _make_decision(
        self,
        neural_output: torch.Tensor,
        features: NeuralFeatures,
        context: ContextSnapshot
    ) -> tuple[VoteDecision, VoteConfidence]:
        """Make decision based on neural output."""
        with tracer.start_as_current_span("make_decision") as span:
            decision, confidence = await self.decision_maker.make_decision(
                neural_output,
                features,
                context
            )
            
            span.set_attribute("decision", decision.value)
            span.set_attribute("confidence", float(confidence))
            
            return decision, confidence
    
    async def _collect_evidence(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        context: ContextSnapshot
    ) -> List[DecisionEvidence]:
        """Collect evidence supporting the decision."""
        with tracer.start_as_current_span("collect_evidence") as span:
            evidence = await self.evidence_collector.collect_evidence(
                request,
                decision,
                confidence,
                context
            )
            
            # Validate evidence
            is_valid = await self.evidence_collector.validate_evidence(evidence)
            
            span.set_attribute("evidence_count", len(evidence))
            span.set_attribute("evidence_valid", is_valid)
            
            return evidence
    
    async def _generate_reasoning(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        evidence: List[DecisionEvidence],
        context: ContextSnapshot
    ) -> str:
        """Generate human-readable reasoning."""
        with tracer.start_as_current_span("generate_reasoning") as span:
            reasoning = await self.reasoning_engine.generate_reasoning(
                request,
                decision,
                confidence,
                evidence,
                context
            )
            
            span.set_attribute("reasoning_length", len(reasoning))
            
            return reasoning
    
    async def _store_in_memory(
        self,
        request: CouncilRequest,
        response: CouncilResponse
    ):
        """Store experience in memory for future learning."""
        try:
            await self.memory_manager.store_experience(
                request,
                response,
                outcome=None  # Will be updated later with actual outcome
            )
        except Exception as e:
            logger.error(
                "Failed to store experience in memory",
                request_id=str(request.request_id),
                error=str(e)
            )