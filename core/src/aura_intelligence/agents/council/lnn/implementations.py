"""
Default Implementations

Default implementations of all LNN Council Agent interfaces.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import numpy as np
import torch
import structlog

from .contracts import (
    CouncilRequest,
    CouncilResponse,
    VoteDecision,
    VoteConfidence,
    DecisionEvidence,
    ContextScope,
    ContextSnapshot,
    NeuralFeatures
)
from .interfaces import (
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

logger = structlog.get_logger()


class DefaultContextProvider(IContextProvider):
    """Default context provider implementation."""
    
    def __init__(self, neo4j_adapter=None):
        self.neo4j_adapter = neo4j_adapter
    
    async def gather_context(
        self,
        request: CouncilRequest,
        scope: str = "local"
    ) -> ContextSnapshot:
        """Gather context for decision making."""
        # Get historical patterns if available
        historical_patterns = []
        recent_decisions = []
        
        if self.neo4j_adapter and scope in ["historical", "global"]:
            try:
                historical_context = await self._get_historical_context(request)
                historical_patterns = historical_context.get("patterns", [])
                recent_decisions = historical_context.get("decisions", [])
            except Exception as e:
                logger.warning(f"Failed to get historical context: {e}")
        
        # Create context snapshot
        context = ContextSnapshot(
            historical_patterns=historical_patterns,
            recent_decisions=recent_decisions,
            entity_relationships={},
            temporal_context={
                "request_id": str(request.request_id),
                "request_type": request.request_type,
                "priority": request.priority,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_data": request.payload,
                "request_context": request.context
            }
        )
        
        return context
    
    async def query_historical(
        self,
        entity_id: str,
        time_window: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query historical data."""
        if not self.neo4j_adapter:
            return []
        
        try:
            query = """
            MATCH (d:Decision {entity_id: $entity_id})
            WHERE ($time_window IS NULL OR d.created_at >= $time_window)
            RETURN d
            ORDER BY d.created_at DESC
            LIMIT $limit
            """
            
            result = await self.neo4j_adapter.query(query, {
                "entity_id": entity_id,
                "time_window": time_window.isoformat() if time_window else None,
                "limit": limit
            })
            
            return [dict(record['d']) for record in result]
        except Exception as e:
            logger.error(f"Failed to query historical data: {e}")
            return []
    
    async def get_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get entity relationships."""
        if not self.neo4j_adapter:
            return []
        
        try:
            if relationship_types:
                type_filter = "AND type(r) IN $relationship_types"
            else:
                type_filter = ""
            
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE a.id = $entity_id {type_filter}
            RETURN type(r) as relationship_type, b as related_entity
            LIMIT 50
            """
            
            result = await self.neo4j_adapter.query(query, {
                "entity_id": entity_id,
                "relationship_types": relationship_types or []
            })
            
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []
    
    async def store_context(
        self,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store context snapshot."""
        if not self.neo4j_adapter:
            return False
        
        try:
            query = """
            CREATE (c:ContextSnapshot {
                id: $id,
                data: $data,
                created_at: $created_at,
                ttl: $ttl
            })
            RETURN c
            """
            
            result = await self.neo4j_adapter.query(query, {
                "id": str(uuid.uuid4()),
                "data": json.dumps(context),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ttl": ttl_seconds
            })
            
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            return False
    
    async def _get_historical_context(self, request: CouncilRequest) -> Dict[str, Any]:
        """Get historical context from Neo4j."""
        # Simple query for similar past decisions
        query = """
        MATCH (d:Decision)
        WHERE d.request_type = $request_type
        RETURN d
        ORDER BY d.created_at DESC
        LIMIT 5
        """
        
        try:
            results = await self.neo4j_adapter.query(query, {
                "request_type": request.request_type
            })
            
            return {
                "patterns": [{"pattern": "similar_request", "data": dict(record["d"])} for record in results],
                "decisions": [dict(record["d"]) for record in results]
            }
        except Exception as e:
            logger.error(f"Historical context query failed: {e}")
            return {}


class DefaultFeatureExtractor(IFeatureExtractor):
    """Default feature extractor implementation."""
    
    def __init__(self, target_size: int = 256):
        self.target_size = target_size
    
    async def extract_features(
        self,
        request: CouncilRequest,
        context: ContextSnapshot
    ) -> NeuralFeatures:
        """Extract features from request and context."""
        features = []
        
        # Basic request features
        features.extend(self._extract_request_features(request))
        
        # Context features
        features.extend(self._extract_context_features(context))
        
        # Pad or truncate to target size
        if len(features) < self.target_size:
            features.extend([0.0] * (self.target_size - len(features)))
        else:
            features = features[:self.target_size]
        
        # Convert to numpy arrays
        raw_features = np.array(features, dtype=np.float32)
        normalized_features = raw_features / (np.linalg.norm(raw_features) + 1e-8)  # L2 normalization
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        return NeuralFeatures(
            raw_features=raw_features,
            normalized_features=normalized_features,
            feature_names=feature_names
        )
    
    def _extract_request_features(self, request: CouncilRequest) -> List[float]:
        """Extract features from the request."""
        features = []
        
        # Priority (normalized)
        features.append(request.priority / 10.0)
        
        # Request type encoding (simple hash)
        type_hash = hash(request.request_type) % 1000
        features.append(type_hash / 1000.0)
        
        # GPU allocation specific features
        if "gpu_allocation" in request.payload:
            gpu_data = request.payload["gpu_allocation"]
            
            # GPU count (normalized)
            gpu_count = gpu_data.get("gpu_count", 0)
            features.append(min(gpu_count / 10.0, 1.0))
            
            # Cost per hour (normalized)
            cost_per_hour = gpu_data.get("cost_per_hour", 0)
            features.append(min(cost_per_hour / 50.0, 1.0))
            
            # Duration (normalized)
            duration = gpu_data.get("duration_hours", 0)
            features.append(min(duration / 24.0, 1.0))
            
            # GPU type encoding
            gpu_type = gpu_data.get("gpu_type", "unknown")
            type_encoding = {"a100": 0.9, "v100": 0.7, "h100": 1.0}.get(gpu_type, 0.5)
            features.append(type_encoding)
        else:
            # Default values if no GPU allocation data
            features.extend([0.0, 0.0, 0.0, 0.5])
        
        return features
    
    def _extract_context_features(self, context: ContextSnapshot) -> List[float]:
        """Extract features from context."""
        features = []
        
        # Historical context features
        decision_count = len(context.recent_decisions)
        features.append(min(decision_count / 10.0, 1.0))
        
        # System context features - extract from temporal_context
        system_context = context.temporal_context.get("request_context", {}).get("system_context", {})
        available_gpus = system_context.get("available_gpus", 0)
        features.append(min(available_gpus / 20.0, 1.0))
        
        # Time-based features
        current_hour = datetime.now(timezone.utc).hour
        features.append(current_hour / 24.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        names = [
            "priority_normalized",
            "request_type_hash",
            "gpu_count_normalized",
            "cost_per_hour_normalized", 
            "duration_normalized",
            "gpu_type_encoding",
            "historical_decision_count",
            "available_gpus_normalized",
            "current_hour_normalized"
        ]
        
        # Pad to match feature vector size
        while len(names) < self.target_size:
            names.append(f"feature_{len(names)}")
        
        return names[:self.target_size]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        # Simple importance scores based on domain knowledge
        importance = {
            "priority_normalized": 0.8,
            "gpu_count_normalized": 0.9,
            "cost_per_hour_normalized": 0.95,
            "duration_normalized": 0.7,
            "gpu_type_encoding": 0.6,
            "available_gpus_normalized": 0.85,
            "historical_decision_count": 0.5,
            "current_hour_normalized": 0.3,
            "request_type_hash": 0.4
        }
        
        # Add default importance for other features
        feature_names = self.get_feature_names()
        for name in feature_names:
            if name not in importance:
                importance[name] = 0.1
        
        return importance


class DefaultDecisionMaker(IDecisionMaker):
    """Default decision maker implementation."""
    
    async def make_decision(
        self,
        neural_output: torch.Tensor,
        features: NeuralFeatures,
        context: ContextSnapshot
    ) -> tuple[VoteDecision, VoteConfidence]:
        """Make decision based on neural output."""
        # Convert tensor to numpy for easier handling
        neural_output_np = neural_output.detach().cpu().numpy()
        
        # Handle batch dimension - take first sample if batch size > 1
        if neural_output_np.ndim > 1:
            neural_output_np = neural_output_np[0]  # Take first sample
        
        # Neural output should be probabilities for each decision type
        if len(neural_output_np) >= 4:
            approve_prob = float(neural_output_np[0])
            reject_prob = float(neural_output_np[1])
            abstain_prob = float(neural_output_np[2])
            delegate_prob = float(neural_output_np[3])
        else:
            # Fallback if neural output is different format
            approve_prob = float(neural_output_np[0]) if len(neural_output_np) > 0 else 0.5
            reject_prob = 1.0 - approve_prob
            abstain_prob = 0.1
            delegate_prob = 0.1
        
        # Find decision with highest probability
        decisions = [
            (VoteDecision.APPROVE, approve_prob),
            (VoteDecision.REJECT, reject_prob),
            (VoteDecision.ABSTAIN, abstain_prob),
            (VoteDecision.DELEGATE, delegate_prob)
        ]
        
        decision, confidence = max(decisions, key=lambda x: x[1])
        
        # Apply business rules
        if confidence < 0.3:
            decision = VoteDecision.ABSTAIN
            confidence = 0.3
        
        return decision, VoteConfidence(confidence)
    
    def __init__(self):
        self._thresholds = {
            "approve": 0.6,
            "reject": 0.6,
            "abstain": 0.3,
            "delegate": 0.4
        }
    
    def get_decision_threshold(self) -> Dict[str, float]:
        """Get decision thresholds."""
        return self._thresholds.copy()
    
    def set_decision_threshold(self, thresholds: Dict[str, float]) -> None:
        """Set decision thresholds."""
        self._thresholds.update(thresholds)


class DefaultEvidenceCollector(IEvidenceCollector):
    """Default evidence collector implementation."""
    
    async def collect_evidence(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        context: ContextSnapshot
    ) -> List[DecisionEvidence]:
        """Collect evidence for the decision."""
        evidence = []
        
        # Decision-based evidence
        evidence.append(DecisionEvidence(
            evidence_type="decision_analysis",
            source="decision_maker",
            confidence=VoteConfidence(float(confidence)),
            data={
                "decision": decision.value,
                "confidence": float(confidence),
                "decision_factors": {
                    "decision_type": decision.value,
                    "confidence_level": float(confidence)
                }
            }
        ))
        
        # Request details evidence
        if "gpu_allocation" in request.payload:
            gpu_data = request.payload["gpu_allocation"]
            evidence.append(DecisionEvidence(
                evidence_type="request_details",
                source="request_parser",
                confidence=VoteConfidence(1.0),
                data={
                    "gpu_type": gpu_data.get("gpu_type"),
                    "gpu_count": gpu_data.get("gpu_count"),
                    "cost_per_hour": gpu_data.get("cost_per_hour"),
                    "duration_hours": gpu_data.get("duration_hours"),
                    "estimated_cost": gpu_data.get("estimated_cost")
                }
            ))
        
        # Historical evidence
        if len(context.recent_decisions) > 0:
            evidence.append(DecisionEvidence(
                type="historical_pattern",
                source="context_provider",
                confidence=0.7,
                data={
                    "similar_decisions": len(context.recent_decisions),
                    "pattern_analysis": "Based on historical decision patterns"
                }
            ))
        
        return evidence
    
    async def validate_evidence(
        self,
        evidence: List[DecisionEvidence]
    ) -> List[DecisionEvidence]:
        """Validate and filter evidence."""
        validated_evidence = []
        
        for item in evidence:
            # Basic validation
            if item.confidence > 0 and item.data:
                # Check for required fields
                if hasattr(item, 'type') and hasattr(item, 'source'):
                    validated_evidence.append(item)
                else:
                    logger.warning(f"Evidence item missing required fields: {item}")
            else:
                logger.warning(f"Evidence item failed validation: {item}")
        
        return validated_evidence


class DefaultReasoningEngine(IReasoningEngine):
    """Default reasoning engine implementation."""
    
    async def generate_reasoning(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        evidence: List[DecisionEvidence],
        context: ContextSnapshot
    ) -> str:
        """Generate human-readable reasoning."""
        reasoning_parts = []
        
        # Decision statement
        reasoning_parts.append(f"Decision: {decision.value.upper()}")
        reasoning_parts.append(f"Confidence: {confidence:.1%}")
        
        # Evidence-based reasoning
        for ev in evidence:
            if ev.evidence_type == "neural_analysis":
                neural_data = ev.data
                max_prob = max(neural_data.get("decision_probabilities", {}).values())
                reasoning_parts.append(f"Neural network analysis shows {max_prob:.1%} confidence in this decision.")
            
            elif ev.evidence_type == "request_details":
                details = ev.data
                gpu_count = details.get("gpu_count", 0)
                cost_per_hour = details.get("cost_per_hour", 0)
                reasoning_parts.append(
                    f"Request analysis: {gpu_count} GPUs at ${cost_per_hour}/hour."
                )
            
            elif ev.evidence_type == "historical_pattern":
                hist_data = ev.data
                similar_count = hist_data.get("similar_decisions", 0)
                reasoning_parts.append(f"Historical analysis based on {similar_count} similar decisions.")
            
            elif ev.evidence_type == "decision_analysis":
                decision_data = ev.data
                decision_type = decision_data.get("decision", "unknown")
                confidence = decision_data.get("confidence", 0.0)
                reasoning_parts.append(f"Decision analysis: {decision_type} with {confidence:.1%} confidence.")
        
        # Decision-specific reasoning
        if decision == VoteDecision.APPROVE:
            reasoning_parts.append("Request meets approval criteria based on cost, availability, and risk assessment.")
        elif decision == VoteDecision.REJECT:
            reasoning_parts.append("Request does not meet approval criteria due to cost, risk, or resource constraints.")
        elif decision == VoteDecision.ABSTAIN:
            reasoning_parts.append("Insufficient information or confidence to make a definitive decision.")
        elif decision == VoteDecision.DELEGATE:
            reasoning_parts.append("Request requires specialized expertise or higher authority approval.")
        
        return " ".join(reasoning_parts)
    
    async def explain_decision(
        self,
        response: CouncilResponse
    ) -> Dict[str, Any]:
        """Provide detailed explanation of the decision."""
        explanation = {
            "decision": response.decision.value,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "factors": {
                "primary_factors": [],
                "supporting_factors": [],
                "risk_factors": []
            },
            "alternatives": {
                "considered": ["approve", "reject", "abstain", "delegate"],
                "why_not_chosen": {}
            },
            "metadata": {
                "processing_time_ms": response.processing_time_ms,
                "agent_id": response.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Analyze evidence for factors
        for evidence in response.evidence:
            if evidence.type == "neural_analysis":
                explanation["factors"]["primary_factors"].append(
                    f"Neural network confidence: {evidence.confidence:.1%}"
                )
            elif evidence.type == "request_details":
                explanation["factors"]["supporting_factors"].append(
                    "Request parameters within acceptable ranges"
                )
            elif evidence.type == "historical_pattern":
                explanation["factors"]["supporting_factors"].append(
                    "Historical patterns support this decision"
                )
        
        # Add decision-specific explanations
        if response.decision == VoteDecision.APPROVE:
            explanation["factors"]["primary_factors"].append("Cost-benefit analysis favorable")
        elif response.decision == VoteDecision.REJECT:
            explanation["factors"]["risk_factors"].append("Risk assessment indicates rejection")
        
        return explanation


class Neo4jStorageAdapter(IStorageAdapter):
    """Neo4j storage adapter implementation."""
    
    def __init__(self, neo4j_adapter):
        self.neo4j_adapter = neo4j_adapter
    
    async def store_decision(
        self,
        response: CouncilResponse,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store decision in Neo4j."""
        try:
            query = """
            CREATE (d:Decision {
                request_id: $request_id,
                agent_id: $agent_id,
                decision: $decision,
                confidence: $confidence,
                reasoning: $reasoning,
                processing_time_ms: $processing_time_ms,
                created_at: $created_at,
                metadata: $metadata
            })
            RETURN d
            """
            
            await self.neo4j_adapter.query(query, {
                "request_id": str(response.request_id),
                "agent_id": response.agent_id,
                "decision": response.decision.value,
                "confidence": response.confidence,
                "reasoning": response.reasoning,
                "processing_time_ms": response.processing_time_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": json.dumps(metadata or {})
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store decision: {e}")
            return False


class KafkaEventPublisher(IEventPublisher):
    """Kafka event publisher implementation."""
    
    def __init__(self, event_producer):
        self.event_producer = event_producer
    
    async def publish_decision(
        self,
        response: CouncilResponse,
        topic: str
    ) -> bool:
        """Publish decision event to Kafka."""
        try:
            event = {
                "event_type": "council_decision",
                "request_id": str(response.request_id),
                "agent_id": response.agent_id,
                "decision": response.decision.value,
                "confidence": response.confidence,
                "reasoning": response.reasoning,
                "processing_time_ms": response.processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.event_producer.send_event(topic, event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish decision event: {e}")
            return False
    
    async def publish_metrics(
        self,
        metrics: Dict[str, Any],
        topic: str
    ) -> bool:
        """Publish metrics event to Kafka."""
        try:
            event = {
                "event_type": "agent_metrics",
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.event_producer.send_event(topic, event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish metrics event: {e}")
            return False


class Mem0MemoryManager(IMemoryManager):
    """Mem0 memory manager implementation."""
    
    def __init__(self, mem0_manager):
        self.mem0_manager = mem0_manager
    
    async def store_experience(
        self,
        request: CouncilRequest,
        response: CouncilResponse,
        context: Dict[str, Any]
    ) -> bool:
        """Store decision experience in memory."""
        try:
            experience = {
                "request": request.dict(),
                "response": response.dict(),
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store in Mem0 (implementation depends on Mem0 API)
            # await self.mem0_manager.store(experience)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return False
    
    async def retrieve_similar_experiences(
        self,
        request: CouncilRequest,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past experiences."""
        try:
            # Query Mem0 for similar experiences (implementation depends on Mem0 API)
            # experiences = await self.mem0_manager.search(request, limit=limit)
            # return experiences
            
            return []  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []


class DefaultResourceManager(IResourceManager):
    """Default resource manager implementation."""
    
    def __init__(self):
        self.resources = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0
        }
    
    async def initialize(self):
        """Initialize resource manager."""
        logger.info("Resource manager initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Resource manager cleaned up")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "resources": self.resources,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def allocate_resources(
        self,
        resource_type: str,
        amount: float
    ) -> bool:
        """Allocate resources."""
        if resource_type in self.resources:
            self.resources[resource_type] += amount
            return True
        return False
    
    async def release_resources(
        self,
        resource_type: str,
        amount: float
    ) -> bool:
        """Release resources."""
        if resource_type in self.resources:
            self.resources[resource_type] = max(0, self.resources[resource_type] - amount)
            return True
        return False