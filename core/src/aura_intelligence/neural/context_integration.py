"""
Context-Aware LNN Integration.

This module provides integration between Liquid Neural Networks and the
existing context management systems (Neo4j, Mem0, Kafka).
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import numpy as np
import torch
from opentelemetry import trace

from ..neural.lnn import LiquidNeuralNetwork, LNNConfig
from ..resilience import resilient, ResilienceLevel
from ..observability import create_tracer

logger = logging.getLogger(__name__)
tracer = create_tracer("context_aware_lnn")


@dataclass
class ContextWindow:
    """Sliding window of relevant context for LNN inference."""
    historical_patterns: List[Dict[str, Any]] = field(default_factory=list)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    entity_relationships: Dict[str, List[str]] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert context window to tensor for LNN input."""
        # Simplified encoding - in production, use proper embeddings
        features = []
        
        # Encode historical patterns (take last 10)
        for pattern in self.historical_patterns[-10:]:
            features.extend(pattern.get("features", [0.0] * 10))
            
        # Encode recent decisions (take last 5)
        for decision in self.recent_decisions[-5:]:
            features.append(decision.get("confidence", 0.0))
            features.append(decision.get("value", 0.0))
            
        # Pad or truncate to fixed size
        target_size = 128
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return torch.tensor(features, dtype=torch.float32)


class ContextAwareLNN:
    """LNN that queries and updates shared context."""
    
    def __init__(
        self,
        lnn_config: LNNConfig,
        memory_manager: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        event_producer: Optional[Any] = None,
        context_window_size: int = 100,
        feature_flags: Optional[Dict[str, bool]] = None
    ):
        self.lnn = LiquidNeuralNetwork(lnn_config)
        self.memory = memory_manager
        self.graph = knowledge_graph
        self.events = event_producer
        self.window_size = context_window_size
        self.flags = feature_flags or {}
        
        # Context cache for performance
        self._context_cache: Dict[str, ContextWindow] = {}
        self._cache_ttl = 300  # 5 minutes
        
    @resilient(level=ResilienceLevel.CRITICAL)
    async def context_aware_inference(
        self,
        input_data: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None,
        trace_context: Optional[trace.Context] = None
    ) -> Dict[str, Any]:
        """Run inference with context retrieval and update."""
        with tracer.start_as_current_span(
            "context_aware_inference",
            context=trace_context
        ) as span:
            span.set_attribute("input.shape", str(input_data.shape))
            span.set_attribute("context.enabled", self.flags.get("enable_context_retrieval", True))
            
            # 1. Retrieve relevant context
            if self.flags.get("enable_context_retrieval", True):
                context = await self._retrieve_context(input_data, query_context)
                span.set_attribute("context.patterns", len(context.historical_patterns))
            else:
                context = ContextWindow()
                
            # 2. Enrich input with context
            enriched_input = self._enrich_input(input_data, context)
            span.set_attribute("enriched.shape", str(enriched_input.shape))
            
            # 3. Run LNN inference
            start_time = datetime.utcnow()
            output, hidden_states = self.lnn(enriched_input)
            inference_time = (datetime.utcnow() - start_time).total_seconds()
            span.set_attribute("inference.latency_ms", inference_time * 1000)
            
            # 4. Post-process with context
            result = self._post_process(output, context, hidden_states)
            result["inference_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "latency_ms": inference_time * 1000,
                "context_used": len(context.historical_patterns) > 0,
                "model_version": self.lnn.config.version if hasattr(self.lnn.config, 'version') else "1.0"
            }
            
            # 5. Update context stores (async, non-blocking)
            if self.flags.get("enable_context_updates", True):
                asyncio.create_task(self._update_context(result))
                
            # 6. Emit decision event
            if self.flags.get("enable_event_emission", True) and self.events:
                await self._emit_decision_event(result)
                
            span.set_attribute("result.confidence", result.get("confidence", 0.0))
            return result
    
    async def _retrieve_context(
        self,
        input_data: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None
    ) -> ContextWindow:
        """Retrieve relevant context from memory and knowledge graph."""
        context = ContextWindow()
        
        # Check cache first
        cache_key = self._compute_cache_key(input_data, query_context)
        if cache_key in self._context_cache:
            cached = self._context_cache[cache_key]
            if (datetime.utcnow() - cached["timestamp"]).total_seconds() < self._cache_ttl:
                return cached["context"]
        
        try:
            # Parallel retrieval from different sources
            tasks = []
            
            if self.memory:
                tasks.append(self._retrieve_from_memory(input_data, query_context))
            if self.graph:
                tasks.append(self._retrieve_from_graph(input_data, query_context))
                
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Context retrieval failed: {result}")
                    elif i == 0 and self.memory:  # Memory results
                        context.historical_patterns = result.get("patterns", [])
                        context.recent_decisions = result.get("decisions", [])
                    elif i == 1 and self.graph:  # Graph results
                        context.entity_relationships = result.get("relationships", {})
                        context.temporal_context = result.get("temporal", {})
                        
            # Calculate confidence scores
            context.confidence_scores = self._calculate_confidence(context)
            
            # Update cache
            self._context_cache[cache_key] = {
                "context": context,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            # Return empty context on error
            
        return context
    
    async def _retrieve_from_memory(
        self,
        input_data: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve patterns and decisions from memory store."""
        if not self.memory:
            return {}
            
        # Convert input to query embedding
        query_embedding = input_data.mean(dim=0).numpy()
        
        # Search for similar patterns
        patterns = await self.memory.search_memories(
            agent_id="lnn",
            query_embedding=query_embedding,
            memory_types=["pattern", "decision"],
            limit=self.window_size
        )
        
        return {
            "patterns": [p for p in patterns if p["type"] == "pattern"],
            "decisions": [d for d in patterns if d["type"] == "decision"]
        }
    
    async def _retrieve_from_graph(
        self,
        input_data: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve relationships and temporal context from knowledge graph."""
        if not self.graph:
            return {}
            
        # Extract entities from query context
        entities = query_context.get("entities", []) if query_context else []
        
        if not entities:
            return {}
            
        # Query relationships
        relationships = {}
        for entity in entities[:5]:  # Limit to prevent explosion
            query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(related)
            RETURN type(r) as relationship, collect(related.id) as related_ids
            LIMIT 20
            """
            results = await self.graph.query(query, {"entity_id": entity})
            relationships[entity] = {
                r["relationship"]: r["related_ids"] 
                for r in results
            }
            
        return {
            "relationships": relationships,
            "temporal": query_context.get("temporal", {}) if query_context else {}
        }
    
    def _enrich_input(
        self,
        input_data: torch.Tensor,
        context: ContextWindow
    ) -> torch.Tensor:
        """Enrich input tensor with context information."""
        # Convert context to tensor
        context_tensor = context.to_tensor()
        
        # Concatenate with input
        # Ensure dimensions match
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
        if len(context_tensor.shape) == 1:
            context_tensor = context_tensor.unsqueeze(0)
            
        # Adjust batch dimensions
        batch_size = input_data.shape[0]
        if context_tensor.shape[0] == 1 and batch_size > 1:
            context_tensor = context_tensor.repeat(batch_size, 1)
            
        # Concatenate along feature dimension
        enriched = torch.cat([input_data, context_tensor], dim=-1)
        
        return enriched
    
    def _post_process(
        self,
        output: torch.Tensor,
        context: ContextWindow,
        hidden_states: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Post-process LNN output with context."""
        # Extract predictions
        if len(output.shape) > 1:
            predictions = output.detach().numpy()
            confidence = float(torch.softmax(output, dim=-1).max().item())
        else:
            predictions = output.detach().numpy()
            confidence = float(torch.sigmoid(output).item())
            
        # Build result
        result = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else float(predictions),
            "confidence": confidence,
            "context_influence": self._calculate_context_influence(hidden_states, context),
            "adaptation_metrics": self.lnn.get_metrics().to_dict() if hasattr(self.lnn, 'get_metrics') else {}
        }
        
        # Add explainability
        if context.historical_patterns:
            result["similar_patterns"] = [
                {
                    "pattern_id": p.get("id"),
                    "similarity": p.get("similarity", 0.0),
                    "outcome": p.get("outcome")
                }
                for p in context.historical_patterns[:3]
            ]
            
        return result
    
    async def _update_context(self, result: Dict[str, Any]):
        """Update context stores with new decision."""
        try:
            # Update memory
            if self.memory:
                await self.memory.add_memory(
                    agent_id="lnn",
                    memory_type="decision",
                    content=result,
                    metadata={
                        "timestamp": datetime.utcnow(),
                        "confidence": result.get("confidence", 0.0),
                        "context_influence": result.get("context_influence", 0.0)
                    }
                )
                
            # Update graph
            if self.graph and "entity_updates" in result:
                for entity_id, updates in result["entity_updates"].items():
                    await self.graph.update_entity(entity_id, updates)
                    
        except Exception as e:
            logger.error(f"Context update failed: {e}")
    
    async def _emit_decision_event(self, result: Dict[str, Any]):
        """Emit decision event to Kafka."""
        try:
            event = {
                "event_type": "lnn.decision",
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": "context_aware_lnn",
                "decision": result,
                "metadata": {
                    "model_version": getattr(self.lnn.config, 'version', '1.0'),
                    "context_enabled": self.flags.get("enable_context_retrieval", True)
                }
            }
            await self.events.publish("agent.decisions", event)
        except Exception as e:
            logger.error(f"Event emission failed: {e}")
    
    def _compute_cache_key(
        self,
        input_data: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Compute cache key for context retrieval."""
        # Simple hash of input shape and context entities
        key_parts = [
            str(input_data.shape),
            str(sorted(query_context.get("entities", []))) if query_context else ""
        ]
        return "|".join(key_parts)
    
    def _calculate_confidence(self, context: ContextWindow) -> Dict[str, float]:
        """Calculate confidence scores for context quality."""
        scores = {}
        
        # Relevance score based on pattern count
        pattern_count = len(context.historical_patterns)
        scores["relevance"] = min(1.0, pattern_count / 10.0)
        
        # Recency score based on decision age
        if context.recent_decisions:
            latest = context.recent_decisions[0]
            age_hours = (datetime.utcnow() - latest.get("timestamp", datetime.utcnow())).total_seconds() / 3600
            scores["recency"] = max(0.0, 1.0 - (age_hours / 24.0))
        else:
            scores["recency"] = 0.0
            
        # Completeness score
        scores["completeness"] = (
            (0.4 if context.historical_patterns else 0.0) +
            (0.3 if context.recent_decisions else 0.0) +
            (0.3 if context.entity_relationships else 0.0)
        )
        
        return scores
    
    def _calculate_context_influence(
        self,
        hidden_states: List[torch.Tensor],
        context: ContextWindow
    ) -> float:
        """Calculate how much context influenced the decision."""
        if not hidden_states or not context.historical_patterns:
            return 0.0
            
        # Simple heuristic: compare activation magnitudes
        # In production, use attention weights or gradient-based attribution
        try:
            final_state = hidden_states[-1]
            activation_magnitude = float(final_state.abs().mean().item())
            
            # Higher activation with context indicates influence
            baseline_activation = 0.1  # Empirically determined
            influence = min(1.0, (activation_magnitude - baseline_activation) / baseline_activation)
            
            return max(0.0, influence)
        except Exception:
            return 0.0