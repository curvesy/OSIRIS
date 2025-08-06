"""
TDA Agent Context API - Production Ready
Provides standardized interface for agents to consume TDA insights
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum

from ..tda.models import TDAResult, PersistenceDiagram
from ..observability.context_managers import AgentContext
from ..utils.logger import get_logger
from ..observability.metrics import metrics_collector
from ..observability.tracing import trace_span

logger = get_logger(__name__)


class TopologicalSignal(Enum):
    """Types of topological signals for agents"""
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_EMERGED = "pattern_emerged"
    STRUCTURE_CHANGED = "structure_changed"
    PERSISTENCE_SPIKE = "persistence_spike"
    DIMENSION_ANOMALY = "dimension_anomaly"


@dataclass
class TDAContextEnrichment:
    """Enriched context with TDA insights for agents"""
    base_context: AgentContext
    topological_features: Dict[str, Any]
    anomaly_signals: List[TopologicalSignal]
    persistence_summary: Dict[str, float]
    recommendations: List[str]
    confidence_score: float
    trace_id: str


class ITDAContextProvider(Protocol):
    """Interface for TDA context providers"""
    
    async def enrich_agent_context(
        self,
        agent_id: str,
        base_context: AgentContext,
        data_id: Optional[str] = None
    ) -> TDAContextEnrichment:
        """Enrich agent context with TDA insights"""
        ...
        
    async def subscribe_to_tda_events(
        self,
        agent_id: str,
        event_types: List[TopologicalSignal],
        callback: callable
    ) -> str:
        """Subscribe agent to TDA event stream"""
        ...


class TDAAgentContextAdapter:
    """
    Production adapter for enriching agent context with TDA insights.
    
    Features:
    - Real-time context enrichment
    - Event subscription for agents
    - Signal interpretation
    - Actionable recommendations
    """
    
    def __init__(self, neo4j_adapter, mem0_adapter, event_bus):
        self.neo4j_adapter = neo4j_adapter
        self.mem0_adapter = mem0_adapter
        self.event_bus = event_bus
        self._subscriptions = {}
        
    @trace_span("enrich_agent_context")
    async def enrich_agent_context(
        self,
        agent_id: str,
        base_context: AgentContext,
        data_id: Optional[str] = None
    ) -> TDAContextEnrichment:
        """
        Enrich agent context with TDA insights.
        
        Args:
            agent_id: ID of requesting agent
            base_context: Base agent context
            data_id: Optional specific data to analyze
            
        Returns:
            Enriched context with TDA insights
        """
        try:
            # Get topological context from Neo4j
            if data_id:
                topo_context = await self.neo4j_adapter.get_topological_context(data_id)
            else:
                # Get general topological context
                topo_context = await self._get_general_topological_context(base_context)
                
            # Get episodic TDA memories
            tda_memories = await self.mem0_adapter.get_tda_context_for_agent(
                agent_id=agent_id,
                current_data_id=data_id or base_context.current_task_id
            )
            
            # Analyze signals
            signals = self._analyze_topological_signals(topo_context, tda_memories)
            
            # Generate persistence summary
            persistence_summary = self._generate_persistence_summary(topo_context)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                signals, persistence_summary, base_context
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(topo_context, tda_memories)
            
            # Create enriched context
            enriched = TDAContextEnrichment(
                base_context=base_context,
                topological_features={
                    "current": topo_context,
                    "historical": tda_memories["topological_summary"],
                    "anomaly_history": tda_memories["topological_summary"]["recent_anomalies"]
                },
                anomaly_signals=signals,
                persistence_summary=persistence_summary,
                recommendations=recommendations,
                confidence_score=confidence,
                trace_id=base_context.trace_id
            )
            
            # Update metrics
            metrics_collector.increment(
                "tda.context.enrichments",
                tags={"agent_id": agent_id, "has_anomaly": len(signals) > 0}
            )
            
            logger.info(f"Enriched context for agent {agent_id} with {len(signals)} signals")
            
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich agent context: {e}")
            raise
            
    async def _get_general_topological_context(
        self,
        base_context: AgentContext
    ) -> Dict[str, Any]:
        """Get general topological context based on agent's current state"""
        # Query recent TDA results relevant to context
        filters = {
            "time_range": (base_context.start_time, datetime.now(timezone.utc)),
            "anomaly_threshold": 0.3  # Include medium and high anomalies
        }
        
        results = await self.neo4j_adapter.query_tda_results(filters=filters, limit=20)
        
        # Aggregate into context
        return {
            "data_id": base_context.current_task_id,
            "topological_features": results,
            "anomalies": [r for r in results if r.get("anomaly_score", 0) > 0.7],
            "summary": {
                "total_analyses": len(results),
                "max_anomaly_score": max((r.get("anomaly_score", 0) for r in results), default=0),
                "algorithms_used": list(set(r["algorithm"] for r in results))
            }
        }
        
    def _analyze_topological_signals(
        self,
        topo_context: Dict[str, Any],
        tda_memories: Dict[str, Any]
    ) -> List[TopologicalSignal]:
        """Analyze and identify topological signals"""
        signals = []
        
        # Check for anomalies
        if topo_context.get("summary", {}).get("max_anomaly_score", 0) > 0.7:
            signals.append(TopologicalSignal.ANOMALY_DETECTED)
            
        # Check for pattern emergence
        recent_features = topo_context.get("topological_features", [])
        if len(recent_features) > 5:
            # Simple pattern detection - in production use more sophisticated methods
            persistence_values = [f.get("total_persistence", 0) for f in recent_features[-5:]]
            if all(p > 1.0 for p in persistence_values):
                signals.append(TopologicalSignal.PATTERN_EMERGED)
                
        # Check for structure changes
        historical = tda_memories.get("topological_summary", {})
        current_algos = set(topo_context.get("summary", {}).get("algorithms_used", []))
        historical_algos = set(historical.get("algorithms_used", []))
        
        if current_algos != historical_algos:
            signals.append(TopologicalSignal.STRUCTURE_CHANGED)
            
        # Check for persistence spikes
        current_max = max(
            (f.get("max_persistence", 0) for f in recent_features),
            default=0
        )
        if current_max > 2.0:  # Threshold for spike
            signals.append(TopologicalSignal.PERSISTENCE_SPIKE)
            
        # Check for dimensional anomalies
        for anomaly in topo_context.get("anomalies", []):
            if anomaly.get("dimension", 0) > 2:  # High-dimensional anomaly
                signals.append(TopologicalSignal.DIMENSION_ANOMALY)
                break
                
        return list(set(signals))  # Remove duplicates
        
    def _generate_persistence_summary(self, topo_context: Dict[str, Any]) -> Dict[str, float]:
        """Generate summary statistics of persistence features"""
        features = topo_context.get("topological_features", [])
        
        if not features:
            return {
                "total_persistence": 0.0,
                "avg_persistence": 0.0,
                "max_persistence": 0.0,
                "feature_count": 0,
                "anomaly_ratio": 0.0
            }
            
        total_persistence = sum(f.get("total_persistence", 0) for f in features)
        max_persistence = max(f.get("max_persistence", 0) for f in features)
        anomaly_count = sum(1 for f in features if f.get("anomaly_score", 0) > 0.7)
        
        return {
            "total_persistence": total_persistence,
            "avg_persistence": total_persistence / len(features) if features else 0.0,
            "max_persistence": max_persistence,
            "feature_count": len(features),
            "anomaly_ratio": anomaly_count / len(features) if features else 0.0
        }
        
    def _generate_recommendations(
        self,
        signals: List[TopologicalSignal],
        persistence_summary: Dict[str, float],
        base_context: AgentContext
    ) -> List[str]:
        """Generate actionable recommendations based on TDA insights"""
        recommendations = []
        
        if TopologicalSignal.ANOMALY_DETECTED in signals:
            recommendations.append(
                "INVESTIGATE: High topological anomaly detected. Review data sources and recent changes."
            )
            
        if TopologicalSignal.PATTERN_EMERGED in signals:
            recommendations.append(
                "MONITOR: Persistent pattern detected. Consider adjusting monitoring thresholds."
            )
            
        if TopologicalSignal.STRUCTURE_CHANGED in signals:
            recommendations.append(
                "ADAPT: Topological structure has changed. Update models and decision criteria."
            )
            
        if TopologicalSignal.PERSISTENCE_SPIKE in signals:
            recommendations.append(
                "ALERT: Significant persistence spike. May indicate system stress or data quality issues."
            )
            
        if TopologicalSignal.DIMENSION_ANOMALY in signals:
            recommendations.append(
                "ANALYZE: High-dimensional anomaly present. Deep analysis recommended."
            )
            
        # Add context-specific recommendations
        if persistence_summary["anomaly_ratio"] > 0.3:
            recommendations.append(
                f"CAUTION: {persistence_summary['anomaly_ratio']:.1%} of recent analyses show anomalies."
            )
            
        return recommendations
        
    def _calculate_confidence(
        self,
        topo_context: Dict[str, Any],
        tda_memories: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for TDA insights"""
        confidence = 1.0
        
        # Reduce confidence if limited data
        feature_count = len(topo_context.get("topological_features", []))
        if feature_count < 5:
            confidence *= 0.7
        elif feature_count < 10:
            confidence *= 0.85
            
        # Reduce confidence if no historical context
        memory_count = tda_memories.get("topological_summary", {}).get("total_analyses", 0)
        if memory_count < 10:
            confidence *= 0.8
            
        # Reduce confidence if mixed signals
        anomaly_scores = [
            f.get("anomaly_score", 0) 
            for f in topo_context.get("topological_features", [])
        ]
        if anomaly_scores:
            score_variance = np.var(anomaly_scores)
            if score_variance > 0.2:  # High variance indicates uncertainty
                confidence *= 0.9
                
        return max(0.1, min(1.0, confidence))
        
    @trace_span("subscribe_to_tda_events")
    async def subscribe_to_tda_events(
        self,
        agent_id: str,
        event_types: List[TopologicalSignal],
        callback: callable
    ) -> str:
        """
        Subscribe agent to TDA event stream.
        
        Args:
            agent_id: ID of subscribing agent
            event_types: Types of events to subscribe to
            callback: Async callback function
            
        Returns:
            subscription_id
        """
        subscription_id = f"tda_sub_{agent_id}_{datetime.now(timezone.utc).timestamp()}"
        
        # Create subscription handler
        async def event_handler(event):
            # Check if event matches subscription
            if event.get("type") in [e.value for e in event_types]:
                # Enrich event with context
                enriched_event = {
                    **event,
                    "agent_id": agent_id,
                    "subscription_id": subscription_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Call agent's callback
                try:
                    await callback(enriched_event)
                except Exception as e:
                    logger.error(f"Error in TDA event callback for agent {agent_id}: {e}")
                    
        # Register with event bus
        await self.event_bus.subscribe(
            topic="tda.signals",
            handler=event_handler,
            subscription_id=subscription_id
        )
        
        # Store subscription
        self._subscriptions[subscription_id] = {
            "agent_id": agent_id,
            "event_types": event_types,
            "created_at": datetime.now(timezone.utc)
        }
        
        logger.info(f"Agent {agent_id} subscribed to TDA events: {event_types}")
        
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from TDA events"""
        if subscription_id in self._subscriptions:
            await self.event_bus.unsubscribe(subscription_id)
            del self._subscriptions[subscription_id]
            logger.info(f"Unsubscribed {subscription_id} from TDA events")