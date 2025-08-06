"""
🧭 Semantic Routing Engine

Advanced semantic routing for intelligent agent selection and task distribution.
Implements 2025 patterns for capability-based routing, TDA-aware decisions,
and dynamic agent selection with performance optimization.

Key Features:
- Capability-based agent matching
- TDA-aware routing decisions with anomaly consideration
- Dynamic load balancing and performance optimization
- Routing accuracy tracking and improvement
- Fallback mechanisms for agent unavailability

TDA Integration:
- Uses TDA patterns for routing optimization
- Considers TDA anomaly data for agent selection
- Integrates with TDA performance metrics
- Supports TDA-driven routing adaptations
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

from .base_interfaces import (
    TDAContext, SemanticAnalysis, OrchestrationStrategy, 
    UrgencyLevel, SemanticOrchestrator
)
from .tda_integration import TDAIntegration

# TDA integration
try:
    from aura_intelligence.observability.tracing import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None

class AgentCapability(Enum):
    """Agent capability types for routing"""
    ANALYSIS = "analysis"
    DECISION_MAKING = "decision_making"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    COORDINATION = "coordination"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class AgentProfile:
    """Agent profile for routing decisions"""
    agent_id: str
    capabilities: Set[AgentCapability]
    performance_score: float
    current_load: float
    availability: bool
    specializations: List[str]
    tda_integration_level: float  # How well integrated with TDA
    last_success_rate: float
    metadata: Dict[str, Any]

@dataclass
class RoutingDecision:
    """Result of semantic routing decision"""
    selected_agents: List[str]
    routing_strategy: OrchestrationStrategy
    confidence: float
    reasoning: str
    fallback_agents: List[str]
    estimated_performance: float
    tda_influence: float

class SemanticRouter:
    """
    2025 semantic routing engine with TDA integration
    """
    
    def __init__(self, tda_integration: Optional[TDAIntegration] = None):
        self.tda_integration = tda_integration
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, float] = {}
        self.capability_weights = {
            AgentCapability.ANALYSIS: 1.0,
            AgentCapability.DECISION_MAKING: 1.2,
            AgentCapability.EXECUTION: 1.1,
            AgentCapability.MONITORING: 0.9,
            AgentCapability.COORDINATION: 1.3,
            AgentCapability.PATTERN_RECOGNITION: 1.0,
            AgentCapability.ANOMALY_DETECTION: 1.1
        }
        
        # Initialize default agent profiles
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent profiles"""
        default_agents = [
            AgentProfile(
                agent_id="supervisor_agent",
                capabilities={AgentCapability.DECISION_MAKING, AgentCapability.COORDINATION},
                performance_score=0.85,
                current_load=0.3,
                availability=True,
                specializations=["strategic_decisions", "workflow_coordination"],
                tda_integration_level=0.8,
                last_success_rate=0.9,
                metadata={"type": "supervisor", "priority": "high"}
            ),
            AgentProfile(
                agent_id="analyst_agent",
                capabilities={AgentCapability.ANALYSIS, AgentCapability.PATTERN_RECOGNITION},
                performance_score=0.9,
                current_load=0.4,
                availability=True,
                specializations=["data_analysis", "pattern_detection", "insights"],
                tda_integration_level=0.95,
                last_success_rate=0.88,
                metadata={"type": "analyst", "priority": "medium"}
            ),
            AgentProfile(
                agent_id="observer_agent",
                capabilities={AgentCapability.MONITORING, AgentCapability.ANOMALY_DETECTION},
                performance_score=0.8,
                current_load=0.2,
                availability=True,
                specializations=["system_monitoring", "anomaly_detection", "alerting"],
                tda_integration_level=0.9,
                last_success_rate=0.92,
                metadata={"type": "observer", "priority": "medium"}
            ),
            AgentProfile(
                agent_id="executor_agent",
                capabilities={AgentCapability.EXECUTION, AgentCapability.COORDINATION},
                performance_score=0.75,
                current_load=0.6,
                availability=True,
                specializations=["task_execution", "action_coordination"],
                tda_integration_level=0.7,
                last_success_rate=0.85,
                metadata={"type": "executor", "priority": "low"}
            )
        ]
        
        for agent in default_agents:
            self.agent_profiles[agent.agent_id] = agent
    
    async def route_to_optimal_agents(
        self,
        analysis: SemanticAnalysis,
        available_agents: Optional[List[str]] = None,
        max_agents: int = 5
    ) -> RoutingDecision:
        """
        Route to optimal agents based on semantic analysis and TDA context
        """
        if tracer:
            with tracer.start_as_current_span("semantic_routing") as span:
                span.set_attributes({
                    "routing.complexity": analysis.complexity_score,
                    "routing.urgency": analysis.urgency_level.value,
                    "routing.strategy": analysis.coordination_pattern.value
                })
        
        # Filter available agents
        candidate_agents = self._filter_available_agents(available_agents)
        
        # Score agents based on semantic analysis
        agent_scores = await self._score_agents_for_task(analysis, candidate_agents)
        
        # Apply TDA-based adjustments
        if analysis.tda_correlation:
            agent_scores = await self._apply_tda_adjustments(
                agent_scores, analysis.tda_correlation
            )
        
        # Select optimal agents
        selected_agents = self._select_agents_by_strategy(
            agent_scores, analysis.coordination_pattern, max_agents
        )
        
        # Generate fallback options
        fallback_agents = self._generate_fallback_agents(
            agent_scores, selected_agents, max_agents // 2
        )
        
        # Calculate routing confidence
        confidence = self._calculate_routing_confidence(
            selected_agents, agent_scores, analysis
        )
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            selected_agents, analysis, agent_scores
        )
        
        # Estimate performance
        estimated_performance = self._estimate_routing_performance(
            selected_agents, analysis
        )
        
        # Calculate TDA influence
        tda_influence = (
            analysis.tda_correlation.pattern_confidence 
            if analysis.tda_correlation else 0.0
        )
        
        routing_decision = RoutingDecision(
            selected_agents=selected_agents,
            routing_strategy=analysis.coordination_pattern,
            confidence=confidence,
            reasoning=reasoning,
            fallback_agents=fallback_agents,
            estimated_performance=estimated_performance,
            tda_influence=tda_influence
        )
        
        # Record routing decision for learning
        await self._record_routing_decision(routing_decision, analysis)
        
        return routing_decision
    
    def _filter_available_agents(self, available_agents: Optional[List[str]]) -> List[str]:
        """Filter agents based on availability and constraints"""
        if available_agents:
            # Use provided list, but filter by availability
            return [
                agent_id for agent_id in available_agents
                if (agent_id in self.agent_profiles and 
                    self.agent_profiles[agent_id].availability)
            ]
        else:
            # Use all available agents
            return [
                agent_id for agent_id, profile in self.agent_profiles.items()
                if profile.availability
            ]
    
    async def _score_agents_for_task(
        self,
        analysis: SemanticAnalysis,
        candidate_agents: List[str]
    ) -> Dict[str, float]:
        """Score agents based on their suitability for the task"""
        agent_scores = {}
        
        for agent_id in candidate_agents:
            if agent_id not in self.agent_profiles:
                continue
                
            profile = self.agent_profiles[agent_id]
            score = 0.0
            
            # Base performance score
            score += profile.performance_score * 0.3
            
            # Capability matching
            capability_score = self._calculate_capability_match(
                profile, analysis
            )
            score += capability_score * 0.4
            
            # Load balancing factor
            load_factor = max(0.1, 1.0 - profile.current_load)
            score += load_factor * 0.2
            
            # Success rate factor
            score += profile.last_success_rate * 0.1
            
            # Normalize score
            agent_scores[agent_id] = min(score, 1.0)
        
        return agent_scores
    
    def _calculate_capability_match(
        self,
        profile: AgentProfile,
        analysis: SemanticAnalysis
    ) -> float:
        """Calculate how well agent capabilities match task requirements"""
        required_capabilities = self._infer_required_capabilities(analysis)
        
        if not required_capabilities:
            return 0.5  # Neutral score if no specific requirements
        
        match_score = 0.0
        total_weight = 0.0
        
        for capability in required_capabilities:
            weight = self.capability_weights.get(capability, 1.0)
            total_weight += weight
            
            if capability in profile.capabilities:
                match_score += weight
        
        return match_score / total_weight if total_weight > 0 else 0.0
    
    def _infer_required_capabilities(self, analysis: SemanticAnalysis) -> Set[AgentCapability]:
        """Infer required capabilities from semantic analysis"""
        capabilities = set()
        
        # Based on coordination pattern
        if analysis.coordination_pattern == OrchestrationStrategy.HIERARCHICAL:
            capabilities.add(AgentCapability.COORDINATION)
            capabilities.add(AgentCapability.DECISION_MAKING)
        elif analysis.coordination_pattern == OrchestrationStrategy.CONSENSUS:
            capabilities.add(AgentCapability.DECISION_MAKING)
            capabilities.add(AgentCapability.ANALYSIS)
        elif analysis.coordination_pattern == OrchestrationStrategy.PARALLEL:
            capabilities.add(AgentCapability.EXECUTION)
            capabilities.add(AgentCapability.COORDINATION)
        
        # Based on urgency level
        if analysis.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            capabilities.add(AgentCapability.MONITORING)
            capabilities.add(AgentCapability.EXECUTION)
        
        # Based on complexity
        if analysis.complexity_score > 0.7:
            capabilities.add(AgentCapability.ANALYSIS)
            capabilities.add(AgentCapability.PATTERN_RECOGNITION)
        
        # Based on TDA correlation
        if analysis.tda_correlation and analysis.tda_correlation.anomaly_severity > 0.6:
            capabilities.add(AgentCapability.ANOMALY_DETECTION)
            capabilities.add(AgentCapability.MONITORING)
        
        return capabilities
    
    async def _apply_tda_adjustments(
        self,
        agent_scores: Dict[str, float],
        tda_context: TDAContext
    ) -> Dict[str, float]:
        """Apply TDA-based adjustments to agent scores"""
        adjusted_scores = agent_scores.copy()
        
        for agent_id, score in agent_scores.items():
            if agent_id not in self.agent_profiles:
                continue
                
            profile = self.agent_profiles[agent_id]
            
            # Boost agents with high TDA integration
            tda_boost = profile.tda_integration_level * 0.1
            adjusted_scores[agent_id] += tda_boost
            
            # Additional boost for anomaly detection capability during anomalies
            if (tda_context.anomaly_severity > 0.6 and 
                AgentCapability.ANOMALY_DETECTION in profile.capabilities):
                adjusted_scores[agent_id] += 0.15
            
            # Boost pattern recognition agents when patterns are strong
            if (tda_context.pattern_confidence > 0.8 and
                AgentCapability.PATTERN_RECOGNITION in profile.capabilities):
                adjusted_scores[agent_id] += 0.1
            
            # Normalize to [0, 1]
            adjusted_scores[agent_id] = min(adjusted_scores[agent_id], 1.0)
        
        return adjusted_scores
    
    def _select_agents_by_strategy(
        self,
        agent_scores: Dict[str, float],
        strategy: OrchestrationStrategy,
        max_agents: int
    ) -> List[str]:
        """Select agents based on orchestration strategy"""
        sorted_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if strategy == OrchestrationStrategy.SEQUENTIAL:
            # For sequential, prefer fewer high-performing agents
            return [agent for agent, _ in sorted_agents[:min(2, max_agents)]]
        
        elif strategy == OrchestrationStrategy.PARALLEL:
            # For parallel, use more agents for load distribution
            return [agent for agent, _ in sorted_agents[:max_agents]]
        
        elif strategy == OrchestrationStrategy.HIERARCHICAL:
            # For hierarchical, ensure we have coordination capability
            selected = []
            coordinator_found = False
            
            for agent_id, score in sorted_agents:
                if len(selected) >= max_agents:
                    break
                    
                profile = self.agent_profiles.get(agent_id)
                if profile:
                    # Prioritize coordinator if not found yet
                    if (not coordinator_found and 
                        AgentCapability.COORDINATION in profile.capabilities):
                        selected.insert(0, agent_id)  # Put coordinator first
                        coordinator_found = True
                    else:
                        selected.append(agent_id)
            
            return selected
        
        elif strategy == OrchestrationStrategy.CONSENSUS:
            # For consensus, prefer odd number of decision-making agents
            decision_makers = []
            others = []
            
            for agent_id, score in sorted_agents:
                profile = self.agent_profiles.get(agent_id)
                if profile and AgentCapability.DECISION_MAKING in profile.capabilities:
                    decision_makers.append(agent_id)
                else:
                    others.append(agent_id)
            
            # Select odd number of decision makers (3 or 5)
            target_decision_makers = 3 if max_agents >= 3 else 1
            selected = decision_makers[:target_decision_makers]
            
            # Fill remaining slots with other agents
            remaining_slots = max_agents - len(selected)
            selected.extend(others[:remaining_slots])
            
            return selected
        
        else:  # EVENT_DRIVEN or default
            # For event-driven, prefer monitoring and responsive agents
            monitoring_agents = []
            others = []
            
            for agent_id, score in sorted_agents:
                profile = self.agent_profiles.get(agent_id)
                if profile and AgentCapability.MONITORING in profile.capabilities:
                    monitoring_agents.append(agent_id)
                else:
                    others.append(agent_id)
            
            # Ensure at least one monitoring agent
            selected = monitoring_agents[:1] if monitoring_agents else []
            remaining_slots = max_agents - len(selected)
            selected.extend(others[:remaining_slots])
            
            return selected
    
    def _generate_fallback_agents(
        self,
        agent_scores: Dict[str, float],
        selected_agents: List[str],
        max_fallbacks: int
    ) -> List[str]:
        """Generate fallback agent options"""
        # Get agents not already selected, sorted by score
        fallback_candidates = [
            (agent_id, score) for agent_id, score in agent_scores.items()
            if agent_id not in selected_agents
        ]
        
        fallback_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [agent for agent, _ in fallback_candidates[:max_fallbacks]]
    
    def _calculate_routing_confidence(
        self,
        selected_agents: List[str],
        agent_scores: Dict[str, float],
        analysis: SemanticAnalysis
    ) -> float:
        """Calculate confidence in the routing decision"""
        if not selected_agents:
            return 0.0
        
        # Base confidence from agent scores
        avg_agent_score = sum(
            agent_scores.get(agent_id, 0.0) for agent_id in selected_agents
        ) / len(selected_agents)
        
        # Boost confidence with semantic analysis confidence
        semantic_confidence = analysis.confidence
        
        # Adjust for strategy appropriateness
        strategy_confidence = 0.8  # Base strategy confidence
        
        # Combine factors
        overall_confidence = (
            avg_agent_score * 0.4 +
            semantic_confidence * 0.4 +
            strategy_confidence * 0.2
        )
        
        return min(overall_confidence, 0.95)  # Cap at 95%
    
    def _generate_routing_reasoning(
        self,
        selected_agents: List[str],
        analysis: SemanticAnalysis,
        agent_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasoning_parts = []
        
        # Strategy reasoning
        strategy = analysis.coordination_pattern.value
        complexity = analysis.complexity_score
        urgency = analysis.urgency_level.value
        
        reasoning_parts.append(
            f"Selected {strategy} strategy based on complexity ({complexity:.2f}) "
            f"and urgency ({urgency})"
        )
        
        # Agent selection reasoning
        if selected_agents:
            top_agent = max(selected_agents, key=lambda a: agent_scores.get(a, 0))
            top_score = agent_scores.get(top_agent, 0)
            reasoning_parts.append(
                f"Top agent '{top_agent}' selected with score {top_score:.2f}"
            )
        
        # TDA influence reasoning
        if analysis.tda_correlation:
            if analysis.tda_correlation.anomaly_severity > 0.6:
                reasoning_parts.append(
                    "TDA anomaly detection influenced agent selection for monitoring"
                )
            if analysis.tda_correlation.pattern_confidence > 0.8:
                reasoning_parts.append(
                    "Strong TDA pattern correlation boosted analytical agents"
                )
        
        return ". ".join(reasoning_parts)
    
    def _estimate_routing_performance(
        self,
        selected_agents: List[str],
        analysis: SemanticAnalysis
    ) -> float:
        """Estimate expected performance of the routing decision"""
        if not selected_agents:
            return 0.0
        
        # Base performance from agent profiles
        agent_performances = []
        for agent_id in selected_agents:
            profile = self.agent_profiles.get(agent_id)
            if profile:
                # Combine performance score and success rate
                perf = (profile.performance_score + profile.last_success_rate) / 2
                agent_performances.append(perf)
        
        if not agent_performances:
            return 0.5
        
        # Strategy-based performance adjustment
        strategy_multiplier = {
            OrchestrationStrategy.SEQUENTIAL: 0.9,  # Single point of failure risk
            OrchestrationStrategy.PARALLEL: 1.1,   # Redundancy benefit
            OrchestrationStrategy.HIERARCHICAL: 1.0,  # Balanced
            OrchestrationStrategy.CONSENSUS: 0.95,  # Coordination overhead
            OrchestrationStrategy.EVENT_DRIVEN: 1.05  # Responsive benefit
        }.get(analysis.coordination_pattern, 1.0)
        
        base_performance = sum(agent_performances) / len(agent_performances)
        return min(base_performance * strategy_multiplier, 1.0)
    
    async def _record_routing_decision(
        self,
        decision: RoutingDecision,
        analysis: SemanticAnalysis
    ):
        """Record routing decision for learning and improvement"""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "selected_agents": decision.selected_agents,
            "strategy": decision.routing_strategy.value,
            "confidence": decision.confidence,
            "complexity": analysis.complexity_score,
            "urgency": analysis.urgency_level.value,
            "tda_influence": decision.tda_influence,
            "estimated_performance": decision.estimated_performance
        }
        
        self.routing_history.append(record)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    async def update_agent_performance(
        self,
        agent_id: str,
        performance_metrics: Dict[str, float]
    ):
        """Update agent performance based on execution results"""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        
        # Update success rate
        if "success_rate" in performance_metrics:
            profile.last_success_rate = performance_metrics["success_rate"]
        
        # Update performance score (weighted average)
        if "performance_score" in performance_metrics:
            new_score = performance_metrics["performance_score"]
            profile.performance_score = (
                profile.performance_score * 0.7 + new_score * 0.3
            )
        
        # Update current load
        if "current_load" in performance_metrics:
            profile.current_load = performance_metrics["current_load"]
        
        # Update availability
        if "availability" in performance_metrics:
            profile.availability = performance_metrics["availability"]
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions"""
        if not self.routing_history:
            return {"message": "No routing history available"}
        
        recent_decisions = self.routing_history[-100:]  # Last 100 decisions
        
        # Calculate statistics
        avg_confidence = sum(d["confidence"] for d in recent_decisions) / len(recent_decisions)
        avg_performance = sum(d["estimated_performance"] for d in recent_decisions) / len(recent_decisions)
        
        # Strategy distribution
        strategy_counts = {}
        for decision in recent_decisions:
            strategy = decision["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Agent usage frequency
        agent_usage = {}
        for decision in recent_decisions:
            for agent in decision["selected_agents"]:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            "total_decisions": len(self.routing_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": avg_confidence,
            "average_estimated_performance": avg_performance,
            "strategy_distribution": strategy_counts,
            "agent_usage_frequency": agent_usage,
            "active_agents": len([a for a in self.agent_profiles.values() if a.availability])
        }