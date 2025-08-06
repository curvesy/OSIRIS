"""
🧩 Semantic Pattern Matcher

Advanced semantic pattern matching with TDA correlation for intelligent
orchestration decisions. Implements 2025 patterns for complexity analysis,
urgency scoring, and coordination pattern selection.

Key Features:
- TDA pattern correlation and amplification
- ML-ready complexity calculation algorithms
- Dynamic urgency scoring with anomaly awareness
- Coordination pattern recommendation engine

TDA Integration:
- Correlates semantic patterns with TDA anomaly data
- Uses TDA confidence scores for pattern amplification
- Integrates with TDA temporal windows for context
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import math
from datetime import datetime, timedelta, timezone

from .base_interfaces import (
    TDAContext, SemanticAnalysis, OrchestrationStrategy, 
    UrgencyLevel, TDAIntegration
)

# TDA integration
try:
    from aura_intelligence.observability.tracing import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None

class SemanticPatternMatcher:
    """
    2025 semantic pattern matching with TDA integration
    """
    
    def __init__(self, tda_integration: Optional[TDAIntegration] = None):
        self.tda_integration = tda_integration
        self.pattern_cache: Dict[str, Any] = {}
        self.complexity_weights = {
            "data_size": 0.3,
            "requirements_count": 0.25,
            "agent_dependencies": 0.2,
            "temporal_constraints": 0.15,
            "consensus_required": 0.1
        }
    
    async def analyze_semantic_patterns(
        self, 
        input_data: Dict[str, Any],
        tda_context: Optional[TDAContext] = None
    ) -> SemanticAnalysis:
        """
        Comprehensive semantic pattern analysis with TDA correlation
        """
        if tracer:
            with tracer.start_as_current_span("semantic_pattern_analysis") as span:
                span.set_attributes({
                    "pattern.analysis_type": "comprehensive",
                    "tda.context_available": tda_context is not None
                })
        
        # Core semantic analysis
        complexity_score = await self._calculate_complexity_score(input_data)
        urgency_level = await self._determine_urgency_level(input_data, tda_context)
        coordination_pattern = await self._select_coordination_pattern(
            complexity_score, urgency_level, tda_context
        )
        suggested_agents = await self._suggest_optimal_agents(
            input_data, tda_context
        )
        
        # TDA amplification
        if tda_context:
            complexity_score = self._amplify_with_tda(complexity_score, tda_context)
            urgency_level = self._escalate_urgency_with_tda(urgency_level, tda_context)
        
        # Calculate overall confidence
        confidence = self._calculate_analysis_confidence(
            complexity_score, urgency_level, tda_context
        )
        
        return SemanticAnalysis(
            complexity_score=min(complexity_score, 1.0),
            urgency_level=urgency_level,
            coordination_pattern=coordination_pattern,
            suggested_agents=suggested_agents,
            confidence=confidence,
            tda_correlation=tda_context
        )
    
    async def _calculate_complexity_score(self, input_data: Dict[str, Any]) -> float:
        """
        Calculate semantic complexity using weighted factors
        """
        factors = {}
        
        # Data size complexity
        data_str = str(input_data)
        factors["data_size"] = min(len(data_str) / 5000, 1.0)
        
        # Requirements complexity
        requirements = input_data.get("requirements", [])
        factors["requirements_count"] = min(len(requirements) / 20, 1.0)
        
        # Agent dependencies complexity
        dependencies = input_data.get("agent_dependencies", [])
        factors["agent_dependencies"] = min(len(dependencies) / 10, 1.0)
        
        # Temporal constraints complexity
        has_deadlines = bool(input_data.get("deadline") or input_data.get("timeout"))
        factors["temporal_constraints"] = 1.0 if has_deadlines else 0.0
        
        # Consensus requirement complexity
        factors["consensus_required"] = 1.0 if input_data.get("requires_consensus") else 0.0
        
        # Weighted sum
        complexity_score = sum(
            factors[factor] * self.complexity_weights[factor]
            for factor in factors
        )
        
        return complexity_score
    
    async def _determine_urgency_level(
        self, 
        input_data: Dict[str, Any],
        tda_context: Optional[TDAContext]
    ) -> UrgencyLevel:
        """
        Determine urgency level with TDA anomaly amplification
        """
        base_urgency = input_data.get("urgency", "medium").lower()
        
        # Check for explicit urgency indicators
        if input_data.get("emergency") or input_data.get("critical"):
            base_urgency = "critical"
        elif input_data.get("deadline"):
            # Check if deadline is soon
            try:
                deadline = datetime.fromisoformat(input_data["deadline"])
                time_to_deadline = deadline - datetime.now(timezone.utc)
                if time_to_deadline < timedelta(hours=1):
                    base_urgency = "critical"
                elif time_to_deadline < timedelta(hours=6):
                    base_urgency = "high"
            except (ValueError, TypeError):
                pass
        
        # TDA anomaly amplification
        if tda_context:
            if tda_context.anomaly_severity > 0.9:
                return UrgencyLevel.CRITICAL
            elif tda_context.anomaly_severity > 0.7:
                # Escalate urgency by one level
                urgency_escalation = {
                    "low": UrgencyLevel.MEDIUM,
                    "medium": UrgencyLevel.HIGH,
                    "high": UrgencyLevel.CRITICAL,
                    "critical": UrgencyLevel.CRITICAL
                }
                return urgency_escalation.get(base_urgency, UrgencyLevel.MEDIUM)
        
        return UrgencyLevel(base_urgency)
    
    async def _select_coordination_pattern(
        self,
        complexity_score: float,
        urgency_level: UrgencyLevel,
        tda_context: Optional[TDAContext]
    ) -> OrchestrationStrategy:
        """
        Select optimal coordination pattern based on analysis
        """
        # High complexity patterns
        if complexity_score > 0.8:
            if urgency_level in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
                return OrchestrationStrategy.PARALLEL
            else:
                return OrchestrationStrategy.HIERARCHICAL
        
        # Medium complexity patterns
        elif complexity_score > 0.5:
            if urgency_level == UrgencyLevel.CRITICAL:
                return OrchestrationStrategy.PARALLEL
            else:
                return OrchestrationStrategy.CONSENSUS
        
        # Low complexity patterns
        else:
            if urgency_level == UrgencyLevel.CRITICAL:
                return OrchestrationStrategy.SEQUENTIAL  # Fast execution
            else:
                return OrchestrationStrategy.EVENT_DRIVEN
    
    async def _suggest_optimal_agents(
        self,
        input_data: Dict[str, Any],
        tda_context: Optional[TDAContext]
    ) -> List[str]:
        """
        Suggest optimal agents based on semantic analysis and TDA context
        """
        suggested_agents = []
        
        # Analyze task requirements
        task_type = input_data.get("task_type", "general")
        domain = input_data.get("domain", "general")
        
        # Basic agent selection logic (can be enhanced with ML)
        if task_type == "analysis":
            suggested_agents.extend(["analyst_agent", "observer_agent"])
        elif task_type == "decision":
            suggested_agents.extend(["supervisor_agent", "analyst_agent"])
        elif task_type == "execution":
            suggested_agents.extend(["executor_agent", "observer_agent"])
        else:
            suggested_agents.extend(["supervisor_agent", "analyst_agent", "observer_agent"])
        
        # TDA-based agent selection enhancement
        if tda_context and tda_context.current_patterns:
            # If anomalies detected, include observer for monitoring
            if tda_context.anomaly_severity > 0.6:
                if "observer_agent" not in suggested_agents:
                    suggested_agents.append("observer_agent")
            
            # If complex patterns detected, include analyst
            if tda_context.pattern_confidence > 0.8:
                if "analyst_agent" not in suggested_agents:
                    suggested_agents.append("analyst_agent")
        
        return suggested_agents[:5]  # Limit to 5 agents for performance
    
    def _amplify_with_tda(self, complexity_score: float, tda_context: TDAContext) -> float:
        """
        Amplify complexity score with TDA pattern confidence
        """
        amplification_factor = 1 + (tda_context.pattern_confidence * 0.3)
        return complexity_score * amplification_factor
    
    def _escalate_urgency_with_tda(
        self, 
        urgency_level: UrgencyLevel, 
        tda_context: TDAContext
    ) -> UrgencyLevel:
        """
        Escalate urgency based on TDA anomaly severity
        """
        if tda_context.anomaly_severity > 0.9:
            return UrgencyLevel.CRITICAL
        elif tda_context.anomaly_severity > 0.7:
            # Escalate by one level
            escalation_map = {
                UrgencyLevel.LOW: UrgencyLevel.MEDIUM,
                UrgencyLevel.MEDIUM: UrgencyLevel.HIGH,
                UrgencyLevel.HIGH: UrgencyLevel.CRITICAL,
                UrgencyLevel.CRITICAL: UrgencyLevel.CRITICAL
            }
            return escalation_map[urgency_level]
        
        return urgency_level
    
    def _calculate_analysis_confidence(
        self,
        complexity_score: float,
        urgency_level: UrgencyLevel,
        tda_context: Optional[TDAContext]
    ) -> float:
        """
        Calculate confidence in the semantic analysis
        """
        base_confidence = 0.7  # Base confidence level
        
        # Boost confidence with TDA correlation
        if tda_context:
            tda_boost = tda_context.pattern_confidence * 0.2
            base_confidence += tda_boost
        
        # Adjust for complexity clarity
        if 0.3 <= complexity_score <= 0.7:
            base_confidence += 0.1  # Clear complexity signals
        
        # Adjust for urgency clarity
        if urgency_level in [UrgencyLevel.LOW, UrgencyLevel.CRITICAL]:
            base_confidence += 0.05  # Clear urgency signals
        
        return min(base_confidence, 0.95)  # Cap at 95%
    
    async def get_pattern_insights(
        self, 
        analysis: SemanticAnalysis
    ) -> Dict[str, Any]:
        """
        Get detailed insights about the semantic patterns
        """
        insights = {
            "complexity_breakdown": {
                "score": analysis.complexity_score,
                "level": "high" if analysis.complexity_score > 0.7 else 
                        "medium" if analysis.complexity_score > 0.4 else "low"
            },
            "urgency_analysis": {
                "level": analysis.urgency_level.value,
                "tda_influenced": analysis.tda_correlation is not None
            },
            "coordination_rationale": {
                "pattern": analysis.coordination_pattern.value,
                "reason": self._get_coordination_rationale(analysis)
            },
            "agent_selection": {
                "suggested": analysis.suggested_agents,
                "count": len(analysis.suggested_agents)
            },
            "confidence_factors": {
                "overall": analysis.confidence,
                "tda_correlation": analysis.tda_correlation.pattern_confidence if analysis.tda_correlation else 0.0
            }
        }
        
        return insights
    
    def _get_coordination_rationale(self, analysis: SemanticAnalysis) -> str:
        """Get human-readable rationale for coordination pattern selection"""
        pattern = analysis.coordination_pattern
        complexity = analysis.complexity_score
        urgency = analysis.urgency_level
        
        if pattern == OrchestrationStrategy.PARALLEL:
            return f"High complexity ({complexity:.2f}) or critical urgency ({urgency.value}) requires parallel execution"
        elif pattern == OrchestrationStrategy.HIERARCHICAL:
            return f"High complexity ({complexity:.2f}) with moderate urgency benefits from hierarchical coordination"
        elif pattern == OrchestrationStrategy.CONSENSUS:
            return f"Medium complexity ({complexity:.2f}) requires consensus for optimal results"
        elif pattern == OrchestrationStrategy.SEQUENTIAL:
            return f"Low complexity ({complexity:.2f}) with critical urgency ({urgency.value}) needs fast sequential execution"
        else:
            return f"Event-driven pattern selected for flexible coordination"