"""
🔍 Analyst Node
Pattern analysis and insight generation for workflows.
"""

from typing import Dict, Any, List, Optional, Protocol
from datetime import datetime, timezone
import time
import numpy as np

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from aura_common.logging import get_logger, with_correlation_id
from aura_common import resilient_operation
from aura_common.config import is_feature_enabled

from ..state import CollectiveState, NodeResult
from ....tda.unified_engine import UnifiedTDAEngine, TDARequest

logger = get_logger(__name__)


class AnalysisStrategy(Protocol):
    """Protocol for analysis strategies."""
    
    async def analyze(
        self,
        evidence: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform analysis on evidence."""
        ...


class AnalystNode:
    """
    Analyst node for pattern detection and insights.
    
    Responsibilities:
    - Analyze evidence patterns
    - Perform TDA analysis when applicable
    - Generate actionable insights
    - Provide recommendations
    
    Contract:
    - Input: CollectiveState with evidence_log
    - Output: Dict with analysis results and recommendations
    - Side effects: May query memory systems, perform TDA
    """
    
    def __init__(
        self,
        llm=None,
        tda_engine: Optional[UnifiedTDAEngine] = None,
        strategies: Optional[List[AnalysisStrategy]] = None
    ):
        """
        Initialize analyst node.
        
        Args:
            llm: Optional LLM for advanced analysis
            tda_engine: TDA engine for topological analysis
            strategies: Custom analysis strategies
        """
        self.llm = llm
        self.tda_engine = tda_engine or UnifiedTDAEngine()
        self.strategies = strategies or []
        self.name = "analyst"
    
    @with_correlation_id()
    @resilient_operation(
        max_retries=3,
        delay=1.0,
        backoff_factor=2.0
    )
    async def __call__(
        self,
        state: CollectiveState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute analyst logic.
        
        Args:
            state: Current workflow state
            config: Optional runtime configuration
            
        Returns:
            Updated state with analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Analyst node starting",
                workflow_id=state["workflow_id"],
                thread_id=state["thread_id"],
                evidence_count=len(state.get("evidence_log", []))
            )
            
            # Extract evidence for analysis
            evidence = state.get("evidence_log", [])
            
            # Perform pattern analysis
            patterns = await self._analyze_patterns(evidence, state)
            
            # Perform TDA if applicable
            tda_results = None
            if is_feature_enabled("tda_analysis") and self._should_run_tda(evidence):
                tda_results = await self._run_tda_analysis(evidence)
            
            # Generate insights
            insights = self._generate_insights(patterns, tda_results)
            
            # Create recommendations
            recommendations = self._create_recommendations(insights, state)
            
            # Build analysis result
            analysis_result = {
                "node": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patterns": patterns,
                "tda_results": tda_results,
                "insights": insights,
                "recommendations": recommendations
            }
            
            # Update state
            updates = {
                "analysis_results": [analysis_result],
                "current_step": "analysis_complete"
            }
            
            # Add message
            message = AIMessage(
                content=f"Analysis complete: {len(insights)} insights, {len(recommendations)} recommendations",
                additional_kwargs={"node": self.name}
            )
            updates["messages"] = [message]
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Analyst node completed",
                workflow_id=state["workflow_id"],
                insights_count=len(insights),
                duration_ms=duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error(
                "Analyst node failed",
                workflow_id=state["workflow_id"],
                error=str(e),
                exc_info=e
            )
            
            return {
                "error_log": [{
                    "node": self.name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }],
                "current_step": "analyst_error"
            }
    
    async def _analyze_patterns(
        self,
        evidence: List[Dict[str, Any]],
        state: CollectiveState
    ) -> Dict[str, Any]:
        """Analyze patterns in evidence."""
        patterns = {
            "temporal": [],
            "statistical": [],
            "anomalies": []
        }
        
        # Temporal patterns
        if len(evidence) > 1:
            patterns["temporal"] = self._find_temporal_patterns(evidence)
        
        # Statistical patterns
        patterns["statistical"] = self._find_statistical_patterns(evidence)
        
        # Anomaly detection
        patterns["anomalies"] = self._detect_anomalies(evidence)
        
        # Apply custom strategies
        for strategy in self.strategies:
            custom_patterns = await strategy.analyze(evidence, {"state": state})
            patterns.update(custom_patterns)
        
        return patterns
    
    def _should_run_tda(self, evidence: List[Dict[str, Any]]) -> bool:
        """Determine if TDA analysis should run."""
        # Run TDA if we have numerical data points
        for e in evidence:
            if "data_points" in e or "metrics" in e:
                return True
        return False
    
    async def _run_tda_analysis(
        self,
        evidence: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Run TDA analysis on evidence data."""
        try:
            # Extract numerical data
            data_points = []
            for e in evidence:
                if "metrics" in e:
                    # Convert metrics to points
                    metrics = e["metrics"]
                    point = [
                        metrics.get("cpu_usage", 0),
                        metrics.get("memory_usage", 0),
                        metrics.get("error_rate", 0)
                    ]
                    data_points.append(point)
            
            if not data_points:
                return None
            
            # Run TDA
            data = np.array(data_points)
            response = await self.tda_engine.analyze(
                data,
                max_dimension=2
            )
            
            return {
                "algorithm": response.algorithm_used,
                "computation_time_ms": response.computation_time_ms,
                "persistence_features": len(response.persistence_diagrams.get(1, [])),
                "topological_summary": "Detected persistent features"
            }
            
        except Exception as e:
            logger.warning(f"TDA analysis failed: {e}")
            return None
    
    def _find_temporal_patterns(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find temporal patterns in evidence."""
        # Placeholder - implement temporal analysis
        return [{"type": "periodic", "confidence": 0.8}]
    
    def _find_statistical_patterns(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find statistical patterns."""
        # Placeholder - implement statistical analysis
        return [{"type": "trending", "direction": "stable"}]
    
    def _detect_anomalies(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in evidence."""
        anomalies = []
        for e in evidence:
            if e.get("risk_indicators"):
                anomalies.append({
                    "type": "risk_indicator",
                    "severity": "medium",
                    "details": e["risk_indicators"]
                })
        return anomalies
    
    def _generate_insights(
        self,
        patterns: Dict[str, Any],
        tda_results: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate insights from analysis."""
        insights = []
        
        # Pattern-based insights
        if patterns["anomalies"]:
            insights.append({
                "type": "anomaly_detected",
                "priority": "high",
                "description": f"Detected {len(patterns['anomalies'])} anomalies"
            })
        
        # TDA-based insights
        if tda_results and tda_results.get("persistence_features", 0) > 5:
            insights.append({
                "type": "complex_topology",
                "priority": "medium",
                "description": "Complex topological structure detected"
            })
        
        return insights
    
    def _create_recommendations(
        self,
        insights: List[Dict[str, Any]],
        state: CollectiveState
    ) -> List[Dict[str, Any]]:
        """Create actionable recommendations."""
        recommendations = []
        
        for insight in insights:
            if insight["type"] == "anomaly_detected":
                recommendations.append({
                    "action": "investigate_anomalies",
                    "priority": insight["priority"],
                    "rationale": insight["description"]
                })
            elif insight["type"] == "complex_topology":
                recommendations.append({
                    "action": "deep_analysis",
                    "priority": "medium",
                    "rationale": "Complex patterns require deeper investigation"
                })
        
        return recommendations


# Factory function
def create_analyst_node(
    llm=None,
    tda_engine: Optional[UnifiedTDAEngine] = None,
    strategies: Optional[List[AnalysisStrategy]] = None
) -> AnalystNode:
    """
    Create an analyst node instance.
    
    Args:
        llm: Optional LLM for analysis
        tda_engine: TDA engine for topological analysis
        strategies: Custom analysis strategies
        
    Returns:
        Configured analyst node
    """
    return AnalystNode(llm=llm, tda_engine=tda_engine, strategies=strategies)