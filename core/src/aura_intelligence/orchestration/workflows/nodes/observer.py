"""
👁️ Observer Node
Evidence collection and system observation for workflows.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import time

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from aura_common.logging import get_logger, with_correlation_id
from aura_common import resilient_operation
from aura_common.config import is_feature_enabled

from ..state import CollectiveState, NodeResult, update_state_safely

logger = get_logger(__name__)


class ObserverNode:
    """
    Observer node for evidence collection.
    
    Responsibilities:
    - Collect system evidence
    - Monitor health metrics
    - Log observations
    - Detect anomalies
    """
    
    def __init__(self, llm=None):
        """
        Initialize observer node.
        
        Args:
            llm: Optional LLM for advanced analysis
        """
        self.llm = llm
        self.name = "observer"
    
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
        Execute observer node logic.
        
        Args:
            state: Current workflow state
            config: Optional runtime configuration
            
        Returns:
            Updated state with observations
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Observer node starting",
                workflow_id=state["workflow_id"],
                thread_id=state["thread_id"],
                current_step=state["current_step"]
            )
            
            # Collect evidence
            evidence = await self._collect_evidence(state)
            
            # Analyze if LLM available
            analysis = None
            if self.llm and is_feature_enabled("llm_analysis"):
                analysis = await self._analyze_with_llm(evidence, state)
            
            # Build observation
            observation = self._build_observation(evidence, analysis)
            
            # Create result
            result = NodeResult(
                success=True,
                node_name=self.name,
                output=observation,
                duration_ms=(time.time() - start_time) * 1000,
                next_node="supervisor"  # Default routing
            )
            
            # Update state
            updates = {
                "evidence_log": [observation],
                "current_step": "observation_complete",
                "system_health": self._get_system_health()
            }
            
            # Add message if we have analysis
            if analysis:
                message = AIMessage(
                    content=f"Observation: {analysis['summary']}",
                    additional_kwargs={"node": self.name}
                )
                updates["messages"] = [message]
            
            logger.info(
                "Observer node completed",
                workflow_id=state["workflow_id"],
                evidence_count=len(evidence),
                duration_ms=result.duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error(
                "Observer node failed",
                workflow_id=state["workflow_id"],
                error=str(e),
                exc_info=e
            )
            
            # Return error state
            return {
                "error_log": [{
                    "node": self.name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }],
                "last_error": {
                    "node": self.name,
                    "message": str(e)
                },
                "current_step": "observer_error"
            }
    
    async def _collect_evidence(
        self,
        state: CollectiveState
    ) -> List[Dict[str, Any]]:
        """Collect evidence from various sources."""
        evidence = []
        
        # System metrics
        evidence.append({
            "type": "system_metrics",
            "data": self._get_system_health(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Message history analysis
        if state["messages"]:
            evidence.append({
                "type": "message_analysis",
                "data": {
                    "message_count": len(state["messages"]),
                    "last_message": state["messages"][-1].content[:100]
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Previous decisions
        if state["supervisor_decisions"]:
            evidence.append({
                "type": "decision_history",
                "data": {
                    "decision_count": len(state["supervisor_decisions"]),
                    "last_decision": state["supervisor_decisions"][-1]
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Error patterns
        if state["error_log"]:
            evidence.append({
                "type": "error_patterns",
                "data": {
                    "error_count": len(state["error_log"]),
                    "recovery_attempts": state["error_recovery_attempts"]
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return evidence
    
    async def _analyze_with_llm(
        self,
        evidence: List[Dict[str, Any]],
        state: CollectiveState
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze evidence."""
        if not self.llm:
            return None
        
        try:
            # Prepare prompt
            evidence_summary = "\n".join([
                f"- {e['type']}: {e['data']}"
                for e in evidence
            ])
            
            prompt = f"""
            Analyze the following system evidence and provide insights:
            
            {evidence_summary}
            
            Provide:
            1. Key observations
            2. Potential risks
            3. Recommended actions
            """
            
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            
            return {
                "summary": response.content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.warning(
                "LLM analysis failed",
                error=str(e)
            )
            return None
    
    def _build_observation(
        self,
        evidence: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build observation from evidence and analysis."""
        observation = {
            "node": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence,
            "evidence_count": len(evidence)
        }
        
        if analysis:
            observation["analysis"] = analysis
        
        # Add risk indicators
        risk_indicators = []
        for e in evidence:
            if e["type"] == "error_patterns" and e["data"]["error_count"] > 5:
                risk_indicators.append("high_error_rate")
            if e["type"] == "system_metrics":
                metrics = e["data"]
                if metrics.get("cpu_usage", 0) > 0.8:
                    risk_indicators.append("high_cpu_usage")
                if metrics.get("memory_usage", 0) > 0.8:
                    risk_indicators.append("high_memory_usage")
        
        if risk_indicators:
            observation["risk_indicators"] = risk_indicators
        
        return observation
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        # This would integrate with real monitoring
        # For now, return mock data
        return {
            "cpu_usage": 0.45,
            "memory_usage": 0.62,
            "active_connections": 42,
            "error_rate": 0.02,
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Factory function for node creation
def create_observer_node(llm=None) -> ObserverNode:
    """
    Create an observer node instance.
    
    Args:
        llm: Optional LLM for analysis
        
    Returns:
        Configured observer node
    """
    return ObserverNode(llm=llm)