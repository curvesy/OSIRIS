"""
🎯 Supervisor Node
Decision-making and coordination for workflow orchestration.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from enum import Enum

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from aura_common.logging import get_logger, with_correlation_id
from aura_common.errors import resilient_operation
from aura_common.config import is_feature_enabled

from ..state import CollectiveState, NodeResult

logger = get_logger(__name__)


class DecisionType(str, Enum):
    """Types of supervisor decisions."""
    CONTINUE = "continue"
    ESCALATE = "escalate"
    RETRY = "retry"
    COMPLETE = "complete"
    ABORT = "abort"


class SupervisorNode:
    """
    Supervisor node for workflow coordination.
    
    Responsibilities:
    - Evaluate evidence and analysis
    - Make routing decisions
    - Assess risks
    - Coordinate agent consensus
    """
    
    def __init__(self, llm=None, risk_threshold: float = 0.7):
        """
        Initialize supervisor node.
        
        Args:
            llm: Optional LLM for decision making
            risk_threshold: Threshold for risk escalation
        """
        self.llm = llm
        self.risk_threshold = risk_threshold
        self.name = "supervisor"
    
    @with_correlation_id()
    @resilient_operation(
        "supervisor_node",
        failure_threshold=3,
        recovery_timeout=30
    )
    async def __call__(
        self,
        state: CollectiveState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute supervisor decision logic.
        
        Args:
            state: Current workflow state
            config: Optional runtime configuration
            
        Returns:
            Updated state with decision
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Supervisor node starting",
                workflow_id=state["workflow_id"],
                thread_id=state["thread_id"],
                evidence_count=len(state.get("evidence_log", []))
            )
            
            # Analyze current state
            analysis = self._analyze_state(state)
            
            # Assess risk
            risk_score = self._assess_risk(state, analysis)
            
            # Make decision
            decision = await self._make_decision(state, analysis, risk_score)
            
            # Build decision record
            decision_record = self._build_decision_record(
                decision, analysis, risk_score
            )
            
            # Create result
            result = NodeResult(
                success=True,
                node_name=self.name,
                output=decision_record,
                duration_ms=(time.time() - start_time) * 1000,
                next_node=self._determine_next_node(decision)
            )
            
            # Update state
            updates = {
                "supervisor_decisions": [decision_record],
                "current_step": f"supervisor_decided_{decision.value}",
                "risk_assessment": {
                    "score": risk_score,
                    "threshold": self.risk_threshold,
                    "high_risk": risk_score > self.risk_threshold
                }
            }
            
            # Add message
            message = AIMessage(
                content=f"Supervisor decision: {decision.value} (risk: {risk_score:.2f})",
                additional_kwargs={"node": self.name, "decision": decision.value}
            )
            updates["messages"] = [message]
            
            logger.info(
                "Supervisor decision made",
                workflow_id=state["workflow_id"],
                decision=decision.value,
                risk_score=risk_score,
                duration_ms=result.duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error(
                "Supervisor node failed",
                workflow_id=state["workflow_id"],
                error=str(e),
                exc_info=e
            )
            
            return {
                "error_log": [{
                    "node": self.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }],
                "last_error": {
                    "node": self.name,
                    "message": str(e)
                },
                "current_step": "supervisor_error"
            }
    
    def _analyze_state(self, state: CollectiveState) -> Dict[str, Any]:
        """Analyze current workflow state."""
        analysis = {
            "evidence_count": len(state.get("evidence_log", [])),
            "error_count": len(state.get("error_log", [])),
            "decision_count": len(state.get("supervisor_decisions", [])),
            "has_risk_indicators": False,
            "completion_indicators": []
        }
        
        # Check for risk indicators in evidence
        for evidence in state.get("evidence_log", []):
            if evidence.get("risk_indicators"):
                analysis["has_risk_indicators"] = True
                analysis["risk_indicators"] = evidence["risk_indicators"]
                break
        
        # Check for completion indicators
        if state.get("execution_results"):
            analysis["completion_indicators"].append("execution_complete")
        
        if state.get("validation_results", {}).get("valid"):
            analysis["completion_indicators"].append("validation_passed")
        
        return analysis
    
    def _assess_risk(
        self,
        state: CollectiveState,
        analysis: Dict[str, Any]
    ) -> float:
        """Assess risk score based on state and analysis."""
        risk_score = 0.0
        
        # Error-based risk
        error_count = analysis["error_count"]
        if error_count > 0:
            risk_score += min(0.3, error_count * 0.1)
        
        # Risk indicators
        if analysis["has_risk_indicators"]:
            risk_indicators = analysis.get("risk_indicators", [])
            if "high_error_rate" in risk_indicators:
                risk_score += 0.3
            if "high_cpu_usage" in risk_indicators:
                risk_score += 0.2
            if "high_memory_usage" in risk_indicators:
                risk_score += 0.2
        
        # Recovery attempts
        recovery_attempts = state.get("error_recovery_attempts", 0)
        if recovery_attempts > 2:
            risk_score += 0.2
        
        # Cap at 1.0
        return min(1.0, risk_score)
    
    async def _make_decision(
        self,
        state: CollectiveState,
        analysis: Dict[str, Any],
        risk_score: float
    ) -> DecisionType:
        """Make supervisor decision."""
        # High risk - escalate
        if risk_score > self.risk_threshold:
            return DecisionType.ESCALATE
        
        # Errors with low recovery attempts - retry
        if (analysis["error_count"] > 0 and 
            state.get("error_recovery_attempts", 0) < 3):
            return DecisionType.RETRY
        
        # Completion indicators - complete
        if len(analysis["completion_indicators"]) >= 2:
            return DecisionType.COMPLETE
        
        # Too many errors - abort
        if analysis["error_count"] > 5:
            return DecisionType.ABORT
        
        # Default - continue
        return DecisionType.CONTINUE
    
    def _build_decision_record(
        self,
        decision: DecisionType,
        analysis: Dict[str, Any],
        risk_score: float
    ) -> Dict[str, Any]:
        """Build decision record for audit trail."""
        return {
            "node": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision.value,
            "risk_score": risk_score,
            "analysis": analysis,
            "reasoning": self._get_decision_reasoning(decision, analysis, risk_score)
        }
    
    def _get_decision_reasoning(
        self,
        decision: DecisionType,
        analysis: Dict[str, Any],
        risk_score: float
    ) -> str:
        """Get human-readable reasoning for decision."""
        if decision == DecisionType.ESCALATE:
            return f"Risk score {risk_score:.2f} exceeds threshold {self.risk_threshold}"
        elif decision == DecisionType.RETRY:
            return f"Errors detected ({analysis['error_count']}), attempting recovery"
        elif decision == DecisionType.COMPLETE:
            return f"Completion criteria met: {', '.join(analysis['completion_indicators'])}"
        elif decision == DecisionType.ABORT:
            return f"Too many errors ({analysis['error_count']}), aborting workflow"
        else:
            return "Continuing normal workflow execution"
    
    def _determine_next_node(self, decision: DecisionType) -> Optional[str]:
        """Determine next node based on decision."""
        if decision == DecisionType.CONTINUE:
            return "analyst"
        elif decision == DecisionType.RETRY:
            return "observer"
        elif decision == DecisionType.COMPLETE:
            return "end"
        elif decision == DecisionType.ESCALATE:
            return "human_review"
        elif decision == DecisionType.ABORT:
            return "error_handler"
        return None


# Factory function
def create_supervisor_node(
    llm=None,
    risk_threshold: float = 0.7
) -> SupervisorNode:
    """
    Create a supervisor node instance.
    
    Args:
        llm: Optional LLM for decision making
        risk_threshold: Risk threshold for escalation
        
    Returns:
        Configured supervisor node
    """
    return SupervisorNode(llm=llm, risk_threshold=risk_threshold)