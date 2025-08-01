#!/usr/bin/env python3
"""
🧠 Collective Supervisor - LangGraph Central Intelligence

Professional supervisor node for multi-agent coordination.
Implements the latest LangGraph patterns with context engineering.
"""

import logging
from typing import Dict, Any, Literal
from pathlib import Path
import sys

# Import schemas
schema_dir = Path(__file__).parent.parent / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

try:
    import enums
    import base
    from production_observer_agent import ProductionAgentState
except ImportError:
    # Fallback for testing
    class ProductionAgentState:
        def __init__(self): pass

logger = logging.getLogger(__name__)


class CollectiveSupervisor:
    """
    Professional LangGraph Supervisor implementing central intelligence.
    
    The supervisor is the brain of the collective - it:
    1. Receives enriched state from context engine
    2. Makes intelligent routing decisions
    3. Coordinates agent interactions
    4. Manages workflow completion
    """
    
    def __init__(self, memory_manager, context_engine):
        self.memory_manager = memory_manager
        self.context_engine = context_engine
        self.agent_id = "collective_supervisor"
        
        logger.info("🧠 Collective Supervisor initialized")
    
    async def supervisor_node(self, state: Any) -> Any:
        """
        Main supervisor node for LangGraph.
        
        This performs context engineering and prepares state for routing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Enriched state ready for routing decision
        """
        
        logger.info(f"🧠 Supervisor processing: {getattr(state, 'workflow_id', 'unknown')}")
        
        try:
            # Step 1: Query collective memory for relevant context
            memory_context = await self.memory_manager.query_relevant_context(state)
            
            # Step 2: Perform context engineering
            enriched_state = await self.context_engine.enrich_state(
                state, 
                memory_context
            )
            
            # Step 3: Add supervisor metadata
            enriched_state = self._add_supervisor_metadata(enriched_state, memory_context)
            
            logger.info(f"✅ State enriched with context: confidence={memory_context.get('confidence', 0.0)}")
            
            return enriched_state
            
        except Exception as e:
            logger.error(f"❌ Supervisor node failed: {e}")
            # Return original state if enrichment fails
            return state
    
    def supervisor_router(self, state: Any) -> str:
        """
        Intelligent routing function - the supervisor's decision brain.
        
        This inspects the enriched state and decides which agent to route to next.
        
        Args:
            state: Enriched workflow state
            
        Returns:
            Next node name for LangGraph routing
        """
        
        try:
            # Get the latest step to understand where we are
            latest_step = self._get_latest_step(state)
            
            logger.info(f"🧠 Routing decision for step: {latest_step}")
            
            # Route based on workflow stage
            if latest_step == "observe":
                return self._route_after_observation(state)
            
            elif latest_step == "analyze":
                return self._route_after_analysis(state)
            
            elif latest_step in ["execute", "human_approval"]:
                return self._route_after_execution(state)
            
            else:
                logger.warning(f"Unknown step: {latest_step}")
                return "workflow_complete"
                
        except Exception as e:
            logger.error(f"❌ Routing failed: {e}")
            return "workflow_complete"
    
    def _route_after_observation(self, state: Any) -> str:
        """Route after observation step."""
        
        # After observation, we always need analysis
        # But check memory context for special cases
        memory_context = self._get_memory_context(state)
        
        if memory_context and memory_context.get("confidence", 0) > 0.9:
            # High confidence from memory - might skip detailed analysis
            recommended_approach = memory_context.get("recommended_approach", "standard_analysis")
            
            if recommended_approach == "conservative_analysis":
                logger.info("🧠 Memory suggests conservative approach")
                return "needs_analysis"
            elif recommended_approach == "standard_analysis":
                logger.info("🧠 Memory suggests standard analysis")
                return "needs_analysis"
            else:
                logger.info("🧠 Memory suggests careful analysis")
                return "needs_analysis"
        
        # Default: always analyze after observation
        return "needs_analysis"
    
    def _route_after_analysis(self, state: Any) -> str:
        """Route after analysis step."""
        
        # Get analysis results
        analysis_evidence = self._get_latest_analysis(state)
        
        if not analysis_evidence:
            logger.warning("No analysis evidence found")
            return "workflow_complete"
        
        # Extract risk score
        risk_score = self._extract_risk_score(analysis_evidence)
        
        logger.info(f"🧠 Analysis risk score: {risk_score}")
        
        # Route based on risk and memory context
        memory_context = self._get_memory_context(state)
        confidence = memory_context.get("confidence", 0.5) if memory_context else 0.5
        
        # Adjust thresholds based on memory confidence
        high_risk_threshold = 0.9 if confidence > 0.8 else 0.8
        medium_risk_threshold = 0.5 if confidence > 0.8 else 0.4
        
        if risk_score > high_risk_threshold:
            logger.info("🚨 High risk - escalating to human")
            return "needs_human_escalation"
        elif risk_score > medium_risk_threshold:
            logger.info("⚡ Medium risk - executing action")
            return "can_execute"
        else:
            logger.info("✅ Low risk - workflow complete")
            return "workflow_complete"
    
    def _route_after_execution(self, state: Any) -> str:
        """Route after execution or human approval."""
        
        # After execution or approval, workflow is typically complete
        # But check if there are any follow-up actions needed
        
        latest_step = self._get_latest_step(state)
        
        if latest_step == "execute":
            # Check execution results
            execution_evidence = self._get_latest_execution(state)
            if execution_evidence:
                success_count = self._extract_success_count(execution_evidence)
                total_actions = self._extract_total_actions(execution_evidence)
                
                if success_count < total_actions:
                    logger.warning(f"⚠️ Partial execution success: {success_count}/{total_actions}")
                    # Could route to retry or human escalation
                    return "workflow_complete"  # For now, complete anyway
        
        logger.info("✅ Workflow complete")
        return "workflow_complete"
    
    def _get_latest_step(self, state: Any) -> str:
        """Determine the latest step from state."""
        
        try:
            # Check if state has step tracking
            if hasattr(state, 'current_step'):
                return state.current_step
            
            # Infer from evidence entries
            if hasattr(state, 'evidence_entries') and state.evidence_entries:
                latest_evidence = state.evidence_entries[-1]
                evidence_type = getattr(latest_evidence, 'evidence_type', None)
                
                if evidence_type:
                    if str(evidence_type) == "EvidenceType.OBSERVATION":
                        return "observe"
                    elif str(evidence_type) == "EvidenceType.PATTERN":
                        return "analyze"
                    elif str(evidence_type) == "EvidenceType.EXECUTION":
                        return "execute"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Failed to get latest step: {e}")
            return "unknown"
    
    def _get_memory_context(self, state: Any) -> Dict[str, Any]:
        """Extract memory context from enriched state."""
        
        try:
            if hasattr(state, 'memory_context'):
                return state.memory_context
            return {}
        except Exception:
            return {}
    
    def _get_latest_analysis(self, state: Any) -> Any:
        """Get latest analysis evidence."""
        
        try:
            if hasattr(state, 'evidence_entries'):
                for evidence in reversed(state.evidence_entries):
                    evidence_type = getattr(evidence, 'evidence_type', None)
                    if evidence_type and str(evidence_type) == "EvidenceType.PATTERN":
                        return evidence
            return None
        except Exception:
            return None
    
    def _get_latest_execution(self, state: Any) -> Any:
        """Get latest execution evidence."""
        
        try:
            if hasattr(state, 'evidence_entries'):
                for evidence in reversed(state.evidence_entries):
                    evidence_type = getattr(evidence, 'evidence_type', None)
                    if evidence_type and str(evidence_type) == "EvidenceType.EXECUTION":
                        return evidence
            return None
        except Exception:
            return None
    
    def _extract_risk_score(self, analysis_evidence: Any) -> float:
        """Extract risk score from analysis evidence."""
        
        try:
            content = getattr(analysis_evidence, 'content', {})
            if isinstance(content, dict):
                return content.get('risk_score', 0.5)
            return 0.5
        except Exception:
            return 0.5
    
    def _extract_success_count(self, execution_evidence: Any) -> int:
        """Extract success count from execution evidence."""
        
        try:
            content = getattr(execution_evidence, 'content', {})
            if isinstance(content, dict):
                return content.get('success_count', 0)
            return 0
        except Exception:
            return 0
    
    def _extract_total_actions(self, execution_evidence: Any) -> int:
        """Extract total actions from execution evidence."""
        
        try:
            content = getattr(execution_evidence, 'content', {})
            if isinstance(content, dict):
                actions_taken = content.get('actions_taken', [])
                return len(actions_taken) if isinstance(actions_taken, list) else 1
            return 1
        except Exception:
            return 1
    
    def _add_supervisor_metadata(self, state: Any, memory_context: Dict[str, Any]) -> Any:
        """Add supervisor metadata to state."""
        
        try:
            # Add memory context if state supports it
            if hasattr(state, '__dict__'):
                state.memory_context = memory_context
                state.supervisor_processed = True
                state.supervisor_timestamp = base.utc_now().isoformat()
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to add supervisor metadata: {e}")
            return state
