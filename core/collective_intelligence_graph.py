#!/usr/bin/env python3
"""
🧠 The Collective Intelligence Graph - Professional 2025 Architecture

This is the cutting-edge implementation using:
- LangGraph StateGraph with Supervisor pattern
- LangMem collective memory integration  
- Advanced context engineering
- Your bulletproof AgentState as the backbone

Built on your proven foundation with enterprise-grade features.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import sys

# LangGraph imports - latest patterns
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

# LangMem integration
try:
    from langmem import Client as LangMemClient
except ImportError:
    # Fallback for development
    class LangMemClient:
        def __init__(self, *args, **kwargs): pass
        async def search(self, *args, **kwargs): return []
        async def add(self, *args, **kwargs): pass

# Add schema directory to path
schema_dir = Path(__file__).parent / "src" / "aura_intelligence" / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

# Import your proven schemas
import enums
import base
from production_observer_agent import ProductionObserverAgent, ProductionAgentState, AgentConfig

logger = logging.getLogger(__name__)


class CollectiveMemoryManager:
    """
    Professional LangMem integration for collective intelligence.
    Manages context engineering and continuous learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = LangMemClient(
            api_key=config.get("langmem_api_key"),
            namespace="aura_collective_intelligence"
        )
        
    async def query_relevant_context(self, state: ProductionAgentState) -> Dict[str, Any]:
        """Query LangMem for relevant context to inform supervisor decisions."""
        
        # Create semantic signature for the current situation
        event_signature = self._create_event_signature(state)
        
        # Query similar past workflows
        memories = await self.client.search(
            query=f"Similar incidents to: {event_signature}",
            limit=5,
            filters={"workflow_type": "collective_intelligence"}
        )
        
        if not memories:
            return {"insight": "No similar past incidents found", "confidence": 0.0}
        
        # Analyze patterns from past workflows
        insights = self._analyze_memory_patterns(memories)
        
        return {
            "similar_incidents_count": len(memories),
            "success_patterns": insights.get("success_patterns", []),
            "failure_patterns": insights.get("failure_patterns", []),
            "recommended_approach": insights.get("recommended_approach"),
            "confidence": insights.get("confidence", 0.5),
            "context_summary": insights.get("summary", "")
        }
    
    async def learn_from_workflow(self, final_state: ProductionAgentState) -> None:
        """Store completed workflow in collective memory for future learning."""
        
        workflow_summary = {
            "workflow_id": final_state.workflow_id,
            "event_type": self._extract_event_type(final_state),
            "evidence_count": len(final_state.evidence_entries),
            "processing_time": self._calculate_processing_time(final_state),
            "outcome": self._determine_outcome(final_state),
            "success": self._was_successful(final_state),
            "patterns": self._extract_patterns(final_state),
            "lessons_learned": self._extract_lessons(final_state)
        }
        
        await self.client.add(
            content=json.dumps(workflow_summary),
            metadata={
                "workflow_type": "collective_intelligence",
                "timestamp": base.utc_now().isoformat(),
                "agent_version": "production_v1.0"
            }
        )
        
        logger.info(f"Stored workflow {final_state.workflow_id} in collective memory")
    
    def _create_event_signature(self, state: ProductionAgentState) -> str:
        """Create semantic signature for event matching."""
        if not state.evidence_entries:
            return "unknown_event"
        
        latest_evidence = state.evidence_entries[-1]
        return f"{latest_evidence.evidence_type.value}_{latest_evidence.content.get('severity', 'unknown')}"
    
    def _analyze_memory_patterns(self, memories: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns from retrieved memories."""
        if not memories:
            return {"confidence": 0.0}
        
        # Simple pattern analysis - in production this would be more sophisticated
        success_count = sum(1 for m in memories if m.get("success", False))
        total_count = len(memories)
        
        return {
            "success_patterns": ["pattern_analysis_needed"],
            "failure_patterns": ["failure_analysis_needed"],
            "recommended_approach": "standard_analysis" if success_count > total_count / 2 else "careful_analysis",
            "confidence": success_count / total_count if total_count > 0 else 0.5,
            "summary": f"Found {total_count} similar cases with {success_count} successes"
        }
    
    def _extract_event_type(self, state: ProductionAgentState) -> str:
        """Extract event type from state."""
        if state.evidence_entries:
            return state.evidence_entries[0].evidence_type.value
        return "unknown"
    
    def _calculate_processing_time(self, state: ProductionAgentState) -> float:
        """Calculate total processing time."""
        return (state.updated_at - state.created_at).total_seconds()
    
    def _determine_outcome(self, state: ProductionAgentState) -> str:
        """Determine workflow outcome."""
        if state.status == enums.TaskStatus.COMPLETED:
            return "completed"
        elif state.status == enums.TaskStatus.FAILED:
            return "failed"
        else:
            return "in_progress"
    
    def _was_successful(self, state: ProductionAgentState) -> bool:
        """Determine if workflow was successful."""
        return state.status == enums.TaskStatus.COMPLETED
    
    def _extract_patterns(self, state: ProductionAgentState) -> List[str]:
        """Extract patterns from workflow."""
        patterns = []
        if len(state.evidence_entries) > 3:
            patterns.append("high_evidence_volume")
        if any("error" in e.content.get("message", "").lower() for e in state.evidence_entries):
            patterns.append("error_pattern")
        return patterns
    
    def _extract_lessons(self, state: ProductionAgentState) -> List[str]:
        """Extract lessons learned from workflow."""
        return ["lesson_extraction_needed"]  # Placeholder for sophisticated analysis


class AnalystAgent:
    """
    Professional analyst agent using your proven patterns.
    Specialized for deep analysis with TDA integration.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = f"analyst_{config.agent_id}"
        
    async def analyze_state(self, state: ProductionAgentState) -> ProductionAgentState:
        """Perform deep analysis on the current state."""
        
        logger.info(f"🔍 AnalystAgent analyzing state: {state.workflow_id}")
        
        # Analyze evidence patterns
        analysis_results = self._perform_analysis(state.evidence_entries)
        
        # Create analysis evidence using your proven schemas
        from production_observer_agent import ProductionEvidence
        
        analysis_evidence = ProductionEvidence(
            evidence_type=enums.EvidenceType.PATTERN,
            content={
                "analysis_type": "collective_intelligence_analysis",
                "risk_score": analysis_results["risk_score"],
                "patterns_detected": analysis_results["patterns"],
                "confidence": analysis_results["confidence"],
                "recommendations": analysis_results["recommendations"],
                "analysis_timestamp": base.utc_now().isoformat(),
                "analyst_id": self.agent_id
            },
            workflow_id=state.workflow_id,
            task_id=state.task_id,
            config=self.config
        )
        
        # Update state immutably
        new_state = state.add_evidence(analysis_evidence, self.config)
        
        logger.info(f"✅ Analysis complete: risk_score={analysis_results['risk_score']}")
        return new_state
    
    def _perform_analysis(self, evidence_entries: List) -> Dict[str, Any]:
        """Perform sophisticated analysis on evidence."""
        
        if not evidence_entries:
            return {
                "risk_score": 0.0,
                "patterns": [],
                "confidence": 0.5,
                "recommendations": ["no_evidence_to_analyze"]
            }
        
        # Sophisticated pattern analysis
        error_count = sum(1 for e in evidence_entries if "error" in str(e.content).lower())
        critical_count = sum(1 for e in evidence_entries if "critical" in str(e.content).lower())
        
        # Calculate risk score
        risk_score = min(1.0, (error_count * 0.3 + critical_count * 0.7) / len(evidence_entries))
        
        # Detect patterns
        patterns = []
        if error_count > len(evidence_entries) * 0.5:
            patterns.append("high_error_rate")
        if critical_count > 0:
            patterns.append("critical_issues_present")
        
        # Generate recommendations
        recommendations = []
        if risk_score > 0.8:
            recommendations.append("immediate_action_required")
        elif risk_score > 0.5:
            recommendations.append("monitoring_recommended")
        else:
            recommendations.append("continue_observation")
        
        return {
            "risk_score": risk_score,
            "patterns": patterns,
            "confidence": 0.85,
            "recommendations": recommendations
        }


class ExecutorAgent:
    """
    Professional executor agent for taking actions based on analysis.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = f"executor_{config.agent_id}"
        
    async def execute_action(self, state: ProductionAgentState) -> ProductionAgentState:
        """Execute actions based on analysis results."""
        
        logger.info(f"⚡ ExecutorAgent executing actions: {state.workflow_id}")
        
        # Find latest analysis
        analysis_evidence = self._get_latest_analysis(state.evidence_entries)
        
        if not analysis_evidence:
            logger.warning("No analysis found for execution")
            return state
        
        # Determine actions based on analysis
        actions = self._determine_actions(analysis_evidence)
        
        # Execute actions
        execution_results = []
        for action in actions:
            result = await self._execute_single_action(action)
            execution_results.append(result)
        
        # Create execution evidence
        from production_observer_agent import ProductionEvidence
        
        execution_evidence = ProductionEvidence(
            evidence_type=enums.EvidenceType.OBSERVATION,
            content={
                "execution_type": "collective_intelligence_execution",
                "actions_taken": [a["type"] for a in actions],
                "execution_results": execution_results,
                "success_count": sum(1 for r in execution_results if r["success"]),
                "execution_timestamp": base.utc_now().isoformat(),
                "executor_id": self.agent_id
            },
            workflow_id=state.workflow_id,
            task_id=state.task_id,
            config=self.config
        )
        
        # Update state
        new_state = state.add_evidence(execution_evidence, self.config)
        
        # Mark as completed if all actions succeeded
        if all(r["success"] for r in execution_results):
            new_state.status = enums.TaskStatus.COMPLETED
        
        logger.info(f"✅ Execution complete: {len(execution_results)} actions")
        return new_state
    
    def _get_latest_analysis(self, evidence_entries: List) -> Optional[Any]:
        """Get the latest analysis evidence."""
        for evidence in reversed(evidence_entries):
            if evidence.evidence_type == enums.EvidenceType.PATTERN:
                return evidence
        return None
    
    def _determine_actions(self, analysis_evidence: Any) -> List[Dict[str, Any]]:
        """Determine actions based on analysis."""
        risk_score = analysis_evidence.content.get("risk_score", 0.0)
        
        actions = []
        
        if risk_score > 0.8:
            actions.append({
                "type": "send_alert",
                "priority": "high",
                "message": "Critical risk detected"
            })
        elif risk_score > 0.5:
            actions.append({
                "type": "create_ticket",
                "priority": "medium",
                "message": "Moderate risk requires attention"
            })
        else:
            actions.append({
                "type": "log_observation",
                "priority": "low",
                "message": "Normal operation observed"
            })
        
        return actions
    
    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        
        # Simulate action execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "action_type": action["type"],
            "success": True,
            "message": f"Successfully executed {action['type']}",
            "execution_time_ms": 100
        }
