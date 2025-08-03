"""
Enhanced Workflow Integration - Connects All Existing AURA Components
Integrates with UnifiedAURABrain, ProductionLNNCouncil, and existing orchestration
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import structlog

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

# Import existing AURA components
from ..unified_brain import UnifiedAURABrain, UnifiedConfig, AnalysisResult
from ..agents.council.production_lnn_council import ProductionLNNCouncilAgent, CouncilState
from ..orchestration.langgraph_workflows import AURACollectiveIntelligence, AgentState
from ..orchestration.tda_coordinator import TDACoordinator
from ..orchestration.checkpoints import CheckpointManager
from ..orchestration.feature_flags import FeatureFlagManager
from ..tda.algorithms import SpecSeqPlusPlus, SimBaGPU, NeuralSurveillance
from ..adapters.tda_neo4j_adapter import TDANeo4jAdapter
from ..adapters.tda_mem0_adapter import TDAMem0Adapter
from ..adapters.tda_agent_context import TDAAgentContextAdapter
from ..observability.dashboard import UnifiedDashboard
from ..feature_flags import FeatureFlags as ModernFeatureFlags

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedWorkflowState(AgentState):
    """Enhanced state that combines all workflow states"""
    # From AgentState
    messages: List[str]
    evidence_log: List[Dict[str, Any]]
    tda_insights: Dict[str, Any]
    current_agent: str
    workflow_context: Dict[str, Any]
    decision_history: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    collective_decision: Dict[str, Any]
    agents_involved: List[str]
    
    # From CouncilState
    task: Optional[Any] = None
    context_window: Optional[Any] = None
    lnn_output: Optional[Any] = None
    vote: Optional[Any] = None
    neo4j_context: Dict[str, Any] = None
    memory_context: Dict[str, Any] = None
    
    # New enhanced fields
    unified_brain_analysis: Optional[AnalysisResult] = None
    adapter_results: Dict[str, Any] = None
    feature_flags_state: Dict[str, bool] = None
    monitoring_metrics: Dict[str, Any] = None


class EnhancedWorkflowIntegration:
    """
    Enhanced integration that connects all AURA components:
    - UnifiedAURABrain for core intelligence
    - ProductionLNNCouncil for agent decisions
    - Existing orchestration systems
    - New production adapters
    - Modern feature flags and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize unified brain
        unified_config = UnifiedConfig(**config.get("unified_brain", {}))
        self.unified_brain = UnifiedAURABrain(unified_config)
        
        # Initialize existing orchestration
        self.collective_intelligence = AURACollectiveIntelligence()
        self.tda_coordinator = TDACoordinator(config.get("tda", {}))
        
        # Initialize production council
        council_config = config.get("council", {})
        self.lnn_council = ProductionLNNCouncilAgent(council_config)
        
        # Initialize new production adapters
        self.neo4j_adapter = TDANeo4jAdapter(
            uri=config.get("neo4j_uri"),
            user=config.get("neo4j_user"),
            password=config.get("neo4j_password")
        )
        self.mem0_adapter = TDAMem0Adapter(config.get("mem0", {}))
        self.context_adapter = TDAAgentContextAdapter(
            neo4j_adapter=self.neo4j_adapter,
            mem0_adapter=self.mem0_adapter,
            event_bus=self.collective_intelligence.event_bus
        )
        
        # Initialize monitoring and feature flags
        self.dashboard = UnifiedDashboard(config.get("monitoring", {}))
        self.modern_features = ModernFeatureFlags(config.get("feature_source"))
        
        # Merge with existing feature flags
        self.feature_manager = FeatureFlagManager()
        
        # Create enhanced workflow
        self.workflow = self._create_enhanced_workflow()
        self.checkpointer = CheckpointManager()
        
        logger.info("Enhanced AURA Integration initialized")
        
    def _create_enhanced_workflow(self) -> StateGraph:
        """Create the enhanced workflow that integrates everything"""
        workflow = StateGraph(EnhancedWorkflowState)
        
        # Add nodes
        workflow.add_node("unified_analysis", self._unified_brain_analysis)
        workflow.add_node("tda_processing", self._enhanced_tda_processing)
        workflow.add_node("context_enrichment", self._context_enrichment)
        workflow.add_node("council_deliberation", self._council_deliberation)
        workflow.add_node("collective_decision", self._collective_decision)
        workflow.add_node("execution", self._execute_decision)
        workflow.add_node("monitoring", self._update_monitoring)
        workflow.add_node("checkpoint", self._create_checkpoint)
        
        # Set entry point
        workflow.set_entry_point("unified_analysis")
        
        # Define flow
        workflow.add_edge("unified_analysis", "tda_processing")
        workflow.add_conditional_edges(
            "tda_processing",
            self._route_after_tda,
            {
                "enrich": "context_enrichment",
                "urgent": "council_deliberation",
                "monitor": "monitoring"
            }
        )
        workflow.add_edge("context_enrichment", "council_deliberation")
        workflow.add_edge("council_deliberation", "collective_decision")
        workflow.add_conditional_edges(
            "collective_decision",
            self._route_after_decision,
            {
                "execute": "execution",
                "monitor": "monitoring",
                "end": END
            }
        )
        workflow.add_edge("execution", "checkpoint")
        workflow.add_edge("monitoring", "checkpoint")
        workflow.add_edge("checkpoint", END)
        
        return workflow.compile(checkpointer=self.checkpointer.memory_saver)
        
    async def _unified_brain_analysis(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Run unified brain analysis"""
        logger.info("Running unified brain analysis")
        
        # Prepare data for analysis
        analysis_data = {
            "messages": state.messages,
            "evidence": state.evidence_log,
            "context": state.workflow_context
        }
        
        # Run unified brain analysis
        analysis_result = await self.unified_brain.analyze(
            data=analysis_data,
            request_type="comprehensive"
        )
        
        state.unified_brain_analysis = analysis_result
        state.risk_assessment = {
            "risk_score": analysis_result.risk_score,
            "ethical_status": analysis_result.ethical_status
        }
        
        # Update metrics
        self.dashboard.agent_confidence.set(
            analysis_result.confidence,
            {"agent_id": "unified_brain"}
        )
        
        return state
        
    async def _enhanced_tda_processing(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Enhanced TDA processing with feature flags and adapters"""
        logger.info("Running enhanced TDA processing")
        
        # Check feature flags for algorithm selection
        algorithms = []
        if await self.modern_features.is_enabled("tda.specseq_plus"):
            algorithms.append(SpecSeqPlusPlus())
        if await self.modern_features.is_enabled("tda.simba_gpu"):
            algorithms.append(SimBaGPU())
        if await self.modern_features.is_enabled("tda.neural_surveillance"):
            algorithms.append(NeuralSurveillance())
            
        # Run TDA with coordinator
        tda_result = await self.tda_coordinator.coordinate_analysis(
            data=state.evidence_log,
            algorithms=algorithms,
            trace_id=state.workflow_context.get("trace_id")
        )
        
        # Store results in adapters
        neo4j_id = await self.neo4j_adapter.store_tda_result(
            result=tda_result,
            parent_data_id=state.workflow_context.get("data_id"),
            context={"workflow_id": state.workflow_context.get("workflow_id")}
        )
        
        memory_id = await self.mem0_adapter.store_tda_memory(
            result=tda_result,
            agent_id="enhanced_workflow",
            context=state.workflow_context
        )
        
        state.tda_insights = {
            "result": tda_result,
            "neo4j_id": neo4j_id,
            "memory_id": memory_id,
            "algorithms_used": [algo.__class__.__name__ for algo in algorithms]
        }
        
        state.adapter_results = {
            "neo4j_stored": True,
            "mem0_stored": True,
            "storage_ids": {"neo4j": neo4j_id, "mem0": memory_id}
        }
        
        # Update dashboard
        self.dashboard.tda_computations.labels(
            algorithm=",".join(state.tda_insights["algorithms_used"]),
            status="success"
        ).inc()
        
        return state
        
    async def _context_enrichment(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Enrich context using TDA insights"""
        logger.info("Enriching context with TDA insights")
        
        # Get enriched context for each agent type
        agent_contexts = {}
        
        for agent_id in ["observer", "analyst", "supervisor", "council"]:
            enriched = await self.context_adapter.enrich_agent_context(
                agent_id=agent_id,
                base_context={
                    "task_id": state.workflow_context.get("task_id"),
                    "trace_id": state.workflow_context.get("trace_id")
                },
                data_id=state.workflow_context.get("data_id")
            )
            agent_contexts[agent_id] = enriched
            
        state.neo4j_context = {
            "topological_features": agent_contexts,
            "aggregated_signals": list(set(
                signal 
                for ctx in agent_contexts.values() 
                for signal in ctx.anomaly_signals
            ))
        }
        
        return state
        
    async def _council_deliberation(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Run LNN council deliberation"""
        logger.info("Running council deliberation")
        
        # Prepare council task
        council_task = {
            "type": "analyze_and_decide",
            "unified_analysis": state.unified_brain_analysis,
            "tda_insights": state.tda_insights,
            "enriched_contexts": state.neo4j_context,
            "risk_assessment": state.risk_assessment
        }
        
        # Run council deliberation
        council_result = await self.lnn_council.deliberate(council_task)
        
        state.vote = council_result
        state.lnn_output = council_result.get("neural_output")
        
        # Update decision history
        state.decision_history.append({
            "timestamp": datetime.utcnow(),
            "agent": "lnn_council",
            "decision": council_result,
            "confidence": council_result.get("confidence", 0.0)
        })
        
        return state
        
    async def _collective_decision(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Make collective decision using all agents"""
        logger.info("Making collective decision")
        
        # Run through existing collective intelligence
        collective_state = await self.collective_intelligence.workflow.ainvoke({
            "messages": state.messages,
            "evidence_log": state.evidence_log,
            "tda_insights": state.tda_insights,
            "current_agent": "collective",
            "workflow_context": state.workflow_context,
            "decision_history": state.decision_history,
            "risk_assessment": state.risk_assessment,
            "collective_decision": {},
            "agents_involved": []
        })
        
        state.collective_decision = collective_state["collective_decision"]
        state.agents_involved = collective_state["agents_involved"]
        
        # Record metrics
        self.dashboard.agent_decisions.labels(
            agent_id="collective",
            decision_type=state.collective_decision.get("type", "unknown")
        ).inc()
        
        return state
        
    async def _execute_decision(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Execute the collective decision"""
        logger.info("Executing decision")
        
        decision = state.collective_decision
        
        # Execute based on decision type
        if decision.get("type") == "mitigate_anomaly":
            # Execute mitigation
            logger.info("Executing anomaly mitigation")
            # Implementation specific to your system
            
        elif decision.get("type") == "optimize_performance":
            # Execute optimization
            logger.info("Executing performance optimization")
            # Implementation specific to your system
            
        elif decision.get("type") == "escalate_human":
            # Escalate to human
            logger.info("Escalating to human operator")
            # Implementation specific to your system
            
        return state
        
    async def _update_monitoring(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Update monitoring dashboard"""
        logger.info("Updating monitoring")
        
        # Collect metrics
        metrics = {
            "workflow_id": state.workflow_context.get("workflow_id"),
            "duration": (datetime.utcnow() - state.workflow_context.get("start_time", datetime.utcnow())).total_seconds(),
            "agents_involved": len(state.agents_involved),
            "confidence_scores": [d.get("confidence", 0) for d in state.decision_history],
            "risk_score": state.risk_assessment.get("risk_score", 0),
            "tda_algorithms": state.tda_insights.get("algorithms_used", [])
        }
        
        state.monitoring_metrics = metrics
        
        # Update dashboard
        await self.dashboard.update_metrics()
        
        # Check alerts
        await self.dashboard.check_alerts()
        
        return state
        
    async def _create_checkpoint(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Create workflow checkpoint"""
        logger.info("Creating checkpoint")
        
        checkpoint_data = {
            "workflow_id": state.workflow_context.get("workflow_id"),
            "timestamp": datetime.utcnow(),
            "state_snapshot": {
                "unified_analysis": state.unified_brain_analysis,
                "tda_insights": state.tda_insights,
                "collective_decision": state.collective_decision,
                "adapter_results": state.adapter_results
            },
            "metrics": state.monitoring_metrics
        }
        
        await self.checkpointer.save_checkpoint(
            workflow_id=state.workflow_context.get("workflow_id"),
            checkpoint_data=checkpoint_data
        )
        
        return state
        
    def _route_after_tda(self, state: EnhancedWorkflowState) -> str:
        """Route after TDA processing"""
        if state.unified_brain_analysis and state.unified_brain_analysis.risk_score > 0.8:
            return "urgent"
        elif state.tda_insights.get("result", {}).get("anomaly_score", 0) > 0.5:
            return "enrich"
        else:
            return "monitor"
            
    def _route_after_decision(self, state: EnhancedWorkflowState) -> str:
        """Route after collective decision"""
        decision_type = state.collective_decision.get("type")
        
        if decision_type in ["mitigate_anomaly", "optimize_performance", "escalate_human"]:
            return "execute"
        elif decision_type == "monitor":
            return "monitor"
        else:
            return "end"
            
    async def process_request(self, request: Dict[str, Any]) -> EnhancedWorkflowState:
        """Process a request through the enhanced workflow"""
        initial_state = EnhancedWorkflowState(
            messages=[request.get("message", "")],
            evidence_log=request.get("evidence", []),
            tda_insights={},
            current_agent="unified_brain",
            workflow_context={
                "workflow_id": f"wf_{datetime.utcnow().timestamp()}",
                "trace_id": request.get("trace_id", f"trace_{datetime.utcnow().timestamp()}"),
                "task_id": request.get("task_id"),
                "data_id": request.get("data_id"),
                "start_time": datetime.utcnow()
            },
            decision_history=[],
            risk_assessment={},
            collective_decision={},
            agents_involved=[],
            neo4j_context={},
            memory_context={},
            adapter_results={},
            feature_flags_state=await self._get_feature_flags_state(),
            monitoring_metrics={}
        )
        
        # Run workflow
        result = await self.workflow.ainvoke(initial_state)
        
        # Update final metrics
        self.dashboard.workflow_executions.labels(status="completed").inc()
        
        return result
        
    async def _get_feature_flags_state(self) -> Dict[str, bool]:
        """Get current state of all feature flags"""
        flags = {}
        
        # Modern feature flags
        for flag_name in ["tda.specseq_plus", "tda.simba_gpu", "tda.neural_surveillance",
                         "workflow.auto_recovery", "monitoring.enhanced_tracing"]:
            flags[flag_name] = await self.modern_features.is_enabled(flag_name)
            
        # Existing feature flags
        existing_flags = self.feature_manager.get_all_flags()
        flags.update(existing_flags)
        
        return flags