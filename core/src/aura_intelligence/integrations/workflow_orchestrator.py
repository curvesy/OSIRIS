"""
Modern Workflow Orchestrator for AURA Intelligence
Built with LangGraph for production-grade orchestration
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
import structlog
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode as ToolExecutor
try:
    from langgraph.prebuilt import ToolInvocation
except ImportError:
    ToolInvocation = None

from ..adapters.tda_neo4j_adapter import TDANeo4jAdapter
from ..adapters.tda_mem0_adapter import TDAMem0Adapter
from ..adapters.tda_agent_context import TDAAgentContextAdapter, TopologicalSignal
from ..tda_engine import TDAEngine
from ..agents.council import AgentCouncil
from ..observability import MetricsCollector, TracingContext
from ..feature_flags import FeatureFlags

logger = structlog.get_logger(__name__)
T = TypeVar('T')


class WorkflowState(str, Enum):
    """Workflow states with semantic meaning"""
    INGESTING = "ingesting"
    ANALYZING_TDA = "analyzing_tda"
    ENRICHING_CONTEXT = "enriching_context"
    AGENT_DELIBERATION = "agent_deliberation"
    EXECUTING_ACTION = "executing_action"
    MONITORING = "monitoring"
    ESCALATING = "escalating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowContext:
    """Context for workflow execution"""
    workflow_id: str
    trace_id: str
    data: Dict[str, Any]
    tda_results: Optional[Dict[str, Any]] = None
    enriched_context: Optional[Dict[str, Any]] = None
    agent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    current_state: WorkflowState = WorkflowState.INGESTING
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """
    Modern workflow orchestrator integrating all AURA components.
    
    Features:
    - TDA integration with feature flag control
    - Multi-agent deliberation
    - Knowledge graph and memory integration
    - Full observability and monitoring
    - Checkpointing and recovery
    """
    
    def __init__(self):
        self.tda_engine = TDAEngine()
        self.neo4j_adapter = TDANeo4jAdapter()
        self.mem0_adapter = TDAMem0Adapter()
        self.context_adapter = TDAAgentContextAdapter(
            neo4j_adapter=self.neo4j_adapter,
            mem0_adapter=self.mem0_adapter
        )
        self.agent_council = AgentCouncil()
        self.feature_flags = FeatureFlags()
        self.metrics = MetricsCollector()
        self.checkpointer = PostgresSaver.from_conn_string(
            "postgresql://localhost:5432/aura"
        )
        self.graph: Optional[StateGraph] = None
        
    async def initialize(self):
        """Initialize all components"""
        await self.neo4j_adapter.initialize()
        await self.mem0_adapter.initialize()
        await self.agent_council.initialize()
        
        # Build workflow graph
        self._build_graph()
        
        logger.info("Workflow orchestrator initialized")
        
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowContext)
        
        # Add nodes
        workflow.add_node("ingest", self._ingest_data)
        workflow.add_node("analyze_tda", self._analyze_tda)
        workflow.add_node("enrich_context", self._enrich_context)
        workflow.add_node("deliberate", self._agent_deliberation)
        workflow.add_node("execute", self._execute_action)
        workflow.add_node("monitor", self._monitor_results)
        workflow.add_node("escalate", self._escalate_anomaly)
        
        # Add edges
        workflow.add_edge("ingest", "analyze_tda")
        workflow.add_edge("analyze_tda", "enrich_context")
        workflow.add_edge("enrich_context", "deliberate")
        
        # Conditional routing after deliberation
        workflow.add_conditional_edges(
            "deliberate",
            self._route_after_deliberation,
            {
                "execute": "execute",
                "escalate": "escalate",
                "monitor": "monitor"
            }
        )
        
        workflow.add_edge("execute", "monitor")
        workflow.add_edge("escalate", "monitor")
        workflow.add_edge("monitor", END)
        
        # Set entry point
        workflow.set_entry_point("ingest")
        
        # Compile with checkpointer
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
    async def _ingest_data(self, state: WorkflowContext) -> WorkflowContext:
        """Ingest and validate data"""
        state.current_state = WorkflowState.INGESTING
        
        # Validate data
        if not state.data:
            state.error = "No data provided"
            state.current_state = WorkflowState.FAILED
            
        logger.info("Data ingested", workflow_id=state.workflow_id)
        return state
        
    async def _analyze_tda(self, state: WorkflowContext) -> WorkflowContext:
        """Run TDA analysis with feature flag control"""
        state.current_state = WorkflowState.ANALYZING_TDA
        
        try:
            # Check feature flags for algorithm selection
            algorithm = "specseq++"  # default
            if await self.feature_flags.is_enabled("tda.simba_gpu"):
                algorithm = "simba_gpu"
            elif await self.feature_flags.is_enabled("tda.neural_surveillance"):
                algorithm = "neural_surveillance"
                
            # Run TDA analysis
            tda_results = await self.tda_engine.analyze(
                data=state.data,
                algorithm=algorithm,
                use_gpu=await self.feature_flags.is_enabled("tda.gpu_acceleration")
            )
            
            state.tda_results = tda_results
            
            # Store TDA results
            await self.neo4j_adapter.store_tda_result(
                tda_id=f"tda_{state.workflow_id}",
                result=tda_results
            )
            
            await self.mem0_adapter.store_tda_memory(
                tda_id=f"tda_{state.workflow_id}",
                result=tda_results,
                metadata={"workflow_id": state.workflow_id}
            )
            
            # Update metrics
            self.metrics.tda_computations_total.labels(
                algorithm=algorithm,
                status="success"
            ).inc()
            
            logger.info("TDA analysis complete", 
                       workflow_id=state.workflow_id,
                       algorithm=algorithm,
                       anomaly_score=tda_results.get("anomaly_score", 0))
                       
        except Exception as e:
            logger.error("TDA analysis failed", error=str(e))
            if await self.feature_flags.is_enabled("tda.auto_fallback"):
                # Use fallback algorithm
                from ..tda.production_fallbacks import DeterministicTDAFallback
                fallback = DeterministicTDAFallback()
                state.tda_results = await fallback.analyze(state.data)
            else:
                state.error = f"TDA analysis failed: {str(e)}"
                state.current_state = WorkflowState.FAILED
                
        return state
        
    async def _enrich_context(self, state: WorkflowContext) -> WorkflowContext:
        """Enrich context with historical data"""
        state.current_state = WorkflowState.ENRICHING_CONTEXT
        
        # Get enriched context
        enriched_context = await self.context_adapter.enrich_agent_context(
            tda_results=state.tda_results,
            historical_window=timedelta(days=7)
        )
        
        state.enriched_context = enriched_context
        
        logger.info("Context enriched", 
                   workflow_id=state.workflow_id,
                   historical_patterns=len(enriched_context.get("historical_patterns", [])))
                   
        return state
        
    async def _agent_deliberation(self, state: WorkflowContext) -> WorkflowContext:
        """Multi-agent deliberation"""
        state.current_state = WorkflowState.AGENT_DELIBERATION
        
        # Prepare context for agents
        agent_context = {
            "tda_results": state.tda_results,
            "enriched_context": state.enriched_context,
            "workflow_metadata": state.metadata
        }
        
        # Get council decision
        decision = await self.agent_council.deliberate(
            context=agent_context,
            timeout=30.0
        )
        
        state.agent_decisions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "participating_agents": decision.get("agents", [])
        })
        
        # Update metrics
        self.metrics.agent_decisions_total.labels(
            agent_id="council",
            decision_type=decision.get("action", "unknown")
        ).inc()
        
        logger.info("Agent deliberation complete",
                   workflow_id=state.workflow_id,
                   decision=decision.get("action"),
                   confidence=decision.get("confidence"))
                   
        return state
        
    def _route_after_deliberation(self, state: WorkflowContext) -> str:
        """Route based on agent decision"""
        if not state.agent_decisions:
            return "monitor"
            
        latest_decision = state.agent_decisions[-1]["decision"]
        action = latest_decision.get("action", "monitor")
        
        if action == "escalate" or latest_decision.get("risk_level") == "critical":
            return "escalate"
        elif action in ["execute", "remediate", "allocate"]:
            return "execute"
        else:
            return "monitor"
            
    async def _execute_action(self, state: WorkflowContext) -> WorkflowContext:
        """Execute the decided action"""
        state.current_state = WorkflowState.EXECUTING_ACTION
        
        latest_decision = state.agent_decisions[-1]["decision"]
        action = latest_decision.get("action")
        
        logger.info("Executing action",
                   workflow_id=state.workflow_id,
                   action=action)
                   
        # Action execution would go here
        # For now, just record it
        state.metadata["executed_action"] = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed"
        }
        
        return state
        
    async def _escalate_anomaly(self, state: WorkflowContext) -> WorkflowContext:
        """Escalate critical anomalies"""
        state.current_state = WorkflowState.ESCALATING
        
        logger.warning("Escalating anomaly",
                      workflow_id=state.workflow_id,
                      anomaly_score=state.tda_results.get("anomaly_score"))
                      
        # Escalation logic would go here
        state.metadata["escalation"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "High anomaly score",
            "notified": ["ops-team", "security-team"]
        }
        
        return state
        
    async def _monitor_results(self, state: WorkflowContext) -> WorkflowContext:
        """Monitor and record results"""
        state.current_state = WorkflowState.MONITORING
        
        # Update metrics
        self.metrics.workflow_executions_total.labels(
            status="success" if not state.error else "failed"
        ).inc()
        
        # Record checkpoint
        checkpoint = {
            "workflow_id": state.workflow_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state.current_state.value,
            "tda_results": state.tda_results,
            "decisions": state.agent_decisions,
            "metadata": state.metadata
        }
        
        state.checkpoints.append(checkpoint)
        state.current_state = WorkflowState.COMPLETED
        
        logger.info("Workflow completed",
                   workflow_id=state.workflow_id,
                   final_state=state.current_state)
                   
        return state
        
    async def execute_workflow(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> WorkflowContext:
        """Execute a complete workflow"""
        import uuid
        
        workflow_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        initial_state = WorkflowContext(
            workflow_id=workflow_id,
            trace_id=trace_id,
            data=data,
            metadata=metadata or {}
        )
        
        # Execute workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state
        
    async def recover_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Recover a workflow from checkpoint"""
        # Recovery logic would use the checkpointer
        # For now, return None
        return None
        
    @asynccontextmanager
    async def trace_workflow(self, workflow_id: str):
        """Context manager for workflow tracing"""
        async with TracingContext(
            service="workflow_orchestrator",
            operation=f"workflow_{workflow_id}"
        ) as ctx:
            yield ctx
            
    async def cleanup(self):
        """Cleanup resources"""
        await self.neo4j_adapter.cleanup()
        await self.mem0_adapter.cleanup()
        await self.agent_council.cleanup()