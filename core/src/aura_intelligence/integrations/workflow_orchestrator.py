"""
Modern Workflow Orchestrator - 2025 Production Standard
Integrates TDA, Agents, Knowledge Graph, and Memory with full observability
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, TypeVar, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import structlog
from contextlib import asynccontextmanager

from langgraph import StateGraph, State, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from ..adapters.tda_neo4j_adapter import TDANeo4jAdapter
from ..adapters.tda_mem0_adapter import TDAMem0Adapter
from ..adapters.tda_agent_context import TDAAgentContextAdapter, TopologicalSignal
from ..tda.engine import TDAEngine
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
class WorkflowContext(State):
    """Rich workflow context with full traceability"""
    workflow_id: str
    trace_id: str
    data: Dict[str, Any]
    tda_results: Optional[Dict[str, Any]] = None
    enriched_context: Optional[Dict[str, Any]] = None
    agent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_signals: List[TopologicalSignal] = field(default_factory=list)
    current_state: WorkflowState = WorkflowState.INGESTING
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class IWorkflowComponent(Protocol):
    """Protocol for workflow components"""
    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        ...


class ModernWorkflowOrchestrator:
    """
    2025-standard workflow orchestrator with:
    - Full TDA integration
    - Dynamic routing based on signals
    - Feature flag control
    - Complete observability
    - Disaster recovery
    """
    
    def __init__(
        self,
        tda_engine: TDAEngine,
        neo4j_adapter: TDANeo4jAdapter,
        mem0_adapter: TDAMem0Adapter,
        context_adapter: TDAAgentContextAdapter,
        agent_council: AgentCouncil,
        feature_flags: FeatureFlags,
        metrics: MetricsCollector
    ):
        self.tda_engine = tda_engine
        self.neo4j_adapter = neo4j_adapter
        self.mem0_adapter = mem0_adapter
        self.context_adapter = context_adapter
        self.agent_council = agent_council
        self.feature_flags = feature_flags
        self.metrics = metrics
        
        # Build workflow graph
        self.graph = self._build_workflow_graph()
        self.checkpointer = MemorySaver()
        
        # Component registry for extensibility
        self.components: Dict[str, IWorkflowComponent] = {}
        self._register_default_components()
        
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing"""
        workflow = StateGraph(WorkflowContext)
        
        # Define nodes
        workflow.add_node("ingest", self._ingest_data)
        workflow.add_node("analyze_tda", self._analyze_tda)
        workflow.add_node("enrich_context", self._enrich_context)
        workflow.add_node("agent_deliberation", self._agent_deliberation)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("monitor", self._monitor_state)
        workflow.add_node("escalate", self._escalate_anomaly)
        workflow.add_node("checkpoint", self._create_checkpoint)
        
        # Set entry point
        workflow.set_entry_point("ingest")
        
        # Define edges with conditions
        workflow.add_edge("ingest", "analyze_tda")
        workflow.add_conditional_edges(
            "analyze_tda",
            self._route_after_tda,
            {
                "normal": "enrich_context",
                "anomaly": "escalate",
                "error": END
            }
        )
        workflow.add_edge("enrich_context", "agent_deliberation")
        workflow.add_conditional_edges(
            "agent_deliberation",
            self._route_after_deliberation,
            {
                "execute": "execute_action",
                "escalate": "escalate",
                "monitor": "monitor"
            }
        )
        workflow.add_edge("execute_action", "checkpoint")
        workflow.add_edge("escalate", "checkpoint")
        workflow.add_edge("monitor", "checkpoint")
        workflow.add_edge("checkpoint", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
        
    async def _ingest_data(self, context: WorkflowContext) -> WorkflowContext:
        """Ingest and validate data"""
        async with self._trace_span("ingest_data", context):
            try:
                # Validate data
                if not context.data:
                    raise ValueError("No data provided")
                    
                # Extract features for TDA
                context.metadata["ingestion_time"] = datetime.utcnow()
                context.current_state = WorkflowState.INGESTING
                
                logger.info("Data ingested", 
                           workflow_id=context.workflow_id,
                           data_size=len(str(context.data)))
                
                return context
                
            except Exception as e:
                context.error = str(e)
                context.current_state = WorkflowState.FAILED
                logger.error("Ingestion failed", error=str(e))
                raise
                
    async def _analyze_tda(self, context: WorkflowContext) -> WorkflowContext:
        """Run TDA analysis with feature flag control"""
        async with self._trace_span("analyze_tda", context):
            try:
                context.current_state = WorkflowState.ANALYZING_TDA
                
                # Check feature flags for algorithm selection
                algorithms = []
                if await self.feature_flags.is_enabled("tda.specseq_plus"):
                    algorithms.append("specseq++")
                if await self.feature_flags.is_enabled("tda.simba_gpu"):
                    algorithms.append("simba_gpu")
                if await self.feature_flags.is_enabled("tda.neural_surveillance"):
                    algorithms.append("neural_surveillance")
                    
                if not algorithms:
                    algorithms = ["deterministic_fallback"]  # Always have fallback
                    
                # Run TDA analysis
                tda_result = await self.tda_engine.analyze(
                    data=context.data.get("features", []),
                    algorithms=algorithms,
                    trace_id=context.trace_id
                )
                
                # Store results
                node_id = await self.neo4j_adapter.store_tda_result(
                    result=tda_result,
                    parent_data_id=context.data.get("id"),
                    context={"workflow_id": context.workflow_id}
                )
                
                memory_id = await self.mem0_adapter.store_tda_memory(
                    result=tda_result,
                    agent_id="workflow_orchestrator",
                    context=context.metadata
                )
                
                context.tda_results = {
                    "result": tda_result.dict(),
                    "neo4j_id": node_id,
                    "memory_id": memory_id,
                    "algorithms_used": algorithms
                }
                
                # Extract signals
                if tda_result.anomaly_score > 0.7:
                    context.anomaly_signals.append(TopologicalSignal.ANOMALY_DETECTED)
                    
                self.metrics.increment("workflow.tda.completed",
                                     tags={"algorithms": ",".join(algorithms)})
                
                return context
                
            except Exception as e:
                logger.error("TDA analysis failed", error=str(e))
                # Use fallback
                if await self.feature_flags.is_enabled("tda.auto_fallback"):
                    return await self._run_fallback_tda(context)
                raise
                
    async def _enrich_context(self, context: WorkflowContext) -> WorkflowContext:
        """Enrich context with TDA insights"""
        async with self._trace_span("enrich_context", context):
            context.current_state = WorkflowState.ENRICHING_CONTEXT
            
            # Get enriched context for each agent
            enriched_contexts = {}
            
            for agent_id in ["observer", "analyst", "supervisor"]:
                enriched = await self.context_adapter.enrich_agent_context(
                    agent_id=agent_id,
                    base_context=self._create_base_context(context),
                    data_id=context.data.get("id")
                )
                enriched_contexts[agent_id] = enriched
                
            context.enriched_context = {
                "agent_contexts": enriched_contexts,
                "aggregated_signals": list(set(
                    signal 
                    for ctx in enriched_contexts.values() 
                    for signal in ctx.anomaly_signals
                )),
                "confidence_scores": {
                    agent_id: ctx.confidence_score 
                    for agent_id, ctx in enriched_contexts.items()
                }
            }
            
            return context
            
    async def _agent_deliberation(self, context: WorkflowContext) -> WorkflowContext:
        """Multi-agent deliberation with council consensus"""
        async with self._trace_span("agent_deliberation", context):
            context.current_state = WorkflowState.AGENT_DELIBERATION
            
            # Prepare task for council
            task = {
                "type": "analyze_and_decide",
                "data": context.data,
                "tda_insights": context.tda_results,
                "enriched_contexts": context.enriched_context,
                "anomaly_signals": context.anomaly_signals
            }
            
            # Get council decision
            decision = await self.agent_council.deliberate(
                task=task,
                timeout=30.0,
                require_consensus=True
            )
            
            context.agent_decisions.append({
                "timestamp": datetime.utcnow(),
                "decision": decision.action,
                "confidence": decision.confidence,
                "votes": decision.votes,
                "reasoning": decision.reasoning
            })
            
            logger.info("Agent council decision",
                       decision=decision.action,
                       confidence=decision.confidence)
            
            return context
            
    async def _execute_action(self, context: WorkflowContext) -> WorkflowContext:
        """Execute the decided action"""
        async with self._trace_span("execute_action", context):
            context.current_state = WorkflowState.EXECUTING_ACTION
            
            if not context.agent_decisions:
                raise ValueError("No agent decision to execute")
                
            latest_decision = context.agent_decisions[-1]
            action = latest_decision["decision"]
            
            # Execute based on action type
            if action == "mitigate_anomaly":
                await self._execute_mitigation(context)
            elif action == "adjust_parameters":
                await self._execute_parameter_adjustment(context)
            elif action == "alert_human":
                await self._execute_human_alert(context)
            else:
                logger.warning(f"Unknown action: {action}")
                
            return context
            
    async def _monitor_state(self, context: WorkflowContext) -> WorkflowContext:
        """Monitor system state"""
        async with self._trace_span("monitor_state", context):
            context.current_state = WorkflowState.MONITORING
            
            # Set up continuous monitoring
            monitor_config = {
                "workflow_id": context.workflow_id,
                "metrics_to_track": ["anomaly_score", "system_load", "error_rate"],
                "alert_thresholds": {
                    "anomaly_score": 0.8,
                    "system_load": 0.9,
                    "error_rate": 0.1
                },
                "check_interval": 60  # seconds
            }
            
            context.metadata["monitoring_config"] = monitor_config
            
            return context
            
    async def _escalate_anomaly(self, context: WorkflowContext) -> WorkflowContext:
        """Escalate high-severity anomalies"""
        async with self._trace_span("escalate_anomaly", context):
            context.current_state = WorkflowState.ESCALATING
            
            escalation = {
                "severity": "high",
                "anomaly_score": context.tda_results.get("result", {}).get("anomaly_score"),
                "signals": [s.value for s in context.anomaly_signals],
                "recommended_actions": [
                    "Immediate investigation required",
                    "Consider system isolation",
                    "Prepare rollback procedures"
                ],
                "notified_parties": ["ops_team", "security_team", "data_science_team"]
            }
            
            context.metadata["escalation"] = escalation
            
            # Send notifications
            await self._send_escalation_notifications(escalation)
            
            return context
            
    async def _create_checkpoint(self, context: WorkflowContext) -> WorkflowContext:
        """Create workflow checkpoint for recovery"""
        async with self._trace_span("create_checkpoint", context):
            checkpoint = {
                "checkpoint_id": f"ckpt_{context.workflow_id}_{datetime.utcnow().timestamp()}",
                "workflow_state": context.current_state.value,
                "timestamp": datetime.utcnow(),
                "data_snapshot": {
                    "tda_results": context.tda_results,
                    "agent_decisions": context.agent_decisions,
                    "anomaly_signals": [s.value for s in context.anomaly_signals]
                },
                "recovery_metadata": {
                    "can_resume": True,
                    "resume_from_state": context.current_state.value,
                    "required_components": ["tda_engine", "agent_council"]
                }
            }
            
            context.checkpoints.append(checkpoint)
            context.completed_at = datetime.utcnow()
            context.current_state = WorkflowState.COMPLETED
            
            # Persist checkpoint
            await self._persist_checkpoint(checkpoint)
            
            return context
            
    def _route_after_tda(self, context: WorkflowContext) -> str:
        """Conditional routing after TDA analysis"""
        if context.error:
            return "error"
        elif TopologicalSignal.ANOMALY_DETECTED in context.anomaly_signals:
            if context.tda_results.get("result", {}).get("anomaly_score", 0) > 0.9:
                return "anomaly"
        return "normal"
        
    def _route_after_deliberation(self, context: WorkflowContext) -> str:
        """Conditional routing after agent deliberation"""
        if not context.agent_decisions:
            return "escalate"
            
        latest_decision = context.agent_decisions[-1]
        action = latest_decision["decision"]
        
        if action in ["mitigate_anomaly", "adjust_parameters", "alert_human"]:
            return "execute"
        elif action == "escalate":
            return "escalate"
        else:
            return "monitor"
            
    @asynccontextmanager
    async def _trace_span(self, operation: str, context: WorkflowContext):
        """Create tracing span for operation"""
        span = TracingContext.start_span(
            operation,
            attributes={
                "workflow_id": context.workflow_id,
                "trace_id": context.trace_id,
                "state": context.current_state.value
            }
        )
        try:
            yield span
        finally:
            span.end()
            
    async def execute_workflow(
        self,
        data: Dict[str, Any],
        workflow_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> WorkflowContext:
        """Execute the complete workflow"""
        context = WorkflowContext(
            workflow_id=workflow_id or f"wf_{datetime.utcnow().timestamp()}",
            trace_id=trace_id or f"trace_{datetime.utcnow().timestamp()}",
            data=data
        )
        
        try:
            # Execute workflow
            result = await self.graph.ainvoke(
                context,
                config={"configurable": {"thread_id": context.workflow_id}}
            )
            
            self.metrics.increment("workflow.completed",
                                 tags={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error("Workflow execution failed",
                        workflow_id=context.workflow_id,
                        error=str(e))
            
            self.metrics.increment("workflow.completed",
                                 tags={"status": "failed"})
            
            # Attempt recovery
            if await self.feature_flags.is_enabled("workflow.auto_recovery"):
                return await self._attempt_recovery(context, e)
                
            raise
            
    async def _attempt_recovery(
        self,
        context: WorkflowContext,
        error: Exception
    ) -> WorkflowContext:
        """Attempt to recover from workflow failure"""
        logger.info("Attempting workflow recovery",
                   workflow_id=context.workflow_id)
        
        # Find last checkpoint
        if context.checkpoints:
            last_checkpoint = context.checkpoints[-1]
            
            # Restore from checkpoint
            context.current_state = WorkflowState(
                last_checkpoint["recovery_metadata"]["resume_from_state"]
            )
            
            # Resume workflow
            return await self.graph.ainvoke(
                context,
                config={
                    "configurable": {
                        "thread_id": context.workflow_id,
                        "checkpoint_id": last_checkpoint["checkpoint_id"]
                    }
                }
            )
            
        # No checkpoint available
        context.error = f"Recovery failed: {str(error)}"
        context.current_state = WorkflowState.FAILED
        return context
        
    def _create_base_context(self, workflow_context: WorkflowContext) -> Any:
        """Create base context for agents"""
        # This would integrate with your actual agent context structure
        return {
            "current_task_id": workflow_context.workflow_id,
            "trace_id": workflow_context.trace_id,
            "start_time": workflow_context.started_at,
            "metadata": workflow_context.metadata
        }
        
    async def _run_fallback_tda(self, context: WorkflowContext) -> WorkflowContext:
        """Run deterministic fallback TDA"""
        from ..tda.production_fallbacks import DETERMINISTIC_FALLBACK
        
        result = DETERMINISTIC_FALLBACK.compute(
            data=context.data.get("features", []),
            trace_id=context.trace_id
        )
        
        context.tda_results = {
            "result": result.dict(),
            "fallback_used": True,
            "algorithms_used": ["deterministic_fallback"]
        }
        
        return context
        
    async def _execute_mitigation(self, context: WorkflowContext):
        """Execute anomaly mitigation"""
        logger.info("Executing anomaly mitigation",
                   workflow_id=context.workflow_id)
        # Implementation specific to your system
        
    async def _execute_parameter_adjustment(self, context: WorkflowContext):
        """Execute parameter adjustment"""
        logger.info("Executing parameter adjustment",
                   workflow_id=context.workflow_id)
        # Implementation specific to your system
        
    async def _execute_human_alert(self, context: WorkflowContext):
        """Send alert to human operators"""
        logger.info("Sending human alert",
                   workflow_id=context.workflow_id)
        # Implementation specific to your system
        
    async def _send_escalation_notifications(self, escalation: Dict[str, Any]):
        """Send escalation notifications"""
        logger.info("Sending escalation notifications",
                   severity=escalation["severity"])
        # Implementation specific to your notification system
        
    async def _persist_checkpoint(self, checkpoint: Dict[str, Any]):
        """Persist checkpoint to storage"""
        # Store in your checkpoint storage system
        pass
        
    def _register_default_components(self):
        """Register default workflow components"""
        # Register extensible components
        pass