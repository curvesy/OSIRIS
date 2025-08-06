"""
Enhanced Workflow Orchestrator with PostgresSaver and Cross-Thread Memory
Production-grade persistence and memory sharing for AURA Intelligence
"""

import asyncio
import os
from typing import Dict, List, Optional, Any, Callable, TypeVar, Protocol
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
import structlog
from contextlib import asynccontextmanager
import uuid
import json

from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    # Fallback for development
    from langgraph.checkpoint.memory import MemorySaver as PostgresSaver
    AsyncPostgresSaver = PostgresSaver

try:
    from langgraph.store.postgres import PostgresStore
except ImportError:
    # Fallback for development
    PostgresStore = None

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError:
    # Fallback for development
    AsyncConnectionPool = None
    
import numpy as np

from ..adapters.tda_neo4j_adapter import TDANeo4jAdapter
from ..adapters.tda_mem0_adapter import TDAMem0Adapter
from ..adapters.tda_agent_context import TDAAgentContextAdapter, TopologicalSignal
from ..tda_engine import TDAEngine
from ..agents.council import AgentCouncil
from ..observability import MetricsCollector, TracingContext
from ..feature_flags import FeatureFlags
from ..config.base import get_config
from ..events.event_bus import EventBus

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


class EnhancedWorkflowContext(TypedDict):
    """Enhanced workflow context with cross-thread memory support"""
    workflow_id: str
    thread_id: str  # For cross-thread memory
    trace_id: str
    data: Dict[str, Any]
    tda_results: Optional[Dict[str, Any]]
    enriched_context: Optional[Dict[str, Any]]
    agent_decisions: List[Dict[str, Any]]
    current_state: WorkflowState
    checkpoints: List[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    # Cross-thread memory references
    memory_keys: List[str]
    shared_context: Dict[str, Any]


# For backwards compatibility with dataclass usage
@dataclass
class WorkflowContextData:
    """Dataclass version of workflow context for easy creation"""
    workflow_id: str
    thread_id: str
    trace_id: str
    data: Dict[str, Any]
    tda_results: Optional[Dict[str, Any]] = None
    enriched_context: Optional[Dict[str, Any]] = None
    agent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    current_state: WorkflowState = WorkflowState.INGESTING
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_keys: List[str] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> EnhancedWorkflowContext:
        """Convert to TypedDict format"""
        return {
            "workflow_id": self.workflow_id,
            "thread_id": self.thread_id,
            "trace_id": self.trace_id,
            "data": self.data,
            "tda_results": self.tda_results,
            "enriched_context": self.enriched_context,
            "agent_decisions": self.agent_decisions,
            "current_state": self.current_state,
            "checkpoints": self.checkpoints,
            "error": self.error,
            "metadata": self.metadata,
            "memory_keys": self.memory_keys,
            "shared_context": self.shared_context
        }


class EnhancedWorkflowOrchestrator:
    """
    Enhanced orchestrator with production-grade persistence and memory.
    
    Key improvements:
    - PostgresSaver for durable checkpointing
    - Cross-thread memory with Store interface
    - Connection pooling for performance
    - Integrated with existing AURA components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.aura_config = get_config()
        
        # Database connection pool
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[Any] = None  # PostgresSaver or MemorySaver
        self.memory_store: Optional[PostgresStore] = None
        
        # Core components
        self.event_bus = EventBus()
        self.tda_engine = TDAEngine()
        self.neo4j_adapter = TDANeo4jAdapter()
        self.mem0_adapter = TDAMem0Adapter()
        self.context_adapter = TDAAgentContextAdapter(
            neo4j_adapter=self.neo4j_adapter,
            mem0_adapter=self.mem0_adapter,
            event_bus=self.event_bus
        )
        self.agent_council = AgentCouncil()
        self.feature_flags = FeatureFlags()
        self.metrics = MetricsCollector  # Already an instance
        
        # Workflow graph
        self.graph: Optional[StateGraph] = None
        
    async def initialize(self):
        """Initialize database connections and components"""
        try:
            # Create connection pool if available
            postgres_url = self.config.get(
                'postgres_url',
                os.getenv('POSTGRES_URL', 'postgresql://localhost:5432/aura')
            )
            
            if AsyncConnectionPool and AsyncPostgresSaver != PostgresSaver and postgres_url and postgres_url != '':
                
                self.pool = AsyncConnectionPool(
                    conninfo=postgres_url,
                    max_size=20,
                    min_size=5
                )
                
                # Initialize PostgresSaver with pool
                self.checkpointer = AsyncPostgresSaver(self.pool)
                await self.checkpointer.setup()  # Create tables if needed
                
                # Initialize Store for cross-thread memory
                if PostgresStore:
                    self.memory_store = PostgresStore(
                        connection_pool=self.pool,
                        table_prefix="aura_memory"
                    )
                    await self.memory_store.setup()
            else:
                # Use in-memory fallback
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
            
            # Initialize other components
            await self.neo4j_adapter.initialize()
            await self.mem0_adapter.initialize()
            await self.agent_council.initialize()
            
            # Build workflow graph
            self._build_graph()
            
            logger.info("Enhanced workflow orchestrator initialized",
                       postgres_available=bool(AsyncConnectionPool),
                       checkpointer=type(self.checkpointer).__name__,
                       memory_store=bool(self.memory_store))
                       
        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise
    
    def _build_graph(self):
        """Build the LangGraph workflow with all nodes"""
        workflow = StateGraph(EnhancedWorkflowContext)
        
        # Add nodes
        workflow.add_node("ingest", self._ingest_data)
        workflow.add_node("analyze_tda", self._analyze_tda)
        workflow.add_node("enrich_context", self._enrich_context)
        workflow.add_node("deliberate", self._agent_deliberation)
        workflow.add_node("execute", self._execute_action)
        workflow.add_node("monitor", self._monitor_results)
        workflow.add_node("escalate", self._escalate_anomaly)
        
        # Add edges with conditions
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
        self.graph = workflow.compile(
            checkpointer=self.checkpointer,
            store=self.memory_store  # Enable cross-thread memory
        )
    
    async def _ingest_data(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Ingest and validate data"""
        state["current_state"] = WorkflowState.INGESTING
        
        # Store in cross-thread memory if available
        if self.memory_store:
            memory_key = f"ingestion_{state['workflow_id']}_{datetime.now(timezone.utc).isoformat()}"
            await self.memory_store.put(
                namespace=("workflows", state["workflow_id"]),
                key=memory_key,
                value={
                    "data": state["data"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "thread_id": state["thread_id"]
                }
            )
            state["memory_keys"].append(memory_key)
        
        logger.info("Data ingested", workflow_id=state["workflow_id"])
        return state
    
    async def _analyze_tda(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Run TDA analysis with feature flag control"""
        state["current_state"] = WorkflowState.ANALYZING_TDA
        
        try:
            # Check feature flags for algorithm selection
            algorithm = "specseq++"  # default
            if await self.feature_flags.is_enabled("tda.simba_gpu"):
                algorithm = "simba_gpu"
            elif await self.feature_flags.is_enabled("tda.neural_surveillance"):
                algorithm = "neural_surveillance"
            
            # Run TDA analysis
            tda_results = await self.tda_engine.analyze(
                data=state["data"],
                algorithm=algorithm,
                use_gpu=await self.feature_flags.is_enabled("tda.gpu_acceleration")
            )
            
            state["tda_results"] = tda_results
            
            # Store TDA results
            await self.neo4j_adapter.store_tda_result(
                tda_id=f"tda_{state['workflow_id']}",
                result=tda_results
            )
            
            # Store in cross-thread memory
            if self.memory_store:
                await self._store_tda_memory(state, tda_results)
            
            logger.info("TDA analysis complete",
                       workflow_id=state["workflow_id"],
                       algorithm=algorithm,
                       anomaly_score=tda_results.get("anomaly_score", 0))
            
        except Exception as e:
            logger.error("TDA analysis failed", error=str(e))
            if await self.feature_flags.is_enabled("tda.auto_fallback"):
                # Use fallback algorithm
                from ..tda.production_fallbacks import DeterministicTDAFallback
                fallback = DeterministicTDAFallback()
                state["tda_results"] = await fallback.analyze(state["data"])
            else:
                state["error"] = f"TDA analysis failed: {str(e)}"
                state["current_state"] = WorkflowState.FAILED
        
        return state
    
    async def _store_tda_memory(self, state: EnhancedWorkflowContext, tda_results: Dict[str, Any]):
        """Store TDA results in cross-thread memory with embeddings"""
        memory_key = f"tda_{state['workflow_id']}_{datetime.now(timezone.utc).isoformat()}"
        
        # Generate embedding for semantic search
        embedding = await self._generate_tda_embedding(tda_results)
        
        await self.memory_store.put(
            namespace=("tda_results", state["workflow_id"]),
            key=memory_key,
            value={
                "results": tda_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "thread_id": state["thread_id"],
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            }
        )
        
        state["memory_keys"].append(memory_key)
    
    async def _generate_tda_embedding(self, tda_results: Dict[str, Any]) -> np.ndarray:
        """Generate embedding vector for TDA results"""
        # Extract key features
        features = []
        features.append(tda_results.get("anomaly_score", 0))
        features.extend(tda_results.get("betti_numbers", [0, 0, 0]))
        features.extend(tda_results.get("persistence_features", {}).values())
        
        # Normalize and pad to fixed size
        embedding = np.array(features[:128])  # Fixed size
        if len(embedding) < 128:
            embedding = np.pad(embedding, (0, 128 - len(embedding)))
        
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    async def _enrich_context(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Enrich context with historical data and cross-thread memories"""
        state["current_state"] = WorkflowState.ENRICHING_CONTEXT
        
        # Get context from adapters
        enriched_context = await self.context_adapter.enrich_agent_context(
            tda_results=state["tda_results"],
            historical_window=timedelta(days=7)
        )
        
        # Search cross-thread memories if available
        if self.memory_store:
            # Search for similar TDA patterns
            similar_memories = await self.memory_store.search(
                namespace=("tda_results",),
                query=state["tda_results"].get("anomaly_score", 0),
                top_k=5
            )
            
            enriched_context["cross_thread_memories"] = [
                mem.value for mem in similar_memories
            ]
            
            # Get memories from other threads
            related_threads = await self._get_related_thread_memories(state)
            enriched_context["related_thread_insights"] = related_threads
        
        state["enriched_context"] = enriched_context
        logger.info("Context enriched",
                   workflow_id=state["workflow_id"],
                   historical_patterns=len(enriched_context.get("historical_patterns", [])),
                   cross_thread_memories=len(enriched_context.get("cross_thread_memories", [])))
        
        return state
    
    async def _get_related_thread_memories(self, state: EnhancedWorkflowContext) -> List[Dict[str, Any]]:
        """Retrieve memories from related workflow threads"""
        related_memories = []
        
        # Search by similar context
        namespace = ("workflows",)
        recent_workflows = await self.memory_store.list(
            namespace=namespace,
            limit=10
        )
        
        for workflow_mem in recent_workflows:
            if workflow_mem.key != state["workflow_id"]:
                # Check similarity
                similarity = self._calculate_context_similarity(
                    state["data"],
                    workflow_mem.value.get("data", {})
                )
                
                if similarity > 0.7:  # Threshold
                    related_memories.append({
                        "workflow_id": workflow_mem.key,
                        "similarity": similarity,
                        "insights": workflow_mem.value.get("insights", {})
                    })
        
        return related_memories
    
    def _calculate_context_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two data contexts"""
        # Simple Jaccard similarity for keys
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    async def _agent_deliberation(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Multi-agent deliberation with memory-aware decisions"""
        state["current_state"] = WorkflowState.AGENT_DELIBERATION
        
        # Prepare agent context with memories
        agent_context = {
            "tda_results": state["tda_results"],
            "enriched_context": state["enriched_context"],
            "historical_decisions": await self._get_historical_decisions(state),
            "workflow_metadata": state["metadata"]
        }
        
        # Get council decision
        decision = await self.agent_council.deliberate(
            context=agent_context,
            timeout=30.0
        )
        
        state["agent_decisions"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "participating_agents": decision.get("agents", [])
        })
        
        # Store decision in memory
        if self.memory_store:
            await self._store_decision_memory(state, decision)
        
        logger.info("Agent deliberation complete",
                   workflow_id=state["workflow_id"],
                   decision=decision.get("action"),
                   confidence=decision.get("confidence"))
        
        return state
    
    async def _get_historical_decisions(self, state: EnhancedWorkflowContext) -> List[Dict[str, Any]]:
        """Retrieve historical decisions from memory"""
        if not self.memory_store:
            return []
        
        decisions = await self.memory_store.search(
            namespace=("decisions",),
            query="similar_context",
            filter={
                "anomaly_score": {
                    "$gte": state["tda_results"].get("anomaly_score", 0) - 0.1,
                    "$lte": state["tda_results"].get("anomaly_score", 0) + 0.1
                }
            },
            top_k=5
        )
        
        return [d.value for d in decisions]
    
    async def _store_decision_memory(self, state: EnhancedWorkflowContext, decision: Dict[str, Any]):
        """Store agent decision in cross-thread memory"""
        await self.memory_store.put(
            namespace=("decisions", state["workflow_id"]),
            key=f"decision_{datetime.now(timezone.utc).isoformat()}",
            value={
                "decision": decision,
                "context": {
                    "anomaly_score": state["tda_results"].get("anomaly_score", 0),
                    "tda_algorithm": state["tda_results"].get("algorithm"),
                    "enrichment_sources": list(state["enriched_context"].keys())
                },
                "thread_id": state["thread_id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def _route_after_deliberation(self, state: EnhancedWorkflowContext) -> str:
        """Route based on agent decision"""
        if not state["agent_decisions"]:
            return "monitor"
        
        latest_decision = state["agent_decisions"][-1]["decision"]
        action = latest_decision.get("action", "monitor")
        
        if action == "escalate" or latest_decision.get("risk_level") == "critical":
            return "escalate"
        elif action in ["execute", "remediate", "allocate"]:
            return "execute"
        else:
            return "monitor"
    
    async def _execute_action(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Execute the decided action"""
        state["current_state"] = WorkflowState.EXECUTING_ACTION
        
        latest_decision = state["agent_decisions"][-1]["decision"]
        action = latest_decision.get("action")
        
        logger.info("Executing action",
                   workflow_id=state["workflow_id"],
                   action=action)
        
        # Action execution would go here
        # For now, just record it
        state["metadata"]["executed_action"] = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed"
        }
        
        return state
    
    async def _escalate_anomaly(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Escalate critical anomalies"""
        state["current_state"] = WorkflowState.ESCALATING
        
        logger.warning("Escalating anomaly",
                      workflow_id=state["workflow_id"],
                      anomaly_score=state["tda_results"].get("anomaly_score"))
        
        # Escalation logic would go here
        state["metadata"]["escalation"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "High anomaly score",
            "notified": ["ops-team", "security-team"]
        }
        
        return state
    
    async def _monitor_results(self, state: EnhancedWorkflowContext) -> EnhancedWorkflowContext:
        """Monitor and record results"""
        state["current_state"] = WorkflowState.MONITORING
        
        # Update metrics
        self.metrics.record_workflow_completion(
            workflow_id=state["workflow_id"],
            duration=(datetime.now(timezone.utc) - datetime.fromisoformat(
                state["metadata"].get("start_time", datetime.now(timezone.utc).isoformat())
            )).total_seconds(),
            success=state["error"] is None
        )
        
        state["current_state"] = WorkflowState.COMPLETED
        logger.info("Workflow completed",
                   workflow_id=state["workflow_id"],
                   final_state=state["current_state"])
        
        return state
    
    async def execute_workflow(
        self,
        data: Dict[str, Any],
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowContextData:
        """Execute a complete workflow with persistence"""
        workflow_id = str(uuid.uuid4())
        thread_id = thread_id or str(uuid.uuid4())
        
        initial_state: EnhancedWorkflowContext = {
            "workflow_id": workflow_id,
            "thread_id": thread_id,
            "trace_id": str(uuid.uuid4()),
            "data": data,
            "tda_results": None,
            "enriched_context": None,
            "agent_decisions": [],
            "current_state": WorkflowState.INGESTING,
            "checkpoints": [],
            "error": None,
            "metadata": metadata or {},
            "memory_keys": [],
            "shared_context": {}
        }
        
        initial_state["metadata"]["start_time"] = datetime.now(timezone.utc).isoformat()
        
        # Execute with checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": f"workflow_{workflow_id}"
            }
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            # Convert back to dataclass for compatibility
            return WorkflowContextData(**final_state)
        except Exception as e:
            logger.error("Workflow execution failed",
                        workflow_id=workflow_id,
                        error=str(e))
            raise
    
    async def get_workflow_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get workflow history from checkpoints"""
        history = []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        async for checkpoint in self.checkpointer.list(config):
            history.append({
                "checkpoint_id": checkpoint["checkpoint_id"],
                "timestamp": checkpoint["timestamp"],
                "state": checkpoint.get("channel_values", {})
            })
        
        return history
    
    async def resume_workflow(self, thread_id: str, checkpoint_id: str) -> WorkflowContextData:
        """Resume workflow from a specific checkpoint"""
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }
        
        # Get checkpoint
        checkpoint = await self.checkpointer.get(config)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Resume execution
        state = checkpoint["channel_values"]
        final_state = await self.graph.ainvoke(state, config)
        
        # Convert back to dataclass
        return WorkflowContextData(**final_state)
    
    async def search_memories(
        self,
        query: str,
        namespace: Optional[tuple] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search cross-thread memories"""
        if not self.memory_store:
            return []
        
        results = await self.memory_store.search(
            namespace=namespace or ("workflows",),
            query=query,
            top_k=top_k
        )
        
        return [
            {
                "key": r.key,
                "value": r.value,
                "score": getattr(r, "score", 0.0)
            }
            for r in results
        ]
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.pool:
            await self.pool.close()
        
        await self.neo4j_adapter.cleanup()
        await self.mem0_adapter.cleanup()
        await self.agent_council.cleanup()


# Export the dataclass version for backward compatibility
EnhancedWorkflowContext = WorkflowContextData