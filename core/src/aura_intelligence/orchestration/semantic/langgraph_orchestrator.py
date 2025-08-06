"""
🧠 LangGraph Semantic Orchestrator

2025 LangGraph StateGraph integration with semantic routing and TDA context.
Implements orchestrator-worker patterns with checkpointing and fault tolerance.

Key Features:
- StateGraph with MemorySaver checkpointing
- Semantic task decomposition with TDA context
- Dynamic agent selection based on capabilities
- Fault-tolerant execution with automatic recovery

TDA Integration:
- Uses TDA patterns for semantic routing decisions
- Correlates orchestration with TDA anomaly data
- Sends results to TDA for pattern analysis
"""

from typing import Dict, Any, List, Optional, Callable
import asyncio
import uuid
from datetime import datetime, timezone

# LangGraph imports with fallback
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    
    # 2025 Enhanced checkpointing and memory
    try:
        from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver
        from langgraph.store.postgres import PostgresStore
        from langgraph.store.redis import RedisStore
        POSTGRES_CHECKPOINTING_AVAILABLE = True
    except ImportError:
        PostgresSaver = None
        AsyncPostgresSaver = None
        PostgresStore = None
        RedisStore = None
        POSTGRES_CHECKPOINTING_AVAILABLE = False
    
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for environments without LangGraph
    BaseMessage = Dict[str, Any]
    HumanMessage = Dict[str, Any] 
    AIMessage = Dict[str, Any]
    StateGraph = None
    MemorySaver = None
    PostgresSaver = None
    AsyncPostgresSaver = None
    PostgresStore = None
    RedisStore = None
    LANGGRAPH_AVAILABLE = False
    POSTGRES_CHECKPOINTING_AVAILABLE = False

from .base_interfaces import (
    AgentState, SemanticOrchestrator, TDAContext, 
    SemanticAnalysis, OrchestrationStrategy, UrgencyLevel
)

# TDA integration
try:
    from aura_intelligence.observability.tracing import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None

class SemanticWorkflowConfig:
    """Configuration for semantic workflows"""
    
    def __init__(
        self,
        workflow_id: str,
        orchestrator_agent: str,
        worker_agents: List[str],
        routing_strategy: OrchestrationStrategy = OrchestrationStrategy.SEMANTIC,
        max_retries: int = 3,
        timeout_seconds: int = 300
    ):
        self.workflow_id = workflow_id
        self.orchestrator_agent = orchestrator_agent
        self.worker_agents = worker_agents
        self.routing_strategy = routing_strategy
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

class LangGraphSemanticOrchestrator(SemanticOrchestrator):
    """
    2025 LangGraph-based semantic orchestrator with enhanced persistence and memory
    """
    
    def __init__(
        self, 
        tda_integration=None,
        postgres_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        enable_cross_thread_memory: bool = True
    ):
        self.tda_integration = tda_integration
        self.active_workflows: Dict[str, Any] = {}
        self.enable_cross_thread_memory = enable_cross_thread_memory
        
        # Initialize enhanced checkpointing
        self.checkpointer = self._initialize_checkpointer(postgres_url)
        self.memory_store = self._initialize_memory_store(postgres_url, redis_url)
    
    def _initialize_checkpointer(self, postgres_url: Optional[str]):
        """Initialize the best available checkpointer"""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        if POSTGRES_CHECKPOINTING_AVAILABLE and postgres_url:
            try:
                # Use PostgresSaver for production-grade persistence
                return PostgresSaver.from_conn_string(
                    postgres_url,
                    pool_size=20  # Connection pooling for production
                )
            except Exception as e:
                print(f"Warning: Failed to initialize PostgresSaver: {e}")
                return MemorySaver()  # Fallback to memory
        else:
            # Fallback to MemorySaver
            return MemorySaver() if LANGGRAPH_AVAILABLE else None
    
    def _initialize_memory_store(self, postgres_url: Optional[str], redis_url: Optional[str]):
        """Initialize cross-thread memory store"""
        if not self.enable_cross_thread_memory or not POSTGRES_CHECKPOINTING_AVAILABLE:
            return None
            
        try:
            # Prefer Redis for vector search capabilities
            if redis_url and RedisStore:
                return RedisStore(
                    redis_url=redis_url,
                    index_embeddings=True,  # Enable vector search
                    namespace_prefix="aura"
                )
            # Fallback to PostgresStore
            elif postgres_url and PostgresStore:
                return PostgresStore(
                    connection_string=postgres_url,
                    namespace_prefix="aura"
                )
        except Exception as e:
            print(f"Warning: Failed to initialize memory store: {e}")
            
        return None
        
    async def create_orchestrator_worker_graph(
        self, 
        config: SemanticWorkflowConfig
    ) -> Optional[Any]:
        """
        Create LangGraph StateGraph with enhanced persistence and memory
        """
        if not LANGGRAPH_AVAILABLE:
            return await self._fallback_orchestration(config)
            
        workflow = StateGraph(AgentState)
        
        # Add orchestrator node with semantic analysis and memory
        workflow.add_node("orchestrator", self._enhanced_orchestrator_node)
        
        # Add worker nodes dynamically with memory access
        for i, agent in enumerate(config.worker_agents):
            node_name = f"worker_{i}"
            workflow.add_node(node_name, self._create_enhanced_worker_node(agent))
        
        # Add semantic routing with conditional edges
        workflow.add_conditional_edges(
            "orchestrator",
            self._semantic_router_decision,
            {f"worker_{i}": f"worker_{i}" for i in range(len(config.worker_agents))}
        )
        
        # Add result aggregation with memory storage
        workflow.add_node("aggregator", self._enhanced_aggregation_node)
        
        # Connect workers to aggregator
        for i in range(len(config.worker_agents)):
            workflow.add_edge(f"worker_{i}", "aggregator")
        
        workflow.set_entry_point("orchestrator")
        workflow.set_finish_point("aggregator")
        
        # Compile with enhanced checkpointer and memory store
        compile_kwargs = {"checkpointer": self.checkpointer}
        if self.memory_store:
            compile_kwargs["store"] = self.memory_store
            
        return workflow.compile(**compile_kwargs)
    
    async def _enhanced_orchestrator_node(self, state: AgentState, *, store=None) -> AgentState:
        """
        Enhanced orchestrator with semantic task decomposition and cross-thread memory
        """
        if tracer:
            with tracer.start_as_current_span("enhanced_semantic_orchestration") as span:
                span.set_attributes({
                    "orchestration.type": "semantic_enhanced",
                    "workflow.id": state.get("workflow_metadata", {}).get("workflow_id", "unknown"),
                    "memory_enabled": store is not None
                })
        
        # Get TDA context for semantic analysis
        tda_context = await self._get_tda_context(state)
        
        # Enhance with cross-thread memory if available
        if store and tda_context:
            tda_context = await self._enhance_with_memory(tda_context, store, state)
        
        # Perform semantic task analysis
        task_analysis = await self.analyze_semantically(
            state.get("context", {}),
            tda_context
        )
        
        # Store current context for future reference
        if store:
            await self._store_orchestration_context(store, state, task_analysis, tda_context)
        
        # Update state with orchestration metadata
        state["workflow_metadata"].update({
            "task_analysis": task_analysis.__dict__,
            "orchestration_strategy": "semantic_decomposition_enhanced",
            "tda_integration": True,
            "memory_enhanced": store is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        state["tda_context"] = tda_context
        
        return state
    
    async def _semantic_router_decision(self, state: AgentState) -> str:
        """
        2025 semantic routing with TDA-aware contextual decisions
        """
        task_analysis = state["workflow_metadata"].get("task_analysis", {})
        
        # Route based on complexity and TDA context
        if task_analysis.get("complexity_score", 0) > 0.8:
            return "parallel_execution"
        elif task_analysis.get("urgency_level") == "critical":
            return "immediate_execution"
        else:
            return "sequential_execution"
    
    async def _create_worker_node(self, agent_name: str) -> Callable:
        """Create worker node for specific agent"""
        async def worker_node(state: AgentState) -> AgentState:
            # Execute agent with TDA context
            result = await self._execute_agent_with_context(
                agent_name, 
                state["context"],
                state.get("tda_context")
            )
            
            state["agent_outputs"][agent_name] = result
            state["execution_trace"].append({
                "agent": agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result_summary": str(result)[:100]  # Truncate for logging
            })
            
            return state
        
        return worker_node
    
    async def _aggregation_node(self, state: AgentState) -> AgentState:
        """Aggregate results from all workers"""
        aggregated_result = {
            "workflow_id": state["workflow_metadata"].get("workflow_id"),
            "agent_outputs": state["agent_outputs"],
            "execution_summary": {
                "total_agents": len(state["agent_outputs"]),
                "execution_time": len(state["execution_trace"]),
                "tda_correlation": state.get("tda_context", {}).get("correlation_id")
            }
        }
        
        # Send result to TDA for pattern analysis
        if self.tda_integration and state.get("tda_context"):
            await self.tda_integration.send_orchestration_result(
                aggregated_result,
                state["tda_context"]["correlation_id"]
            )
        
        state["final_result"] = aggregated_result
        return state
    
    async def analyze_semantically(
        self, 
        input_data: Dict[str, Any],
        tda_context: Optional[TDAContext] = None
    ) -> SemanticAnalysis:
        """Perform semantic analysis with TDA correlation"""
        
        # Basic semantic analysis (can be enhanced with ML models)
        complexity_score = self._calculate_complexity(input_data)
        urgency_level = self._determine_urgency(input_data, tda_context)
        
        # TDA-enhanced pattern recognition
        if tda_context:
            complexity_score *= (1 + tda_context.pattern_confidence)
            if tda_context.anomaly_severity > 0.7:
                urgency_level = UrgencyLevel.HIGH
        
        return SemanticAnalysis(
            complexity_score=min(complexity_score, 1.0),
            urgency_level=urgency_level,
            coordination_pattern=self._select_coordination_pattern(complexity_score),
            suggested_agents=self._suggest_agents(input_data, tda_context),
            confidence=0.85,  # Base confidence, can be ML-enhanced
            tda_correlation=tda_context
        )
    
    def _calculate_complexity(self, input_data: Dict[str, Any]) -> float:
        """Calculate task complexity score"""
        # Simple heuristic - can be replaced with ML model
        factors = [
            len(str(input_data)) / 1000,  # Data size factor
            len(input_data.get("requirements", [])) / 10,  # Requirements factor
            1.0 if input_data.get("requires_consensus") else 0.0  # Consensus factor
        ]
        return min(sum(factors) / len(factors), 1.0)
    
    def _determine_urgency(
        self, 
        input_data: Dict[str, Any], 
        tda_context: Optional[TDAContext]
    ) -> UrgencyLevel:
        """Determine urgency level with TDA amplification"""
        base_urgency = input_data.get("urgency", "medium")
        
        # TDA anomaly amplification
        if tda_context and tda_context.anomaly_severity > 0.8:
            return UrgencyLevel.CRITICAL
        elif tda_context and tda_context.anomaly_severity > 0.6:
            return UrgencyLevel.HIGH
        
        return UrgencyLevel(base_urgency.lower())
    
    async def _get_tda_context(self, state: AgentState) -> Optional[TDAContext]:
        """Get TDA context for workflow"""
        if not self.tda_integration:
            return None
            
        correlation_id = state.get("workflow_metadata", {}).get("correlation_id")
        if correlation_id:
            return await self.tda_integration.get_context(correlation_id)
        
        return None
    
    async def _enhance_with_memory(self, tda_context: TDAContext, store, state: AgentState) -> TDAContext:
        """Enhance TDA context with historical patterns from cross-thread memory"""
        try:
            user_id = state.get("workflow_metadata", {}).get("user_id", "system")
            workflow_type = state.get("workflow_metadata", {}).get("workflow_type", "unknown")
            
            # Search for similar historical contexts
            namespace = ("tda", "contexts", user_id)
            
            # Query for similar anomaly patterns
            query_context = {
                "anomaly_severity": tda_context.anomaly_severity,
                "complexity_score": getattr(tda_context, 'complexity_score', 0.5),
                "workflow_type": workflow_type
            }
            
            similar_contexts = await store.search(
                namespace=namespace,
                query=f"anomaly_severity:{tda_context.anomaly_severity:.2f} type:{workflow_type}",
                top_k=3
            )
            
            # Enhance context with historical patterns
            if similar_contexts:
                historical_patterns = []
                for context in similar_contexts:
                    if context.value.get("outcome") == "success":
                        historical_patterns.append({
                            "pattern": context.value.get("successful_strategy"),
                            "confidence": context.value.get("confidence", 0.5),
                            "timestamp": context.value.get("timestamp")
                        })
                
                # Add historical patterns to TDA context
                tda_context.historical_patterns = historical_patterns
                tda_context.pattern_confidence = min(
                    tda_context.pattern_confidence + 0.1 * len(historical_patterns), 1.0
                )
            
        except Exception as e:
            print(f"Warning: Failed to enhance with memory: {e}")
            
        return tda_context
    
    async def _store_orchestration_context(self, store, state: AgentState, task_analysis, tda_context):
        """Store current orchestration context for future learning"""
        try:
            user_id = state.get("workflow_metadata", {}).get("user_id", "system")
            workflow_id = state.get("workflow_metadata", {}).get("workflow_id", "unknown")
            
            namespace = ("tda", "contexts", user_id)
            
            context_data = {
                "workflow_id": workflow_id,
                "workflow_type": state.get("workflow_metadata", {}).get("workflow_type", "unknown"),
                "anomaly_severity": tda_context.anomaly_severity if tda_context else 0.0,
                "complexity_score": task_analysis.complexity_score,
                "urgency_level": task_analysis.urgency_level.value,
                "coordination_pattern": task_analysis.coordination_pattern.value,
                "suggested_agents": task_analysis.suggested_agents,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tda_correlation_id": getattr(tda_context, 'correlation_id', None) if tda_context else None
            }
            
            # Store with unique key
            key = f"context_{workflow_id}_{datetime.now(timezone.utc).timestamp()}"
            await store.put(namespace, key, context_data)
            
        except Exception as e:
            print(f"Warning: Failed to store orchestration context: {e}")
    
    async def _create_enhanced_worker_node(self, agent_name: str) -> Callable:
        """Create enhanced worker node with memory access"""
        async def enhanced_worker_node(state: AgentState, *, store=None) -> AgentState:
            # Get relevant memory for this agent
            agent_memory = None
            if store:
                agent_memory = await self._get_agent_memory(store, agent_name, state)
            
            # Execute agent with TDA context and memory
            result = await self._execute_agent_with_context_and_memory(
                agent_name, 
                state["context"],
                state.get("tda_context"),
                agent_memory
            )
            
            # Store agent result for future reference
            if store:
                await self._store_agent_result(store, agent_name, state, result)
            
            state["agent_outputs"][agent_name] = result
            state["execution_trace"].append({
                "agent": agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result_summary": str(result)[:100],  # Truncate for logging
                "memory_enhanced": agent_memory is not None
            })
            
            return state
        
        return enhanced_worker_node
    
    async def _enhanced_aggregation_node(self, state: AgentState, *, store=None) -> AgentState:
        """Enhanced aggregation with memory storage and learning"""
        aggregated_result = {
            "workflow_id": state["workflow_metadata"].get("workflow_id"),
            "agent_outputs": state["agent_outputs"],
            "execution_summary": {
                "total_agents": len(state["agent_outputs"]),
                "execution_time": len(state["execution_trace"]),
                "tda_correlation": state.get("tda_context", {}).get("correlation_id") if isinstance(state.get("tda_context"), dict) else getattr(state.get("tda_context"), 'correlation_id', None),
                "memory_enhanced": store is not None
            }
        }
        
        # Store successful workflow patterns for learning
        if store:
            await self._store_successful_workflow_pattern(store, state, aggregated_result)
        
        # Send result to TDA for pattern analysis
        if self.tda_integration and state.get("tda_context"):
            tda_correlation_id = state["tda_context"].correlation_id if hasattr(state["tda_context"], 'correlation_id') else state.get("tda_context", {}).get("correlation_id")
            if tda_correlation_id:
                await self.tda_integration.send_orchestration_result(
                    aggregated_result,
                    tda_correlation_id
                )
        
        state["final_result"] = aggregated_result
        return state
    
    async def _get_agent_memory(self, store, agent_name: str, state: AgentState):
        """Get relevant memory for specific agent"""
        try:
            user_id = state.get("workflow_metadata", {}).get("user_id", "system")
            namespace = ("agents", agent_name, user_id)
            
            # Search for relevant past experiences
            workflow_type = state.get("workflow_metadata", {}).get("workflow_type", "unknown")
            recent_memories = await store.search(
                namespace=namespace,
                query=f"type:{workflow_type}",
                top_k=5
            )
            
            return [memory.value for memory in recent_memories] if recent_memories else None
            
        except Exception as e:
            print(f"Warning: Failed to get agent memory for {agent_name}: {e}")
            return None
    
    async def _store_agent_result(self, store, agent_name: str, state: AgentState, result):
        """Store agent result for future learning"""
        try:
            user_id = state.get("workflow_metadata", {}).get("user_id", "system")
            workflow_id = state.get("workflow_metadata", {}).get("workflow_id", "unknown")
            
            namespace = ("agents", agent_name, user_id)
            
            memory_data = {
                "workflow_id": workflow_id,
                "workflow_type": state.get("workflow_metadata", {}).get("workflow_type", "unknown"),
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context_summary": str(state.get("context", {}))[:200]  # Truncated context
            }
            
            key = f"result_{workflow_id}_{datetime.now(timezone.utc).timestamp()}"
            await store.put(namespace, key, memory_data)
            
        except Exception as e:
            print(f"Warning: Failed to store agent result for {agent_name}: {e}")
    
    async def _store_successful_workflow_pattern(self, store, state: AgentState, result):
        """Store successful workflow patterns for future optimization"""
        try:
            user_id = state.get("workflow_metadata", {}).get("user_id", "system")
            workflow_type = state.get("workflow_metadata", {}).get("workflow_type", "unknown")
            
            namespace = ("workflows", "patterns", user_id)
            
            pattern_data = {
                "workflow_type": workflow_type,
                "agent_sequence": list(state["agent_outputs"].keys()),
                "execution_time": result["execution_summary"]["execution_time"],
                "success_indicators": {
                    "agents_completed": result["execution_summary"]["total_agents"],
                    "memory_enhanced": result["execution_summary"]["memory_enhanced"]
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome": "success"  # Mark as successful pattern
            }
            
            key = f"pattern_{workflow_type}_{datetime.now(timezone.utc).timestamp()}"
            await store.put(namespace, key, pattern_data)
            
        except Exception as e:
            print(f"Warning: Failed to store workflow pattern: {e}")
    
    async def _execute_agent_with_context_and_memory(
        self, 
        agent_name: str, 
        context: Dict[str, Any], 
        tda_context: Optional[TDAContext],
        agent_memory: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute agent with enhanced context and memory"""
        # Enhanced execution with memory context
        execution_context = {
            "agent_name": agent_name,
            "context": context,
            "tda_context": tda_context.__dict__ if tda_context else None,
            "agent_memory": agent_memory,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Mock enhanced execution (replace with actual agent calls)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "status": "completed",
            "result": f"Enhanced {agent_name} execution with memory",
            "memory_utilized": agent_memory is not None,
            "memory_entries_used": len(agent_memory) if agent_memory else 0,
            "tda_enhanced": tda_context is not None,
            "execution_context": execution_context
        }
    
    async def _fallback_orchestration(self, config: SemanticWorkflowConfig) -> Dict[str, Any]:
        """Fallback orchestration when LangGraph is not available"""
        return {
            "type": "fallback",
            "message": "LangGraph not available, using simple orchestration",
            "config": config.__dict__,
            "enhancement_available": False
        }