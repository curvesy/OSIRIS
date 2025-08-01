"""
🧪 Tests for LangGraph Semantic Orchestrator

Comprehensive unit tests for LangGraph integration with TDA context.
Tests semantic analysis, routing decisions, and state transitions.

Test Coverage:
- StateGraph creation and compilation
- Semantic analysis with TDA correlation
- Routing decisions based on complexity
- State transitions and checkpointing
- Error handling and fallbacks
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from aura_intelligence.orchestration.semantic.langgraph_orchestrator import (
    LangGraphSemanticOrchestrator,
    SemanticWorkflowConfig,
    LANGGRAPH_AVAILABLE
)
from aura_intelligence.orchestration.semantic.base_interfaces import (
    AgentState, TDAContext, UrgencyLevel, OrchestrationStrategy
)

class TestLangGraphSemanticOrchestrator:
    """Test suite for LangGraph Semantic Orchestrator"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration for testing"""
        mock = AsyncMock()
        mock.get_context.return_value = TDAContext(
            correlation_id="test-correlation-123",
            pattern_confidence=0.8,
            anomaly_severity=0.6,
            current_patterns={"pattern1": "value1"},
            temporal_window="1h",
            metadata={"test": True}
        )
        mock.send_orchestration_result.return_value = True
        return mock
    
    @pytest.fixture
    def orchestrator(self, mock_tda_integration):
        """Create orchestrator instance for testing"""
        return LangGraphSemanticOrchestrator(tda_integration=mock_tda_integration)
    
    @pytest.fixture
    def workflow_config(self):
        """Sample workflow configuration"""
        return SemanticWorkflowConfig(
            workflow_id="test-workflow-123",
            orchestrator_agent="orchestrator",
            worker_agents=["agent1", "agent2", "agent3"],
            routing_strategy=OrchestrationStrategy.PARALLEL,
            max_retries=3,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def sample_state(self):
        """Sample agent state for testing"""
        return {
            "messages": [{"role": "user", "content": "test message"}],
            "context": {"task": "analyze data", "complexity": "medium"},
            "agent_outputs": {},
            "workflow_metadata": {
                "workflow_id": "test-workflow-123",
                "correlation_id": "test-correlation-123"
            },
            "execution_trace": [],
            "tda_context": None
        }
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_basic(self, orchestrator):
        """Test basic semantic analysis without TDA context"""
        input_data = {
            "task": "simple analysis",
            "requirements": ["req1", "req2"],
            "urgency": "medium"
        }
        
        analysis = await orchestrator.analyze_semantically(input_data)
        
        assert analysis.complexity_score >= 0.0
        assert analysis.complexity_score <= 1.0
        assert analysis.urgency_level == UrgencyLevel.MEDIUM
        assert analysis.confidence == 0.85
        assert analysis.tda_correlation is None
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_with_tda(self, orchestrator, mock_tda_integration):
        """Test semantic analysis with TDA context enhancement"""
        input_data = {
            "task": "complex analysis",
            "requirements": ["req1", "req2", "req3"],
            "urgency": "low"
        }
        
        tda_context = TDAContext(
            correlation_id="test-123",
            pattern_confidence=0.9,
            anomaly_severity=0.8,  # High anomaly should elevate urgency
            current_patterns={},
            temporal_window="1h",
            metadata={}
        )
        
        analysis = await orchestrator.analyze_semantically(input_data, tda_context)
        
        # TDA context should enhance complexity and elevate urgency
        assert analysis.complexity_score > 0.5  # Enhanced by TDA confidence
        assert analysis.urgency_level == UrgencyLevel.HIGH  # Elevated by anomaly
        assert analysis.tda_correlation == tda_context
    
    @pytest.mark.asyncio
    async def test_orchestrator_node_execution(self, orchestrator, sample_state):
        """Test orchestrator node execution with state updates"""
        result_state = await orchestrator._orchestrator_node(sample_state)
        
        # Verify state updates
        assert "task_analysis" in result_state["workflow_metadata"]
        assert result_state["workflow_metadata"]["orchestration_strategy"] == "semantic_decomposition"
        assert result_state["workflow_metadata"]["tda_integration"] is True
        assert "timestamp" in result_state["workflow_metadata"]
    
    @pytest.mark.asyncio
    async def test_semantic_routing_decisions(self, orchestrator, sample_state):
        """Test semantic routing decision logic"""
        # Test high complexity routing
        sample_state["workflow_metadata"]["task_analysis"] = {
            "complexity_score": 0.9,
            "urgency_level": "medium"
        }
        
        decision = await orchestrator._semantic_router_decision(sample_state)
        assert decision == "parallel_execution"
        
        # Test critical urgency routing
        sample_state["workflow_metadata"]["task_analysis"] = {
            "complexity_score": 0.3,
            "urgency_level": "critical"
        }
        
        decision = await orchestrator._semantic_router_decision(sample_state)
        assert decision == "immediate_execution"
        
        # Test default routing
        sample_state["workflow_metadata"]["task_analysis"] = {
            "complexity_score": 0.3,
            "urgency_level": "low"
        }
        
        decision = await orchestrator._semantic_router_decision(sample_state)
        assert decision == "sequential_execution"
    
    @pytest.mark.asyncio
    async def test_aggregation_node(self, orchestrator, sample_state, mock_tda_integration):
        """Test result aggregation and TDA integration"""
        # Setup state with agent outputs
        sample_state["agent_outputs"] = {
            "agent1": {"result": "analysis1"},
            "agent2": {"result": "analysis2"}
        }
        sample_state["execution_trace"] = [
            {"agent": "agent1", "timestamp": "2025-01-01T00:00:00"},
            {"agent": "agent2", "timestamp": "2025-01-01T00:01:00"}
        ]
        sample_state["tda_context"] = TDAContext(
            correlation_id="test-123",
            pattern_confidence=0.8,
            anomaly_severity=0.5,
            current_patterns={},
            temporal_window="1h",
            metadata={}
        )
        
        result_state = await orchestrator._aggregation_node(sample_state)
        
        # Verify aggregation
        assert "final_result" in result_state
        final_result = result_state["final_result"]
        assert final_result["workflow_id"] == "test-workflow-123"
        assert len(final_result["agent_outputs"]) == 2
        assert final_result["execution_summary"]["total_agents"] == 2
        assert final_result["execution_summary"]["tda_correlation"] == "test-123"
        
        # Verify TDA integration call
        mock_tda_integration.send_orchestration_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complexity_calculation(self, orchestrator):
        """Test complexity calculation heuristics"""
        # Simple task
        simple_data = {"task": "simple", "requirements": []}
        complexity = orchestrator._calculate_complexity(simple_data)
        assert 0.0 <= complexity <= 1.0
        
        # Complex task
        complex_data = {
            "task": "very complex task with lots of details and requirements",
            "requirements": ["req1", "req2", "req3", "req4", "req5"],
            "requires_consensus": True
        }
        complex_complexity = orchestrator._calculate_complexity(complex_data)
        assert complex_complexity > complexity
        assert complex_complexity <= 1.0
    
    @pytest.mark.asyncio
    async def test_urgency_determination(self, orchestrator):
        """Test urgency determination with TDA amplification"""
        input_data = {"urgency": "low"}
        
        # Without TDA context
        urgency = orchestrator._determine_urgency(input_data, None)
        assert urgency == UrgencyLevel.LOW
        
        # With high anomaly TDA context
        high_anomaly_context = TDAContext(
            correlation_id="test",
            pattern_confidence=0.5,
            anomaly_severity=0.9,  # High anomaly
            current_patterns={},
            temporal_window="1h",
            metadata={}
        )
        
        urgency = orchestrator._determine_urgency(input_data, high_anomaly_context)
        assert urgency == UrgencyLevel.CRITICAL  # Elevated by TDA
    
    @pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
    @pytest.mark.asyncio
    async def test_graph_creation(self, orchestrator, workflow_config):
        """Test StateGraph creation when LangGraph is available"""
        graph = await orchestrator.create_orchestrator_worker_graph(workflow_config)
        
        # Verify graph was created (exact structure depends on LangGraph version)
        assert graph is not None
    
    @pytest.mark.asyncio
    async def test_fallback_orchestration(self, orchestrator, workflow_config):
        """Test fallback orchestration when LangGraph is not available"""
        with patch('aura_intelligence.orchestration.semantic.langgraph_orchestrator.LANGGRAPH_AVAILABLE', False):
            result = await orchestrator._fallback_orchestration(workflow_config)
            
            assert result["type"] == "fallback"
            assert "LangGraph not available" in result["message"]
            assert result["config"]["workflow_id"] == "test-workflow-123"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling in semantic analysis"""
        # Test with invalid input data
        invalid_data = None
        
        with pytest.raises(Exception):
            await orchestrator.analyze_semantically(invalid_data)
    
    @pytest.mark.asyncio
    async def test_tda_context_retrieval(self, orchestrator, sample_state, mock_tda_integration):
        """Test TDA context retrieval"""
        context = await orchestrator._get_tda_context(sample_state)
        
        assert context is not None
        assert context.correlation_id == "test-correlation-123"
        assert context.pattern_confidence == 0.8
        mock_tda_integration.get_context.assert_called_once_with("test-correlation-123")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, orchestrator):
        """Test performance characteristics"""
        import time
        
        input_data = {"task": "performance test", "requirements": ["req1"]}
        
        start_time = time.time()
        analysis = await orchestrator.analyze_semantically(input_data)
        end_time = time.time()
        
        # Semantic analysis should be fast (<100ms)
        execution_time = end_time - start_time
        assert execution_time < 0.1  # 100ms threshold
        assert analysis.confidence > 0.0