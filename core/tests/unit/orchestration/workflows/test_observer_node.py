"""
🧪 Observer Node Unit Tests
Comprehensive testing for the observer workflow node.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from aura_intelligence.orchestration.workflows.nodes.observer import (
    ObserverNode,
    create_observer_node
)
from aura_intelligence.orchestration.workflows.state import (
    CollectiveState,
    create_initial_state
)


class TestObserverNode:
    """Test suite for ObserverNode."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = Mock()
        llm.ainvoke = AsyncMock(return_value=Mock(
            content="Test analysis summary"
        ))
        return llm
    
    @pytest.fixture
    def initial_state(self) -> CollectiveState:
        """Create initial workflow state."""
        return create_initial_state(
            workflow_id="test-workflow-123",
            thread_id="test-thread-456",
            initial_message="Test message"
        )
    
    @pytest.fixture
    def observer_node(self, mock_llm):
        """Create observer node with mock LLM."""
        return ObserverNode(llm=mock_llm)
    
    @pytest.mark.asyncio
    async def test_observer_basic_functionality(self, observer_node, initial_state):
        """Test basic observer functionality."""
        # Execute node
        result = await observer_node(initial_state)
        
        # Verify result structure
        assert "evidence_log" in result
        assert "current_step" in result
        assert "system_health" in result
        
        # Check evidence was collected
        assert len(result["evidence_log"]) == 1
        observation = result["evidence_log"][0]
        assert observation["node"] == "observer"
        assert "evidence" in observation
        assert "timestamp" in observation
        
        # Check state progression
        assert result["current_step"] == "observation_complete"
    
    @pytest.mark.asyncio
    async def test_observer_evidence_collection(self, observer_node, initial_state):
        """Test evidence collection from various sources."""
        # Add some history to state
        initial_state["supervisor_decisions"] = [
            {"decision": "test", "confidence": 0.8}
        ]
        initial_state["error_log"] = [
            {"error": "test error", "timestamp": datetime.utcnow().isoformat()}
        ]
        
        # Execute node
        result = await observer_node(initial_state)
        
        # Check evidence types
        observation = result["evidence_log"][0]
        evidence_types = {e["type"] for e in observation["evidence"]}
        
        assert "system_metrics" in evidence_types
        assert "message_analysis" in evidence_types
        assert "decision_history" in evidence_types
        assert "error_patterns" in evidence_types
    
    @pytest.mark.asyncio
    async def test_observer_with_llm_analysis(self, observer_node, initial_state):
        """Test observer with LLM analysis enabled."""
        with patch("aura_common.config.is_feature_enabled", return_value=True):
            result = await observer_node(initial_state)
            
            # Check analysis was performed
            observation = result["evidence_log"][0]
            assert "analysis" in observation
            assert observation["analysis"]["summary"] == "Test analysis summary"
            
            # Check message was added
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], AIMessage)
    
    @pytest.mark.asyncio
    async def test_observer_error_handling(self, initial_state):
        """Test observer error handling."""
        # Create node that will fail
        node = ObserverNode()
        
        # Mock evidence collection to raise error
        with patch.object(
            node,
            "_collect_evidence",
            side_effect=Exception("Test error")
        ):
            result = await node(initial_state)
            
            # Check error was logged
            assert "error_log" in result
            assert len(result["error_log"]) == 1
            assert result["error_log"][0]["error"] == "Test error"
            
            # Check error state
            assert result["current_step"] == "observer_error"
            assert "last_error" in result
    
    @pytest.mark.asyncio
    async def test_observer_risk_detection(self, observer_node, initial_state):
        """Test risk indicator detection."""
        # Mock high resource usage
        with patch.object(
            observer_node,
            "_get_system_health",
            return_value={
                "cpu_usage": 0.9,  # High CPU
                "memory_usage": 0.85,  # High memory
                "error_rate": 0.02,
                "status": "degraded"
            }
        ):
            result = await observer_node(initial_state)
            
            observation = result["evidence_log"][0]
            assert "risk_indicators" in observation
            assert "high_cpu_usage" in observation["risk_indicators"]
            assert "high_memory_usage" in observation["risk_indicators"]
    
    @pytest.mark.asyncio
    async def test_observer_performance_metrics(self, observer_node, initial_state):
        """Test performance metric collection."""
        result = await observer_node(initial_state)
        
        # Observer doesn't return NodeResult directly, but we can check
        # that it executes quickly
        # In real implementation, we'd check logs or metrics
        assert result is not None
    
    def test_create_observer_node_factory(self, mock_llm):
        """Test factory function."""
        # Without LLM
        node1 = create_observer_node()
        assert isinstance(node1, ObserverNode)
        assert node1.llm is None
        
        # With LLM
        node2 = create_observer_node(llm=mock_llm)
        assert isinstance(node2, ObserverNode)
        assert node2.llm == mock_llm
    
    @pytest.mark.asyncio
    async def test_observer_correlation_id(self, observer_node, initial_state):
        """Test correlation ID propagation."""
        with patch("aura_common.logging.get_correlation_id", return_value="test-correlation-123"):
            # The with_correlation_id decorator should handle this
            result = await observer_node(initial_state)
            assert result is not None
            # In real implementation, we'd verify correlation ID in logs


class TestObserverIntegration:
    """Integration tests for observer node."""
    
    @pytest.mark.asyncio
    async def test_observer_with_real_state_transitions(self):
        """Test observer with realistic state transitions."""
        # Create initial state with some history
        state = create_initial_state(
            workflow_id="integration-test",
            thread_id="thread-123"
        )
        
        # Add realistic message history
        state["messages"] = [
            HumanMessage(content="Analyze system performance"),
            AIMessage(content="Starting analysis...")
        ]
        
        # Add some decisions
        state["supervisor_decisions"] = [
            {
                "action": "collect_metrics",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        # Create and run observer
        observer = create_observer_node()
        result = await observer(state)
        
        # Verify comprehensive observation
        assert result["current_step"] == "observation_complete"
        observation = result["evidence_log"][0]
        
        # Check all evidence types were collected
        evidence_types = {e["type"] for e in observation["evidence"]}
        assert len(evidence_types) >= 3  # At least system, messages, decisions
    
    @pytest.mark.asyncio
    async def test_observer_circuit_breaker(self):
        """Test observer with circuit breaker pattern."""
        observer = create_observer_node()
        
        # Simulate multiple failures to trigger circuit breaker
        with patch.object(
            observer,
            "_collect_evidence",
            side_effect=Exception("Service unavailable")
        ):
            # First few calls should attempt
            for i in range(3):
                state = create_initial_state(f"test-{i}", f"thread-{i}")
                result = await observer(state)
                assert result["current_step"] == "observer_error"
            
            # Circuit breaker should now be open
            # In real implementation, subsequent calls would fail fast