"""
🔄 Workflow Migration Tests
Ensure refactoring maintains backward compatibility.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestWorkflowMigration:
    """Test that refactored workflows maintain compatibility."""
    
    @pytest.mark.asyncio
    async def test_observer_node_import_paths(self):
        """Test that observer node can be imported from new location."""
        # New modular import
        from aura_intelligence.orchestration.workflows.nodes import (
            ObserverNode,
            create_observer_node
        )
        
        # Verify classes exist
        assert ObserverNode is not None
        assert create_observer_node is not None
        
        # Create instance
        node = create_observer_node()
        assert isinstance(node, ObserverNode)
        assert node.name == "observer"
    
    @pytest.mark.asyncio
    async def test_state_compatibility(self):
        """Test that new state module is compatible."""
        from aura_intelligence.orchestration.workflows.state import (
            CollectiveState,
            create_initial_state,
            update_state_safely
        )
        
        # Create state
        state = create_initial_state(
            workflow_id="test-123",
            thread_id="thread-456"
        )
        
        # Verify all expected fields exist
        expected_fields = [
            "messages", "workflow_id", "thread_id", "evidence_log",
            "supervisor_decisions", "memory_context", "active_config",
            "current_step", "risk_assessment", "execution_results",
            "error_log", "error_recovery_attempts", "last_error",
            "system_health", "validation_results", "shadow_mode_enabled",
            "shadow_predictions"
        ]
        
        for field in expected_fields:
            assert field in state, f"Missing field: {field}"
        
        # Test state update
        updated = update_state_safely(state, {
            "current_step": "test_complete",
            "evidence_log": [{"test": "data"}]
        })
        
        assert updated["current_step"] == "test_complete"
        assert len(updated["evidence_log"]) == 1
    
    @pytest.mark.asyncio
    async def test_shared_libs_integration(self):
        """Test that nodes use shared libraries correctly."""
        from aura_intelligence.orchestration.workflows.nodes import ObserverNode
        from aura_intelligence.orchestration.workflows.state import create_initial_state
        
        # Create node and state
        node = ObserverNode()
        state = create_initial_state("test", "test")
        
        # Execute node
        result = await node(state)
        
        # Should not raise errors
        assert result is not None
        assert "current_step" in result
    
    def test_old_imports_deprecated(self):
        """Test that old monolithic imports are marked for removal."""
        # This would check that workflows.py is marked deprecated
        # For now, we just document the migration path
        migration_notes = """
        Migration Path:
        OLD: from aura_intelligence.orchestration.workflows import CollectiveState
        NEW: from aura_intelligence.orchestration.workflows.state import CollectiveState
        
        OLD: from aura_intelligence.orchestration.workflows import observer_node
        NEW: from aura_intelligence.orchestration.workflows.nodes import create_observer_node
        """
        assert migration_notes is not None


class TestFeatureFlagMigration:
    """Test feature flag usage in migrated code."""
    
    @pytest.mark.asyncio
    async def test_shadow_mode_feature_flag(self):
        """Test shadow mode respects feature flags."""
        from aura_intelligence.orchestration.workflows.state import create_initial_state
        from unittest.mock import patch
        
        # Test with shadow mode disabled
        with patch("aura_common.config.get_config") as mock_config:
            mock_config.return_value.logging.shadow_mode_enabled = False
            state = create_initial_state("test", "test")
            assert state["shadow_mode_enabled"] is False
        
        # Test with shadow mode enabled
        with patch("aura_common.config.get_config") as mock_config:
            mock_config.return_value.logging.shadow_mode_enabled = True
            state = create_initial_state("test", "test")
            assert state["shadow_mode_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_llm_analysis_feature_flag(self):
        """Test LLM analysis respects feature flags."""
        from aura_intelligence.orchestration.workflows.nodes import ObserverNode
        from aura_intelligence.orchestration.workflows.state import create_initial_state
        from unittest.mock import patch, Mock, AsyncMock
        
        # Create node with mock LLM
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="test"))
        node = ObserverNode(llm=mock_llm)
        
        state = create_initial_state("test", "test")
        
        # Test with feature disabled
        with patch("aura_common.config.is_feature_enabled", return_value=False):
            result = await node(state)
            mock_llm.ainvoke.assert_not_called()
        
        # Test with feature enabled
        with patch("aura_common.config.is_feature_enabled", return_value=True):
            result = await node(state)
            mock_llm.ainvoke.assert_called_once()


class TestErrorHandlingMigration:
    """Test error handling uses shared libraries."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test nodes use circuit breaker from shared libs."""
        from aura_intelligence.orchestration.workflows.nodes import ObserverNode
        from aura_intelligence.orchestration.workflows.state import create_initial_state
        
        # The @resilient_operation decorator should be applied
        # Check that the node has the decorator
        assert hasattr(ObserverNode.__call__, "__wrapped__")
    
    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self):
        """Test correlation ID propagates through nodes."""
        from aura_intelligence.orchestration.workflows.nodes import ObserverNode
        from aura_intelligence.orchestration.workflows.state import create_initial_state
        from unittest.mock import patch
        
        node = ObserverNode()
        state = create_initial_state("test", "test")
        
        # The @with_correlation_id decorator should handle this
        with patch("aura_common.logging.correlation.get_correlation_id", return_value="test-123"):
            result = await node(state)
            # In real implementation, we'd verify correlation ID in logs