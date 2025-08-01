"""
🔄 Hybrid Checkpointer Tests

Tests for the hybrid checkpoint manager that combines LangGraph PostgresSaver
with Temporal.io durable workflows for ultimate reliability.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from aura_intelligence.orchestration.durable.hybrid_checkpointer import (
    HybridCheckpointManager,
    HybridCheckpointConfig,
    HybridCheckpointResult,
    CheckpointLevel,
    RecoveryMode
)

class TestHybridCheckpointManager:
    """Test suite for HybridCheckpointManager"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            anomaly_severity=0.6,
            complexity_score=0.7,
            pattern_confidence=0.8
        ))
        mock.send_orchestration_result = AsyncMock()
        return mock
    
    @pytest.fixture
    def hybrid_config(self):
        """Hybrid checkpoint configuration"""
        return HybridCheckpointConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            temporal_namespace="test-namespace",
            checkpoint_level=CheckpointLevel.HYBRID,
            recovery_mode=RecoveryMode.INTELLIGENT_RECOVERY,
            enable_tda_optimization=True
        )
    
    @pytest.fixture
    def checkpoint_manager(self, hybrid_config, mock_tda_integration):
        """Create hybrid checkpoint manager instance"""
        return HybridCheckpointManager(
            config=hybrid_config,
            tda_integration=mock_tda_integration
        )
    
    @pytest.fixture
    def sample_conversation_state(self):
        """Sample conversation state"""
        return {
            "messages": [
                {"role": "user", "content": "Analyze system anomaly"},
                {"role": "assistant", "content": "Starting analysis..."}
            ],
            "context": {
                "user_id": "test-user",
                "session_id": "test-session",
                "workflow_type": "anomaly_analysis"
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "step_count": 2
            }
        }
    
    @pytest.fixture
    def sample_workflow_state(self):
        """Sample workflow state"""
        return {
            "current_step": "analysis",
            "completed_steps": ["observation", "data_collection"],
            "step_results": {
                "observation": {"status": "completed", "anomalies_detected": 3},
                "data_collection": {"status": "completed", "data_points": 1500}
            },
            "workflow_metadata": {
                "workflow_id": "test-workflow-001",
                "start_time": datetime.utcnow().isoformat(),
                "expected_duration": 1800
            }
        }
    
    def test_hybrid_checkpoint_manager_initialization(self, hybrid_config, mock_tda_integration):
        """Test hybrid checkpoint manager initialization"""
        manager = HybridCheckpointManager(
            config=hybrid_config,
            tda_integration=mock_tda_integration
        )
        
        assert manager.config == hybrid_config
        assert manager.tda_integration == mock_tda_integration
        assert manager.active_checkpoints == {}
        assert manager.checkpoint_metrics["total_checkpoints"] == 0
    
    @pytest.mark.asyncio
    async def test_create_hybrid_checkpoint_conversation_only(
        self, 
        checkpoint_manager, 
        sample_conversation_state
    ):
        """Test creating conversation-only checkpoint"""
        # Mock LangGraph checkpointer
        checkpoint_manager.langgraph_checkpointer = Mock()
        checkpoint_manager.langgraph_checkpointer.put = AsyncMock()
        
        # Set configuration to conversation only
        checkpoint_manager.config.checkpoint_level = CheckpointLevel.CONVERSATION
        
        result = await checkpoint_manager.create_hybrid_checkpoint(
            workflow_id="test-workflow-001",
            conversation_state=sample_conversation_state,
            tda_correlation_id="test-correlation"
        )
        
        assert result.checkpoint_level == CheckpointLevel.CONVERSATION
        assert result.conversation_checkpoint_id is not None
        assert result.workflow_checkpoint_id is None
        assert result.size_bytes > 0
        assert result.tda_correlation_id == "test-correlation"
    
    @pytest.mark.asyncio
    async def test_create_hybrid_checkpoint_workflow_only(
        self, 
        checkpoint_manager, 
        sample_workflow_state
    ):
        """Test creating workflow-only checkpoint"""
        # Mock Temporal client
        checkpoint_manager.temporal_client = Mock()
        checkpoint_manager.temporal_client.get_workflow_handle = Mock()
        workflow_handle = Mock()
        workflow_handle.signal = AsyncMock()
        checkpoint_manager.temporal_client.get_workflow_handle.return_value = workflow_handle
        
        # Set configuration to workflow only
        checkpoint_manager.config.checkpoint_level = CheckpointLevel.WORKFLOW
        
        result = await checkpoint_manager.create_hybrid_checkpoint(
            workflow_id="test-workflow-001",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        assert result.checkpoint_level == CheckpointLevel.WORKFLOW
        assert result.conversation_checkpoint_id is None
        assert result.workflow_checkpoint_id is not None
        assert result.size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_create_full_hybrid_checkpoint(
        self, 
        checkpoint_manager, 
        sample_conversation_state, 
        sample_workflow_state,
        mock_tda_integration
    ):
        """Test creating full hybrid checkpoint with both conversation and workflow state"""
        # Mock both checkpointers
        checkpoint_manager.langgraph_checkpointer = Mock()
        checkpoint_manager.langgraph_checkpointer.put = AsyncMock()
        
        checkpoint_manager.temporal_client = Mock()
        checkpoint_manager.temporal_client.get_workflow_handle = Mock()
        workflow_handle = Mock()
        workflow_handle.signal = AsyncMock()
        checkpoint_manager.temporal_client.get_workflow_handle.return_value = workflow_handle
        
        result = await checkpoint_manager.create_hybrid_checkpoint(
            workflow_id="test-workflow-001",
            conversation_state=sample_conversation_state,
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        assert result.checkpoint_level == CheckpointLevel.HYBRID
        assert result.conversation_checkpoint_id is not None
        assert result.workflow_checkpoint_id is not None
        assert result.size_bytes > 0
        assert result.tda_correlation_id == "test-correlation"
        
        # Verify TDA integration was called
        mock_tda_integration.get_context.assert_called_with("test-correlation")
        mock_tda_integration.send_orchestration_result.assert_called()
        
        # Verify metrics were updated
        assert checkpoint_manager.checkpoint_metrics["total_checkpoints"] == 1
        assert checkpoint_manager.checkpoint_metrics["hybrid_checkpoints"] == 1
    
    @pytest.mark.asyncio
    async def test_recover_from_hybrid_checkpoint_parallel(self, checkpoint_manager):
        """Test parallel recovery from hybrid checkpoint"""
        # Create a mock checkpoint
        checkpoint_info = HybridCheckpointResult(
            checkpoint_id="test-checkpoint-001",
            conversation_checkpoint_id="conv_001",
            workflow_checkpoint_id="wf_001",
            checkpoint_level=CheckpointLevel.HYBRID,
            timestamp=datetime.utcnow(),
            size_bytes=1024
        )
        checkpoint_manager.active_checkpoints["test-checkpoint-001"] = checkpoint_info
        
        # Mock recovery methods
        checkpoint_manager._recover_conversation_state = AsyncMock(return_value={
            "type": "conversation",
            "status": "recovered",
            "state": {"conversation": "recovered"}
        })
        checkpoint_manager._recover_workflow_state = AsyncMock(return_value={
            "type": "workflow", 
            "status": "recovered",
            "state": {"workflow": "recovered"}
        })
        
        # Set recovery mode to parallel
        checkpoint_manager.config.recovery_mode = RecoveryMode.PARALLEL_RECOVERY
        
        result = await checkpoint_manager.recover_from_hybrid_checkpoint(
            checkpoint_id="test-checkpoint-001",
            tda_correlation_id="test-correlation"
        )
        
        assert result["status"] == "success"
        assert result["recovery_mode"] == "parallel"
        assert len(result["recovery_results"]) == 2
        assert result["recovery_time"] > 0
    
    @pytest.mark.asyncio
    async def test_intelligent_recovery_mode_selection(self, checkpoint_manager, mock_tda_integration):
        """Test intelligent recovery mode selection based on TDA context"""
        # Test high urgency scenario (should prefer workflow-first)
        mock_tda_integration.get_context.return_value.anomaly_severity = 0.9
        mode = checkpoint_manager._determine_optimal_recovery_mode(
            mock_tda_integration.get_context.return_value
        )
        assert mode == RecoveryMode.WORKFLOW_FIRST
        
        # Test high complexity scenario (should prefer conversation-first)
        mock_tda_integration.get_context.return_value.anomaly_severity = 0.5
        mock_tda_integration.get_context.return_value.complexity_score = 0.8
        mode = checkpoint_manager._determine_optimal_recovery_mode(
            mock_tda_integration.get_context.return_value
        )
        assert mode == RecoveryMode.CONVERSATION_FIRST
        
        # Test balanced scenario (should use parallel)
        mock_tda_integration.get_context.return_value.anomaly_severity = 0.5
        mock_tda_integration.get_context.return_value.complexity_score = 0.5
        mode = checkpoint_manager._determine_optimal_recovery_mode(
            mock_tda_integration.get_context.return_value
        )
        assert mode == RecoveryMode.PARALLEL_RECOVERY
    
    @pytest.mark.asyncio
    async def test_conversation_first_recovery(self, checkpoint_manager):
        """Test conversation-first recovery strategy"""
        checkpoint_info = HybridCheckpointResult(
            checkpoint_id="test-checkpoint-002",
            conversation_checkpoint_id="conv_002",
            workflow_checkpoint_id="wf_002",
            checkpoint_level=CheckpointLevel.HYBRID,
            timestamp=datetime.utcnow(),
            size_bytes=1024
        )
        
        # Mock recovery methods to track order
        recovery_order = []
        
        async def mock_conv_recovery(checkpoint_id):
            recovery_order.append("conversation")
            return {"type": "conversation", "status": "recovered"}
        
        async def mock_wf_recovery(checkpoint_id):
            recovery_order.append("workflow")
            return {"type": "workflow", "status": "recovered"}
        
        checkpoint_manager._recover_conversation_state = mock_conv_recovery
        checkpoint_manager._recover_workflow_state = mock_wf_recovery
        
        result = await checkpoint_manager._conversation_first_recovery(
            checkpoint_info, None
        )
        
        assert result["status"] == "success"
        assert result["recovery_mode"] == "conversation_first"
        assert recovery_order == ["conversation", "workflow"]
    
    @pytest.mark.asyncio
    async def test_workflow_first_recovery(self, checkpoint_manager):
        """Test workflow-first recovery strategy"""
        checkpoint_info = HybridCheckpointResult(
            checkpoint_id="test-checkpoint-003",
            conversation_checkpoint_id="conv_003",
            workflow_checkpoint_id="wf_003",
            checkpoint_level=CheckpointLevel.HYBRID,
            timestamp=datetime.utcnow(),
            size_bytes=1024
        )
        
        # Mock recovery methods to track order
        recovery_order = []
        
        async def mock_conv_recovery(checkpoint_id):
            recovery_order.append("conversation")
            return {"type": "conversation", "status": "recovered"}
        
        async def mock_wf_recovery(checkpoint_id):
            recovery_order.append("workflow")
            return {"type": "workflow", "status": "recovered"}
        
        checkpoint_manager._recover_conversation_state = mock_conv_recovery
        checkpoint_manager._recover_workflow_state = mock_wf_recovery
        
        result = await checkpoint_manager._workflow_first_recovery(
            checkpoint_info, None
        )
        
        assert result["status"] == "success"
        assert result["recovery_mode"] == "workflow_first"
        assert recovery_order == ["workflow", "conversation"]
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation_failure_handling(self, checkpoint_manager):
        """Test handling of checkpoint creation failures"""
        # Mock checkpointer to raise exception
        checkpoint_manager.langgraph_checkpointer = Mock()
        checkpoint_manager.langgraph_checkpointer.put = AsyncMock(side_effect=Exception("Checkpoint failed"))
        
        result = await checkpoint_manager.create_hybrid_checkpoint(
            workflow_id="test-workflow-fail",
            conversation_state={"test": "data"}
        )
        
        assert result.metadata.get("creation_failed") is True
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_recovery_failure_handling(self, checkpoint_manager):
        """Test handling of recovery failures"""
        # Create checkpoint that doesn't exist
        result = await checkpoint_manager.recover_from_hybrid_checkpoint(
            checkpoint_id="nonexistent-checkpoint"
        )
        
        assert result["status"] == "failed"
        assert "not found" in result["error"]
        assert result["recovery_time"] > 0
    
    def test_checkpoint_metrics_tracking(self, checkpoint_manager):
        """Test checkpoint metrics tracking"""
        # Initial metrics
        metrics = checkpoint_manager.get_checkpoint_metrics()
        assert metrics["total_checkpoints"] == 0
        assert metrics["active_checkpoints"] == 0
        
        # Add mock checkpoint
        checkpoint_manager.active_checkpoints["test"] = Mock()
        checkpoint_manager.checkpoint_metrics["total_checkpoints"] = 5
        checkpoint_manager.checkpoint_metrics["hybrid_checkpoints"] = 3
        
        metrics = checkpoint_manager.get_checkpoint_metrics()
        assert metrics["total_checkpoints"] == 5
        assert metrics["hybrid_checkpoints"] == 3
        assert metrics["active_checkpoints"] == 1
    
    @pytest.mark.asyncio
    async def test_tda_optimization_integration(self, checkpoint_manager, mock_tda_integration):
        """Test TDA optimization integration"""
        # Create checkpoint with TDA correlation
        checkpoint_manager.langgraph_checkpointer = Mock()
        checkpoint_manager.langgraph_checkpointer.put = AsyncMock()
        
        result = await checkpoint_manager.create_hybrid_checkpoint(
            workflow_id="test-tda-optimization",
            conversation_state={"test": "data"},
            tda_correlation_id="test-correlation"
        )
        
        # Verify TDA integration was used
        mock_tda_integration.get_context.assert_called_with("test-correlation")
        assert result.metadata.get("tda_optimized") is True
        
        # Test recovery with TDA optimization
        checkpoint_manager.active_checkpoints[result.checkpoint_id] = result
        checkpoint_manager._recover_conversation_state = AsyncMock(return_value={
            "type": "conversation", "status": "recovered"
        })
        
        recovery_result = await checkpoint_manager.recover_from_hybrid_checkpoint(
            checkpoint_id=result.checkpoint_id,
            recovery_mode=RecoveryMode.INTELLIGENT_RECOVERY,
            tda_correlation_id="test-correlation"
        )
        
        assert recovery_result["status"] == "success"
        assert checkpoint_manager.checkpoint_metrics["tda_optimizations"] > 0

class TestHybridCheckpointConfig:
    """Test suite for HybridCheckpointConfig"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = HybridCheckpointConfig()
        
        assert config.postgres_url is None
        assert config.temporal_namespace == "aura-production"
        assert config.checkpoint_level == CheckpointLevel.HYBRID
        assert config.recovery_mode == RecoveryMode.INTELLIGENT_RECOVERY
        assert config.conversation_checkpoint_interval == 30
        assert config.workflow_checkpoint_interval == 300
        assert config.enable_tda_optimization is True
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = HybridCheckpointConfig(
            postgres_url="postgresql://custom:custom@localhost:5432/custom",
            temporal_namespace="custom-namespace",
            checkpoint_level=CheckpointLevel.CONVERSATION,
            recovery_mode=RecoveryMode.PARALLEL_RECOVERY,
            conversation_checkpoint_interval=60,
            workflow_checkpoint_interval=600,
            enable_tda_optimization=False
        )
        
        assert config.postgres_url == "postgresql://custom:custom@localhost:5432/custom"
        assert config.temporal_namespace == "custom-namespace"
        assert config.checkpoint_level == CheckpointLevel.CONVERSATION
        assert config.recovery_mode == RecoveryMode.PARALLEL_RECOVERY
        assert config.conversation_checkpoint_interval == 60
        assert config.workflow_checkpoint_interval == 600
        assert config.enable_tda_optimization is False

class TestHybridCheckpointResult:
    """Test suite for HybridCheckpointResult"""
    
    def test_checkpoint_result_creation(self):
        """Test checkpoint result creation"""
        timestamp = datetime.utcnow()
        result = HybridCheckpointResult(
            checkpoint_id="test-checkpoint",
            conversation_checkpoint_id="conv-123",
            workflow_checkpoint_id="wf-456",
            checkpoint_level=CheckpointLevel.HYBRID,
            timestamp=timestamp,
            size_bytes=2048,
            tda_correlation_id="tda-789",
            metadata={"test": "metadata"}
        )
        
        assert result.checkpoint_id == "test-checkpoint"
        assert result.conversation_checkpoint_id == "conv-123"
        assert result.workflow_checkpoint_id == "wf-456"
        assert result.checkpoint_level == CheckpointLevel.HYBRID
        assert result.timestamp == timestamp
        assert result.size_bytes == 2048
        assert result.tda_correlation_id == "tda-789"
        assert result.metadata == {"test": "metadata"}
    
    def test_checkpoint_result_defaults(self):
        """Test checkpoint result with default values"""
        result = HybridCheckpointResult(
            checkpoint_id="test-checkpoint",
            conversation_checkpoint_id=None,
            workflow_checkpoint_id=None,
            checkpoint_level=CheckpointLevel.CONVERSATION,
            timestamp=datetime.utcnow(),
            size_bytes=1024
        )
        
        assert result.tda_correlation_id is None
        assert result.metadata is None