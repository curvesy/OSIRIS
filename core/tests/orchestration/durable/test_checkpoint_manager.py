"""
💾 Checkpoint Manager Tests

Tests for checkpoint management including state persistence,
recovery strategies, and TDA integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from aura_intelligence.orchestration.durable.checkpoint_manager import (
    CheckpointManager,
    WorkflowCheckpoint,
    RecoveryStrategy,
    CheckpointStatus
)

class TestCheckpointManager:
    """Test suite for CheckpointManager"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            complexity_score=0.8,
            urgency_score=0.6,
            context_data={"checkpoint": "context"}
        ))
        mock.send_orchestration_result = AsyncMock()
        return mock
    
    @pytest.fixture
    def checkpoint_manager(self, mock_tda_integration):
        """Create checkpoint manager instance"""
        return CheckpointManager(
            storage_backend="memory",
            tda_integration=mock_tda_integration,
            checkpoint_retention_hours=24
        )
    
    @pytest.fixture
    def sample_workflow_state(self):
        """Sample workflow state for testing"""
        return {
            "current_step": 2,
            "completed_steps": ["step1", "step2"],
            "step_results": {
                "step1": {"status": "completed", "result": "data1"},
                "step2": {"status": "completed", "result": "data2"}
            },
            "workflow_context": {
                "start_time": datetime.utcnow().isoformat(),
                "input_data": {"test": "input"}
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_manager, sample_workflow_state):
        """Test checkpoint creation"""
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-001",
            step_index=2,
            step_name="data_analysis",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        assert checkpoint.workflow_id == "test-workflow-001"
        assert checkpoint.step_index == 2
        assert checkpoint.step_name == "data_analysis"
        assert checkpoint.workflow_state == sample_workflow_state
        assert checkpoint.status == CheckpointStatus.CREATED
        assert checkpoint.checksum is not None
        assert checkpoint.size_bytes > 0
        assert checkpoint.tda_context is not None
    
    @pytest.mark.asyncio
    async def test_create_checkpoint_with_tda_integration(self, checkpoint_manager, sample_workflow_state, mock_tda_integration):
        """Test checkpoint creation with TDA integration"""
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-tda",
            step_index=1,
            step_name="preparation",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        # Verify TDA integration was called
        mock_tda_integration.get_context.assert_called_with("test-correlation")
        mock_tda_integration.send_orchestration_result.assert_called()
        
        # Check that TDA context was stored
        assert checkpoint.tda_context is not None
        assert checkpoint.metadata["tda_correlation_id"] == "test-correlation"
    
    @pytest.mark.asyncio
    async def test_recover_from_checkpoint_rollback(self, checkpoint_manager, sample_workflow_state):
        """Test rollback recovery from checkpoint"""
        # Create checkpoint first
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-recovery",
            step_index=2,
            step_name="analysis",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        # Recover from checkpoint
        recovery_result = await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            recovery_strategy=RecoveryStrategy.ROLLBACK_TO_CHECKPOINT,
            tda_correlation_id="test-correlation"
        )
        
        assert recovery_result["status"] == "success"
        assert recovery_result["checkpoint_id"] == checkpoint.checkpoint_id
        assert recovery_result["recovery_strategy"] == "rollback_to_checkpoint"
        assert recovery_result["workflow_state"] == sample_workflow_state
        assert recovery_result["recovery_time"] > 0
    
    @pytest.mark.asyncio
    async def test_recover_from_checkpoint_forward_recovery(self, checkpoint_manager, sample_workflow_state, mock_tda_integration):
        """Test forward recovery from checkpoint"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-forward",
            step_index=1,
            step_name="preparation",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        # Recover with forward recovery
        recovery_result = await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            recovery_strategy=RecoveryStrategy.FORWARD_RECOVERY,
            tda_correlation_id="test-correlation"
        )
        
        assert recovery_result["status"] == "success"
        assert recovery_result["recovery_strategy"] == "forward_recovery"
        
        # Forward recovery should enhance state with current TDA context
        enhanced_state = recovery_result["workflow_state"]
        assert "recovery_enhancement" in enhanced_state
        assert enhanced_state["recovery_enhancement"]["current_tda_context"] is not None
    
    @pytest.mark.asyncio
    async def test_recover_from_checkpoint_hybrid(self, checkpoint_manager, sample_workflow_state):
        """Test hybrid recovery strategy"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-hybrid",
            step_index=1,
            step_name="preparation",
            workflow_state=sample_workflow_state,
            tda_correlation_id="test-correlation"
        )
        
        # Recover with hybrid strategy
        recovery_result = await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            recovery_strategy=RecoveryStrategy.HYBRID_RECOVERY,
            tda_correlation_id="test-correlation"
        )
        
        assert recovery_result["status"] == "success"
        assert recovery_result["recovery_strategy"] == "hybrid"
        
        # Hybrid recovery should combine rollback and forward elements
        hybrid_state = recovery_result["workflow_state"]
        assert "hybrid_recovery" in hybrid_state
        assert hybrid_state["hybrid_recovery"]["strategy"] == "hybrid"
    
    @pytest.mark.asyncio
    async def test_recover_from_checkpoint_manual_intervention(self, checkpoint_manager, sample_workflow_state):
        """Test manual intervention recovery strategy"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-workflow-manual",
            step_index=1,
            step_name="preparation",
            workflow_state=sample_workflow_state
        )
        
        # Recover with manual intervention
        recovery_result = await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
        )
        
        assert recovery_result["status"] == "success"
        assert recovery_result["recovery_strategy"] == "manual_intervention"
        
        # Manual intervention should include instructions
        manual_state = recovery_result["workflow_state"]
        assert manual_state["requires_manual_action"] is True
        assert "intervention_instructions" in manual_state
        assert len(manual_state["intervention_instructions"]) > 0
    
    @pytest.mark.asyncio
    async def test_recover_from_nonexistent_checkpoint(self, checkpoint_manager):
        """Test recovery from non-existent checkpoint"""
        recovery_result = await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id="nonexistent-checkpoint",
            recovery_strategy=RecoveryStrategy.ROLLBACK_TO_CHECKPOINT
        )
        
        assert recovery_result["status"] == "failed"
        assert "not found" in recovery_result["error"]
    
    @pytest.mark.asyncio
    async def test_checkpoint_integrity_verification(self, checkpoint_manager, sample_workflow_state):
        """Test checkpoint integrity verification"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-integrity",
            step_index=1,
            step_name="test",
            workflow_state=sample_workflow_state
        )
        
        # Verify integrity
        is_valid = checkpoint_manager._verify_checkpoint_integrity(checkpoint)
        assert is_valid is True
        
        # Corrupt the checkpoint
        checkpoint.workflow_state["corrupted"] = "data"
        is_valid = checkpoint_manager._verify_checkpoint_integrity(checkpoint)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_checkpoints(self, checkpoint_manager, sample_workflow_state):
        """Test cleanup of expired checkpoints"""
        # Create checkpoint with past timestamp
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="test-expired",
            step_index=1,
            step_name="test",
            workflow_state=sample_workflow_state
        )
        
        # Manually set timestamp to past
        checkpoint.timestamp = datetime.utcnow() - timedelta(hours=25)  # Older than retention
        checkpoint_manager.checkpoints[checkpoint.checkpoint_id] = checkpoint
        
        # Run cleanup
        cleaned_count = await checkpoint_manager.cleanup_expired_checkpoints()
        
        assert cleaned_count == 1
        assert checkpoint.checkpoint_id not in checkpoint_manager.checkpoints
    
    def test_get_workflow_checkpoints(self, checkpoint_manager):
        """Test getting checkpoints for a specific workflow"""
        # Create mock checkpoints
        workflow_id = "test-workflow"
        checkpoint1 = WorkflowCheckpoint(
            checkpoint_id="cp1",
            workflow_id=workflow_id,
            step_index=1,
            step_name="step1",
            workflow_state={},
            tda_context=None,
            timestamp=datetime.utcnow()
        )
        checkpoint2 = WorkflowCheckpoint(
            checkpoint_id="cp2",
            workflow_id=workflow_id,
            step_index=2,
            step_name="step2",
            workflow_state={},
            tda_context=None,
            timestamp=datetime.utcnow()
        )
        
        # Store checkpoints
        checkpoint_manager.checkpoints["cp1"] = checkpoint1
        checkpoint_manager.checkpoints["cp2"] = checkpoint2
        checkpoint_manager.workflow_checkpoints[workflow_id] = ["cp1", "cp2"]
        
        # Get workflow checkpoints
        checkpoints = checkpoint_manager.get_workflow_checkpoints(workflow_id)
        
        assert len(checkpoints) == 2
        assert checkpoints[0].checkpoint_id in ["cp1", "cp2"]
        assert checkpoints[1].checkpoint_id in ["cp1", "cp2"]
    
    def test_get_latest_checkpoint(self, checkpoint_manager):
        """Test getting the latest checkpoint for a workflow"""
        workflow_id = "test-workflow"
        
        # Create checkpoints with different timestamps
        older_checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            workflow_id=workflow_id,
            step_index=1,
            step_name="step1",
            workflow_state={},
            tda_context=None,
            timestamp=datetime.utcnow() - timedelta(minutes=10)
        )
        newer_checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp2",
            workflow_id=workflow_id,
            step_index=2,
            step_name="step2",
            workflow_state={},
            tda_context=None,
            timestamp=datetime.utcnow()
        )
        
        # Store checkpoints
        checkpoint_manager.checkpoints["cp1"] = older_checkpoint
        checkpoint_manager.checkpoints["cp2"] = newer_checkpoint
        checkpoint_manager.workflow_checkpoints[workflow_id] = ["cp1", "cp2"]
        
        # Get latest checkpoint
        latest = checkpoint_manager.get_latest_checkpoint(workflow_id)
        
        assert latest is not None
        assert latest.checkpoint_id == "cp2"  # Should be the newer one
    
    def test_get_checkpoint_metrics(self, checkpoint_manager):
        """Test checkpoint metrics calculation"""
        # Add mock checkpoints
        checkpoint_manager.checkpoints = {
            "cp1": WorkflowCheckpoint(
                checkpoint_id="cp1",
                workflow_id="wf1",
                step_index=1,
                step_name="step1",
                workflow_state={},
                tda_context=None,
                timestamp=datetime.utcnow(),
                status=CheckpointStatus.ACTIVE
            ),
            "cp2": WorkflowCheckpoint(
                checkpoint_id="cp2",
                workflow_id="wf2",
                step_index=1,
                step_name="step1",
                workflow_state={},
                tda_context=None,
                timestamp=datetime.utcnow(),
                status=CheckpointStatus.CORRUPTED
            )
        }
        
        # Add mock recovery stats
        checkpoint_manager.recovery_stats = {
            "total_recoveries": 10,
            "successful_recoveries": 8,
            "recovery_times": [1.0, 2.0, 1.5],
            "strategy_usage": {
                "rollback_to_checkpoint": 5,
                "forward_recovery": 3,
                "hybrid_recovery": 2,
                "manual_intervention": 0
            }
        }
        
        metrics = checkpoint_manager.get_checkpoint_metrics()
        
        assert metrics["total_checkpoints"] == 2
        assert metrics["active_checkpoints"] == 1
        assert metrics["corrupted_checkpoints"] == 1
        assert metrics["total_recoveries"] == 10
        assert metrics["successful_recoveries"] == 8
        assert metrics["recovery_success_rate"] == 0.8
        assert metrics["average_recovery_time"] == 1.5  # (1.0 + 2.0 + 1.5) / 3
    
    def test_calculate_checksum(self, checkpoint_manager):
        """Test checksum calculation for integrity verification"""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="test-cp",
            workflow_id="test-wf",
            step_index=1,
            step_name="test",
            workflow_state={"test": "data"},
            tda_context={"context": "data"},
            timestamp=datetime.utcnow()
        )
        
        checksum1 = checkpoint_manager._calculate_checksum(checkpoint)
        checksum2 = checkpoint_manager._calculate_checksum(checkpoint)
        
        # Same checkpoint should produce same checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest length
        
        # Different checkpoint should produce different checksum
        checkpoint.workflow_state["different"] = "data"
        checksum3 = checkpoint_manager._calculate_checksum(checkpoint)
        assert checksum1 != checksum3
    
    @pytest.mark.asyncio
    async def test_checkpoint_storage_and_retrieval(self, checkpoint_manager, sample_workflow_state):
        """Test checkpoint storage and retrieval"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="storage-test",
            step_index=1,
            step_name="test",
            workflow_state=sample_workflow_state
        )
        
        # Retrieve checkpoint
        retrieved = await checkpoint_manager._retrieve_checkpoint(checkpoint.checkpoint_id)
        
        assert retrieved is not None
        assert retrieved.checkpoint_id == checkpoint.checkpoint_id
        assert retrieved.workflow_state == sample_workflow_state
    
    @pytest.mark.asyncio
    async def test_recovery_statistics_tracking(self, checkpoint_manager, sample_workflow_state):
        """Test that recovery statistics are properly tracked"""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            workflow_id="stats-test",
            step_index=1,
            step_name="test",
            workflow_state=sample_workflow_state
        )
        
        initial_total = checkpoint_manager.recovery_stats["total_recoveries"]
        initial_successful = checkpoint_manager.recovery_stats["successful_recoveries"]
        
        # Perform successful recovery
        await checkpoint_manager.recover_from_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            recovery_strategy=RecoveryStrategy.ROLLBACK_TO_CHECKPOINT
        )
        
        # Check statistics were updated
        assert checkpoint_manager.recovery_stats["total_recoveries"] == initial_total + 1
        assert checkpoint_manager.recovery_stats["successful_recoveries"] == initial_successful + 1
        assert len(checkpoint_manager.recovery_stats["recovery_times"]) > 0
        assert checkpoint_manager.recovery_stats["strategy_usage"]["rollback_to_checkpoint"] > 0

class TestWorkflowCheckpoint:
    """Test suite for WorkflowCheckpoint"""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation with all fields"""
        timestamp = datetime.utcnow()
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="test-cp-001",
            workflow_id="test-wf-001",
            step_index=2,
            step_name="analysis",
            workflow_state={"step": "data"},
            tda_context={"context": "data"},
            timestamp=timestamp,
            status=CheckpointStatus.CREATED,
            metadata={"test": "metadata"},
            checksum="test-checksum",
            size_bytes=1024
        )
        
        assert checkpoint.checkpoint_id == "test-cp-001"
        assert checkpoint.workflow_id == "test-wf-001"
        assert checkpoint.step_index == 2
        assert checkpoint.step_name == "analysis"
        assert checkpoint.status == CheckpointStatus.CREATED
        assert checkpoint.timestamp == timestamp
        assert checkpoint.checksum == "test-checksum"
        assert checkpoint.size_bytes == 1024
    
    def test_checkpoint_defaults(self):
        """Test checkpoint creation with default values"""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="test-cp",
            workflow_id="test-wf",
            step_index=1,
            step_name="test",
            workflow_state={},
            tda_context=None,
            timestamp=datetime.utcnow()
        )
        
        assert checkpoint.status == CheckpointStatus.CREATED  # Default value
        assert checkpoint.metadata is None  # Default value
        assert checkpoint.checksum is None  # Default value
        assert checkpoint.size_bytes == 0  # Default value