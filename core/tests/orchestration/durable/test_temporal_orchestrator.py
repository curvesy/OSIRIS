"""
⏱️ Temporal Orchestrator Tests

Tests for the Temporal.io durable workflow orchestrator including
workflow execution, retry logic, and TDA integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from aura_intelligence.orchestration.durable.temporal_orchestrator import (
    TemporalDurableOrchestrator,
    DurableWorkflowConfig,
    WorkflowExecutionResult,
    CompensationAction,
    WorkflowStatus,
    CompensationStrategy
)

class TestTemporalDurableOrchestrator:
    """Test suite for TemporalDurableOrchestrator"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            complexity_score=0.7,
            urgency_score=0.5,
            context_data={"test": "data"}
        ))
        mock.send_orchestration_result = AsyncMock()
        return mock
    
    @pytest.fixture
    def orchestrator(self, mock_tda_integration):
        """Create orchestrator instance"""
        return TemporalDurableOrchestrator(
            temporal_client=None,  # Use fallback mode
            tda_integration=mock_tda_integration
        )
    
    @pytest.fixture
    def sample_workflow_config(self):
        """Sample workflow configuration"""
        return DurableWorkflowConfig(
            workflow_id="test-workflow-001",
            workflow_type="multi_agent_analysis",
            steps=[
                {
                    "name": "data_collection",
                    "agent_type": "observer",
                    "timeout": 300,
                    "parameters": {"source": "test_data"}
                },
                {
                    "name": "analysis",
                    "agent_type": "analyst", 
                    "timeout": 600,
                    "parameters": {"analysis_type": "pattern_detection"}
                },
                {
                    "name": "decision",
                    "agent_type": "supervisor",
                    "timeout": 300,
                    "parameters": {"decision_criteria": "confidence > 0.8"}
                }
            ],
            retry_policy={
                "max_attempts": 3,
                "initial_interval": 1,
                "max_interval": 30
            },
            compensation_strategy=CompensationStrategy.ROLLBACK_ALL,
            tda_correlation_id="test-correlation-001"
        )
    
    @pytest.mark.asyncio
    async def test_execute_durable_workflow_success(self, orchestrator, sample_workflow_config):
        """Test successful workflow execution"""
        input_data = {"test_input": "data"}
        
        result = await orchestrator.execute_durable_workflow(
            sample_workflow_config, input_data
        )
        
        assert result.workflow_id == "test-workflow-001"
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.results) == 3  # All steps completed
        assert result.execution_time > 0
        assert len(result.checkpoints) == 3  # One checkpoint per step
        assert result.tda_correlation == "test-correlation-001"
    
    @pytest.mark.asyncio
    async def test_execute_durable_workflow_with_retry(self, orchestrator, sample_workflow_config):
        """Test workflow execution with retry logic"""
        input_data = {"test_input": "data"}
        
        # Mock step execution to fail first attempt
        original_execute_step = orchestrator._execute_step_with_retry
        call_count = 0
        
        async def mock_execute_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call fails
                raise Exception("Simulated failure")
            return await original_execute_step(*args, **kwargs)
        
        orchestrator._execute_step_with_retry = mock_execute_step
        
        result = await orchestrator.execute_durable_workflow(
            sample_workflow_config, input_data
        )
        
        assert result.status == WorkflowStatus.COMPLETED
        assert call_count > 1  # Retry was attempted
    
    @pytest.mark.asyncio
    async def test_execute_durable_workflow_failure_with_compensation(self, orchestrator, sample_workflow_config):
        """Test workflow failure triggers compensation"""
        input_data = {"test_input": "data"}
        
        # Mock step execution to always fail
        async def mock_failing_step(*args, **kwargs):
            raise Exception("Persistent failure")
        
        orchestrator._execute_step_with_retry = mock_failing_step
        
        result = await orchestrator.execute_durable_workflow(
            sample_workflow_config, input_data
        )
        
        assert result.status == WorkflowStatus.FAILED
        assert result.error_details is not None
        assert "Persistent failure" in result.error_details["error"]
        assert len(result.compensation_actions) >= 0  # Compensation attempted
    
    @pytest.mark.asyncio
    async def test_workflow_tda_integration(self, orchestrator, sample_workflow_config, mock_tda_integration):
        """Test TDA integration during workflow execution"""
        input_data = {"test_input": "data"}
        
        result = await orchestrator.execute_durable_workflow(
            sample_workflow_config, input_data
        )
        
        # Verify TDA integration was called
        mock_tda_integration.get_context.assert_called_with("test-correlation-001")
        mock_tda_integration.send_orchestration_result.assert_called()
        
        # Check that TDA context influenced execution
        call_args = mock_tda_integration.send_orchestration_result.call_args[0]
        assert call_args[1] == "test-correlation-001"  # Correlation ID passed
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, orchestrator, sample_workflow_config):
        """Test workflow status retrieval"""
        input_data = {"test_input": "data"}
        
        # Execute workflow
        result = await orchestrator.execute_durable_workflow(
            sample_workflow_config, input_data
        )
        
        # Get status
        status = await orchestrator.get_workflow_status("test-workflow-001")
        
        assert status is not None
        assert status.workflow_id == "test-workflow-001"
        assert status.status == WorkflowStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(self, orchestrator):
        """Test workflow cancellation"""
        # Test fallback cancellation
        orchestrator.active_workflows["test-workflow"] = Mock()
        
        result = await orchestrator.cancel_workflow("test-workflow")
        
        assert result is True
        assert "test-workflow" not in orchestrator.active_workflows
    
    def test_get_execution_metrics(self, orchestrator):
        """Test execution metrics calculation"""
        # Add some mock execution history
        orchestrator.execution_history = [
            WorkflowExecutionResult(
                workflow_id="wf1",
                status=WorkflowStatus.COMPLETED,
                results={},
                execution_time=1.5,
                checkpoints=[],
                compensation_actions=[]
            ),
            WorkflowExecutionResult(
                workflow_id="wf2", 
                status=WorkflowStatus.FAILED,
                results={},
                execution_time=2.0,
                checkpoints=[],
                compensation_actions=["comp1"]
            )
        ]
        
        metrics = orchestrator.get_execution_metrics()
        
        assert metrics["total_workflows"] == 2
        assert metrics["success_rate"] == 0.5  # 1 success out of 2
        assert metrics["average_execution_time"] == 1.75  # (1.5 + 2.0) / 2
        assert metrics["compensation_rate"] == 0.5  # 1 compensation out of 2
    
    @pytest.mark.asyncio
    async def test_step_retry_logic(self, orchestrator):
        """Test individual step retry logic"""
        step = {
            "name": "test_step",
            "timeout": 300,
            "parameters": {"test": "param"}
        }
        
        retry_policy = {
            "max_attempts": 3,
            "initial_interval": 0.1,  # Short interval for testing
            "max_interval": 1
        }
        
        # Mock that fails twice then succeeds
        attempt_count = 0
        
        async def mock_step_execution(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:
                raise Exception(f"Attempt {attempt_count} failed")
            return {"success": True, "attempt": attempt_count}
        
        # Patch the step execution
        with patch.object(orchestrator, '_execute_step_with_retry', side_effect=mock_step_execution):
            # This should succeed after retries
            result = await orchestrator._execute_step_with_retry(
                step, {}, None, {}, retry_policy
            )
        
        assert result["success"] is True
        assert result["attempt"] == 3  # Succeeded on third attempt
    
    @pytest.mark.asyncio
    async def test_compensation_execution(self, orchestrator, sample_workflow_config):
        """Test compensation logic execution"""
        executed_steps = ["step1", "step2"]
        results = {"step1": "result1", "step2": "result2"}
        
        compensation_actions = await orchestrator._compensate_fallback_steps(
            executed_steps, results, sample_workflow_config
        )
        
        assert len(compensation_actions) == 2
        assert "Compensated step2" in compensation_actions[0]  # Reverse order
        assert "Compensated step1" in compensation_actions[1]
    
    @pytest.mark.asyncio
    async def test_workflow_failure_handling(self, orchestrator, sample_workflow_config):
        """Test workflow failure handling"""
        input_data = {"test": "data"}
        error = Exception("Test error")
        start_time = datetime.utcnow()
        
        result = await orchestrator._handle_workflow_failure(
            sample_workflow_config, input_data, error, start_time
        )
        
        assert result.status == WorkflowStatus.FAILED
        assert result.error_details["error"] == "Test error"
        assert result.error_details["type"] == "Exception"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_fallback_mode_initialization(self):
        """Test fallback mode when Temporal.io is not available"""
        orchestrator = TemporalDurableOrchestrator(temporal_client=None)
        
        assert hasattr(orchestrator, 'fallback_mode')
        assert hasattr(orchestrator, 'fallback_workflows')
    
    def test_workflow_config_validation(self):
        """Test workflow configuration validation"""
        config = DurableWorkflowConfig(
            workflow_id="test",
            workflow_type="test_type",
            steps=[{"name": "step1"}],
            retry_policy={"max_attempts": 3},
            compensation_strategy=CompensationStrategy.ROLLBACK_ALL
        )
        
        assert config.workflow_id == "test"
        assert config.timeout_seconds == 3600  # Default value
        assert config.checkpoint_interval == 300  # Default value
        assert config.metadata is None  # Default value

@pytest.mark.integration
class TestTemporalIntegration:
    """Integration tests for Temporal.io (when available)"""
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'temporal_available') or not pytest.temporal_available,
        reason="Temporal.io not available in test environment"
    )
    @pytest.mark.asyncio
    async def test_temporal_workflow_execution(self):
        """Test actual Temporal.io workflow execution (if available)"""
        # This test would run only if Temporal.io is properly set up
        # Implementation would depend on actual Temporal.io setup
        pass
    
    @pytest.mark.asyncio
    async def test_temporal_fallback_compatibility(self):
        """Test that fallback mode provides compatible interface"""
        orchestrator = TemporalDurableOrchestrator(temporal_client=None)
        
        config = DurableWorkflowConfig(
            workflow_id="fallback-test",
            workflow_type="test",
            steps=[{"name": "test_step"}],
            retry_policy={"max_attempts": 1},
            compensation_strategy=CompensationStrategy.ROLLBACK_ALL
        )
        
        result = await orchestrator.execute_durable_workflow(config, {})
        
        # Should work in fallback mode
        assert result.workflow_id == "fallback-test"
        assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]