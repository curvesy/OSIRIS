"""
🔄 Saga Patterns Tests

Tests for saga pattern implementation including compensation logic,
rollback mechanisms, and TDA integration.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from aura_intelligence.orchestration.durable.saga_patterns import (
    SagaOrchestrator,
    SagaStep,
    CompensationHandler,
    SagaStepStatus,
    CompensationType
)

class TestSagaOrchestrator:
    """Test suite for SagaOrchestrator"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            complexity_score=0.6,
            urgency_score=0.4,
            context_data={"saga": "context"}
        ))
        mock.send_orchestration_result = AsyncMock()
        return mock
    
    @pytest.fixture
    def saga_orchestrator(self, mock_tda_integration):
        """Create saga orchestrator instance"""
        return SagaOrchestrator(tda_integration=mock_tda_integration)
    
    @pytest.fixture
    def sample_saga_steps(self):
        """Sample saga steps for testing"""
        async def step1_action(input_data):
            return {"step1": "completed", "data": input_data}
        
        async def step1_compensation(input_data):
            return {"step1": "compensated"}
        
        async def step2_action(input_data):
            return {"step2": "completed", "data": input_data}
        
        async def step2_compensation(input_data):
            return {"step2": "compensated"}
        
        return [
            SagaStep(
                step_id="step-001",
                name="data_preparation",
                action=step1_action,
                compensation_action=step1_compensation,
                parameters={"source": "test_data"},
                compensation_parameters={"cleanup": True}
            ),
            SagaStep(
                step_id="step-002", 
                name="data_processing",
                action=step2_action,
                compensation_action=step2_compensation,
                parameters={"process_type": "analysis"},
                compensation_parameters={"rollback": True}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_execute_saga_success(self, saga_orchestrator, sample_saga_steps):
        """Test successful saga execution"""
        result = await saga_orchestrator.execute_saga(
            saga_id="test-saga-001",
            steps=sample_saga_steps,
            tda_correlation_id="test-correlation"
        )
        
        assert result["saga_id"] == "test-saga-001"
        assert result["status"] == "completed"
        assert result["steps_executed"] == 2
        assert result["execution_time"] > 0
        assert "data_preparation" in result["results"]
        assert "data_processing" in result["results"]
        assert result["tda_correlation_id"] == "test-correlation"
    
    @pytest.mark.asyncio
    async def test_execute_saga_with_failure_and_compensation(self, saga_orchestrator):
        """Test saga execution with failure triggering compensation"""
        async def successful_action(input_data):
            return {"success": True}
        
        async def failing_action(input_data):
            raise Exception("Step failed")
        
        async def compensation_action(input_data):
            return {"compensated": True}
        
        steps = [
            SagaStep(
                step_id="step-001",
                name="successful_step",
                action=successful_action,
                compensation_action=compensation_action,
                parameters={"test": "param"}
            ),
            SagaStep(
                step_id="step-002",
                name="failing_step", 
                action=failing_action,
                compensation_action=compensation_action,
                parameters={"test": "param"}
            )
        ]
        
        result = await saga_orchestrator.execute_saga(
            saga_id="test-saga-failure",
            steps=steps
        )
        
        assert result["saga_id"] == "test-saga-failure"
        assert result["status"] == "failed"
        assert result["steps_executed"] == 1  # Only first step executed
        assert "error" in result
        assert "compensated_steps" in result
    
    @pytest.mark.asyncio
    async def test_saga_step_retry_logic(self, saga_orchestrator):
        """Test retry logic for saga steps"""
        attempt_count = 0
        
        async def flaky_action(input_data):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise Exception(f"Attempt {attempt_count} failed")
            return {"success": True, "attempts": attempt_count}
        
        steps = [
            SagaStep(
                step_id="step-001",
                name="flaky_step",
                action=flaky_action,
                compensation_action=None,
                parameters={"test": "retry"},
                max_retries=3
            )
        ]
        
        result = await saga_orchestrator.execute_saga(
            saga_id="test-saga-retry",
            steps=steps
        )
        
        assert result["status"] == "completed"
        assert attempt_count == 3  # Should have retried and succeeded
        assert result["results"]["flaky_step"]["attempts"] == 3
    
    @pytest.mark.asyncio
    async def test_saga_tda_integration(self, saga_orchestrator, sample_saga_steps, mock_tda_integration):
        """Test TDA integration during saga execution"""
        result = await saga_orchestrator.execute_saga(
            saga_id="test-saga-tda",
            steps=sample_saga_steps,
            tda_correlation_id="test-correlation"
        )
        
        # Verify TDA integration was called
        mock_tda_integration.get_context.assert_called_with("test-correlation")
        mock_tda_integration.send_orchestration_result.assert_called()
        
        # Check that result was sent to TDA
        call_args = mock_tda_integration.send_orchestration_result.call_args[0]
        assert call_args[1] == "test-correlation"  # Correlation ID
        assert call_args[0]["saga_id"] == "test-saga-tda"
    
    @pytest.mark.asyncio
    async def test_compensation_execution_order(self, saga_orchestrator):
        """Test that compensation executes in reverse order"""
        compensation_order = []
        
        async def action1(input_data):
            return {"step": 1}
        
        async def action2(input_data):
            return {"step": 2}
        
        async def action3(input_data):
            raise Exception("Step 3 failed")  # This will trigger compensation
        
        async def compensation1(input_data):
            compensation_order.append("comp1")
            return {"compensated": 1}
        
        async def compensation2(input_data):
            compensation_order.append("comp2")
            return {"compensated": 2}
        
        steps = [
            SagaStep("s1", "step1", action1, compensation1, {}),
            SagaStep("s2", "step2", action2, compensation2, {}),
            SagaStep("s3", "step3", action3, None, {})
        ]
        
        result = await saga_orchestrator.execute_saga("test-compensation-order", steps)
        
        assert result["status"] == "failed"
        # Compensation should execute in reverse order: comp2, then comp1
        assert compensation_order == ["comp2", "comp1"]
    
    def test_register_compensation_handler(self, saga_orchestrator):
        """Test compensation handler registration"""
        handler = CompensationHandler(
            handler_id="handler-001",
            step_name="test_step",
            compensation_type=CompensationType.ROLLBACK,
            handler_function=lambda x: {"handled": True},
            parameters={"test": "param"},
            priority=1
        )
        
        saga_orchestrator.register_compensation_handler("test_step", handler)
        
        assert "test_step" in saga_orchestrator.compensation_handlers
        assert len(saga_orchestrator.compensation_handlers["test_step"]) == 1
        assert saga_orchestrator.compensation_handlers["test_step"][0] == handler
    
    @pytest.mark.asyncio
    async def test_execute_custom_compensation(self, saga_orchestrator):
        """Test custom compensation execution"""
        async def custom_handler(input_data):
            return {
                "custom_compensation": True,
                "saga_id": input_data["saga_id"],
                "parameters": input_data["parameters"]
            }
        
        handler = CompensationHandler(
            handler_id="custom-handler",
            step_name="custom_step",
            compensation_type=CompensationType.FORWARD_RECOVERY,
            handler_function=custom_handler,
            parameters={"custom": "param"}
        )
        
        saga_orchestrator.register_compensation_handler("custom_step", handler)
        
        result = await saga_orchestrator.execute_custom_compensation(
            saga_id="test-saga",
            step_name="custom_step",
            compensation_type=CompensationType.FORWARD_RECOVERY,
            parameters={"test": "data"}
        )
        
        assert result["status"] == "completed"
        assert result["handler_id"] == "custom-handler"
        assert result["result"]["custom_compensation"] is True
    
    def test_get_saga_status_active(self, saga_orchestrator, sample_saga_steps):
        """Test getting status of active saga"""
        saga_id = "active-saga"
        saga_orchestrator.active_sagas[saga_id] = sample_saga_steps
        
        # Set one step as executing
        sample_saga_steps[0].status = SagaStepStatus.EXECUTING
        sample_saga_steps[1].status = SagaStepStatus.PENDING
        
        status = saga_orchestrator.get_saga_status(saga_id)
        
        assert status["saga_id"] == saga_id
        assert status["status"] == "running"
        assert status["total_steps"] == 2
        assert status["completed_steps"] == 0
        assert status["current_step"] == "data_preparation"
    
    def test_get_saga_status_completed(self, saga_orchestrator):
        """Test getting status of completed saga"""
        # Add completed saga to history
        saga_orchestrator.saga_history.append({
            "saga_id": "completed-saga",
            "status": "completed",
            "steps_executed": 2,
            "execution_time": 1.5
        })
        
        status = saga_orchestrator.get_saga_status("completed-saga")
        
        assert status["saga_id"] == "completed-saga"
        assert status["status"] == "completed"
        assert status["execution_time"] == 1.5
    
    def test_get_saga_metrics(self, saga_orchestrator):
        """Test saga metrics calculation"""
        # Add mock saga history
        saga_orchestrator.saga_history = [
            {
                "saga_id": "saga1",
                "status": "completed",
                "execution_time": 1.0,
                "compensated_steps": []
            },
            {
                "saga_id": "saga2",
                "status": "failed",
                "execution_time": 2.0,
                "compensated_steps": ["step1"]
            },
            {
                "saga_id": "saga3",
                "status": "completed",
                "execution_time": 1.5,
                "compensated_steps": []
            }
        ]
        
        metrics = saga_orchestrator.get_saga_metrics()
        
        assert metrics["total_sagas"] == 3
        assert metrics["success_rate"] == 2/3  # 2 successful out of 3
        assert metrics["average_execution_time"] == 1.5  # (1.0 + 2.0 + 1.5) / 3
        assert metrics["compensation_rate"] == 1/3  # 1 saga with compensation
    
    @pytest.mark.asyncio
    async def test_saga_step_status_transitions(self, saga_orchestrator):
        """Test saga step status transitions"""
        async def test_action(input_data):
            return {"result": "success"}
        
        step = SagaStep(
            step_id="status-test",
            name="status_step",
            action=test_action,
            compensation_action=None,
            parameters={}
        )
        
        # Initial status should be PENDING
        assert step.status == SagaStepStatus.PENDING
        
        # Execute saga with this step
        result = await saga_orchestrator.execute_saga(
            saga_id="status-test-saga",
            steps=[step]
        )
        
        # After successful execution, status should be COMPLETED
        assert step.status == SagaStepStatus.COMPLETED
        assert step.result is not None
        assert step.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_compensation_failure_handling(self, saga_orchestrator):
        """Test handling of compensation failures"""
        async def successful_action(input_data):
            return {"success": True}
        
        async def failing_action(input_data):
            raise Exception("Action failed")
        
        async def failing_compensation(input_data):
            raise Exception("Compensation failed")
        
        steps = [
            SagaStep(
                step_id="step-001",
                name="successful_step",
                action=successful_action,
                compensation_action=failing_compensation,  # This will fail
                parameters={}
            ),
            SagaStep(
                step_id="step-002",
                name="failing_step",
                action=failing_action,
                compensation_action=None,
                parameters={}
            )
        ]
        
        result = await saga_orchestrator.execute_saga(
            saga_id="compensation-failure-test",
            steps=steps
        )
        
        assert result["status"] == "failed"
        # Should still attempt compensation even if it fails
        assert steps[0].error is not None  # Compensation error recorded

class TestSagaStep:
    """Test suite for SagaStep"""
    
    def test_saga_step_creation(self):
        """Test saga step creation with defaults"""
        async def test_action(input_data):
            return {"test": "result"}
        
        step = SagaStep(
            step_id="test-step",
            name="test_step",
            action=test_action,
            compensation_action=None,
            parameters={"param": "value"}
        )
        
        assert step.step_id == "test-step"
        assert step.name == "test_step"
        assert step.status == SagaStepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.retry_count == 0
        assert step.max_retries == 3  # Default value
    
    def test_saga_step_with_compensation(self):
        """Test saga step with compensation action"""
        async def action(input_data):
            return {"action": "result"}
        
        async def compensation(input_data):
            return {"compensation": "result"}
        
        step = SagaStep(
            step_id="comp-step",
            name="compensation_step",
            action=action,
            compensation_action=compensation,
            parameters={"param": "value"},
            compensation_parameters={"comp_param": "comp_value"}
        )
        
        assert step.compensation_action is not None
        assert step.compensation_parameters == {"comp_param": "comp_value"}

class TestCompensationHandler:
    """Test suite for CompensationHandler"""
    
    def test_compensation_handler_creation(self):
        """Test compensation handler creation"""
        def handler_func(input_data):
            return {"handled": True}
        
        handler = CompensationHandler(
            handler_id="handler-001",
            step_name="test_step",
            compensation_type=CompensationType.ROLLBACK,
            handler_function=handler_func,
            parameters={"param": "value"},
            tda_context_required=True,
            priority=5
        )
        
        assert handler.handler_id == "handler-001"
        assert handler.step_name == "test_step"
        assert handler.compensation_type == CompensationType.ROLLBACK
        assert handler.tda_context_required is True
        assert handler.priority == 5