"""
Unit tests for workflow state definitions.

Tests the state management components using modern pytest patterns
and Python 3.13+ features.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from aura_intelligence.orchestration.workflows.state import (
    WorkflowStatus,
    RiskLevel,
    SystemHealth,
    ErrorContext,
    CollectiveState
)


class TestWorkflowStatus:
    """Test cases for WorkflowStatus enum."""
    
    def test_workflow_status_values(self):
        """Test that all workflow statuses are defined."""
        assert WorkflowStatus.INITIALIZING
        assert WorkflowStatus.RUNNING
        assert WorkflowStatus.ERROR
        assert WorkflowStatus.COMPLETED
        assert WorkflowStatus.CANCELLED
    
    def test_workflow_status_uniqueness(self):
        """Test that workflow status values are unique."""
        values = [status.value for status in WorkflowStatus]
        assert len(values) == len(set(values))


class TestRiskLevel:
    """Test cases for RiskLevel enum."""
    
    @pytest.mark.parametrize("risk_level,expected_threshold", [
        (RiskLevel.CRITICAL, 0.9),
        (RiskLevel.HIGH, 0.7),
        (RiskLevel.MEDIUM, 0.4),
        (RiskLevel.LOW, 0.1),
    ])
    def test_risk_level_thresholds(self, risk_level: RiskLevel, expected_threshold: float):
        """Test risk level threshold values."""
        assert risk_level.threshold == expected_threshold
    
    def test_risk_level_values(self):
        """Test risk level string values."""
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"
    
    def test_risk_level_pattern_matching(self):
        """Test pattern matching with risk levels (Python 3.13+)."""
        def get_action(risk: RiskLevel) -> str:
            match risk:
                case RiskLevel.CRITICAL:
                    return "immediate_action"
                case RiskLevel.HIGH:
                    return "urgent_review"
                case RiskLevel.MEDIUM:
                    return "scheduled_review"
                case RiskLevel.LOW:
                    return "monitor"
        
        assert get_action(RiskLevel.CRITICAL) == "immediate_action"
        assert get_action(RiskLevel.HIGH) == "urgent_review"
        assert get_action(RiskLevel.MEDIUM) == "scheduled_review"
        assert get_action(RiskLevel.LOW) == "monitor"


class TestSystemHealth:
    """Test cases for SystemHealth dataclass."""
    
    def test_system_health_defaults(self):
        """Test default values for SystemHealth."""
        health = SystemHealth()
        assert health.cpu_usage == 0.0
        assert health.memory_usage == 0.0
        assert health.active_connections == 0
        assert health.error_rate == 0.0
        assert isinstance(health.last_check, datetime)
    
    @pytest.mark.parametrize("cpu,memory,error_rate,expected", [
        (0.5, 0.5, 0.05, True),   # All healthy
        (0.9, 0.5, 0.05, False),  # High CPU
        (0.5, 0.9, 0.05, False),  # High memory
        (0.5, 0.5, 0.15, False),  # High error rate
        (0.79, 0.79, 0.09, True), # Just under thresholds
    ])
    def test_is_healthy_property(
        self,
        cpu: float,
        memory: float,
        error_rate: float,
        expected: bool
    ):
        """Test is_healthy property with various conditions."""
        health = SystemHealth(
            cpu_usage=cpu,
            memory_usage=memory,
            error_rate=error_rate
        )
        assert health.is_healthy == expected
    
    def test_system_health_immutability(self):
        """Test that SystemHealth fields can be modified (not frozen)."""
        health = SystemHealth(cpu_usage=0.5)
        health.cpu_usage = 0.8
        assert health.cpu_usage == 0.8


class TestErrorContext:
    """Test cases for ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test creating ErrorContext instances."""
        error = ErrorContext(
            error_type="ValueError",
            message="Invalid input",
            stack_trace="Traceback..."
        )
        assert error.error_type == "ValueError"
        assert error.message == "Invalid input"
        assert error.stack_trace == "Traceback..."
        assert error.recovery_attempted is False
        assert isinstance(error.timestamp, datetime)
    
    def test_error_context_to_dict(self):
        """Test converting ErrorContext to dictionary."""
        error = ErrorContext(
            error_type="RuntimeError",
            message="Something went wrong",
            recovery_attempted=True
        )
        
        result = error.to_dict()
        
        assert result["error_type"] == "RuntimeError"
        assert result["message"] == "Something went wrong"
        assert result["recovery_attempted"] is True
        assert result["stack_trace"] is None
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)  # ISO format string


class TestCollectiveState:
    """Test cases for CollectiveState TypedDict."""
    
    def test_collective_state_type_annotations(self):
        """Test that CollectiveState has proper type annotations."""
        annotations = CollectiveState.__annotations__
        
        # Core workflow state
        assert "messages" in annotations
        assert "workflow_id" in annotations
        assert "thread_id" in annotations
        assert "current_step" in annotations
        assert "workflow_status" in annotations
        
        # Evidence and decision tracking
        assert "evidence_log" in annotations
        assert "supervisor_decisions" in annotations
        assert "execution_results" in annotations
        
        # Memory and configuration
        assert "memory_context" in annotations
        assert "active_config" in annotations
        
        # Risk assessment
        assert "risk_assessment" in annotations
        assert "risk_level" in annotations
        
        # Error handling
        assert "error_log" in annotations
        assert "error_recovery_attempts" in annotations
        assert "last_error" in annotations
        assert "system_health" in annotations
        
        # Shadow mode
        assert "shadow_mode_enabled" in annotations
        assert "shadow_predictions" in annotations
    
    def test_collective_state_creation(self):
        """Test creating a CollectiveState instance."""
        # TypedDict allows dict creation with type hints
        state: CollectiveState = {
            "messages": [],
            "workflow_id": "test-123",
            "thread_id": "thread-456",
            "current_step": "initializing",
            "workflow_status": WorkflowStatus.INITIALIZING,
            "evidence_log": [],
            "supervisor_decisions": [],
            "execution_results": [],
            "memory_context": {},
            "active_config": {},
            "risk_assessment": None,
            "risk_level": None,
            "error_log": [],
            "error_recovery_attempts": 0,
            "last_error": None,
            "system_health": SystemHealth(),
            "shadow_mode_enabled": True,
            "shadow_predictions": []
        }
        
        assert state["workflow_id"] == "test-123"
        assert state["workflow_status"] == WorkflowStatus.INITIALIZING
        assert state["shadow_mode_enabled"] is True


@pytest.fixture
def sample_error_context() -> ErrorContext:
    """Fixture providing a sample ErrorContext."""
    return ErrorContext(
        error_type="TestError",
        message="This is a test error",
        stack_trace="Test stack trace"
    )


@pytest.fixture
def sample_system_health() -> SystemHealth:
    """Fixture providing a sample SystemHealth."""
    return SystemHealth(
        cpu_usage=0.45,
        memory_usage=0.60,
        active_connections=100,
        error_rate=0.02
    )


@pytest.fixture
def sample_collective_state(
    sample_system_health: SystemHealth,
    sample_error_context: ErrorContext
) -> CollectiveState:
    """Fixture providing a sample CollectiveState."""
    return {
        "messages": [],
        "workflow_id": "fixture-workflow-123",
        "thread_id": "fixture-thread-456",
        "current_step": "processing",
        "workflow_status": WorkflowStatus.RUNNING,
        "evidence_log": [
            {"type": "observation", "data": "test"}
        ],
        "supervisor_decisions": [],
        "execution_results": [],
        "memory_context": {"key": "value"},
        "active_config": {"enable_streaming": True},
        "risk_assessment": {"score": 0.5},
        "risk_level": RiskLevel.MEDIUM,
        "error_log": [sample_error_context],
        "error_recovery_attempts": 1,
        "last_error": sample_error_context,
        "system_health": sample_system_health,
        "shadow_mode_enabled": False,
        "shadow_predictions": []
    }


class TestIntegration:
    """Integration tests for state components."""
    
    def test_state_with_fixtures(self, sample_collective_state: CollectiveState):
        """Test working with a complete state object."""
        state = sample_collective_state
        
        # Verify state properties
        assert state["workflow_status"] == WorkflowStatus.RUNNING
        assert state["risk_level"] == RiskLevel.MEDIUM
        assert state["system_health"].is_healthy is True
        assert len(state["error_log"]) == 1
        assert state["error_log"][0].error_type == "TestError"
    
    def test_state_modification(self, sample_collective_state: CollectiveState):
        """Test modifying state values."""
        state = sample_collective_state
        
        # Modify state
        state["workflow_status"] = WorkflowStatus.COMPLETED
        state["risk_level"] = RiskLevel.HIGH
        state["error_recovery_attempts"] += 1
        
        # Verify modifications
        assert state["workflow_status"] == WorkflowStatus.COMPLETED
        assert state["risk_level"] == RiskLevel.HIGH
        assert state["error_recovery_attempts"] == 2