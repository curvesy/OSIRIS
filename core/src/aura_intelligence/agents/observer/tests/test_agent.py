"""
🧪 ObserverAgent Tests - Comprehensive Test Suite

Tests that prove our world-class modular architecture works:
- Evidence signature verification
- Immutable state updates
- Trace context propagation
- End-to-end workflow processing
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from ..agent import ObserverAgent
from ...schemas.enums import TaskStatus, EvidenceType, SignatureAlgorithm
from ...schemas.crypto import get_crypto_provider


class TestObserverAgent:
    """Comprehensive test suite for ObserverAgent."""
    
    @pytest.fixture
    def crypto_keys(self):
        """Generate test cryptographic keys."""
        provider = get_crypto_provider(SignatureAlgorithm.HMAC_SHA256)
        private_key = "test_private_key_12345"
        public_key = "test_public_key_12345"
        return private_key, public_key
    
    @pytest.fixture
    def observer_agent(self, crypto_keys):
        """Create test ObserverAgent instance."""
        private_key, public_key = crypto_keys
        
        agent = ObserverAgent(
            agent_id="test_observer_001",
            private_key=private_key,
            public_key=public_key,
            config={"max_concurrent": 5}
        )
        
        return agent
    
    @pytest.fixture
    def sample_event(self):
        """Create realistic test event."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "error",
            "message": "Database connection timeout after 30s",
            "source": "api-gateway",
            "type": "database_error",
            "fields": {
                "database": "user_db",
                "timeout_ms": 30000,
                "connection_pool": "primary"
            },
            "correlation_id": "test_correlation_123"
        }
    
    @pytest.mark.asyncio
    async def test_process_event_creates_valid_state(self, observer_agent, sample_event):
        """Test that process_event returns a valid AgentState."""
        # Act
        result_state = await observer_agent.process_event(sample_event)
        
        # Assert
        assert result_state is not None
        assert result_state.task_type == "error_investigation"
        assert result_state.status == TaskStatus.PENDING
        assert result_state.state_version == 2  # Initial + evidence update
        assert result_state.workflow_id.startswith("wf_")
        assert result_state.task_id.startswith("task_")
        assert result_state.correlation_id == "test_correlation_123"
        
        # Verify evidence was added
        assert len(result_state.context_dossier) == 1
        evidence = result_state.context_dossier[0]
        assert evidence.evidence_type == EvidenceType.LOG_ENTRY
        assert evidence.confidence == 0.95
        
        # Verify decision was made
        assert len(result_state.decision_points) == 1
        decision = result_state.decision_points[0]
        assert decision.decision_type == "workflow_routing"
        assert decision.chosen_option_id in ["escalate", "auto_investigate"]
    
    @pytest.mark.asyncio
    async def test_state_immutability(self, observer_agent, sample_event):
        """Test that state updates are truly immutable."""
        # Act
        state1 = await observer_agent.process_event(sample_event)
        
        # Process another event to get a different state
        sample_event["message"] = "Different error message"
        state2 = await observer_agent.process_event(sample_event)
        
        # Assert
        assert state1 is not state2  # Different objects
        assert state1.state_version != state2.state_version  # Different versions
        assert state1.workflow_id != state2.workflow_id  # Different workflows
        assert state1.updated_at != state2.updated_at  # Different timestamps
        
        # Original state should be unchanged
        assert "Database connection timeout" in str(state1.context_dossier[0].content)
        assert "Different error message" in str(state2.context_dossier[0].content)
    
    @pytest.mark.asyncio
    async def test_evidence_signature_verification(self, observer_agent, sample_event):
        """Test that evidence signatures are valid and verifiable."""
        # Act
        result_state = await observer_agent.process_event(sample_event)
        evidence = result_state.context_dossier[0]
        
        # Assert
        assert evidence.signature != "placeholder"
        assert evidence.signature_algorithm == SignatureAlgorithm.HMAC_SHA256
        assert evidence.collecting_agent_id == observer_agent.agent_id
        
        # Verify signature
        provider = get_crypto_provider(evidence.signature_algorithm)
        evidence_bytes = evidence.get_canonical_representation().encode('utf-8')
        is_valid = provider.verify(evidence_bytes, evidence.signature, observer_agent.private_key)
        assert is_valid, "Evidence signature should be valid"
    
    @pytest.mark.asyncio
    async def test_state_signature_verification(self, observer_agent, sample_event):
        """Test that state signatures are valid and verifiable."""
        # Act
        result_state = await observer_agent.process_event(sample_event)
        
        # Assert
        assert result_state.state_signature != "placeholder"
        assert result_state.signature_algorithm == SignatureAlgorithm.HMAC_SHA256
        assert result_state.last_modifier_agent_id == observer_agent.agent_id
        
        # Verify signature using the state's own method
        is_valid = result_state.verify_signature(observer_agent.private_key)
        assert is_valid, "State signature should be valid"
    
    @pytest.mark.asyncio
    async def test_decision_signature_verification(self, observer_agent, sample_event):
        """Test that decision signatures are valid and verifiable."""
        # Act
        result_state = await observer_agent.process_event(sample_event)
        decision = result_state.decision_points[0]
        
        # Assert
        assert decision.signature != "placeholder"
        assert decision.signature_algorithm == SignatureAlgorithm.HMAC_SHA256
        assert decision.deciding_agent_id == observer_agent.agent_id
        
        # Verify signature
        provider = get_crypto_provider(decision.signature_algorithm)
        decision_bytes = decision.get_canonical_representation().encode('utf-8')
        is_valid = provider.verify(decision_bytes, decision.signature, observer_agent.private_key)
        assert is_valid, "Decision signature should be valid"
    
    @pytest.mark.asyncio
    async def test_task_type_determination(self, observer_agent):
        """Test intelligent task type determination."""
        # Security event
        security_event = {
            "message": "Unauthorized access attempt detected",
            "level": "warning",
            "source": "security_monitor"
        }
        state = await observer_agent.process_event(security_event)
        assert state.task_type == "security_investigation"
        
        # Performance event
        perf_event = {
            "message": "High CPU usage detected",
            "level": "info",
            "metrics": {"cpu": 95.5, "memory": 80.2}
        }
        state = await observer_agent.process_event(perf_event)
        assert state.task_type == "performance_analysis"
        
        # Error event
        error_event = {
            "message": "Application crashed",
            "level": "critical",
            "source": "app_server"
        }
        state = await observer_agent.process_event(error_event)
        assert state.task_type == "error_investigation"
        
        # General event
        general_event = {
            "message": "User logged in",
            "level": "info",
            "source": "auth_service"
        }
        state = await observer_agent.process_event(general_event)
        assert state.task_type == "general_observation"
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, observer_agent, sample_event):
        """Test retry mechanism with exponential backoff."""
        # Mock the evidence creation to fail twice, then succeed
        original_method = observer_agent._create_evidence_from_event
        call_count = 0
        
        async def failing_evidence_creation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return await original_method(*args, **kwargs)
        
        observer_agent._create_evidence_from_event = failing_evidence_creation
        
        # Act - should succeed after retries
        result_state = await observer_agent.process_event(sample_event)
        
        # Assert
        assert result_state is not None
        assert call_count == 3  # Failed twice, succeeded on third try
        assert observer_agent.metrics["events_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, observer_agent, sample_event):
        """Test that performance metrics are tracked correctly."""
        # Initial metrics
        initial_processed = observer_agent.metrics["events_processed"]
        initial_evidence = observer_agent.metrics["evidence_created"]
        initial_workflows = observer_agent.metrics["workflows_initiated"]
        
        # Act
        await observer_agent.process_event(sample_event)
        
        # Assert
        assert observer_agent.metrics["events_processed"] == initial_processed + 1
        assert observer_agent.metrics["evidence_created"] == initial_evidence + 1
        assert observer_agent.metrics["workflows_initiated"] == initial_workflows + 1
        assert observer_agent.metrics["avg_processing_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_health_status(self, observer_agent):
        """Test health status reporting."""
        # Act
        health = await observer_agent.get_health_status()
        
        # Assert
        assert health["agent_id"] == observer_agent.agent_id
        assert health["status"] == "healthy"
        assert "metrics" in health
        assert "timestamp" in health
        assert health["schema_version"] == "2.0"
        
        # Verify metrics structure
        metrics = health["metrics"]
        expected_metrics = [
            "events_processed", "errors", "avg_processing_time_ms",
            "evidence_created", "workflows_initiated"
        ]
        for metric in expected_metrics:
            assert metric in metrics
    
    @pytest.mark.asyncio
    async def test_trace_context_integration(self, observer_agent, sample_event):
        """Test OpenTelemetry trace context integration."""
        # Add trace context to event
        sample_event["traceparent"] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        
        # Act
        with patch('opentelemetry.trace.get_tracer') as mock_tracer:
            mock_span = Mock()
            mock_span.get_span_context.return_value.trace_id = "test_trace_id"
            mock_tracer.return_value.start_as_current_span.return_value.__enter__.return_value = mock_span
            
            result_state = await observer_agent.process_event(sample_event)
        
        # Assert
        assert result_state is not None
        # Verify that tracing methods were called
        mock_tracer.assert_called()
    
    def test_agent_string_representation(self, observer_agent):
        """Test agent string representation."""
        agent_str = str(observer_agent)
        assert "ObserverAgent[test_observer_001]" in agent_str
        assert "Events:" in agent_str
        assert "Errors:" in agent_str


# Integration Tests
class TestObserverAgentIntegration:
    """Integration tests for end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow processing."""
        # Setup
        provider = get_crypto_provider(SignatureAlgorithm.HMAC_SHA256)
        private_key = "integration_test_key"
        public_key = "integration_test_public"
        
        agent = ObserverAgent(
            agent_id="integration_test_agent",
            private_key=private_key,
            public_key=public_key
        )
        
        # Create realistic event
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "error",
            "message": "Payment processing failed for user 12345",
            "source": "payment_service",
            "type": "payment_failure",
            "fields": {
                "user_id": "12345",
                "amount": 99.99,
                "currency": "USD",
                "error_code": "CARD_DECLINED"
            },
            "environment": "production",
            "priority": "high"
        }
        
        # Act
        final_state = await agent.process_event(event)
        
        # Assert comprehensive state
        assert final_state.task_type == "error_investigation"
        assert final_state.priority == "high"
        assert final_state.status == TaskStatus.PENDING
        assert len(final_state.context_dossier) == 1
        assert len(final_state.decision_points) == 1
        
        # Verify evidence quality
        evidence = final_state.context_dossier[0]
        assert evidence.confidence >= 0.9
        assert evidence.reliability >= 0.8
        assert evidence.freshness == 1.0
        
        # Verify decision quality
        decision = final_state.decision_points[0]
        assert decision.confidence_in_decision >= 0.8
        assert decision.chosen_option_id in ["escalate", "auto_investigate"]
        
        # Verify all signatures are valid
        assert final_state.verify_signature(private_key)
        
        print(f"✅ End-to-end test passed: {final_state}")


if __name__ == "__main__":
    # Run a quick test
    import asyncio
    
    async def quick_test():
        provider = get_crypto_provider(SignatureAlgorithm.HMAC_SHA256)
        agent = ObserverAgent(
            agent_id="quick_test_agent",
            private_key="test_key",
            public_key="test_public"
        )
        
        event = {
            "message": "Test event",
            "level": "info",
            "source": "test"
        }
        
        state = await agent.process_event(event)
        print(f"Quick test result: {state}")
        print(f"Agent metrics: {agent.metrics}")
    
    asyncio.run(quick_test())
