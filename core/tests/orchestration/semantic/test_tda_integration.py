"""
🧪 Tests for TDA Context Integration Layer

Comprehensive test suite for TDA integration including context retrieval,
result sending, pattern correlation, and caching functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from aura_intelligence.orchestration.semantic.tda_integration import (
    TDAContextIntegration, MockTDAIntegration
)
from aura_intelligence.orchestration.semantic.base_interfaces import TDAContext

@pytest.fixture
def tda_integration():
    """Create TDA integration with mock configuration"""
    config = {
        "brokers": ["localhost:9092"],
        "client_id": "test-orchestration"
    }
    return TDAContextIntegration(config)

@pytest.fixture
def mock_tda_integration():
    """Create mock TDA integration for testing"""
    return MockTDAIntegration()

@pytest.fixture
def sample_tda_context():
    """Sample TDA context for testing"""
    return TDAContext(
        correlation_id="test-correlation-123",
        pattern_confidence=0.8,
        anomaly_severity=0.6,
        current_patterns={"anomaly_type": "spike", "confidence": 0.8},
        temporal_window="1h",
        metadata={"source": "test", "timestamp": datetime.utcnow().isoformat()}
    )

class TestTDAContextRetrieval:
    """Test TDA context retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_context_retrieval_with_cache_miss(self, tda_integration):
        """Test context retrieval when not in cache"""
        correlation_id = "test-correlation-456"
        
        with patch('aura_intelligence.orchestration.semantic.tda_integration.get_current_patterns') as mock_patterns:
            mock_patterns.return_value = {
                "patterns": {"test_pattern": 0.7},
                "anomalies": {"severity": 0.4, "anomalies": []},
                "window": "1h"
            }
            
            with patch('aura_intelligence.orchestration.semantic.tda_integration.analyze_correlation') as mock_correlation:
                mock_correlation.return_value = {"confidence": 0.8}
                
                context = await tda_integration.get_context(correlation_id)
                
                assert context is not None
                assert context.correlation_id == correlation_id
                assert context.pattern_confidence == 0.8
                assert context.anomaly_severity == 0.4
    
    @pytest.mark.asyncio
    async def test_context_retrieval_with_cache_hit(self, tda_integration, sample_tda_context):
        """Test context retrieval from cache"""
        correlation_id = "cached-correlation"
        
        # Manually cache the context
        tda_integration._cache_context(correlation_id, sample_tda_context)
        
        # Retrieve should hit cache
        context = await tda_integration.get_context(correlation_id)
        
        assert context == sample_tda_context
        assert context.correlation_id == correlation_id
    
    @pytest.mark.asyncio
    async def test_context_cache_expiration(self, tda_integration, sample_tda_context):
        """Test that cached contexts expire correctly"""
        correlation_id = "expiring-correlation"
        
        # Cache with expired timestamp
        expired_time = datetime.utcnow() - timedelta(seconds=400)  # Beyond TTL
        tda_integration.context_cache[correlation_id] = (sample_tda_context, expired_time)
        
        # Should not return expired cache
        cached_context = tda_integration._get_cached_context(correlation_id)
        assert cached_context is None
        assert correlation_id not in tda_integration.context_cache
    
    @pytest.mark.asyncio
    async def test_fallback_context_creation(self, tda_integration):
        """Test fallback context when TDA is unavailable"""
        correlation_id = "fallback-test"
        
        fallback_context = tda_integration._create_fallback_context(correlation_id)
        
        assert fallback_context.correlation_id == correlation_id
        assert fallback_context.pattern_confidence == 0.5
        assert fallback_context.anomaly_severity == 0.0
        assert fallback_context.metadata["source"] == "fallback"
        assert fallback_context.metadata["tda_available"] is False

class TestOrchestrationResultSending:
    """Test sending orchestration results to TDA"""
    
    @pytest.mark.asyncio
    async def test_successful_result_sending(self, tda_integration):
        """Test successful result sending to TDA"""
        correlation_id = "result-test-123"
        result = {
            "workflow_id": "test-workflow",
            "agent_outputs": {"agent1": "result1", "agent2": "result2"},
            "execution_summary": {"execution_time": 150},
            "success": True
        }
        
        with patch.object(tda_integration, 'kafka_client') as mock_kafka:
            mock_kafka.send_message = AsyncMock(return_value=True)
            
            success = await tda_integration.send_orchestration_result(result, correlation_id)
            
            assert success is True
            mock_kafka.send_message.assert_called_once()
            
            # Check message structure
            call_args = mock_kafka.send_message.call_args
            assert call_args[1]["topic"] == "orchestration-results"
            assert call_args[1]["key"] == correlation_id
    
    @pytest.mark.asyncio
    async def test_result_sending_failure(self, tda_integration):
        """Test result sending failure handling"""
        correlation_id = "failing-result"
        result = {"test": "data"}
        
        with patch.object(tda_integration, 'kafka_client') as mock_kafka:
            mock_kafka.send_message = AsyncMock(side_effect=Exception("Kafka error"))
            
            success = await tda_integration.send_orchestration_result(result, correlation_id)
            
            assert success is False
    
    @pytest.mark.asyncio
    async def test_result_sending_without_kafka(self, tda_integration):
        """Test result sending when Kafka is unavailable"""
        correlation_id = "no-kafka-test"
        result = {"test": "data"}
        
        # Set kafka_client to None
        tda_integration.kafka_client = None
        
        success = await tda_integration.send_orchestration_result(result, correlation_id)
        
        assert success is False

class TestPatternRetrieval:
    """Test TDA pattern retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_current_patterns_retrieval(self, tda_integration):
        """Test retrieval of current TDA patterns"""
        window = "30m"
        
        with patch('aura_intelligence.orchestration.semantic.tda_integration.get_current_patterns') as mock_patterns:
            expected_patterns = {
                "patterns": {"pattern1": 0.8, "pattern2": 0.6},
                "anomalies": {"severity": 0.3},
                "window": window
            }
            mock_patterns.return_value = expected_patterns
            
            patterns = await tda_integration.get_current_patterns(window)
            
            assert patterns == expected_patterns
            mock_patterns.assert_called_once_with(window=window)
    
    @pytest.mark.asyncio
    async def test_patterns_caching(self, tda_integration):
        """Test that patterns are cached correctly"""
        window = "1h"
        
        with patch('aura_intelligence.orchestration.semantic.tda_integration.get_current_patterns') as mock_patterns:
            expected_patterns = {"patterns": {"cached_pattern": 0.9}}
            mock_patterns.return_value = expected_patterns
            
            # First call should fetch from TDA
            patterns1 = await tda_integration.get_current_patterns(window)
            
            # Second call should use cache
            patterns2 = await tda_integration.get_current_patterns(window)
            
            assert patterns1 == patterns2
            # Should only call TDA once due to caching
            mock_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_patterns_cache_expiration(self, tda_integration):
        """Test that pattern cache expires correctly"""
        window = "15m"
        
        # Manually set expired cache
        expired_time = datetime.utcnow() - timedelta(minutes=10)
        tda_integration.pattern_cache[f"patterns_{window}"] = (expired_time, {"old": "data"})
        
        with patch('aura_intelligence.orchestration.semantic.tda_integration.get_current_patterns') as mock_patterns:
            fresh_patterns = {"patterns": {"fresh_pattern": 0.7}}
            mock_patterns.return_value = fresh_patterns
            
            patterns = await tda_integration.get_current_patterns(window)
            
            assert patterns == fresh_patterns
            mock_patterns.assert_called_once()

class TestAnomalyCorrelation:
    """Test anomaly correlation functionality"""
    
    @pytest.mark.asyncio
    async def test_anomaly_correlation_with_time_match(self, tda_integration):
        """Test correlation when orchestration and anomaly times match"""
        orchestration_time = datetime.utcnow()
        orchestration_data = {
            "timestamp": orchestration_time.isoformat(),
            "workflow_id": "test-workflow"
        }
        
        # Mock anomaly data with close timestamp
        anomaly_time = orchestration_time - timedelta(minutes=2)
        mock_anomalies = {
            "anomalies": [
                {
                    "timestamp": anomaly_time.isoformat(),
                    "severity": 0.8,
                    "type": "spike"
                }
            ]
        }
        
        with patch.object(tda_integration, 'get_current_patterns') as mock_patterns:
            mock_patterns.return_value = {"anomalies": mock_anomalies}
            
            correlation = await tda_integration.correlate_with_anomalies(orchestration_data)
            
            assert correlation["correlation_strength"] > 0.5  # Should be high due to time proximity
            assert len(correlation["anomalies"]) == 1
    
    @pytest.mark.asyncio
    async def test_anomaly_correlation_with_time_mismatch(self, tda_integration):
        """Test correlation when times don't match well"""
        orchestration_time = datetime.utcnow()
        orchestration_data = {
            "timestamp": orchestration_time.isoformat(),
            "workflow_id": "test-workflow"
        }
        
        # Mock anomaly data with distant timestamp
        anomaly_time = orchestration_time - timedelta(hours=2)
        mock_anomalies = {
            "anomalies": [
                {
                    "timestamp": anomaly_time.isoformat(),
                    "severity": 0.8,
                    "type": "spike"
                }
            ]
        }
        
        with patch.object(tda_integration, 'get_current_patterns') as mock_patterns:
            mock_patterns.return_value = {"anomalies": mock_anomalies}
            
            correlation = await tda_integration.correlate_with_anomalies(orchestration_data)
            
            assert correlation["correlation_strength"] < 0.3  # Should be low due to time distance

class TestCacheManagement:
    """Test cache management functionality"""
    
    def test_cache_cleanup_on_overflow(self, tda_integration):
        """Test that cache is cleaned up when it overflows"""
        # Fill cache beyond limit
        for i in range(1100):  # Beyond 1000 limit
            context = TDAContext(
                correlation_id=f"test-{i}",
                pattern_confidence=0.5,
                anomaly_severity=0.0,
                current_patterns={},
                temporal_window="1h",
                metadata={}
            )
            tda_integration._cache_context(f"test-{i}", context)
        
        # Cache should be cleaned up to reasonable size
        assert len(tda_integration.context_cache) <= 1000
    
    def test_cache_hit_ratio_calculation(self, tda_integration):
        """Test cache hit ratio calculation"""
        # Add some cache entries
        for i in range(5):
            context = TDAContext(
                correlation_id=f"test-{i}",
                pattern_confidence=0.5,
                anomaly_severity=0.0,
                current_patterns={},
                temporal_window="1h",
                metadata={}
            )
            tda_integration._cache_context(f"test-{i}", context)
        
        hit_ratio = tda_integration._calculate_cache_hit_ratio()
        assert 0.0 <= hit_ratio <= 1.0

class TestHealthCheck:
    """Test health check functionality"""
    
    @pytest.mark.asyncio
    async def test_healthy_status(self, tda_integration):
        """Test health check when everything is working"""
        with patch.object(tda_integration, 'kafka_client') as mock_kafka:
            mock_kafka.__bool__ = Mock(return_value=True)
            
            health = await tda_integration.health_check()
            
            assert health["status"] == "healthy"
            assert health["cache_operational"] is True
    
    @pytest.mark.asyncio
    async def test_degraded_status_with_cache_overflow(self, tda_integration):
        """Test degraded status when cache is overflowing"""
        # Simulate cache overflow
        for i in range(15000):  # Way beyond healthy limit
            tda_integration.context_cache[f"test-{i}"] = (None, datetime.utcnow())
        
        health = await tda_integration.health_check()
        
        assert health["status"] == "degraded"
        assert health["cache_operational"] is False

class TestMockTDAIntegration:
    """Test mock TDA integration for development/testing"""
    
    @pytest.mark.asyncio
    async def test_mock_context_retrieval(self, mock_tda_integration):
        """Test mock context retrieval"""
        correlation_id = "mock-test-123"
        
        context = await mock_tda_integration.get_context(correlation_id)
        
        assert context is not None
        assert context.correlation_id == correlation_id
        assert context.pattern_confidence == 0.7
        assert context.metadata["source"] == "mock"
    
    @pytest.mark.asyncio
    async def test_mock_result_sending(self, mock_tda_integration):
        """Test mock result sending"""
        result = {"test": "result"}
        correlation_id = "mock-result-test"
        
        success = await mock_tda_integration.send_orchestration_result(result, correlation_id)
        
        assert success is True
        assert len(mock_tda_integration.results) == 1
        assert mock_tda_integration.results[0] == (result, correlation_id)
    
    @pytest.mark.asyncio
    async def test_mock_pattern_retrieval(self, mock_tda_integration):
        """Test mock pattern retrieval"""
        patterns_1h = await mock_tda_integration.get_current_patterns("1h")
        patterns_30m = await mock_tda_integration.get_current_patterns("30m")
        
        assert "patterns" in patterns_1h
        assert "patterns" in patterns_30m
        assert patterns_1h != patterns_30m  # Different windows should return different data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])