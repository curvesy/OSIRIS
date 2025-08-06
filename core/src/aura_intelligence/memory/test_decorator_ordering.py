"""
Test Decorator Ordering for Resilience Patterns
==============================================

Verifies that circuit breaker and retry decorators work correctly together.
"""

import time
import unittest
from unittest.mock import Mock, patch, call
from pybreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from redis.exceptions import ConnectionError, TimeoutError


class TestDecoratorOrdering(unittest.TestCase):
    """Test that decorators are applied in the correct order."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.call_count = 0
        self.breaker = CircuitBreaker(
            fail_max=2,
            reset_timeout=1,
            exclude=[KeyError]  # Don't break on logical errors
        )
    
    def test_retry_inside_breaker(self):
        """Test that retry logic works inside circuit breaker."""
        # This simulates our corrected decorator order:
        # @breaker (outer)
        # @retry (inner)
        # def method():
        
        @self.breaker
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.1),
            retry=retry_if_exception_type(ConnectionError)
        )
        def flaky_operation():
            self.call_count += 1
            if self.call_count < 3:
                raise ConnectionError("Transient error")
            return "success"
        
        # First call: should retry 2 times then succeed
        result = flaky_operation()
        self.assertEqual(result, "success")
        self.assertEqual(self.call_count, 3)
        
        # Breaker should still be closed (only saw 1 "failure" from its perspective)
        self.assertEqual(self.breaker.current_state, "closed")
    
    def test_breaker_opens_after_retry_exhaustion(self):
        """Test that breaker opens when retries are exhausted."""
        
        @self.breaker
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01),
            retry=retry_if_exception_type(ConnectionError)
        )
        def always_fails():
            raise ConnectionError("Permanent error")
        
        # First call: retries 3 times, then breaker sees 1 failure
        with self.assertRaises(RetryError):
            always_fails()
        
        # Second call: retries 3 times, then breaker sees 2nd failure and opens
        with self.assertRaises(RetryError):
            always_fails()
        
        # Breaker should now be open
        self.assertEqual(self.breaker.current_state, "open")
        
        # Third call: breaker is open, so it fails immediately without retry
        from pybreaker import CircuitBreakerError
        with self.assertRaises(CircuitBreakerError):
            always_fails()
    
    def test_wrong_order_bypasses_retry(self):
        """Demonstrate the bug when decorators are in wrong order."""
        # This simulates the WRONG order:
        # @retry (outer)
        # @breaker (inner)
        # def method():
        
        call_count = 0
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01),
            retry=retry_if_exception_type(ConnectionError)
        )
        @self.breaker
        def wrong_order_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Error")
        
        # The breaker will open after 2 calls
        with self.assertRaises(RetryError):
            wrong_order_operation()
        
        # With wrong order, retry sees CircuitBreakerError (not ConnectionError)
        # so it doesn't retry properly
        # This demonstrates why order matters!
        self.assertGreaterEqual(call_count, 2)  # At least 2 attempts before breaker opens


class TestRedisStoreDecoratorIntegration(unittest.TestCase):
    """Test the actual Redis store decorator implementation."""
    
    @patch('redis.Redis')
    def test_redis_store_handles_transient_errors(self, mock_redis):
        """Test that Redis store retries transient errors correctly."""
        from circle.core.src.aura_intelligence.memory.redis_store import (
            RedisVectorStore, RedisConfig
        )
        
        # Set up mock to fail twice then succeed
        mock_instance = Mock()
        mock_redis.return_value = mock_instance
        
        # Create a mock that fails twice then succeeds
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient Redis error")
            return True
        
        # Mock the pipeline execution
        mock_pipeline = Mock()
        mock_pipeline.execute.side_effect = side_effect
        mock_instance.pipeline.return_value = mock_pipeline
        
        # Create store (this will try to create index, which we'll mock)
        mock_instance.ft.return_value.info.side_effect = Exception("No index")
        mock_instance.ft.return_value.create_index.return_value = None
        
        config = RedisConfig(url="redis://localhost:6379")
        store = RedisVectorStore(config)
        
        # Reset call count for our test
        call_count = 0
        
        # This should retry the transient errors and eventually succeed
        import numpy as np
        result = store.add(
            memory_id="test-123",
            embedding=np.array([1.0] * 128),
            content={"test": "data"},
            context_type="test"
        )
        
        # Should have succeeded after retries
        self.assertTrue(result)
        self.assertEqual(call_count, 3)  # Failed twice, succeeded on third


def retry_if_exception_type(exc_type):
    """Helper to create retry predicate."""
    def predicate(retry_state):
        return retry_state.outcome.failed and \
               isinstance(retry_state.outcome.exception(), exc_type)
    return predicate


if __name__ == '__main__':
    unittest.main()