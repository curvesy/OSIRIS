"""
Tests for AURA Intelligence decorators.

Tests circuit breaker, retry, rate limiting, and other decorators.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from aura_intelligence.utils.decorators import (
    circuit_breaker,
    retry,
    rate_limit,
    timeout,
    log_performance,
    handle_errors,
    timer,
    CircuitState,
)


class TestCircuitBreaker:
    """Test circuit breaker decorator."""
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=3, recovery_timeout=1)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        # Should work normally
        assert test_func() == "success"
        assert call_count == 1
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=3, recovery_timeout=1)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Test error")
        
        # First 3 calls should fail normally
        for i in range(3):
            with pytest.raises(Exception, match="Test error"):
                test_func()
        assert call_count == 3
        
        # Circuit should now be open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            test_func()
        assert call_count == 3  # No additional call
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        fail_count = 0
        
        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1, success_threshold=1)
        def test_func():
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                raise Exception("Test error")
            return "success"
        
        # Trigger circuit breaker
        for _ in range(2):
            with pytest.raises(Exception):
                test_func()
        
        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            test_func()
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should enter half-open state and succeed
        assert test_func() == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self):
        """Test circuit breaker with async functions."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=2)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test error")
            return "success"
        
        # Trigger failures
        for _ in range(2):
            with pytest.raises(Exception, match="Test error"):
                await test_func()
        
        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await test_func()


class TestRetry:
    """Test retry decorator."""
    
    def test_retry_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        call_count = 0
        
        @retry(max_attempts=3)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        assert test_func() == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test retry succeeds after initial failures."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test error")
            return "success"
        
        assert test_func() == "success"
        assert call_count == 3
    
    def test_retry_max_attempts_exceeded(self):
        """Test retry fails after max attempts."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            test_func()
        assert call_count == 3
    
    def test_retry_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        call_count = 0
        
        @retry(max_attempts=3, exceptions=(ValueError,))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            raise TypeError("Don't retry this")
        
        with pytest.raises(TypeError):
            test_func()
        assert call_count == 2  # One retry for ValueError, then TypeError
    
    @pytest.mark.asyncio
    async def test_retry_async(self):
        """Test retry with async functions."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Test error")
            return "success"
        
        assert await test_func() == "success"
        assert call_count == 2


class TestRateLimit:
    """Test rate limit decorator."""
    
    def test_rate_limit_within_limit(self):
        """Test rate limit allows calls within limit."""
        call_count = 0
        
        @rate_limit(calls=3, period=1.0)
        def test_func():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Should allow 3 calls
        for i in range(3):
            assert test_func() == i + 1
    
    def test_rate_limit_exceeded(self):
        """Test rate limit blocks calls over limit."""
        @rate_limit(calls=2, period=1.0)
        def test_func():
            return "success"
        
        # First 2 calls should succeed
        assert test_func() == "success"
        assert test_func() == "success"
        
        # Third call should fail
        with pytest.raises(Exception, match="Rate limit exceeded"):
            test_func()
    
    def test_rate_limit_reset_after_period(self):
        """Test rate limit resets after period."""
        call_count = 0
        
        @rate_limit(calls=2, period=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Use up the limit
        assert test_func() == 1
        assert test_func() == 2
        
        # Should be blocked
        with pytest.raises(Exception, match="Rate limit exceeded"):
            test_func()
        
        # Wait for period to reset
        time.sleep(0.15)
        
        # Should work again
        assert test_func() == 3
    
    @pytest.mark.asyncio
    async def test_rate_limit_async(self):
        """Test rate limit with async functions."""
        @rate_limit(calls=1, period=1.0)
        async def test_func():
            return "success"
        
        assert await test_func() == "success"
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await test_func()


class TestTimeout:
    """Test timeout decorator."""
    
    @pytest.mark.asyncio
    async def test_timeout_success(self):
        """Test timeout with successful completion."""
        @timeout(1.0)
        async def test_func():
            await asyncio.sleep(0.1)
            return "success"
        
        assert await test_func() == "success"
    
    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test timeout when time limit exceeded."""
        @timeout(0.1)
        async def test_func():
            await asyncio.sleep(0.5)
            return "success"
        
        with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
            await test_func()
    
    def test_timeout_sync_function_error(self):
        """Test timeout raises error on sync functions."""
        with pytest.raises(TypeError, match="can only be used on async functions"):
            @timeout(1.0)
            def test_func():
                return "success"


class TestLogPerformance:
    """Test log performance decorator."""
    
    def test_log_performance_fast(self, caplog):
        """Test performance logging for fast functions."""
        @log_performance(threshold_ms=100)
        def test_func():
            time.sleep(0.01)
            return "success"
        
        with caplog.at_level("DEBUG"):
            assert test_func() == "success"
        
        # Should log at DEBUG level (not WARNING)
        assert len([r for r in caplog.records if r.levelname == "DEBUG"]) > 0
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 0
    
    def test_log_performance_slow(self, caplog):
        """Test performance logging for slow functions."""
        @log_performance(threshold_ms=10)
        def test_func():
            time.sleep(0.02)
            return "success"
        
        with caplog.at_level("WARNING"):
            assert test_func() == "success"
        
        # Should log at WARNING level
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) > 0
        assert "took" in caplog.records[-1].message
    
    @pytest.mark.asyncio
    async def test_log_performance_async(self, caplog):
        """Test performance logging for async functions."""
        @log_performance(threshold_ms=50)
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"
        
        with caplog.at_level("DEBUG"):
            assert await test_func() == "success"
        
        assert any("test_func took" in r.message for r in caplog.records)


class TestHandleErrors:
    """Test handle errors decorator."""
    
    def test_handle_errors_success(self):
        """Test handle errors with successful function."""
        @handle_errors(default_return="default")
        def test_func():
            return "success"
        
        assert test_func() == "success"
    
    def test_handle_errors_with_exception(self, caplog):
        """Test handle errors catches exceptions."""
        @handle_errors(default_return="default", log_errors=True)
        def test_func():
            raise ValueError("Test error")
        
        with caplog.at_level("ERROR"):
            assert test_func() == "default"
        
        assert len(caplog.records) > 0
        assert "ValueError: Test error" in caplog.records[-1].message
    
    def test_handle_errors_reraise(self):
        """Test handle errors can reraise exceptions."""
        @handle_errors(default_return="default", reraise=True)
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            test_func()
    
    def test_handle_errors_no_logging(self, caplog):
        """Test handle errors without logging."""
        @handle_errors(default_return=None, log_errors=False)
        def test_func():
            raise ValueError("Test error")
        
        with caplog.at_level("ERROR"):
            assert test_func() is None
        
        assert len(caplog.records) == 0
    
    @pytest.mark.asyncio
    async def test_handle_errors_async(self):
        """Test handle errors with async functions."""
        @handle_errors(default_return=[])
        async def test_func():
            raise Exception("Test error")
        
        assert await test_func() == []


class TestTimer:
    """Test timer context manager."""
    
    def test_timer_basic(self, caplog):
        """Test basic timer functionality."""
        with caplog.at_level("INFO"):
            with timer("Test operation"):
                time.sleep(0.01)
        
        assert len(caplog.records) > 0
        assert "Test operation completed in" in caplog.records[-1].message
    
    def test_timer_custom_logger(self):
        """Test timer with custom logger function."""
        logged_messages = []
        
        def custom_logger(msg):
            logged_messages.append(msg)
        
        with timer("Custom operation", logger_func=custom_logger):
            time.sleep(0.01)
        
        assert len(logged_messages) == 1
        assert "Custom operation completed in" in logged_messages[0]
    
    def test_timer_with_exception(self, caplog):
        """Test timer logs even with exception."""
        with caplog.at_level("INFO"):
            try:
                with timer("Error operation"):
                    time.sleep(0.01)
                    raise ValueError("Test error")
            except ValueError:
                pass
        
        # Timer should still log completion
        assert any("Error operation completed in" in r.message for r in caplog.records)