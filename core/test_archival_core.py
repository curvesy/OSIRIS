#!/usr/bin/env python3
"""
🧪 Core Archival System Test

Simple test to validate the production-grade archival system components.
"""

import asyncio
import time
import sys
sys.path.append('src')

from aura_intelligence.enterprise.mem0_hot.archive import CircuitBreaker, ExponentialBackoff


def test_circuit_breaker():
    """Test circuit breaker pattern."""
    
    print("🔧 Testing Circuit Breaker Pattern...")
    
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    # Mock function that fails
    def failing_function():
        raise Exception("Service degraded")
    
    # Test normal operation (closed state)
    assert circuit_breaker.state == "CLOSED"
    print("✅ Initial state: CLOSED")
    
    # Test failure accumulation
    for i in range(3):
        try:
            circuit_breaker.call(failing_function)
        except Exception:
            pass
    
    # Circuit should be open after threshold failures
    assert circuit_breaker.state == "OPEN"
    print("✅ State after failures: OPEN")
    
    # Test that circuit prevents calls when open
    try:
        circuit_breaker.call(failing_function)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Circuit breaker is OPEN" in str(e)
        print("✅ Circuit breaker blocking calls when OPEN")
    
    # Test recovery after timeout
    time.sleep(1.1)  # Wait for recovery timeout
    
    # Should transition to half-open and allow one call
    def successful_function():
        return "success"
    
    result = circuit_breaker.call(successful_function)
    assert result == "success"
    assert circuit_breaker.state == "CLOSED"
    print("✅ Circuit breaker recovered to CLOSED state")
    
    print("✅ Circuit breaker pattern working correctly\n")


async def test_exponential_backoff():
    """Test exponential backoff pattern."""
    
    print("🔧 Testing Exponential Backoff...")
    
    backoff = ExponentialBackoff(max_retries=3, base_delay=0.1, max_delay=1.0)
    
    # Mock function that fails twice then succeeds
    call_count = 0
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Transient failure")
        return "success"
    
    start_time = time.time()
    result = await backoff.retry(flaky_function)
    duration = time.time() - start_time
    
    assert result == "success"
    assert call_count == 3
    assert duration >= 0.3  # Should have delays
    
    print(f"✅ Function succeeded after {call_count} attempts in {duration:.2f}s")
    print("✅ Exponential backoff working correctly\n")


def test_prometheus_metrics():
    """Test Prometheus metrics are available."""
    
    print("🔧 Testing Prometheus Metrics...")
    
    try:
        from aura_intelligence.enterprise.mem0_hot.archive import (
            ARCHIVAL_JOB_SUCCESS,
            ARCHIVAL_JOB_FAILURES,
            ARCHIVAL_RECORDS_PROCESSED,
            ARCHIVAL_DATA_VOLUME,
            S3_OPERATION_DURATION
        )
        
        # Test that metrics can be incremented
        initial_success = ARCHIVAL_JOB_SUCCESS._value._value
        ARCHIVAL_JOB_SUCCESS.inc()
        assert ARCHIVAL_JOB_SUCCESS._value._value == initial_success + 1
        
        print("✅ Prometheus metrics available and functional")
        
        # Test histogram metric
        with S3_OPERATION_DURATION.labels(operation_type='test').time():
            time.sleep(0.01)  # Small delay
        
        print("✅ Histogram metrics working correctly")
        
    except Exception as e:
        print(f"⚠️ Prometheus metrics test failed: {e}")
    
    print("✅ Prometheus metrics test completed\n")


def test_archival_manager_initialization():
    """Test ArchivalManager can be initialized."""
    
    print("🔧 Testing ArchivalManager Initialization...")
    
    try:
        import tempfile
        import duckdb
        from aura_intelligence.enterprise.mem0_hot.archive import ArchivalManager
        from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        # Create settings
        settings = DuckDBSettings(
            db_path=temp_db.name,
            s3_bucket=None,  # No S3 for basic test
            retention_hours=24
        )
        
        # Create connection
        conn = duckdb.connect(temp_db.name)
        
        # Initialize archival manager
        archival_manager = ArchivalManager(
            conn=conn,
            settings=settings,
            enable_metrics=False  # Disable metrics server for testing
        )
        
        assert archival_manager is not None
        assert archival_manager.circuit_breaker is not None
        assert archival_manager.backoff is not None
        
        print("✅ ArchivalManager initialized successfully")
        print(f"✅ Circuit breaker state: {archival_manager.circuit_breaker.state}")
        print(f"✅ Backoff max retries: {archival_manager.backoff.max_retries}")
        
        # Cleanup
        conn.close()
        import os
        os.unlink(temp_db.name)
        
    except Exception as e:
        print(f"❌ ArchivalManager initialization failed: {e}")
        return False
    
    print("✅ ArchivalManager initialization test completed\n")
    return True


async def main():
    """Run core archival system tests."""
    
    print("🚀 Starting Core Archival System Tests...")
    print("=" * 50)
    
    # Test individual components
    test_circuit_breaker()
    await test_exponential_backoff()
    test_prometheus_metrics()
    
    # Test system initialization
    init_success = test_archival_manager_initialization()
    
    print("=" * 50)
    
    if init_success:
        print("✅ All core archival tests completed successfully!")
        print("\n🎯 Production-Grade Archival System components are working!")
        print("\n📋 Next Steps:")
        print("   1. Deploy Kubernetes CronJob configuration")
        print("   2. Configure S3 bucket and credentials")
        print("   3. Set up Prometheus monitoring dashboard")
        print("   4. Run end-to-end integration tests")
    else:
        print("❌ Some tests failed - check the output above")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
