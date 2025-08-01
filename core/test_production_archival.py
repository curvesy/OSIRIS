#!/usr/bin/env python3
"""
🧪 Production-Grade Archival System Test Suite

Comprehensive testing for the enhanced archival system with:
- Circuit breaker pattern validation
- Exponential backoff testing
- Prometheus metrics verification
- Kubernetes CronJob simulation
- S3 resilience testing
- End-to-end archival pipeline validation
"""

import asyncio
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import pandas as pd
import duckdb

# Import the production archival system
import sys
sys.path.append('src')

try:
    from aura_intelligence.enterprise.mem0_hot.archive import (
        ArchivalManager,
        CircuitBreaker,
        ExponentialBackoff,
        ARCHIVAL_JOB_SUCCESS,
        ARCHIVAL_JOB_FAILURES,
        ARCHIVAL_RECORDS_PROCESSED
    )
    from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings
    from aura_intelligence.enterprise.mem0_hot.schema import create_tables
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_AVAILABLE = False

# Optional imports for full testing
try:
    import boto3
    from moto import mock_s3
    S3_TESTING_AVAILABLE = True
except ImportError:
    S3_TESTING_AVAILABLE = False
    print("⚠️ S3 testing dependencies not available - skipping S3 tests")


class TestProductionArchivalSystem:
    """Production-grade archival system test suite."""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Set up comprehensive test environment."""
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_hot_memory.db"
        
        # Initialize DuckDB with test data
        self.conn = duckdb.connect(str(self.db_path))
        await create_tables(self.conn)
        
        # Create test settings
        self.settings = DuckDBSettings(
            db_path=str(self.db_path),
            s3_bucket="test-aura-intelligence-archive",
            retention_hours=24
        )
        
        # Insert test data with various timestamps
        test_data = []
        base_time = datetime.now()
        
        for i in range(100):
            # Create data spanning 48 hours (some old, some recent)
            timestamp = base_time - timedelta(hours=48 - i * 0.5)
            test_data.append({
                'signature_hash': f'test_hash_{i:03d}',
                'betti_0': i % 5,
                'betti_1': (i * 2) % 3,
                'betti_2': (i * 3) % 2,
                'anomaly_score': 0.1 + (i % 10) * 0.05,
                'timestamp': timestamp,
                'agent_id': f'agent_{i % 5}',
                'event_type': f'event_type_{i % 3}',
                'archived': False
            })
        
        # Insert test data
        df = pd.DataFrame(test_data)
        self.conn.execute("INSERT INTO topological_signatures SELECT * FROM df")
        
        print(f"✅ Test environment setup complete with {len(test_data)} records")
        
        yield
        
        # Cleanup
        self.conn.close()
        shutil.rmtree(self.temp_dir)
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for S3 resilience."""
        
        print("\n🔧 Testing Circuit Breaker Pattern...")
        
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Mock function that fails
        def failing_function():
            raise Exception("S3 service degraded")
        
        # Test normal operation (closed state)
        assert circuit_breaker.state == "CLOSED"
        
        # Test failure accumulation
        for i in range(3):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        # Circuit should be open after threshold failures
        assert circuit_breaker.state == "OPEN"
        
        # Test that circuit prevents calls when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            circuit_breaker.call(failing_function)
        
        # Test recovery after timeout
        time.sleep(1.1)  # Wait for recovery timeout
        
        # Should transition to half-open and allow one call
        def successful_function():
            return "success"
        
        result = circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        
        print("✅ Circuit breaker pattern working correctly")
    
    async def test_exponential_backoff(self):
        """Test exponential backoff for transient failures."""
        
        print("\n🔧 Testing Exponential Backoff...")
        
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
        
        print("✅ Exponential backoff working correctly")
    
    @mock_s3
    async def test_production_archival_pipeline(self, setup_test_environment):
        """Test complete production archival pipeline."""
        
        print("\n🔧 Testing Production Archival Pipeline...")
        
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket=self.settings.s3_bucket)
        
        # Initialize archival manager
        archival_manager = ArchivalManager(
            conn=self.conn,
            settings=self.settings,
            enable_metrics=False  # Disable metrics server for testing
        )
        
        # Mock S3 client to use moto
        archival_manager.s3_client = s3_client
        
        # Run archival process
        start_time = time.time()
        result = await archival_manager.archive_old_data()
        duration = time.time() - start_time
        
        # Verify results
        assert result['status'] == 'success'
        assert result['records_archived'] > 0
        assert result['partitions_archived'] > 0
        assert duration < 30  # Should complete within 30 seconds
        
        # Verify data was archived in database
        archived_count = self.conn.execute(
            "SELECT COUNT(*) FROM topological_signatures WHERE archived = true"
        ).fetchone()[0]
        
        assert archived_count == result['records_archived']
        
        # Verify S3 exports
        s3_objects = s3_client.list_objects_v2(Bucket=self.settings.s3_bucket)
        assert 'Contents' in s3_objects
        assert len(s3_objects['Contents']) == result['partitions_archived']
        
        print(f"✅ Production archival pipeline completed successfully:")
        print(f"   - Records archived: {result['records_archived']}")
        print(f"   - Partitions created: {result['partitions_archived']}")
        print(f"   - Duration: {duration:.2f}s")
        print(f"   - S3 objects created: {len(s3_objects['Contents'])}")
    
    async def test_partition_processing(self, setup_test_environment):
        """Test Hive-style partition processing."""
        
        print("\n🔧 Testing Partition Processing...")
        
        archival_manager = ArchivalManager(
            conn=self.conn,
            settings=self.settings,
            enable_metrics=False
        )
        
        # Get archival data with partitions
        cutoff_time = datetime.now() - timedelta(hours=24)
        archive_data = await archival_manager._get_archival_data_with_partitions(cutoff_time)
        
        assert not archive_data.empty
        assert 'partition_year' in archive_data.columns
        assert 'partition_month' in archive_data.columns
        assert 'partition_day' in archive_data.columns
        assert 'partition_hour' in archive_data.columns
        
        # Test partition grouping
        partitions = archival_manager._group_data_by_partitions(archive_data)
        
        assert len(partitions) > 0
        
        # Verify partition key format (Hive-style)
        for partition_key in partitions.keys():
            assert partition_key.startswith('year=')
            assert '/month=' in partition_key
            assert '/day=' in partition_key
            assert '/hour=' in partition_key
        
        print(f"✅ Partition processing working correctly:")
        print(f"   - Records to archive: {len(archive_data)}")
        print(f"   - Partitions created: {len(partitions)}")
        print(f"   - Sample partition key: {list(partitions.keys())[0]}")
    
    @mock_s3
    async def test_s3_resilience(self, setup_test_environment):
        """Test S3 resilience with circuit breaker."""
        
        print("\n🔧 Testing S3 Resilience...")
        
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket=self.settings.s3_bucket)
        
        archival_manager = ArchivalManager(
            conn=self.conn,
            settings=self.settings,
            enable_metrics=False
        )
        
        # Mock S3 client with intermittent failures
        original_put_object = s3_client.put_object
        call_count = 0
        
        def flaky_put_object(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 calls
                raise Exception("S3 service temporarily unavailable")
            return original_put_object(*args, **kwargs)
        
        s3_client.put_object = flaky_put_object
        archival_manager.s3_client = s3_client
        
        # Test that archival still succeeds with retries
        result = await archival_manager.archive_old_data()
        
        # Should succeed despite initial failures
        assert result['status'] == 'success'
        assert call_count > 2  # Should have retried
        
        print("✅ S3 resilience testing completed successfully")
    
    async def test_prometheus_metrics(self, setup_test_environment):
        """Test Prometheus metrics collection."""
        
        print("\n🔧 Testing Prometheus Metrics...")
        
        # Reset metrics
        ARCHIVAL_JOB_SUCCESS._value._value = 0
        ARCHIVAL_JOB_FAILURES._value._value = 0
        ARCHIVAL_RECORDS_PROCESSED._value._value = 0
        
        archival_manager = ArchivalManager(
            conn=self.conn,
            settings=self.settings,
            enable_metrics=False
        )
        
        # Mock successful archival
        with patch.object(archival_manager, '_archive_partition_with_resilience') as mock_archive:
            mock_archive.return_value = (True, 1024, 's3://test/key')
            
            result = await archival_manager.archive_old_data()
        
        # Verify metrics were updated
        assert ARCHIVAL_JOB_SUCCESS._value._value > 0
        assert ARCHIVAL_RECORDS_PROCESSED._value._value > 0
        
        print("✅ Prometheus metrics collection working correctly")


async def main():
    """Run comprehensive production archival tests."""
    
    print("🚀 Starting Production-Grade Archival System Tests...")
    print("=" * 60)
    
    test_suite = TestProductionArchivalSystem()
    
    # Run individual tests
    test_suite.test_circuit_breaker_pattern()
    await test_suite.test_exponential_backoff()
    
    # Run integration tests with setup
    async with test_suite.setup_test_environment():
        await test_suite.test_production_archival_pipeline()
        await test_suite.test_partition_processing()
        await test_suite.test_s3_resilience()
        await test_suite.test_prometheus_metrics()
    
    print("\n" + "=" * 60)
    print("✅ All production archival tests completed successfully!")
    print("\n🎯 Production-Grade Archival System is ready for deployment!")


if __name__ == "__main__":
    asyncio.run(main())
