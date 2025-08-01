"""
🧪 Pytest Configuration & Fixtures - Production-Grade Testing

Comprehensive test fixtures for end-to-end pipeline validation:
- DuckDB hot memory testing with real schemas
- Redis vector search with proper indices
- MinIO S3-compatible storage testing
- Async test support with proper event loops
- Data generation utilities for realistic testing
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, AsyncGenerator, Generator
import json

# Database and storage imports
import duckdb
import redis
from minio import Minio
from minio.error import S3Error
import boto3
from botocore.exceptions import ClientError

# Data processing imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Testing utilities
from faker import Faker
import factory
from factory import fuzzy

# Logging
import logging
from loguru import logger


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load/performance tests"
    )
    config.addinivalue_line(
        "markers", "quality: marks tests as data quality tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger.add("test-results/test.log", rotation="10 MB")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark load tests
        if "load" in str(item.fspath):
            item.add_marker(pytest.mark.load)
        
        # Mark quality tests
        if "quality" in str(item.fspath):
            item.add_marker(pytest.mark.quality)


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
async def test_db() -> AsyncGenerator[duckdb.DuckDBPyConnection, None]:
    """In-memory DuckDB for hot memory testing."""
    
    conn = duckdb.connect(':memory:')
    
    try:
        # Install and load VSS extension for vector similarity
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        # Create hot tier schema matching production
        conn.execute("""
            CREATE TABLE recent_activity (
                id UUID DEFAULT gen_random_uuid(),
                timestamp TIMESTAMP NOT NULL,
                signature BLOB NOT NULL,
                metadata JSON,
                partition_hour INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(timestamp),
                INDEX(partition_hour)
            )
        """)
        
        # Create vector similarity index
        conn.execute("""
            CREATE INDEX idx_signature_similarity 
            ON recent_activity 
            USING vss (signature)
        """)
        
        logger.info("✅ Test DuckDB initialized with VSS extension")
        yield conn
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize test DuckDB: {e}")
        # Fallback without VSS if not available
        conn.execute("""
            CREATE TABLE recent_activity (
                id UUID DEFAULT gen_random_uuid(),
                timestamp TIMESTAMP NOT NULL,
                signature BLOB NOT NULL,
                metadata JSON,
                partition_hour INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        yield conn
    
    finally:
        conn.close()


@pytest.fixture
async def test_redis() -> AsyncGenerator[redis.Redis, None]:
    """Redis with vector search for semantic memory testing."""
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    r = redis.from_url(redis_url, decode_responses=True)
    
    try:
        # Test connection
        r.ping()
        
        # Create vector index for semantic search
        try:
            r.execute_command(
                'FT.CREATE', 'semantic_idx', 
                'ON', 'HASH', 
                'PREFIX', '1', 'sig:',
                'SCHEMA', 
                'vector', 'VECTOR', 'HNSW', '6', 
                'TYPE', 'FLOAT32', 
                'DIM', '768', 
                'DISTANCE_METRIC', 'COSINE',
                'metadata', 'TEXT',
                'timestamp', 'NUMERIC', 'SORTABLE',
                'cluster_id', 'TAG'
            )
            logger.info("✅ Test Redis initialized with vector search index")
        except redis.ResponseError as e:
            if "Index already exists" not in str(e):
                logger.warning(f"⚠️ Redis index creation failed: {e}")
        
        yield r
        
    except redis.ConnectionError:
        logger.error("❌ Redis connection failed - using fake Redis")
        # Fallback to fake Redis for local testing
        import fakeredis
        fake_r = fakeredis.FakeRedis(decode_responses=True)
        yield fake_r
    
    finally:
        # Cleanup
        try:
            r.flushall()
        except:
            pass


@pytest.fixture
async def test_s3() -> AsyncGenerator[Minio, None]:
    """MinIO client for S3-compatible cold storage testing."""
    
    endpoint = os.getenv('S3_ENDPOINT', 'localhost:9000').replace('http://', '')
    access_key = os.getenv('S3_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('S3_SECRET_KEY', 'minioadmin')
    bucket_name = os.getenv('S3_BUCKET', 'test-forge')
    
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    try:
        # Create test bucket
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"✅ Created test bucket: {bucket_name}")
        
        yield client
        
    except Exception as e:
        logger.error(f"❌ MinIO connection failed: {e}")
        # Create a mock S3 client for local testing
        yield None
    
    finally:
        # Cleanup - remove all test objects
        try:
            objects = client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                client.remove_object(bucket_name, obj.object_name)
            logger.info(f"🧹 Cleaned up test bucket: {bucket_name}")
        except:
            pass


# ============================================================================
# Data Generation Fixtures
# ============================================================================

class SignatureFactory(factory.Factory):
    """Factory for generating realistic topological signatures."""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: f"test_sig_{n}")
    timestamp = factory.LazyFunction(
        lambda: datetime.now() - timedelta(
            hours=fuzzy.FuzzyInteger(0, 48).fuzz(),
            minutes=fuzzy.FuzzyInteger(0, 59).fuzz()
        )
    )
    signature = factory.LazyFunction(
        lambda: np.random.rand(768).astype(np.float32).tobytes()
    )
    metadata = factory.LazyFunction(
        lambda: {
            "source": fuzzy.FuzzyChoice(["agent_1", "agent_2", "system"]).fuzz(),
            "confidence": fuzzy.FuzzyFloat(0.1, 1.0).fuzz(),
            "betti_numbers": [
                fuzzy.FuzzyInteger(0, 10).fuzz(),
                fuzzy.FuzzyInteger(0, 5).fuzz(),
                fuzzy.FuzzyInteger(0, 2).fuzz()
            ],
            "persistence_diagram": {
                "birth": fuzzy.FuzzyFloat(0.0, 1.0).fuzz(),
                "death": fuzzy.FuzzyFloat(0.0, 1.0).fuzz()
            }
        }
    )


@pytest.fixture
def signature_factory():
    """Provide signature factory for test data generation."""
    return SignatureFactory


@pytest.fixture
async def sample_signatures(signature_factory) -> List[Dict[str, Any]]:
    """Generate sample signatures for testing."""
    
    signatures = []
    base_time = datetime.now() - timedelta(hours=48)
    
    for i in range(100):
        sig = signature_factory.build()
        sig['timestamp'] = base_time + timedelta(minutes=i * 10)
        sig['partition_hour'] = sig['timestamp'].hour
        signatures.append(sig)
    
    logger.info(f"📊 Generated {len(signatures)} sample signatures")
    return signatures


@pytest.fixture
async def large_dataset(signature_factory) -> List[Dict[str, Any]]:
    """Generate large dataset for performance testing."""
    
    signatures = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(10000):
        sig = signature_factory.build()
        sig['timestamp'] = base_time + timedelta(minutes=i)
        sig['partition_hour'] = sig['timestamp'].hour
        signatures.append(sig)
    
    logger.info(f"📊 Generated large dataset with {len(signatures)} signatures")
    return signatures


# ============================================================================
# Test Environment Fixtures
# ============================================================================

@pytest.fixture
async def test_environment(test_db, test_redis, test_s3):
    """Complete test environment with all services."""
    
    environment = {
        'db': test_db,
        'redis': test_redis,
        's3': test_s3,
        'bucket': os.getenv('S3_BUCKET', 'test-forge'),
        'config': {
            'hot_retention_hours': 24,
            'archival_batch_size': 1000,
            'consolidation_threshold': 0.85,
            'search_timeout_ms': 100
        }
    }
    
    logger.info("🧪 Test environment initialized")
    return environment


@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    
    temp_dir = tempfile.mkdtemp(prefix="aura_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = psutil.Process().memory_info().rss
        
        def stop(self):
            if self.start_time:
                duration = time.perf_counter() - self.start_time
                memory_delta = psutil.Process().memory_info().rss - self.start_memory
                
                self.metrics = {
                    'duration_seconds': duration,
                    'memory_delta_bytes': memory_delta,
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent
                }
                
                return self.metrics
    
    return PerformanceMonitor()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    
    yield  # Run the test
    
    # Cleanup logic runs after test
    test_results_dir = Path("test-results")
    if test_results_dir.exists():
        # Keep only the latest 10 test result files
        result_files = sorted(test_results_dir.glob("*.json"), key=os.path.getmtime)
        for old_file in result_files[:-10]:
            old_file.unlink(missing_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def assert_signature_valid(signature: Dict[str, Any]):
    """Assert that a signature has valid structure."""
    
    required_fields = ['id', 'timestamp', 'signature', 'metadata']
    for field in required_fields:
        assert field in signature, f"Missing required field: {field}"
    
    assert isinstance(signature['signature'], (bytes, np.ndarray))
    assert isinstance(signature['metadata'], dict)
    assert isinstance(signature['timestamp'], datetime)


def assert_search_results_valid(results: List[Dict[str, Any]], min_score: float = 0.0):
    """Assert that search results are valid."""
    
    assert isinstance(results, list)
    
    for result in results:
        assert 'id' in result
        assert 'score' in result
        assert 'content' in result or 'signature' in result
        assert result['score'] >= min_score
        assert result['score'] <= 1.0
    
    # Results should be sorted by score (descending)
    scores = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True)
