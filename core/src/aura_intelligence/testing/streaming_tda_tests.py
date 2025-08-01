"""
🧪 Comprehensive Testing Framework for Streaming TDA
Includes unit tests, integration tests, performance benchmarks, and chaos tests
"""

import asyncio
import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
import random
from unittest.mock import Mock, AsyncMock, patch

import structlog
from prometheus_client import REGISTRY

from ..tda.streaming import StreamingTDAProcessor, StreamingWindow
from ..tda.streaming.incremental_persistence import VineyardAlgorithm
from ..tda.streaming.parallel_processor import MultiScaleProcessor, ScaleConfig, RaceConditionDetector
from ..tda.streaming.event_adapters import TDAEventProcessor, PointCloudAdapter
from ..infrastructure.kafka_event_mesh import KafkaEventMesh, EventMessage
from ..common.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)


@dataclass
class TestMetrics:
    """Metrics collected during tests"""
    throughput: float  # points/second
    latency_p50: float  # median latency in ms
    latency_p99: float  # 99th percentile latency
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    

class StreamingTDATestSuite:
    """Comprehensive test suite for streaming TDA"""
    
    @pytest.fixture
    def streaming_processor(self):
        """Create a streaming TDA processor for testing"""
        return StreamingTDAProcessor(
            window_size=1000,
            slide_interval=100,
            epsilon=0.1
        )
        
    @pytest.fixture
    def multi_scale_processor(self):
        """Create a multi-scale processor for testing"""
        scales = [
            ScaleConfig("fast", window_size=100, slide_interval=10),
            ScaleConfig("medium", window_size=500, slide_interval=50),
            ScaleConfig("slow", window_size=1000, slide_interval=100)
        ]
        return MultiScaleProcessor(scales, max_workers=2)
        
    @pytest.fixture
    def mock_kafka_mesh(self):
        """Create a mock Kafka event mesh"""
        mesh = AsyncMock(spec=KafkaEventMesh)
        mesh.send = AsyncMock()
        mesh.consume_batch = AsyncMock(return_value=[])
        return mesh
        

class TestStreamingWindow:
    """Unit tests for streaming window"""
    
    def test_window_initialization(self):
        """Test window initialization"""
        window = StreamingWindow(max_size=100, dimension=3)
        assert window.max_size == 100
        assert window.dimension == 3
        assert window.current_size == 0
        
    def test_add_single_point(self):
        """Test adding a single point"""
        window = StreamingWindow(max_size=10, dimension=2)
        point = np.array([1.0, 2.0])
        
        window.add_point(point)
        assert window.current_size == 1
        assert np.array_equal(window.get_data()[0], point)
        
    def test_add_batch(self):
        """Test adding batch of points"""
        window = StreamingWindow(max_size=100, dimension=3)
        batch = np.random.randn(50, 3)
        
        window.add_batch(batch)
        assert window.current_size == 50
        assert np.array_equal(window.get_data(), batch)
        
    def test_window_overflow(self):
        """Test window behavior when exceeding max size"""
        window = StreamingWindow(max_size=10, dimension=2)
        batch = np.random.randn(15, 2)
        
        window.add_batch(batch)
        assert window.current_size == 10
        # Should keep only the last 10 points
        assert np.array_equal(window.get_data(), batch[-10:])
        
    def test_window_slide(self):
        """Test sliding window"""
        window = StreamingWindow(max_size=100, dimension=2)
        window.add_batch(np.random.randn(100, 2))
        
        window.slide(20)
        assert window.current_size == 80
        
    def test_memory_tracking(self):
        """Test memory usage tracking"""
        window = StreamingWindow(max_size=1000, dimension=3)
        window.add_batch(np.random.randn(500, 3))
        
        stats = window.get_stats()
        assert stats.memory_bytes > 0
        assert stats.total_points_seen == 500
        assert stats.total_slides == 0
        

class TestIncrementalPersistence:
    """Unit tests for incremental persistence algorithms"""
    
    def test_vineyard_initialization(self):
        """Test Vineyard algorithm initialization"""
        algo = VineyardAlgorithm(epsilon=0.1, max_features=100)
        assert algo.epsilon == 0.1
        assert algo.max_features == 100
        
    @pytest.mark.asyncio
    async def test_incremental_update(self):
        """Test incremental persistence update"""
        algo = VineyardAlgorithm(epsilon=0.1)
        
        # Initial points
        points1 = np.random.randn(50, 3)
        update1 = algo.incremental_update(points1, "window_1")
        
        assert len(update1.added_features) > 0
        assert len(update1.removed_features) == 0
        
        # Add more points
        points2 = np.random.randn(50, 3)
        update2 = algo.incremental_update(
            np.vstack([points1, points2]),
            "window_2"
        )
        
        # Should have some changes
        assert update2.update_id == "window_2"
        
    def test_feature_stability(self):
        """Test feature stability across updates"""
        algo = VineyardAlgorithm(epsilon=0.05)
        
        # Generate stable point cloud
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        
        # Add noise incrementally
        for i in range(5):
            noisy_points = points + np.random.randn(*points.shape) * 0.01
            update = algo.incremental_update(noisy_points, f"update_{i}")
            
            # Features should remain relatively stable
            if i > 0:
                assert len(update.modified_features) < len(update.added_features)
                

class TestMultiScaleProcessing:
    """Tests for multi-scale parallel processing"""
    
    @pytest.mark.asyncio
    async def test_multi_scale_initialization(self):
        """Test multi-scale processor initialization"""
        scales = [
            ScaleConfig("small", 100, 10),
            ScaleConfig("large", 1000, 100)
        ]
        processor = MultiScaleProcessor(scales)
        
        assert len(processor.scale_states) == 2
        assert "small" in processor.scale_states
        assert "large" in processor.scale_states
        
    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test parallel processing across scales"""
        scales = [
            ScaleConfig("s1", 100, 50),
            ScaleConfig("s2", 200, 100),
            ScaleConfig("s3", 300, 150)
        ]
        processor = MultiScaleProcessor(scales, max_workers=3)
        
        # Add points
        points = np.random.randn(200, 3)
        updates = await processor.add_points(points)
        
        # Should get updates for scales that hit their slide interval
        assert len(updates) > 0
        
        # Verify no race conditions
        stats = processor.get_scale_stats()
        for scale_name, stat in stats.items():
            assert stat['total_points_seen'] == 200
            
    @pytest.mark.asyncio
    async def test_race_condition_detection(self):
        """Test race condition detection"""
        detector = RaceConditionDetector()
        
        # Simulate concurrent access
        detector.log_access("scale1", "write")
        detector.log_access("scale1", "write")  # Conflict!
        detector.log_access("scale2", "read")
        
        conflicts = detector.detect_conflicts()
        assert len(conflicts) > 0
        assert conflicts[0]['type'] == 'write-write'
        

class TestEventAdapters:
    """Tests for Kafka event adapters"""
    
    @pytest.mark.asyncio
    async def test_point_cloud_adapter(self):
        """Test point cloud event adapter"""
        adapter = PointCloudAdapter()
        
        # Create test event
        event = EventMessage(
            key="test_1",
            value=json.dumps({
                "timestamp": datetime.now().isoformat(),
                "source_id": "sensor_1",
                "points": [[1, 2, 3], [4, 5, 6]],
                "metadata": {"sensor_type": "lidar"}
            }).encode(),
            headers={"content-type": "application/json"}
        )
        
        # Process event
        cloud = await adapter.process_event(event)
        
        assert cloud is not None
        assert cloud.source_id == "sensor_1"
        assert len(cloud.points) == 2
        assert cloud.metadata["sensor_type"] == "lidar"
        
    @pytest.mark.asyncio
    async def test_event_processor_integration(self):
        """Test full event processing pipeline"""
        # Create mocks
        kafka_mesh = AsyncMock(spec=KafkaEventMesh)
        kafka_mesh.consume_batch = AsyncMock(return_value=[])
        
        # Create processor
        scales = [ScaleConfig("test", 100, 10)]
        tda_processor = MultiScaleProcessor(scales)
        
        event_processor = TDAEventProcessor(
            kafka_mesh=kafka_mesh,
            tda_processor=tda_processor,
            input_topic="input",
            output_topic="output"
        )
        
        # Add test hook
        hook_called = False
        async def test_hook(event):
            nonlocal hook_called
            hook_called = True
            return event
            
        event_processor.add_pre_process_hook(test_hook)
        
        # Process one batch
        test_event = EventMessage(
            key="test",
            value=json.dumps({
                "timestamp": datetime.now().isoformat(),
                "source_id": "test",
                "points": np.random.randn(10, 3).tolist()
            }).encode(),
            headers={}
        )
        
        kafka_mesh.consume_batch.return_value = [test_event]
        
        # Run one iteration
        await event_processor._process_batch([test_event])
        
        assert hook_called
        

class PerformanceBenchmarks:
    """Performance benchmarks for streaming TDA"""
    
    @pytest.mark.benchmark
    async def benchmark_single_scale_throughput(self):
        """Benchmark single scale throughput"""
        processor = StreamingTDAProcessor(
            window_size=10000,
            slide_interval=1000
        )
        
        # Generate test data
        num_batches = 100
        batch_size = 1000
        dimension = 3
        
        start_time = time.time()
        total_points = 0
        
        for i in range(num_batches):
            points = np.random.randn(batch_size, dimension)
            await processor.process_batch(points)
            total_points += batch_size
            
        elapsed = time.time() - start_time
        throughput = total_points / elapsed
        
        logger.info(
            "single_scale_benchmark",
            throughput=throughput,
            total_points=total_points,
            elapsed=elapsed
        )
        
        assert throughput > 10000  # Should process >10k points/second
        
    @pytest.mark.benchmark
    async def benchmark_multi_scale_scaling(self):
        """Benchmark multi-scale processing scaling"""
        results = {}
        
        for num_scales in [1, 2, 4, 8]:
            scales = [
                ScaleConfig(
                    f"scale_{i}",
                    window_size=1000 * (i + 1),
                    slide_interval=100 * (i + 1)
                )
                for i in range(num_scales)
            ]
            
            processor = MultiScaleProcessor(scales)
            
            # Measure throughput
            start_time = time.time()
            total_points = 0
            
            for _ in range(50):
                points = np.random.randn(1000, 3)
                await processor.add_points(points)
                total_points += 1000
                
            elapsed = time.time() - start_time
            throughput = total_points / elapsed
            
            results[num_scales] = throughput
            
            await processor.shutdown()
            
        # Verify reasonable scaling
        logger.info("multi_scale_scaling", results=results)
        
        # Should maintain at least 50% efficiency when doubling scales
        assert results[2] > results[1] * 0.5
        assert results[4] > results[2] * 0.5
        

class ChaosTests:
    """Chaos engineering tests for streaming TDA"""
    
    @pytest.mark.chaos
    async def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        processor = StreamingTDAProcessor(
            window_size=100000,  # Large window
            slide_interval=10000
        )
        
        # Generate large batches
        for i in range(10):
            large_batch = np.random.randn(50000, 10)  # High dimension
            
            try:
                await processor.process_batch(large_batch)
            except MemoryError:
                # Should handle gracefully
                logger.warning("memory_error_handled", iteration=i)
                break
                
        # Should still be functional
        small_batch = np.random.randn(100, 3)
        result = await processor.process_batch(small_batch)
        assert result is not None
        
    @pytest.mark.chaos
    async def test_rapid_schema_evolution(self):
        """Test handling of rapid schema changes"""
        adapter = PointCloudAdapter()
        
        # Simulate schema evolution
        schemas = [
            {"points": [[1, 2]], "timestamp": "2024-01-01T00:00:00"},
            {"points": [[1, 2, 3]], "timestamp": "2024-01-01T00:00:00"},  # 2D -> 3D
            {"points": [[1, 2, 3, 4]], "timestamp": "2024-01-01T00:00:00"},  # 3D -> 4D
        ]
        
        for i, schema in enumerate(schemas):
            event = EventMessage(
                key=f"test_{i}",
                value=json.dumps(schema).encode(),
                headers={}
            )
            
            # Should handle dimension changes
            result = await adapter.process_event(event)
            if result:
                assert len(result.points[0]) == len(schema["points"][0])
                

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "chaos: mark test as a chaos engineering test"
    )
    

# Example test runner
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",  # Run only benchmarks
        "-k", "benchmark"
    ])