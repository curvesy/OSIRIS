"""
🚀 Load Testing Framework for Streaming TDA
High-performance load generation with realistic streaming patterns
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Callable, AsyncIterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

import structlog
from prometheus_client import Counter, Histogram, Gauge
import aiofiles

from ..tda.streaming.windows import StreamingWindow
from ..observability.tracing import get_tracer

logger = structlog.get_logger(__name__)

# Metrics
REQUESTS_SENT = Counter('load_test_requests_sent_total', 'Total requests sent', ['scenario', 'type'])
REQUEST_LATENCY = Histogram('load_test_request_latency_seconds', 'Request latency', ['scenario', 'type'])
ACTIVE_CONNECTIONS = Gauge('load_test_active_connections', 'Active connections', ['scenario'])
ERRORS = Counter('load_test_errors_total', 'Total errors', ['scenario', 'error_type'])
THROUGHPUT = Gauge('load_test_throughput_rps', 'Requests per second', ['scenario'])


class LoadPattern(Enum):
    """Load generation patterns"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    WAVE = "wave"
    REALISTIC = "realistic"
    CHAOS = "chaos"


@dataclass
class LoadScenario:
    """Configuration for a load test scenario"""
    name: str
    pattern: LoadPattern
    duration_seconds: int
    target_rps: float
    ramp_time_seconds: int = 60
    spike_multiplier: float = 5.0
    wave_period_seconds: int = 300
    chaos_probability: float = 0.1
    data_generator: Optional[Callable[[], np.ndarray]] = None
    window_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000])
    point_dimensions: int = 3
    
    def __post_init__(self):
        if self.data_generator is None:
            self.data_generator = self._default_data_generator
    
    def _default_data_generator(self) -> np.ndarray:
        """Generate default streaming data points"""
        # Simulate sensor data with noise
        base = np.random.randn(self.point_dimensions)
        noise = np.random.randn(self.point_dimensions) * 0.1
        drift = np.sin(time.time() / 100) * 0.5
        return base + noise + drift


@dataclass
class LoadTestResult:
    """Results from a load test run"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    average_throughput_rps: float
    peak_throughput_rps: float
    error_rate: float
    memory_usage_mb: Dict[str, float]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class LoadGenerator:
    """Generates load according to specified patterns"""
    
    def __init__(self, scenario: LoadScenario):
        self.scenario = scenario
        self.current_rps = 0.0
        self.start_time = None
        self._running = False
        self.tracer = get_tracer()
        
    async def generate_load(self) -> AsyncIterator[float]:
        """Generate load according to pattern"""
        self.start_time = time.time()
        self._running = True
        
        while self._running:
            elapsed = time.time() - self.start_time
            if elapsed >= self.scenario.duration_seconds:
                break
                
            # Calculate target RPS based on pattern
            target_rps = self._calculate_target_rps(elapsed)
            self.current_rps = target_rps
            
            # Calculate delay between requests
            if target_rps > 0:
                delay = 1.0 / target_rps
                yield delay
            else:
                yield 1.0  # Default delay
                
    def _calculate_target_rps(self, elapsed_seconds: float) -> float:
        """Calculate target RPS based on load pattern"""
        pattern = self.scenario.pattern
        
        if pattern == LoadPattern.CONSTANT:
            return self.scenario.target_rps
            
        elif pattern == LoadPattern.RAMP_UP:
            if elapsed_seconds < self.scenario.ramp_time_seconds:
                return (elapsed_seconds / self.scenario.ramp_time_seconds) * self.scenario.target_rps
            return self.scenario.target_rps
            
        elif pattern == LoadPattern.SPIKE:
            # Spike at 25%, 50%, and 75% of duration
            spike_times = [0.25, 0.5, 0.75]
            for spike_time in spike_times:
                spike_point = self.scenario.duration_seconds * spike_time
                if abs(elapsed_seconds - spike_point) < 10:  # 10 second spike
                    return self.scenario.target_rps * self.scenario.spike_multiplier
            return self.scenario.target_rps
            
        elif pattern == LoadPattern.WAVE:
            # Sine wave pattern
            phase = (elapsed_seconds / self.scenario.wave_period_seconds) * 2 * np.pi
            multiplier = 0.5 + 0.5 * np.sin(phase)
            return self.scenario.target_rps * multiplier
            
        elif pattern == LoadPattern.REALISTIC:
            # Simulate realistic traffic patterns
            hour_of_day = (elapsed_seconds / 3600) % 24
            
            # Business hours pattern
            if 9 <= hour_of_day <= 17:
                base_multiplier = 1.0
            elif 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
                base_multiplier = 0.7
            else:
                base_multiplier = 0.3
                
            # Add some randomness
            noise = random.uniform(0.8, 1.2)
            return self.scenario.target_rps * base_multiplier * noise
            
        elif pattern == LoadPattern.CHAOS:
            # Random spikes and drops
            if random.random() < self.scenario.chaos_probability:
                return self.scenario.target_rps * random.uniform(0.1, 3.0)
            return self.scenario.target_rps
            
        return self.scenario.target_rps
    
    def stop(self):
        """Stop load generation"""
        self._running = False


class StreamingTDALoadTester:
    """Load tester specifically for streaming TDA operations"""
    
    def __init__(self, scenario: LoadScenario):
        self.scenario = scenario
        self.generator = LoadGenerator(scenario)
        self.windows: Dict[int, StreamingWindow] = {}
        self.latencies: List[float] = []
        self.errors: List[Tuple[str, str]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self._active_tasks = 0
        self.tracer = get_tracer()
        
    async def initialize(self):
        """Initialize streaming windows"""
        for size in self.scenario.window_sizes:
            self.windows[size] = StreamingWindow(
                capacity=size,
                dimensions=self.scenario.point_dimensions
            )
            await self.windows[size].initialize()
            
    async def run_test(self) -> LoadTestResult:
        """Run the load test"""
        await self.initialize()
        
        self.start_time = datetime.now()
        logger.info("Starting load test", 
                   scenario=self.scenario.name,
                   pattern=self.scenario.pattern.value,
                   duration=self.scenario.duration_seconds)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_throughput())
        
        # Generate load
        tasks = []
        request_count = 0
        
        async for delay in self.generator.generate_load():
            # Create request task
            task = asyncio.create_task(self._execute_request(request_count))
            tasks.append(task)
            request_count += 1
            
            # Control concurrency
            if len(tasks) >= 1000:  # Max concurrent requests
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(tasks)
            
            await asyncio.sleep(delay)
        
        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.end_time = datetime.now()
        monitor_task.cancel()
        
        # Calculate results
        return self._calculate_results(request_count)
    
    async def _execute_request(self, request_id: int):
        """Execute a single streaming TDA request"""
        async with self.tracer.trace_async_operation(
            "load_test_request",
            scenario=self.scenario.name,
            request_id=request_id
        ):
            ACTIVE_CONNECTIONS.labels(scenario=self.scenario.name).inc()
            self._active_tasks += 1
            
            try:
                start_time = time.time()
                
                # Generate data point
                point = self.scenario.data_generator()
                
                # Process through windows
                window_size = random.choice(self.scenario.window_sizes)
                window = self.windows[window_size]
                
                # Add point and compute persistence
                async with self.tracer.trace_async_operation("add_point"):
                    await window.add_point(point)
                
                # Simulate persistence computation
                async with self.tracer.trace_async_operation("compute_persistence"):
                    await asyncio.sleep(random.uniform(0.001, 0.01))  # Simulate computation
                
                # Record success
                latency = (time.time() - start_time) * 1000  # Convert to ms
                self.latencies.append(latency)
                
                REQUESTS_SENT.labels(
                    scenario=self.scenario.name,
                    type="streaming_tda"
                ).inc()
                
                REQUEST_LATENCY.labels(
                    scenario=self.scenario.name,
                    type="streaming_tda"
                ).observe(latency / 1000)  # Convert back to seconds
                
            except Exception as e:
                self.errors.append((str(e), type(e).__name__))
                ERRORS.labels(
                    scenario=self.scenario.name,
                    error_type=type(e).__name__
                ).inc()
                self.tracer.record_exception(e)
                
            finally:
                ACTIVE_CONNECTIONS.labels(scenario=self.scenario.name).dec()
                self._active_tasks -= 1
    
    async def _monitor_throughput(self):
        """Monitor and record throughput"""
        last_count = 0
        
        while True:
            await asyncio.sleep(1)
            current_count = REQUESTS_SENT.labels(
                scenario=self.scenario.name,
                type="streaming_tda"
            )._value.get()
            
            throughput = current_count - last_count
            THROUGHPUT.labels(scenario=self.scenario.name).set(throughput)
            last_count = current_count
    
    def _calculate_results(self, total_requests: int) -> LoadTestResult:
        """Calculate test results"""
        successful_requests = len(self.latencies)
        failed_requests = len(self.errors)
        
        if self.latencies:
            latencies_array = np.array(self.latencies)
            avg_latency = np.mean(latencies_array)
            p50_latency = np.percentile(latencies_array, 50)
            p95_latency = np.percentile(latencies_array, 95)
            p99_latency = np.percentile(latencies_array, 99)
            max_latency = np.max(latencies_array)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = max_latency = 0
        
        duration = (self.end_time - self.start_time).total_seconds()
        avg_throughput = successful_requests / duration if duration > 0 else 0
        
        # Get memory usage from windows
        memory_usage = {}
        for size, window in self.windows.items():
            stats = window.get_stats()
            memory_usage[f"window_{size}"] = stats.memory_bytes / (1024 * 1024)  # MB
        
        return LoadTestResult(
            scenario_name=self.scenario.name,
            start_time=self.start_time,
            end_time=self.end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            average_throughput_rps=avg_throughput,
            peak_throughput_rps=self.scenario.target_rps,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            memory_usage_mb=memory_usage
        )
    
    async def save_results(self, result: LoadTestResult, filepath: str):
        """Save test results to file"""
        result_dict = {
            "scenario_name": result.scenario_name,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration_seconds": (result.end_time - result.start_time).total_seconds(),
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "average_latency_ms": result.average_latency_ms,
            "p50_latency_ms": result.p50_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "p99_latency_ms": result.p99_latency_ms,
            "max_latency_ms": result.max_latency_ms,
            "average_throughput_rps": result.average_throughput_rps,
            "peak_throughput_rps": result.peak_throughput_rps,
            "error_rate": result.error_rate,
            "memory_usage_mb": result.memory_usage_mb,
            "custom_metrics": result.custom_metrics
        }
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(result_dict, indent=2))
            
        logger.info("Load test results saved", filepath=filepath)