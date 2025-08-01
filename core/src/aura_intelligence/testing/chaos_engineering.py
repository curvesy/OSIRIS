"""
🔥 Chaos Engineering Suite for Streaming TDA
Advanced fault injection, network partitions, and resilience testing
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import os
import signal
import psutil
import numpy as np
from enum import Enum

import structlog
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
from toxiproxy import Toxiproxy

from ..tda.streaming import StreamingTDAProcessor
from ..tda.streaming.parallel_processor import MultiScaleProcessor, ScaleConfig
from ..infrastructure.kafka_event_mesh import KafkaEventMesh
from ..common.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

# Chaos metrics
FAULTS_INJECTED = Counter('chaos_faults_injected_total', 'Total faults injected', ['fault_type'])
RECOVERY_TIME = Histogram('chaos_recovery_time_seconds', 'Time to recover from fault', ['fault_type'])
DATA_LOSS = Counter('chaos_data_loss_total', 'Data lost during chaos', ['scenario'])
DEGRADATION_DETECTED = Counter('chaos_degradation_detected_total', 'Performance degradation events')


class FaultType(Enum):
    """Types of faults to inject"""
    NETWORK_PARTITION = "network_partition"
    KAFKA_SLOWDOWN = "kafka_slowdown"
    MESSAGE_CORRUPTION = "message_corruption"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    SCHEMA_DRIFT = "schema_drift"
    CLOCK_SKEW = "clock_skew"
    PARTIAL_FAILURE = "partial_failure"
    CASCADE_FAILURE = "cascade_failure"


@dataclass
class ChaosScenario:
    """Configuration for a chaos scenario"""
    name: str
    fault_types: List[FaultType]
    duration: timedelta
    intensity: float  # 0.0 to 1.0
    targets: List[str]  # Components to target
    recovery_validation: Optional[Callable] = None
    

@dataclass
class ChaosResult:
    """Results from a chaos test"""
    scenario: str
    start_time: datetime
    end_time: datetime
    faults_injected: int
    data_loss: int
    max_latency_ms: float
    recovery_time_s: float
    errors: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    

class NetworkChaosInjector:
    """
    Injects network-level chaos using toxiproxy
    """
    
    def __init__(self, toxiproxy_host: str = "localhost:8474"):
        self.toxiproxy = Toxiproxy(toxiproxy_host)
        self.active_toxics: Dict[str, Any] = {}
        
    async def inject_partition(
        self,
        service: str,
        duration: timedelta,
        direction: str = "both"
    ) -> None:
        """Inject network partition"""
        logger.info(
            "injecting_network_partition",
            service=service,
            duration=duration.total_seconds()
        )
        
        proxy = self.toxiproxy.get_proxy(service)
        
        # Add timeout toxic
        toxic = proxy.add_toxic(
            name=f"partition_{service}",
            type="timeout",
            stream=direction,
            attributes={"timeout": 0}
        )
        self.active_toxics[service] = toxic
        FAULTS_INJECTED.labels(fault_type=FaultType.NETWORK_PARTITION.value).inc()
        
        # Wait for duration
        await asyncio.sleep(duration.total_seconds())
        
        # Remove toxic
        toxic.destroy()
        del self.active_toxics[service]
        
    async def inject_latency(
        self,
        service: str,
        latency_ms: int,
        jitter_ms: int,
        duration: timedelta
    ) -> None:
        """Inject network latency"""
        logger.info(
            "injecting_latency",
            service=service,
            latency_ms=latency_ms,
            jitter_ms=jitter_ms
        )
        
        proxy = self.toxiproxy.get_proxy(service)
        
        # Add latency toxic
        toxic = proxy.add_toxic(
            name=f"latency_{service}",
            type="latency",
            stream="downstream",
            attributes={
                "latency": latency_ms,
                "jitter": jitter_ms
            }
        )
        self.active_toxics[f"{service}_latency"] = toxic
        FAULTS_INJECTED.labels(fault_type=FaultType.KAFKA_SLOWDOWN.value).inc()
        
        await asyncio.sleep(duration.total_seconds())
        
        toxic.destroy()
        del self.active_toxics[f"{service}_latency"]
        
    async def inject_packet_loss(
        self,
        service: str,
        loss_percentage: float,
        duration: timedelta
    ) -> None:
        """Inject packet loss"""
        proxy = self.toxiproxy.get_proxy(service)
        
        toxic = proxy.add_toxic(
            name=f"loss_{service}",
            type="limit_data",
            stream="downstream",
            attributes={
                "bytes": int(1000000 * (1 - loss_percentage))
            }
        )
        
        await asyncio.sleep(duration.total_seconds())
        toxic.destroy()
        

class KafkaChaosInjector:
    """
    Kafka-specific chaos injection
    """
    
    def __init__(self, kafka_admin_url: str):
        self.admin_url = kafka_admin_url
        self.session = aiohttp.ClientSession()
        
    async def inject_broker_failure(
        self,
        broker_id: int,
        duration: timedelta
    ) -> None:
        """Simulate broker failure"""
        # Send shutdown command to broker
        async with self.session.post(
            f"{self.admin_url}/brokers/{broker_id}/shutdown"
        ) as response:
            if response.status == 200:
                logger.info("broker_shutdown", broker_id=broker_id)
                FAULTS_INJECTED.labels(fault_type=FaultType.PARTIAL_FAILURE.value).inc()
                
        await asyncio.sleep(duration.total_seconds())
        
        # Restart broker
        async with self.session.post(
            f"{self.admin_url}/brokers/{broker_id}/start"
        ) as response:
            if response.status == 200:
                logger.info("broker_restarted", broker_id=broker_id)
                
    async def inject_topic_corruption(
        self,
        topic: str,
        corruption_rate: float
    ) -> None:
        """Inject corrupted messages into topic"""
        # This would interact with a test producer that sends corrupted data
        pass
        
    async def cleanup(self) -> None:
        await self.session.close()
        

class SystemChaosInjector:
    """
    System-level chaos injection
    """
    
    @staticmethod
    async def inject_memory_pressure(
        target_mb: int,
        duration: timedelta
    ) -> None:
        """Consume memory to create pressure"""
        logger.info("injecting_memory_pressure", target_mb=target_mb)
        
        # Allocate memory
        memory_hog = []
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        chunks_needed = target_mb // 10
        
        try:
            for _ in range(chunks_needed):
                # Allocate and touch memory to ensure it's resident
                chunk = bytearray(chunk_size)
                for i in range(0, chunk_size, 4096):
                    chunk[i] = 1
                memory_hog.append(chunk)
                
            FAULTS_INJECTED.labels(fault_type=FaultType.MEMORY_PRESSURE.value).inc()
            await asyncio.sleep(duration.total_seconds())
            
        finally:
            # Release memory
            memory_hog.clear()
            
    @staticmethod
    async def inject_cpu_spike(
        target_percent: int,
        duration: timedelta
    ) -> None:
        """Create CPU load"""
        logger.info("injecting_cpu_spike", target_percent=target_percent)
        
        async def cpu_burn():
            end_time = time.time() + duration.total_seconds()
            while time.time() < end_time:
                # Busy loop
                _ = sum(i * i for i in range(1000))
                await asyncio.sleep(0.001)  # Yield occasionally
                
        # Start multiple tasks to burn CPU
        num_tasks = int(psutil.cpu_count() * target_percent / 100)
        tasks = [asyncio.create_task(cpu_burn()) for _ in range(num_tasks)]
        
        FAULTS_INJECTED.labels(fault_type=FaultType.CPU_SPIKE.value).inc()
        await asyncio.gather(*tasks)
        
    @staticmethod
    async def inject_clock_skew(
        skew_seconds: int,
        duration: timedelta
    ) -> None:
        """Simulate clock skew (requires root)"""
        logger.warning(
            "clock_skew_simulation",
            skew_seconds=skew_seconds,
            note="Would require root access to actually change system time"
        )
        FAULTS_INJECTED.labels(fault_type=FaultType.CLOCK_SKEW.value).inc()
        await asyncio.sleep(duration.total_seconds())
        

class ChaosOrchestrator:
    """
    Orchestrates complex chaos scenarios
    """
    
    def __init__(
        self,
        network_injector: NetworkChaosInjector,
        kafka_injector: KafkaChaosInjector,
        system_injector: SystemChaosInjector
    ):
        self.network = network_injector
        self.kafka = kafka_injector
        self.system = system_injector
        self.active_scenarios: List[ChaosScenario] = []
        
    async def run_scenario(
        self,
        scenario: ChaosScenario,
        target_system: Any
    ) -> ChaosResult:
        """Run a complete chaos scenario"""
        logger.info("starting_chaos_scenario", scenario=scenario.name)
        
        result = ChaosResult(
            scenario=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            faults_injected=0,
            data_loss=0,
            max_latency_ms=0,
            recovery_time_s=0,
            errors=[]
        )
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            self._monitor_system(target_system, result)
        )
        
        try:
            # Inject faults based on scenario
            fault_tasks = []
            
            for fault_type in scenario.fault_types:
                if fault_type == FaultType.NETWORK_PARTITION:
                    for target in scenario.targets:
                        task = self.network.inject_partition(
                            target,
                            scenario.duration
                        )
                        fault_tasks.append(task)
                        
                elif fault_type == FaultType.KAFKA_SLOWDOWN:
                    task = self.network.inject_latency(
                        "kafka",
                        latency_ms=int(1000 * scenario.intensity),
                        jitter_ms=int(500 * scenario.intensity),
                        duration=scenario.duration
                    )
                    fault_tasks.append(task)
                    
                elif fault_type == FaultType.MEMORY_PRESSURE:
                    task = self.system.inject_memory_pressure(
                        target_mb=int(1000 * scenario.intensity),
                        duration=scenario.duration
                    )
                    fault_tasks.append(task)
                    
                elif fault_type == FaultType.CASCADE_FAILURE:
                    # Simulate cascading failures
                    task = self._inject_cascade_failure(
                        scenario.targets,
                        scenario.duration,
                        scenario.intensity
                    )
                    fault_tasks.append(task)
                    
                result.faults_injected += 1
                
            # Wait for all faults to complete
            await asyncio.gather(*fault_tasks, return_exceptions=True)
            
            # Measure recovery
            recovery_start = datetime.now()
            if scenario.recovery_validation:
                await scenario.recovery_validation(target_system)
            recovery_time = (datetime.now() - recovery_start).total_seconds()
            
            result.recovery_time_s = recovery_time
            RECOVERY_TIME.labels(fault_type=scenario.name).observe(recovery_time)
            
        except Exception as e:
            logger.error("chaos_scenario_error", error=str(e))
            result.errors.append(str(e))
            
        finally:
            monitor_task.cancel()
            result.end_time = datetime.now()
            
        return result
        
    async def _inject_cascade_failure(
        self,
        targets: List[str],
        duration: timedelta,
        intensity: float
    ) -> None:
        """Inject cascading failures across components"""
        delay_between_failures = 5 * (1 - intensity)  # Faster cascade with higher intensity
        
        for i, target in enumerate(targets):
            await asyncio.sleep(i * delay_between_failures)
            
            # Inject progressive failures
            if i == 0:
                # First component gets network partition
                asyncio.create_task(
                    self.network.inject_partition(target, duration)
                )
            elif i == 1:
                # Second gets high latency
                asyncio.create_task(
                    self.network.inject_latency(
                        target, 5000, 1000, duration
                    )
                )
            else:
                # Others get packet loss
                asyncio.create_task(
                    self.network.inject_packet_loss(
                        target, 0.5, duration
                    )
                )
                
    async def _monitor_system(
        self,
        target_system: Any,
        result: ChaosResult
    ) -> None:
        """Monitor system during chaos"""
        while True:
            try:
                # Collect metrics
                if hasattr(target_system, 'get_stats'):
                    stats = await target_system.get_stats()
                    
                    # Track max latency
                    if 'latency_ms' in stats:
                        result.max_latency_ms = max(
                            result.max_latency_ms,
                            stats['latency_ms']
                        )
                        
                    # Track data loss
                    if 'dropped_messages' in stats:
                        result.data_loss += stats['dropped_messages']
                        
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitoring_error", error=str(e))
                

class RealisticDataGenerator:
    """
    Generates realistic data patterns for testing
    """
    
    @staticmethod
    def generate_iot_sensor_data(
        num_sensors: int,
        duration: timedelta,
        anomaly_rate: float = 0.01
    ) -> AsyncIterator[np.ndarray]:
        """Generate realistic IoT sensor data"""
        # Base patterns for different sensor types
        patterns = {
            'temperature': lambda t: 20 + 5 * np.sin(2 * np.pi * t / 86400) + np.random.randn() * 0.5,
            'pressure': lambda t: 1013 + 10 * np.sin(2 * np.pi * t / 3600) + np.random.randn() * 2,
            'vibration': lambda t: 0.1 + 0.05 * np.sin(2 * np.pi * t / 60) + np.random.randn() * 0.02
        }
        
        start_time = time.time()
        while time.time() - start_time < duration.total_seconds():
            data = []
            
            for sensor_id in range(num_sensors):
                t = time.time() - start_time
                pattern = list(patterns.values())[sensor_id % len(patterns)]
                
                # Normal reading
                value = pattern(t)
                
                # Inject anomalies
                if random.random() < anomaly_rate:
                    value *= random.uniform(2, 5)  # Spike
                    
                data.append([sensor_id, t, value])
                
            yield np.array(data)
            await asyncio.sleep(0.1)  # 10Hz sampling
            
    @staticmethod
    def generate_financial_data(
        num_instruments: int,
        duration: timedelta,
        volatility: float = 0.02
    ) -> AsyncIterator[np.ndarray]:
        """Generate realistic financial market data"""
        prices = np.random.uniform(50, 200, num_instruments)
        volumes = np.random.uniform(1000, 10000, num_instruments)
        
        start_time = time.time()
        while time.time() - start_time < duration.total_seconds():
            data = []
            
            for i in range(num_instruments):
                # Random walk with drift
                price_change = np.random.randn() * volatility * prices[i]
                prices[i] += price_change
                
                # Volume follows price volatility
                volume = volumes[i] * (1 + abs(price_change) / prices[i] * 10)
                
                data.append([
                    i,  # instrument_id
                    time.time(),  # timestamp
                    prices[i],  # price
                    volume  # volume
                ])
                
            yield np.array(data)
            await asyncio.sleep(0.01)  # 100Hz for HFT simulation
            

class LongRunningStressTest:
    """
    Extended stress tests for memory leaks and degradation
    """
    
    def __init__(
        self,
        target_system: Any,
        duration: timedelta,
        checkpoint_interval: timedelta = timedelta(minutes=30)
    ):
        self.target_system = target_system
        self.duration = duration
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints: List[Dict[str, Any]] = []
        
    async def run(self) -> Dict[str, Any]:
        """Run extended stress test"""
        logger.info(
            "starting_long_stress_test",
            duration_hours=self.duration.total_seconds() / 3600
        )
        
        start_time = datetime.now()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Data generator
        data_gen = RealisticDataGenerator.generate_iot_sensor_data(
            num_sensors=100,
            duration=self.duration
        )
        
        checkpoint_task = asyncio.create_task(self._collect_checkpoints())
        
        try:
            # Process data continuously
            async for data in data_gen:
                if hasattr(self.target_system, 'add_points'):
                    await self.target_system.add_points(data)
                else:
                    await self.target_system.process_batch(data)
                    
        finally:
            checkpoint_task.cancel()
            
        # Final analysis
        end_time = datetime.now()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Detect memory leak
        memory_growth = final_memory - initial_memory
        memory_growth_rate = memory_growth / (self.duration.total_seconds() / 3600)  # MB/hour
        
        # Analyze performance degradation
        degradation = self._analyze_degradation()
        
        return {
            'duration': (end_time - start_time).total_seconds(),
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_hour': memory_growth_rate,
            'performance_degradation': degradation,
            'checkpoints': self.checkpoints,
            'memory_leak_detected': memory_growth_rate > 10  # 10MB/hour threshold
        }
        
    async def _collect_checkpoints(self) -> None:
        """Collect system metrics at regular intervals"""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval.total_seconds())
                
                checkpoint = {
                    'timestamp': datetime.now(),
                    'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.Process().cpu_percent(interval=1),
                    'num_threads': psutil.Process().num_threads(),
                }
                
                # Get system-specific metrics
                if hasattr(self.target_system, 'get_stats'):
                    checkpoint['system_stats'] = await self.target_system.get_stats()
                    
                self.checkpoints.append(checkpoint)
                
            except asyncio.CancelledError:
                break
                
    def _analyze_degradation(self) -> Dict[str, float]:
        """Analyze performance degradation over time"""
        if len(self.checkpoints) < 2:
            return {}
            
        # Compare first and last quartiles
        n = len(self.checkpoints)
        first_quartile = self.checkpoints[:n//4]
        last_quartile = self.checkpoints[3*n//4:]
        
        degradation = {}
        
        # Memory degradation
        first_memory = np.mean([c['memory_mb'] for c in first_quartile])
        last_memory = np.mean([c['memory_mb'] for c in last_quartile])
        degradation['memory_percent'] = (last_memory - first_memory) / first_memory * 100
        
        # CPU degradation
        first_cpu = np.mean([c['cpu_percent'] for c in first_quartile])
        last_cpu = np.mean([c['cpu_percent'] for c in last_quartile])
        degradation['cpu_percent'] = (last_cpu - first_cpu) / max(first_cpu, 1) * 100
        
        # Detect if degradation is significant
        if degradation['memory_percent'] > 20 or degradation['cpu_percent'] > 30:
            DEGRADATION_DETECTED.inc()
            
        return degradation


# Example chaos test scenarios
CHAOS_SCENARIOS = [
    ChaosScenario(
        name="kafka_network_partition",
        fault_types=[FaultType.NETWORK_PARTITION],
        duration=timedelta(minutes=5),
        intensity=1.0,
        targets=["kafka"]
    ),
    ChaosScenario(
        name="multi_component_slowdown",
        fault_types=[FaultType.KAFKA_SLOWDOWN, FaultType.CPU_SPIKE],
        duration=timedelta(minutes=10),
        intensity=0.7,
        targets=["kafka", "tda_processor"]
    ),
    ChaosScenario(
        name="cascade_failure",
        fault_types=[FaultType.CASCADE_FAILURE],
        duration=timedelta(minutes=15),
        intensity=0.8,
        targets=["kafka", "schema_registry", "tda_processor"]
    ),
    ChaosScenario(
        name="resource_exhaustion",
        fault_types=[FaultType.MEMORY_PRESSURE, FaultType.CPU_SPIKE, FaultType.DISK_FULL],
        duration=timedelta(minutes=20),
        intensity=0.9,
        targets=["system"]
    )
]


if __name__ == "__main__":
    # Example usage
    async def run_chaos_suite():
        # Initialize injectors
        network_injector = NetworkChaosInjector()
        kafka_injector = KafkaChaosInjector("http://localhost:8080")
        system_injector = SystemChaosInjector()
        
        orchestrator = ChaosOrchestrator(
            network_injector,
            kafka_injector,
            system_injector
        )
        
        # Create test system
        scales = [
            ScaleConfig("fast", 1000, 100),
            ScaleConfig("slow", 10000, 1000)
        ]
        tda_processor = MultiScaleProcessor(scales)
        
        # Run chaos scenarios
        for scenario in CHAOS_SCENARIOS:
            logger.info("running_scenario", name=scenario.name)
            result = await orchestrator.run_scenario(scenario, tda_processor)
            
            logger.info(
                "scenario_complete",
                name=result.scenario,
                faults=result.faults_injected,
                data_loss=result.data_loss,
                recovery_time=result.recovery_time_s
            )
            
        # Cleanup
        await kafka_injector.cleanup()
        await tda_processor.shutdown()
        
    asyncio.run(run_chaos_suite())