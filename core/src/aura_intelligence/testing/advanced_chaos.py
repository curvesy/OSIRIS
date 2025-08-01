"""
🌪️ Advanced Chaos Engineering for Production-Grade Testing
Simulates real-world failure modes including cloud outages and Kafka storms
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque
import json

import structlog
from prometheus_client import Counter, Histogram, Gauge
import aiohttp

from .chaos_engineering import (
    FaultType, ChaosScenario, ChaosResult,
    NetworkChaosInjector, KafkaChaosInjector, SystemChaosInjector
)
from ..infrastructure.kafka_event_mesh import KafkaEventMesh
from ..common.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

# Advanced chaos metrics
INTERMITTENT_FAULTS = Counter('chaos_intermittent_faults_total', 'Intermittent faults injected')
CLOUD_OUTAGES = Counter('chaos_cloud_outages_total', 'Cloud service outages simulated')
KAFKA_STORMS = Counter('chaos_kafka_storms_total', 'Kafka rebalance storms triggered')
DEGRADATION_PATTERNS = Histogram('chaos_degradation_patterns', 'Performance degradation patterns')


class AdvancedFaultType(Enum):
    """Extended fault types for production scenarios"""
    INTERMITTENT_NETWORK = "intermittent_network"
    CLOUD_SERVICE_OUTAGE = "cloud_service_outage"
    KAFKA_REBALANCE_STORM = "kafka_rebalance_storm"
    DNS_RESOLUTION_FAILURE = "dns_resolution_failure"
    SSL_CERTIFICATE_EXPIRY = "ssl_certificate_expiry"
    RATE_LIMITING = "rate_limiting"
    NOISY_NEIGHBOR = "noisy_neighbor"
    COLD_START_LATENCY = "cold_start_latency"
    REGIONAL_FAILOVER = "regional_failover"
    DATA_CORRUPTION_DRIFT = "data_corruption_drift"


@dataclass
class CloudProvider:
    """Cloud provider configuration"""
    name: str  # AWS, Azure, GCP
    region: str
    availability_zones: List[str]
    services: Dict[str, Any]  # Service configurations


class IntermittentNetworkChaos:
    """
    Simulates realistic intermittent network issues
    """
    
    def __init__(self, network_injector: NetworkChaosInjector):
        self.network = network_injector
        self.patterns = {
            'flapping': self._flapping_connection,
            'degrading': self._degrading_performance,
            'burst_loss': self._burst_packet_loss,
            'jittery': self._jittery_latency
        }
        
    async def inject_intermittent_failure(
        self,
        target: str,
        pattern: str,
        duration: timedelta,
        intensity: float
    ) -> None:
        """Inject intermittent network issues with various patterns"""
        logger.info(
            "injecting_intermittent_failure",
            target=target,
            pattern=pattern,
            duration=duration.total_seconds()
        )
        
        INTERMITTENT_FAULTS.inc()
        
        if pattern in self.patterns:
            await self.patterns[pattern](target, duration, intensity)
        else:
            logger.error("unknown_pattern", pattern=pattern)
            
    async def _flapping_connection(
        self,
        target: str,
        duration: timedelta,
        intensity: float
    ) -> None:
        """Connection that alternates between working and failing"""
        end_time = time.time() + duration.total_seconds()
        
        # Flapping intervals based on intensity
        up_time = 30 * (1 - intensity)  # Less intense = longer up time
        down_time = 10 * intensity  # More intense = longer down time
        
        while time.time() < end_time:
            # Connection down
            await self.network.inject_partition(
                target,
                timedelta(seconds=down_time),
                "both"
            )
            
            # Connection up
            await asyncio.sleep(up_time)
            
    async def _degrading_performance(
        self,
        target: str,
        duration: timedelta,
        intensity: float
    ) -> None:
        """Gradually degrading network performance"""
        steps = 10
        step_duration = duration.total_seconds() / steps
        
        for i in range(steps):
            # Exponentially increasing latency
            latency = int(10 * (2 ** (i * intensity)))
            jitter = int(latency * 0.3)
            
            await self.network.inject_latency(
                target,
                latency,
                jitter,
                timedelta(seconds=step_duration)
            )
            
    async def _burst_packet_loss(
        self,
        target: str,
        duration: timedelta,
        intensity: float
    ) -> None:
        """Burst packet loss patterns"""
        end_time = time.time() + duration.total_seconds()
        
        while time.time() < end_time:
            # Random burst of packet loss
            if random.random() < intensity:
                loss_percent = random.uniform(0.1, 0.5)
                burst_duration = random.uniform(1, 5)
                
                await self.network.inject_packet_loss(
                    target,
                    loss_percent,
                    timedelta(seconds=burst_duration)
                )
            else:
                await asyncio.sleep(1)
                
    async def _jittery_latency(
        self,
        target: str,
        duration: timedelta,
        intensity: float
    ) -> None:
        """Highly variable latency"""
        end_time = time.time() + duration.total_seconds()
        
        while time.time() < end_time:
            # Random latency spikes
            base_latency = 10
            spike_latency = int(base_latency + random.exponential(100 * intensity))
            jitter = int(spike_latency * 0.5)
            
            await self.network.inject_latency(
                target,
                spike_latency,
                jitter,
                timedelta(seconds=random.uniform(0.5, 2))
            )


class CloudServiceChaos:
    """
    Simulates cloud service-specific failures
    """
    
    def __init__(self, cloud_provider: CloudProvider):
        self.provider = cloud_provider
        self.outage_history: List[Dict[str, Any]] = []
        
    async def inject_service_outage(
        self,
        service: str,
        scope: str,  # 'regional', 'zonal', 'global'
        duration: timedelta,
        partial: bool = True
    ) -> None:
        """Simulate cloud service outage"""
        logger.info(
            "injecting_cloud_outage",
            provider=self.provider.name,
            service=service,
            scope=scope,
            duration=duration.total_seconds()
        )
        
        CLOUD_OUTAGES.inc()
        
        outage = {
            'service': service,
            'scope': scope,
            'start_time': datetime.now(),
            'duration': duration,
            'partial': partial
        }
        self.outage_history.append(outage)
        
        if scope == 'regional':
            await self._regional_outage(service, duration, partial)
        elif scope == 'zonal':
            await self._zonal_outage(service, duration, partial)
        else:  # global
            await self._global_outage(service, duration, partial)
            
    async def _regional_outage(
        self,
        service: str,
        duration: timedelta,
        partial: bool
    ) -> None:
        """Simulate regional service outage"""
        # Affect all AZs in the region
        affected_resources = []
        
        for az in self.provider.availability_zones:
            if service in self.provider.services:
                resources = self.provider.services[service].get(az, [])
                affected_resources.extend(resources)
                
        # Simulate outage effects
        if partial:
            # 50% of requests fail
            await self._simulate_partial_outage(affected_resources, duration, 0.5)
        else:
            # Complete outage
            await self._simulate_complete_outage(affected_resources, duration)
            
    async def _zonal_outage(
        self,
        service: str,
        duration: timedelta,
        partial: bool
    ) -> None:
        """Simulate single AZ outage"""
        # Pick random AZ
        affected_az = random.choice(self.provider.availability_zones)
        
        if service in self.provider.services:
            resources = self.provider.services[service].get(affected_az, [])
            
            if partial:
                await self._simulate_partial_outage(resources, duration, 0.3)
            else:
                await self._simulate_complete_outage(resources, duration)
                
    async def _simulate_partial_outage(
        self,
        resources: List[str],
        duration: timedelta,
        failure_rate: float
    ) -> None:
        """Simulate partial service degradation"""
        # This would integrate with actual service mocking
        logger.info(
            "partial_outage",
            resources=len(resources),
            failure_rate=failure_rate
        )
        await asyncio.sleep(duration.total_seconds())
        
    async def _simulate_complete_outage(
        self,
        resources: List[str],
        duration: timedelta
    ) -> None:
        """Simulate complete service failure"""
        logger.info("complete_outage", resources=len(resources))
        await asyncio.sleep(duration.total_seconds())
        
    async def inject_cold_start_latency(
        self,
        service: str,
        latency_ms: int,
        duration: timedelta
    ) -> None:
        """Simulate serverless cold start latency"""
        logger.info(
            "injecting_cold_start",
            service=service,
            latency_ms=latency_ms
        )
        
        # Simulate cold starts at random intervals
        end_time = time.time() + duration.total_seconds()
        
        while time.time() < end_time:
            # Cold start probability decreases over time (warming up)
            cold_start_prob = 0.3 * (1 - (time.time() - end_time + duration.total_seconds()) / duration.total_seconds())
            
            if random.random() < cold_start_prob:
                # Inject cold start latency
                await asyncio.sleep(latency_ms / 1000)
                
            await asyncio.sleep(random.uniform(5, 15))  # Between requests


class KafkaRebalanceStorm:
    """
    Simulates Kafka rebalance storms and related issues
    """
    
    def __init__(self, kafka_injector: KafkaChaosInjector):
        self.kafka = kafka_injector
        self.rebalance_count = 0
        
    async def inject_rebalance_storm(
        self,
        consumer_group: str,
        intensity: float,  # 0.0 to 1.0
        duration: timedelta
    ) -> None:
        """Trigger repeated Kafka rebalances"""
        logger.info(
            "injecting_rebalance_storm",
            consumer_group=consumer_group,
            intensity=intensity
        )
        
        KAFKA_STORMS.inc()
        
        # Calculate rebalance frequency based on intensity
        rebalance_interval = 30 * (1 - intensity)  # More intense = more frequent
        
        end_time = time.time() + duration.total_seconds()
        
        while time.time() < end_time:
            # Trigger rebalance by simulating consumer join/leave
            await self._trigger_rebalance(consumer_group)
            self.rebalance_count += 1
            
            # Wait before next rebalance
            await asyncio.sleep(rebalance_interval)
            
    async def _trigger_rebalance(self, consumer_group: str) -> None:
        """Trigger a consumer group rebalance"""
        # Simulate consumer churn
        actions = ['add_consumer', 'remove_consumer', 'consumer_crash']
        action = random.choice(actions)
        
        logger.info(
            "triggering_rebalance",
            consumer_group=consumer_group,
            action=action,
            count=self.rebalance_count
        )
        
        if action == 'add_consumer':
            # Simulate new consumer joining
            await self._simulate_consumer_join(consumer_group)
        elif action == 'remove_consumer':
            # Simulate consumer leaving gracefully
            await self._simulate_consumer_leave(consumer_group)
        else:
            # Simulate consumer crash
            await self._simulate_consumer_crash(consumer_group)
            
    async def _simulate_consumer_join(self, group: str) -> None:
        """Simulate consumer joining group"""
        # This would interact with Kafka admin API
        await asyncio.sleep(0.1)
        
    async def _simulate_consumer_leave(self, group: str) -> None:
        """Simulate consumer leaving group"""
        await asyncio.sleep(0.1)
        
    async def _simulate_consumer_crash(self, group: str) -> None:
        """Simulate consumer crash"""
        # Abrupt disconnection
        await asyncio.sleep(0.01)
        
    async def inject_partition_skew(
        self,
        topic: str,
        skew_factor: float,
        duration: timedelta
    ) -> None:
        """Create uneven partition load"""
        logger.info(
            "injecting_partition_skew",
            topic=topic,
            skew_factor=skew_factor
        )
        
        # This would interact with a test producer to send
        # disproportionate data to certain partitions
        await asyncio.sleep(duration.total_seconds())


class ProductionDataGenerator:
    """
    Generates realistic production data with domain-specific patterns
    """
    
    @staticmethod
    async def generate_financial_trading_data(
        duration: timedelta,
        include_anomalies: bool = True
    ) -> AsyncIterator[np.ndarray]:
        """Generate realistic financial trading data"""
        # Market hours and patterns
        market_open = 9.5  # 9:30 AM
        market_close = 16  # 4:00 PM
        
        # Instruments with different characteristics
        instruments = {
            'AAPL': {'volatility': 0.02, 'volume': 50000000, 'price': 150},
            'SPY': {'volatility': 0.01, 'volume': 80000000, 'price': 450},
            'TSLA': {'volatility': 0.04, 'volume': 30000000, 'price': 250},
            'GLD': {'volatility': 0.008, 'volume': 10000000, 'price': 180},
            'NVDA': {'volatility': 0.03, 'volume': 40000000, 'price': 500},
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration.total_seconds():
            current_hour = (time.time() % 86400) / 3600  # Hour of day
            
            # Market activity patterns
            if market_open <= current_hour <= market_close:
                activity_level = 1.0
                
                # Opening/closing spikes
                if current_hour < market_open + 0.5:
                    activity_level = 2.0  # Opening spike
                elif current_hour > market_close - 0.5:
                    activity_level = 1.5  # Closing activity
                    
            else:
                activity_level = 0.1  # After-hours trading
                
            data = []
            
            for symbol, props in instruments.items():
                # Price movement with momentum
                price_change = np.random.randn() * props['volatility'] * activity_level
                
                # Volume patterns
                base_volume = props['volume'] * activity_level
                volume = int(base_volume * (1 + np.random.exponential(0.1)))
                
                # Bid-ask spread
                spread = props['price'] * 0.0001 * (1 + 1/activity_level)
                
                # Anomalies
                if include_anomalies and random.random() < 0.001:
                    # Flash crash/spike
                    price_change *= random.uniform(5, 10)
                    volume *= 10
                    
                data.append([
                    hash(symbol) % 1000,  # instrument_id
                    time.time(),
                    props['price'] + price_change,
                    volume,
                    spread
                ])
                
            yield np.array(data)
            
            # Realistic tick rate
            await asyncio.sleep(0.01 if activity_level > 0.5 else 0.1)
            
    @staticmethod
    async def generate_iot_manufacturing_data(
        duration: timedelta,
        include_anomalies: bool = True
    ) -> AsyncIterator[np.ndarray]:
        """Generate manufacturing sensor data with realistic patterns"""
        # Production line configuration
        production_lines = 5
        sensors_per_line = 20
        
        # Shift patterns (3 shifts)
        shift_schedule = {
            0: {'efficiency': 0.95, 'defect_rate': 0.01},   # Night
            1: {'efficiency': 0.98, 'defect_rate': 0.008},  # Day
            2: {'efficiency': 0.96, 'defect_rate': 0.012},  # Evening
        }
        
        # Machine states
        machine_states = {
            'running': 0.85,
            'idle': 0.10,
            'maintenance': 0.04,
            'fault': 0.01
        }
        
        start_time = time.time()
        
        # Initialize machine states
        current_states = [
            random.choices(
                list(machine_states.keys()),
                weights=list(machine_states.values())
            )[0] for _ in range(production_lines * sensors_per_line)
        ]
        
        while time.time() - start_time < duration.total_seconds():
            current_hour = int((time.time() % 86400) / 3600)
            shift = current_hour // 8
            shift_params = shift_schedule[shift % 3]
            
            data = []
            
            for line in range(production_lines):
                for sensor in range(sensors_per_line):
                    sensor_id = line * sensors_per_line + sensor
                    state = current_states[sensor_id]
                    
                    # Base readings based on state
                    if state == 'running':
                        temperature = 75 + np.random.randn() * 2
                        vibration = 0.5 + np.random.randn() * 0.1
                        pressure = 100 + np.random.randn() * 5
                        efficiency = shift_params['efficiency'] * (1 + np.random.randn() * 0.02)
                        
                    elif state == 'idle':
                        temperature = 65 + np.random.randn() * 1
                        vibration = 0.1 + np.random.randn() * 0.02
                        pressure = 90 + np.random.randn() * 2
                        efficiency = 0
                        
                    elif state == 'maintenance':
                        temperature = 70 + np.random.randn() * 3
                        vibration = 0.3 + np.random.randn() * 0.2
                        pressure = 95 + np.random.randn() * 10
                        efficiency = 0
                        
                    else:  # fault
                        temperature = 85 + np.random.randn() * 5
                        vibration = 1.0 + np.random.randn() * 0.3
                        pressure = 110 + np.random.randn() * 15
                        efficiency = shift_params['efficiency'] * 0.5
                        
                    # Anomalies
                    if include_anomalies:
                        # Degradation pattern
                        if random.random() < 0.0001:
                            temperature += random.uniform(10, 20)
                            vibration *= 2
                            
                        # Sudden failure
                        if state == 'running' and random.random() < 0.00001:
                            current_states[sensor_id] = 'fault'
                            temperature = 95
                            vibration = 2.0
                            
                    # State transitions
                    if random.random() < 0.001:
                        current_states[sensor_id] = random.choices(
                            list(machine_states.keys()),
                            weights=list(machine_states.values())
                        )[0]
                        
                    data.append([
                        sensor_id,
                        time.time(),
                        temperature,
                        vibration,
                        pressure,
                        efficiency
                    ])
                    
            yield np.array(data)
            await asyncio.sleep(1)  # 1 Hz sampling
            
    @staticmethod
    async def generate_network_traffic_data(
        duration: timedelta,
        include_anomalies: bool = True
    ) -> AsyncIterator[np.ndarray]:
        """Generate network traffic data with realistic patterns"""
        # Network topology
        num_nodes = 100
        num_services = 10
        
        # Traffic patterns by time of day
        traffic_patterns = {
            'business_hours': {'base_rate': 1000, 'burst_prob': 0.1},
            'peak_hours': {'base_rate': 2000, 'burst_prob': 0.2},
            'off_hours': {'base_rate': 200, 'burst_prob': 0.02},
            'maintenance': {'base_rate': 50, 'burst_prob': 0.01}
        }
        
        # Service characteristics
        services = {
            'web': {'packet_size': 1500, 'latency': 50},
            'api': {'packet_size': 500, 'latency': 20},
            'database': {'packet_size': 4000, 'latency': 10},
            'streaming': {'packet_size': 8000, 'latency': 100},
            'monitoring': {'packet_size': 200, 'latency': 5}
        }
        
        start_time = time.time()
        
        # DDoS attack state
        ddos_active = False
        ddos_target = None
        
        while time.time() - start_time < duration.total_seconds():
            current_hour = int((time.time() % 86400) / 3600)
            
            # Determine traffic pattern
            if 9 <= current_hour <= 17:
                pattern = traffic_patterns['business_hours']
            elif 18 <= current_hour <= 20:
                pattern = traffic_patterns['peak_hours']
            elif 2 <= current_hour <= 4:
                pattern = traffic_patterns['maintenance']
            else:
                pattern = traffic_patterns['off_hours']
                
            data = []
            
            # Generate traffic flows
            num_flows = int(np.random.poisson(pattern['base_rate'] / 10))
            
            for _ in range(num_flows):
                src_node = random.randint(0, num_nodes - 1)
                dst_node = random.randint(0, num_nodes - 1)
                service_type = random.choice(list(services.keys()))
                service_params = services[service_type]
                
                # Normal traffic
                packet_count = np.random.poisson(100)
                bytes_transferred = packet_count * service_params['packet_size']
                latency = service_params['latency'] * (1 + np.random.exponential(0.1))
                
                # Anomalies
                if include_anomalies:
                    # DDoS attack
                    if not ddos_active and random.random() < 0.0001:
                        ddos_active = True
                        ddos_target = random.randint(0, num_nodes - 1)
                        logger.warning("ddos_attack_started", target=ddos_target)
                        
                    if ddos_active:
                        if dst_node == ddos_target:
                            packet_count *= 1000
                            bytes_transferred *= 1000
                            
                        # End DDoS
                        if random.random() < 0.01:
                            ddos_active = False
                            logger.info("ddos_attack_ended")
                            
                    # Port scan
                    if random.random() < 0.0001:
                        # Scanning pattern: many destinations, few packets each
                        for port in range(20):
                            data.append([
                                src_node,
                                (dst_node + port) % num_nodes,
                                time.time(),
                                1,  # Single packet
                                64,  # Small probe
                                1,  # Very low latency
                                hash(f"scan_{port}")
                            ])
                            
                data.append([
                    src_node,
                    dst_node,
                    time.time(),
                    packet_count,
                    bytes_transferred,
                    latency,
                    hash(service_type)
                ])
                
            yield np.array(data)
            await asyncio.sleep(0.1)  # 10 Hz monitoring


class LongRunningDegradationTest:
    """
    Extended tests for subtle performance degradation
    """
    
    def __init__(
        self,
        target_system: Any,
        duration: timedelta,
        checkpoint_interval: timedelta = timedelta(minutes=15)
    ):
        self.target_system = target_system
        self.duration = duration
        self.checkpoint_interval = checkpoint_interval
        self.degradation_indicators: List[Dict[str, Any]] = []
        self.baseline_established = False
        self.baseline_metrics: Dict[str, float] = {}
        
    async def run_with_degradation_detection(self) -> Dict[str, Any]:
        """Run test with advanced degradation detection"""
        logger.info(
            "starting_degradation_test",
            duration_hours=self.duration.total_seconds() / 3600
        )
        
        # Use production-like data
        data_generator = ProductionDataGenerator.generate_financial_trading_data(
            self.duration,
            include_anomalies=True
        )
        
        # Establish baseline
        await self._establish_baseline(data_generator)
        
        # Run main test
        start_time = datetime.now()
        checkpoint_task = asyncio.create_task(self._collect_degradation_metrics())
        
        try:
            async for data in data_generator:
                await self.target_system.process_batch(data)
                
                # Check for immediate issues
                await self._check_immediate_degradation()
                
        finally:
            checkpoint_task.cancel()
            
        # Analyze results
        end_time = datetime.now()
        analysis = self._analyze_degradation_patterns()
        
        return {
            'duration': (end_time - start_time).total_seconds(),
            'baseline_metrics': self.baseline_metrics,
            'degradation_indicators': self.degradation_indicators,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
        
    async def _establish_baseline(self, data_generator: AsyncIterator) -> None:
        """Establish performance baseline"""
        logger.info("establishing_baseline")
        
        baseline_duration = 300  # 5 minutes
        metrics = []
        
        start = time.time()
        async for data in data_generator:
            begin = time.perf_counter()
            await self.target_system.process_batch(data)
            elapsed = time.perf_counter() - begin
            
            metrics.append({
                'latency': elapsed,
                'throughput': len(data) / elapsed,
                'memory': psutil.Process().memory_info().rss / 1024 / 1024
            })
            
            if time.time() - start > baseline_duration:
                break
                
        # Calculate baseline
        self.baseline_metrics = {
            'latency_p50': np.percentile([m['latency'] for m in metrics], 50),
            'latency_p99': np.percentile([m['latency'] for m in metrics], 99),
            'throughput_mean': np.mean([m['throughput'] for m in metrics]),
            'memory_mean': np.mean([m['memory'] for m in metrics])
        }
        
        self.baseline_established = True
        logger.info("baseline_established", metrics=self.baseline_metrics)
        
    async def _collect_degradation_metrics(self) -> None:
        """Collect detailed degradation metrics"""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval.total_seconds())
                
                # Collect current metrics
                current_metrics = await self._get_current_metrics()
                
                # Compare with baseline
                degradation = self._calculate_degradation(current_metrics)
                
                if degradation:
                    self.degradation_indicators.append({
                        'timestamp': datetime.now(),
                        'metrics': current_metrics,
                        'degradation': degradation
                    })
                    
                    DEGRADATION_PATTERNS.observe(degradation['severity'])
                    
            except asyncio.CancelledError:
                break
                
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        # Run a small benchmark
        test_data = np.random.randn(1000, 3)
        
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            await self.target_system.process_batch(test_data)
            latencies.append(time.perf_counter() - start)
            
        return {
            'latency_p50': np.percentile(latencies, 50),
            'latency_p99': np.percentile(latencies, 99),
            'throughput_mean': len(test_data) / np.mean(latencies),
            'memory_current': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent(interval=1),
            'thread_count': threading.active_count(),
            'gc_stats': gc.get_stats()
        }
        
    def _calculate_degradation(self, current: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Calculate degradation from baseline"""
        if not self.baseline_established:
            return None
            
        degradation = {}
        severity = 0
        
        # Latency degradation
        latency_increase = (current['latency_p99'] - self.baseline_metrics['latency_p99']) / self.baseline_metrics['latency_p99']
        if latency_increase > 0.1:  # 10% increase
            degradation['latency_increase_percent'] = latency_increase * 100
            severity += latency_increase
            
        # Throughput degradation
        throughput_decrease = (self.baseline_metrics['throughput_mean'] - current['throughput_mean']) / self.baseline_metrics['throughput_mean']
        if throughput_decrease > 0.1:
            degradation['throughput_decrease_percent'] = throughput_decrease * 100
            severity += throughput_decrease
            
        # Memory growth
        memory_growth = current['memory_current'] - self.baseline_metrics['memory_mean']
        if memory_growth > 100:  # 100MB growth
            degradation['memory_growth_mb'] = memory_growth
            severity += memory_growth / 1000
            
        if degradation:
            degradation['severity'] = severity
            return degradation
            
        return None
        
    async def _check_immediate_degradation(self) -> None:
        """Check for immediate performance issues"""
        # This would integrate with system monitoring
        pass
        
    def _analyze_degradation_patterns(self) -> Dict[str, Any]:
        """Analyze degradation patterns over time"""
        if not self.degradation_indicators:
            return {'status': 'healthy', 'patterns': []}
            
        patterns = []
        
        # Linear degradation
        timestamps = [d['timestamp'] for d in self.degradation_indicators]
        severities = [d['degradation']['severity'] for d in self.degradation_indicators]
        
        if len(severities) > 2:
            # Calculate trend
            x = np.arange(len(severities))
            slope, intercept = np.polyfit(x, severities, 1)
            
            if slope > 0.01:  # Positive trend
                patterns.append({
                    'type': 'linear_degradation',
                    'rate': slope,
                    'projected_failure_hours': (1.0 - intercept) / slope if slope > 0 else None
                })
                
        # Sudden jumps
        for i in range(1, len(severities)):
            if severities[i] - severities[i-1] > 0.2:
                patterns.append({
                    'type': 'sudden_degradation',
                    'timestamp': timestamps[i],
                    'severity_jump': severities[i] - severities[i-1]
                })
                
        # Cyclic patterns
        if len(severities) > 10:
            # Simple FFT to detect cycles
            fft = np.fft.fft(severities)
            frequencies = np.fft.fftfreq(len(severities))
            
            # Find dominant frequency
            dominant_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            if np.abs(fft[dominant_idx]) > len(severities) * 0.1:
                patterns.append({
                    'type': 'cyclic_degradation',
                    'period_hours': 1 / frequencies[dominant_idx] * self.checkpoint_interval.total_seconds() / 3600
                })
                
        return {
            'status': 'degrading' if patterns else 'stable',
            'patterns': patterns,
            'max_severity': max(severities) if severities else 0,
            'final_severity': severities[-1] if severities else 0
        }
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis['status'] == 'degrading':
            for pattern in analysis['patterns']:
                if pattern['type'] == 'linear_degradation':
                    if pattern.get('projected_failure_hours', float('inf')) < 24:
                        recommendations.append(
                            "CRITICAL: Linear degradation detected. "
                            f"Projected failure in {pattern['projected_failure_hours']:.1f} hours. "
                            "Immediate investigation required."
                        )
                    else:
                        recommendations.append(
                            "WARNING: Gradual performance degradation detected. "
                            f"Rate: {pattern['rate']:.3f} severity/checkpoint. "
                            "Schedule maintenance window."
                        )
                        
                elif pattern['type'] == 'sudden_degradation':
                    recommendations.append(
                        f"ALERT: Sudden degradation at {pattern['timestamp']}. "
                        f"Severity jump: {pattern['severity_jump']:.2f}. "
                        "Check logs for errors or resource constraints."
                    )
                    
                elif pattern['type'] == 'cyclic_degradation':
                    recommendations.append(
                        f"INFO: Cyclic degradation pattern detected. "
                        f"Period: {pattern['period_hours']:.1f} hours. "
                        "May be related to scheduled jobs or traffic patterns."
                    )
                    
        # Memory specific
        memory_indicators = [
            d for d in self.degradation_indicators
            if 'memory_growth_mb' in d.get('degradation', {})
        ]
        if memory_indicators:
            total_growth = memory_indicators[-1]['degradation']['memory_growth_mb']
            recommendations.append(
                f"MEMORY: Total growth: {total_growth:.1f}MB. "
                "Investigate potential memory leaks."
            )
            
        return recommendations


# Advanced chaos scenarios
PRODUCTION_CHAOS_SCENARIOS = [
    ChaosScenario(
        name="cloud_regional_outage",
        fault_types=[AdvancedFaultType.CLOUD_SERVICE_OUTAGE],
        duration=timedelta(minutes=30),
        intensity=0.8,
        targets=["us-east-1"]
    ),
    ChaosScenario(
        name="kafka_rebalance_storm",
        fault_types=[AdvancedFaultType.KAFKA_REBALANCE_STORM],
        duration=timedelta(minutes=15),
        intensity=0.9,
        targets=["consumer-group-1"]
    ),
    ChaosScenario(
        name="intermittent_network_issues",
        fault_types=[AdvancedFaultType.INTERMITTENT_NETWORK],
        duration=timedelta(hours=1),
        intensity=0.6,
        targets=["kafka", "schema-registry", "monitoring"]
    ),
    ChaosScenario(
        name="noisy_neighbor_impact",
        fault_types=[AdvancedFaultType.NOISY_NEIGHBOR],
        duration=timedelta(hours=2),
        intensity=0.7,
        targets=["compute", "network"]
    )
]


if __name__ == "__main__":
    import asyncio
    import threading
    import gc
    
    async def run_advanced_chaos():
        # Initialize components
        network = NetworkChaosInjector()
        kafka = KafkaChaosInjector("http://localhost:8080")
        system = SystemChaosInjector()
        
        # Cloud provider config
        aws_provider = CloudProvider(
            name="AWS",
            region="us-east-1",
            availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
            services={
                "kafka": {
                    "us-east-1a": ["broker-1", "broker-2"],
                    "us-east-1b": ["broker-3", "broker-4"],
                    "us-east-1c": ["broker-5", "broker-6"]
                }
            }
        )
        
        # Advanced chaos components
        intermittent = IntermittentNetworkChaos(network)
        cloud_chaos = CloudServiceChaos(aws_provider)
        kafka_storm = KafkaRebalanceStorm(kafka)
        
        # Run scenarios
        logger.info("starting_advanced_chaos_suite")
        
        # Test 1: Intermittent network issues
        await intermittent.inject_intermittent_failure(
            "kafka",
            "flapping",
            timedelta(minutes=10),
            0.7
        )
        
        # Test 2: Cloud service outage
        await cloud_chaos.inject_service_outage(
            "kafka",
            "zonal",
            timedelta(minutes=5),
            partial=True
        )
        
        # Test 3: Kafka rebalance storm
        await kafka_storm.inject_rebalance_storm(
            "streaming-tda-consumers",
            0.8,
            timedelta(minutes=10)
        )
        
        logger.info("advanced_chaos_suite_complete")
        
    asyncio.run(run_advanced_chaos())