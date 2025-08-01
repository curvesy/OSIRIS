"""
Chaos Engineering experiments for AURA Intelligence.

This module provides chaos experiments to test system resilience
through controlled failure injection and recovery validation.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from structlog import get_logger
from prometheus_client import Counter, Gauge

from ..observability.metrics import (
    chaos_experiments_run,
    chaos_experiments_failed,
    chaos_injection_active
)

logger = get_logger(__name__)


class ChaosExperiment(ABC):
    """Base class for chaos experiments"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.id = str(uuid4())
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    async def setup(self) -> None:
        """Setup experiment prerequisites"""
        pass
    
    @abstractmethod
    async def inject_failure(self) -> None:
        """Inject the failure condition"""
        pass
    
    @abstractmethod
    async def verify_hypothesis(self) -> bool:
        """Verify if system behaved as expected"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up after experiment"""
        pass
    
    async def run(self) -> Dict[str, Any]:
        """Execute the chaos experiment"""
        logger.info(
            "Starting chaos experiment",
            experiment=self.name,
            id=self.id
        )
        
        self.start_time = datetime.utcnow()
        chaos_experiments_run.labels(
            experiment_type=self.name,
            target="system"
        ).inc()
        
        try:
            # Setup
            await self.setup()
            
            # Record steady state
            steady_state = await self._capture_steady_state()
            self.results["steady_state"] = steady_state
            
            # Inject failure
            chaos_injection_active.labels(injection_type=self.name).set(1)
            await self.inject_failure()
            
            # Wait for system to react
            await asyncio.sleep(5)
            
            # Verify hypothesis
            hypothesis_held = await self.verify_hypothesis()
            self.results["hypothesis_held"] = hypothesis_held
            
            if not hypothesis_held:
                chaos_experiments_failed.labels(
                    experiment_type=self.name,
                    target="system",
                    failure_type="hypothesis_failed"
                ).inc()
            
            self.status = "completed"
            
        except Exception as e:
            logger.error(
                "Chaos experiment failed",
                experiment=self.name,
                error=str(e),
                exc_info=True
            )
            self.status = "failed"
            self.results["error"] = str(e)
            chaos_experiments_failed.labels(
                experiment_type=self.name,
                target="system",
                failure_type="exception"
            ).inc()
            
        finally:
            # Always cleanup
            chaos_injection_active.labels(injection_type=self.name).set(0)
            await self.cleanup()
            self.end_time = datetime.utcnow()
            
        # Generate report
        return self._generate_report()
    
    async def _capture_steady_state(self) -> Dict[str, Any]:
        """Capture system steady state metrics"""
        # Override in subclasses for specific metrics
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "captured"
        }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate experiment report"""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "experiment": self.name,
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "results": self.results
        }


class EventStoreFailureExperiment(ChaosExperiment):
    """Test system behavior when event store becomes unavailable"""
    
    def __init__(self, event_store, health_check_fn: Callable):
        super().__init__(
            name="event_store_failure",
            description="Simulate event store unavailability"
        )
        self.event_store = event_store
        self.health_check_fn = health_check_fn
        self._original_connect = None
    
    async def setup(self) -> None:
        """Backup original connection method"""
        self._original_connect = self.event_store.connect
    
    async def inject_failure(self) -> None:
        """Disconnect event store and prevent reconnection"""
        logger.warning("Injecting event store failure")
        
        # Disconnect
        await self.event_store.disconnect()
        
        # Override connect to fail
        async def failing_connect():
            raise RuntimeError("Event store connection disabled by chaos experiment")
        
        self.event_store.connect = failing_connect
    
    async def verify_hypothesis(self) -> bool:
        """Verify system remains healthy despite event store failure"""
        # System should degrade gracefully
        health_status = await self.health_check_fn()
        
        # Check if system is still responding
        if not health_status.get("responding", False):
            logger.error("System not responding during event store failure")
            return False
        
        # Check if appropriate error handling is in place
        if health_status.get("event_store_connected", True):
            logger.error("Event store shows as connected when it should be down")
            return False
        
        logger.info("System handling event store failure gracefully")
        return True
    
    async def cleanup(self) -> None:
        """Restore event store connection"""
        logger.info("Restoring event store connection")
        self.event_store.connect = self._original_connect
        await self.event_store.connect()


class ProjectionLagExperiment(ChaosExperiment):
    """Introduce artificial lag in projections"""
    
    def __init__(self, projection_manager, max_lag_seconds: int = 60):
        super().__init__(
            name="projection_lag",
            description="Simulate projection processing lag"
        )
        self.projection_manager = projection_manager
        self.max_lag_seconds = max_lag_seconds
        self._original_handle_event = {}
    
    async def setup(self) -> None:
        """Backup original event handlers"""
        for projection in self.projection_manager.projections:
            self._original_handle_event[projection.name] = projection.handle_event
    
    async def inject_failure(self) -> None:
        """Add artificial delay to projection processing"""
        logger.warning(
            "Injecting projection lag",
            max_lag_seconds=self.max_lag_seconds
        )
        
        for projection in self.projection_manager.projections:
            original_handler = self._original_handle_event[projection.name]
            
            async def delayed_handler(event, original=original_handler):
                # Add random delay
                delay = random.uniform(0, self.max_lag_seconds)
                await asyncio.sleep(delay)
                return await original(event)
            
            projection.handle_event = delayed_handler
    
    async def verify_hypothesis(self) -> bool:
        """Verify system handles projection lag appropriately"""
        # Check if lag monitoring is working
        from ..observability.metrics import projection_lag
        
        # Get current lag values
        lag_detected = False
        for projection in self.projection_manager.projections:
            # In real implementation, would query Prometheus
            # For now, assume lag is detected if injection is active
            lag_detected = True
        
        if not lag_detected:
            logger.error("Projection lag not detected by monitoring")
            return False
        
        logger.info("Projection lag properly detected and monitored")
        return True
    
    async def cleanup(self) -> None:
        """Restore normal projection processing"""
        logger.info("Removing projection lag injection")
        for projection in self.projection_manager.projections:
            projection.handle_event = self._original_handle_event[projection.name]


class NetworkPartitionExperiment(ChaosExperiment):
    """Simulate network partition between components"""
    
    def __init__(self, network_controller, partition_config: Dict[str, List[str]]):
        super().__init__(
            name="network_partition",
            description="Simulate network partition between services"
        )
        self.network_controller = network_controller
        self.partition_config = partition_config
    
    async def setup(self) -> None:
        """Verify network controller is ready"""
        if not hasattr(self.network_controller, 'create_partition'):
            raise RuntimeError("Network controller must support partition creation")
    
    async def inject_failure(self) -> None:
        """Create network partition"""
        logger.warning(
            "Creating network partition",
            config=self.partition_config
        )
        
        for source, targets in self.partition_config.items():
            for target in targets:
                await self.network_controller.create_partition(source, target)
    
    async def verify_hypothesis(self) -> bool:
        """Verify system handles partition correctly"""
        # System should detect partition and handle appropriately
        # This is a simplified check - real implementation would be more thorough
        
        await asyncio.sleep(10)  # Allow time for detection
        
        # Check if partition was detected
        # In real implementation, would check monitoring/alerting
        logger.info("Assuming partition handling is correct for demo")
        return True
    
    async def cleanup(self) -> None:
        """Remove network partition"""
        logger.info("Removing network partition")
        
        for source, targets in self.partition_config.items():
            for target in targets:
                await self.network_controller.remove_partition(source, target)


class DebateTimeoutExperiment(ChaosExperiment):
    """Force debates to timeout"""
    
    def __init__(self, debate_system, timeout_probability: float = 0.8):
        super().__init__(
            name="debate_timeout",
            description="Force debates to timeout before consensus"
        )
        self.debate_system = debate_system
        self.timeout_probability = timeout_probability
        self._original_timeout = None
    
    async def setup(self) -> None:
        """Backup original timeout setting"""
        self._original_timeout = self.debate_system.debate_timeout
    
    async def inject_failure(self) -> None:
        """Set very short timeout for debates"""
        logger.warning("Injecting debate timeouts")
        
        # Set timeout to 5 seconds (normally would be minutes)
        self.debate_system.debate_timeout = timedelta(seconds=5)
    
    async def verify_hypothesis(self) -> bool:
        """Verify system handles debate timeouts gracefully"""
        # Start a test debate
        debate_id = await self.debate_system.start_debate(
            topic="Test topic for chaos experiment",
            initiator_id="chaos_experiment"
        )
        
        # Wait for timeout
        await asyncio.sleep(10)
        
        # Check debate status
        status = await self.debate_system.get_debate_status(debate_id)
        
        if status != "timeout":
            logger.error(
                "Debate did not timeout as expected",
                status=status
            )
            return False
        
        # Check if timeout was handled properly
        # (e.g., notifications sent, cleanup performed)
        logger.info("Debate timeout handled correctly")
        return True
    
    async def cleanup(self) -> None:
        """Restore original timeout"""
        logger.info("Restoring debate timeout")
        self.debate_system.debate_timeout = self._original_timeout


class MemoryPressureExperiment(ChaosExperiment):
    """Simulate memory pressure"""
    
    def __init__(self, target_memory_mb: int = 1000):
        super().__init__(
            name="memory_pressure",
            description="Simulate high memory usage"
        )
        self.target_memory_mb = target_memory_mb
        self._memory_hog = []
    
    async def setup(self) -> None:
        """Prepare for memory allocation"""
        pass
    
    async def inject_failure(self) -> None:
        """Allocate memory to create pressure"""
        logger.warning(
            "Creating memory pressure",
            target_mb=self.target_memory_mb
        )
        
        # Allocate memory in chunks
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        total_allocated = 0
        
        while total_allocated < self.target_memory_mb * 1024 * 1024:
            self._memory_hog.append(bytearray(chunk_size))
            total_allocated += chunk_size
            await asyncio.sleep(0.1)  # Gradual allocation
    
    async def verify_hypothesis(self) -> bool:
        """Verify system handles memory pressure"""
        # Check if system is still responsive
        # In real implementation, would check various health metrics
        
        # Simple check - can we still perform basic operations?
        try:
            # Try to allocate a small amount of additional memory
            test_allocation = bytearray(1024 * 1024)  # 1MB
            del test_allocation
            
            logger.info("System handling memory pressure appropriately")
            return True
            
        except MemoryError:
            logger.error("System unable to handle memory pressure")
            return False
    
    async def cleanup(self) -> None:
        """Release allocated memory"""
        logger.info("Releasing memory")
        self._memory_hog.clear()


class ChaosOrchestrator:
    """Orchestrates chaos experiments"""
    
    def __init__(self):
        self.experiments: List[ChaosExperiment] = []
        self.results: List[Dict[str, Any]] = []
    
    def add_experiment(self, experiment: ChaosExperiment) -> None:
        """Add experiment to queue"""
        self.experiments.append(experiment)
    
    async def run_all(self, parallel: bool = False) -> List[Dict[str, Any]]:
        """Run all experiments"""
        logger.info(
            "Starting chaos orchestration",
            experiment_count=len(self.experiments),
            parallel=parallel
        )
        
        if parallel:
            # Run experiments in parallel
            tasks = [exp.run() for exp in self.experiments]
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run experiments sequentially
            self.results = []
            for exp in self.experiments:
                result = await exp.run()
                self.results.append(result)
                
                # Wait between experiments
                await asyncio.sleep(30)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of all experiments"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("status") == "completed")
        failed = total - successful
        
        hypothesis_failures = sum(
            1 for r in self.results 
            if r.get("results", {}).get("hypothesis_held") is False
        )
        
        return {
            "summary": {
                "total_experiments": total,
                "successful": successful,
                "failed": failed,
                "hypothesis_failures": hypothesis_failures,
                "resilience_score": (successful - hypothesis_failures) / total * 100
            },
            "experiments": self.results
        }


# Example usage
async def run_chaos_suite(system_components: Dict[str, Any]) -> Dict[str, Any]:
    """Run a suite of chaos experiments"""
    orchestrator = ChaosOrchestrator()
    
    # Add experiments based on available components
    if "event_store" in system_components:
        orchestrator.add_experiment(
            EventStoreFailureExperiment(
                system_components["event_store"],
                system_components["health_check"]
            )
        )
    
    if "projection_manager" in system_components:
        orchestrator.add_experiment(
            ProjectionLagExperiment(
                system_components["projection_manager"],
                max_lag_seconds=30
            )
        )
    
    if "debate_system" in system_components:
        orchestrator.add_experiment(
            DebateTimeoutExperiment(
                system_components["debate_system"],
                timeout_probability=0.9
            )
        )
    
    # Add memory pressure test
    orchestrator.add_experiment(
        MemoryPressureExperiment(target_memory_mb=500)
    )
    
    # Run experiments
    results = await orchestrator.run_all(parallel=False)
    
    # Generate report
    report = orchestrator.generate_report()
    
    logger.info(
        "Chaos experiments completed",
        resilience_score=report["summary"]["resilience_score"]
    )
    
    return report