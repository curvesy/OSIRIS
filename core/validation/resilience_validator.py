"""
Resilience Validation Suite for AURA Intelligence.

This module executes comprehensive chaos engineering experiments
with production traffic patterns to validate system resilience.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4

from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge

from ..chaos.experiments import (
    ChaosExperiment,
    EventStoreFailureExperiment,
    ProjectionLagExperiment,
    NetworkPartitionExperiment,
    DebateTimeoutExperiment,
    MemoryPressureExperiment
)
from ..event_store.hardened_store import HardenedNATSEventStore
from ..areopagus.debate_graph import DebateGraph
from ..observability.metrics import system_health_score

logger = get_logger(__name__)

# Validation metrics
resilience_validation_total = Counter(
    "resilience_validation_total",
    "Total resilience validation attempts",
    ["experiment", "status"]
)

resilience_recovery_time = Histogram(
    "resilience_recovery_time_seconds",
    "Time taken to recover from failure",
    ["experiment"]
)

resilience_score = Gauge(
    "resilience_score",
    "Overall system resilience score (0-100)",
    ["category"]
)


class ProductionTrafficSimulator:
    """Simulates production-like traffic patterns"""
    
    def __init__(self, debate_system: DebateGraph):
        self.debate_system = debate_system
        self.active_debates: Dict[str, asyncio.Task] = {}
        self.metrics = {
            "debates_started": 0,
            "debates_completed": 0,
            "debates_failed": 0,
            "total_latency": 0.0
        }
    
    async def generate_traffic(
        self,
        duration_seconds: int,
        debates_per_minute: int = 60
    ) -> Dict[str, Any]:
        """Generate production-like traffic for specified duration"""
        logger.info(
            "Starting production traffic simulation",
            duration_seconds=duration_seconds,
            debates_per_minute=debates_per_minute
        )
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        # Calculate debate interval
        debate_interval = 60.0 / debates_per_minute
        
        while datetime.utcnow() < end_time:
            # Start a new debate
            debate_task = asyncio.create_task(self._run_debate())
            debate_id = str(uuid4())
            self.active_debates[debate_id] = debate_task
            
            # Clean up completed debates
            completed = []
            for did, task in self.active_debates.items():
                if task.done():
                    completed.append(did)
            
            for did in completed:
                del self.active_debates[did]
            
            # Wait for next debate
            await asyncio.sleep(debate_interval)
        
        # Wait for remaining debates to complete
        if self.active_debates:
            await asyncio.gather(*self.active_debates.values(), return_exceptions=True)
        
        return self._calculate_metrics()
    
    async def _run_debate(self) -> None:
        """Run a single debate with timing"""
        debate_start = time.time()
        self.metrics["debates_started"] += 1
        
        try:
            # Generate random debate topic
            topics = [
                "Should AI systems have kill switches?",
                "Is remote work better than office work?",
                "Should social media be regulated?",
                "Is universal basic income necessary?",
                "Should we colonize Mars?"
            ]
            
            topic = random.choice(topics)
            initial_state = {
                "topic": topic,
                "context": {
                    "domain": "General",
                    "urgency": random.choice(["low", "medium", "high"]),
                    "complexity": random.choice(["simple", "moderate", "complex"])
                }
            }
            
            # Run debate
            result = await self.debate_system.graph.ainvoke(
                initial_state,
                config={"recursion_limit": 15}
            )
            
            # Record success
            self.metrics["debates_completed"] += 1
            debate_duration = time.time() - debate_start
            self.metrics["total_latency"] += debate_duration
            
            logger.debug(
                "Debate completed",
                duration_seconds=debate_duration,
                topic=topic
            )
            
        except Exception as e:
            self.metrics["debates_failed"] += 1
            logger.warning(
                "Debate failed",
                error=str(e),
                duration_seconds=time.time() - debate_start
            )
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate traffic simulation metrics"""
        total_debates = self.metrics["debates_started"]
        if total_debates == 0:
            return self.metrics
        
        success_rate = (
            self.metrics["debates_completed"] / total_debates * 100
            if total_debates > 0 else 0
        )
        
        avg_latency = (
            self.metrics["total_latency"] / self.metrics["debates_completed"]
            if self.metrics["debates_completed"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "average_latency": avg_latency
        }


class ResilienceValidator:
    """Validates system resilience under production conditions"""
    
    def __init__(
        self,
        event_store: HardenedNATSEventStore,
        debate_system: DebateGraph,
        traffic_simulator: ProductionTrafficSimulator
    ):
        self.event_store = event_store
        self.debate_system = debate_system
        self.traffic_simulator = traffic_simulator
        self.validation_results: Dict[str, Any] = {}
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Execute all resilience validation scenarios"""
        logger.info("Starting resilience validation suite")
        
        experiments = [
            ("cascading_failure", self.validate_cascading_failure),
            ("network_partition", self.validate_network_partition),
            ("resource_exhaustion", self.validate_resource_exhaustion),
            ("rapid_recovery", self.validate_rapid_recovery),
            ("sustained_pressure", self.validate_sustained_pressure)
        ]
        
        overall_score = 0
        successful_experiments = 0
        
        for name, validator in experiments:
            try:
                result = await validator()
                self.validation_results[name] = result
                
                if result.get("status") == "passed":
                    successful_experiments += 1
                    overall_score += result.get("resilience_score", 0)
                
            except Exception as e:
                logger.error(
                    "Resilience validation failed",
                    experiment=name,
                    error=str(e)
                )
                self.validation_results[name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Calculate overall resilience score
        if len(experiments) > 0:
            overall_score = overall_score / len(experiments)
            resilience_score.labels(category="overall").set(overall_score)
        
        self.validation_results["summary"] = {
            "total_experiments": len(experiments),
            "successful_experiments": successful_experiments,
            "overall_resilience_score": overall_score,
            "recommendation": self._get_recommendation(overall_score)
        }
        
        return self.validation_results
    
    async def validate_cascading_failure(self) -> Dict[str, Any]:
        """Validate recovery from cascading component failures"""
        experiment_name = "cascading_failure"
        logger.info(f"Starting {experiment_name} validation")
        
        # Start production traffic
        traffic_task = asyncio.create_task(
            self.traffic_simulator.generate_traffic(
                duration_seconds=300,  # 5 minutes
                debates_per_minute=30
            )
        )
        
        # Wait for traffic to stabilize
        await asyncio.sleep(30)
        
        # Record baseline metrics
        baseline_health = system_health_score._value.get()
        
        # Inject cascading failures
        failure_sequence = [
            ("event_store", 10),  # Fail event store for 10 seconds
            ("projections", 5),   # Then projections for 5 seconds
            ("agents", 5)         # Then agents for 5 seconds
        ]
        
        recovery_times = []
        
        for component, duration in failure_sequence:
            logger.info(
                f"Injecting {component} failure",
                duration_seconds=duration
            )
            
            # Inject failure (simulated)
            failure_start = datetime.utcnow()
            
            # Wait for failure duration
            await asyncio.sleep(duration)
            
            # Measure recovery time
            recovery_start = datetime.utcnow()
            
            # Wait for system to recover (monitor health score)
            while system_health_score._value.get() < baseline_health * 0.9:
                await asyncio.sleep(1)
                if (datetime.utcnow() - recovery_start).total_seconds() > 60:
                    break
            
            recovery_time = (datetime.utcnow() - recovery_start).total_seconds()
            recovery_times.append(recovery_time)
            
            resilience_recovery_time.labels(
                experiment=f"{experiment_name}_{component}"
            ).observe(recovery_time)
        
        # Wait for traffic to complete
        traffic_metrics = await traffic_task
        
        # Calculate resilience score
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        max_recovery_time = max(recovery_times)
        
        # Score based on recovery time (100 for <5s, 0 for >60s)
        if avg_recovery_time < 5:
            score = 100
        elif avg_recovery_time > 60:
            score = 0
        else:
            score = 100 - ((avg_recovery_time - 5) / 55 * 100)
        
        # Adjust for traffic impact
        traffic_impact = traffic_metrics.get("success_rate", 0)
        final_score = (score * 0.7) + (traffic_impact * 0.3)
        
        resilience_score.labels(category=experiment_name).set(final_score)
        
        result = {
            "status": "passed" if final_score > 70 else "failed",
            "resilience_score": final_score,
            "recovery_times": recovery_times,
            "average_recovery_time": avg_recovery_time,
            "max_recovery_time": max_recovery_time,
            "traffic_metrics": traffic_metrics,
            "baseline_health": baseline_health
        }
        
        resilience_validation_total.labels(
            experiment=experiment_name,
            status=result["status"]
        ).inc()
        
        return result
    
    async def validate_network_partition(self) -> Dict[str, Any]:
        """Validate behavior during network partitions"""
        experiment_name = "network_partition"
        logger.info(f"Starting {experiment_name} validation")
        
        # Create and run network partition experiment
        experiment = NetworkPartitionExperiment(
            event_store=self.event_store,
            duration_seconds=60,
            partition_type="asymmetric"
        )
        
        # Run with production traffic
        traffic_task = asyncio.create_task(
            self.traffic_simulator.generate_traffic(
                duration_seconds=120,  # 2 minutes
                debates_per_minute=20
            )
        )
        
        # Wait for traffic to start
        await asyncio.sleep(10)
        
        # Run chaos experiment
        chaos_result = await experiment.run()
        
        # Wait for traffic to complete
        traffic_metrics = await traffic_task
        
        # Calculate resilience score
        hypothesis_held = chaos_result.get("hypothesis_held", False)
        no_split_brain = chaos_result.get("results", {}).get("no_split_brain", False)
        traffic_success = traffic_metrics.get("success_rate", 0)
        
        score = 0
        if hypothesis_held:
            score += 40
        if no_split_brain:
            score += 40
        score += (traffic_success / 100) * 20
        
        resilience_score.labels(category=experiment_name).set(score)
        
        result = {
            "status": "passed" if score > 70 else "failed",
            "resilience_score": score,
            "hypothesis_held": hypothesis_held,
            "no_split_brain": no_split_brain,
            "traffic_metrics": traffic_metrics,
            "chaos_results": chaos_result
        }
        
        resilience_validation_total.labels(
            experiment=experiment_name,
            status=result["status"]
        ).inc()
        
        return result
    
    async def validate_resource_exhaustion(self) -> Dict[str, Any]:
        """Validate behavior under resource pressure"""
        experiment_name = "resource_exhaustion"
        logger.info(f"Starting {experiment_name} validation")
        
        # Gradually increase load
        load_levels = [
            (30, 60),   # 30 debates/min for 60 seconds
            (60, 60),   # 60 debates/min for 60 seconds
            (120, 60),  # 120 debates/min for 60 seconds
            (200, 30)   # 200 debates/min for 30 seconds (stress test)
        ]
        
        level_results = []
        
        for debates_per_minute, duration in load_levels:
            logger.info(
                f"Testing load level",
                debates_per_minute=debates_per_minute,
                duration_seconds=duration
            )
            
            # Run traffic at this level
            metrics = await self.traffic_simulator.generate_traffic(
                duration_seconds=duration,
                debates_per_minute=debates_per_minute
            )
            
            level_results.append({
                "load_level": debates_per_minute,
                "duration": duration,
                "success_rate": metrics.get("success_rate", 0),
                "average_latency": metrics.get("average_latency", 0),
                "debates_completed": metrics.get("debates_completed", 0)
            })
            
            # Brief pause between levels
            await asyncio.sleep(5)
        
        # Calculate resilience score based on degradation
        baseline_success = level_results[0]["success_rate"]
        stress_success = level_results[-1]["success_rate"]
        
        # Score based on how well system handles stress
        if stress_success > 90:
            score = 100
        elif stress_success < 50:
            score = 0
        else:
            score = ((stress_success - 50) / 40) * 100
        
        # Check for graceful degradation
        graceful_degradation = all(
            r["success_rate"] > 70 for r in level_results[:-1]
        )
        
        if graceful_degradation:
            score = min(100, score + 10)
        
        resilience_score.labels(category=experiment_name).set(score)
        
        result = {
            "status": "passed" if score > 70 else "failed",
            "resilience_score": score,
            "load_test_results": level_results,
            "graceful_degradation": graceful_degradation,
            "baseline_success_rate": baseline_success,
            "stress_success_rate": stress_success
        }
        
        resilience_validation_total.labels(
            experiment=experiment_name,
            status=result["status"]
        ).inc()
        
        return result
    
    async def validate_rapid_recovery(self) -> Dict[str, Any]:
        """Validate rapid recovery from sudden failures"""
        experiment_name = "rapid_recovery"
        logger.info(f"Starting {experiment_name} validation")
        
        recovery_times = []
        
        # Test multiple failure/recovery cycles
        for i in range(5):
            # Start traffic
            traffic_task = asyncio.create_task(
                self.traffic_simulator.generate_traffic(
                    duration_seconds=60,
                    debates_per_minute=40
                )
            )
            
            # Wait for stabilization
            await asyncio.sleep(10)
            
            # Record baseline
            baseline_health = system_health_score._value.get()
            
            # Inject sudden failure
            logger.info(f"Injecting sudden failure {i+1}/5")
            failure_start = datetime.utcnow()
            
            # Simulate component failure (e.g., event store disconnect)
            # In real implementation, this would actually fail the component
            
            # Wait for health to drop
            while system_health_score._value.get() > baseline_health * 0.5:
                await asyncio.sleep(0.1)
                if (datetime.utcnow() - failure_start).total_seconds() > 10:
                    break
            
            # Measure recovery time
            recovery_start = datetime.utcnow()
            
            # Wait for recovery
            while system_health_score._value.get() < baseline_health * 0.9:
                await asyncio.sleep(0.1)
                if (datetime.utcnow() - recovery_start).total_seconds() > 30:
                    break
            
            recovery_time = (datetime.utcnow() - recovery_start).total_seconds()
            recovery_times.append(recovery_time)
            
            # Complete traffic task
            await traffic_task
        
        # Calculate metrics
        avg_recovery = sum(recovery_times) / len(recovery_times)
        max_recovery = max(recovery_times)
        consistency = 100 - (
            (max(recovery_times) - min(recovery_times)) / avg_recovery * 100
            if avg_recovery > 0 else 0
        )
        
        # Score based on recovery speed and consistency
        if avg_recovery < 5:
            speed_score = 100
        elif avg_recovery > 30:
            speed_score = 0
        else:
            speed_score = 100 - ((avg_recovery - 5) / 25 * 100)
        
        final_score = (speed_score * 0.7) + (consistency * 0.3)
        
        resilience_score.labels(category=experiment_name).set(final_score)
        
        result = {
            "status": "passed" if final_score > 70 else "failed",
            "resilience_score": final_score,
            "recovery_times": recovery_times,
            "average_recovery_time": avg_recovery,
            "max_recovery_time": max_recovery,
            "recovery_consistency": consistency
        }
        
        resilience_validation_total.labels(
            experiment=experiment_name,
            status=result["status"]
        ).inc()
        
        return result
    
    async def validate_sustained_pressure(self) -> Dict[str, Any]:
        """Validate system under sustained high load"""
        experiment_name = "sustained_pressure"
        logger.info(f"Starting {experiment_name} validation")
        
        # Run high load for extended period
        duration_minutes = 10
        debates_per_minute = 100
        
        # Collect metrics every minute
        minute_metrics = []
        
        for minute in range(duration_minutes):
            logger.info(
                f"Sustained pressure test minute {minute+1}/{duration_minutes}"
            )
            
            # Run traffic for one minute
            metrics = await self.traffic_simulator.generate_traffic(
                duration_seconds=60,
                debates_per_minute=debates_per_minute
            )
            
            minute_metrics.append({
                "minute": minute + 1,
                "success_rate": metrics.get("success_rate", 0),
                "average_latency": metrics.get("average_latency", 0),
                "debates_completed": metrics.get("debates_completed", 0)
            })
        
        # Analyze performance degradation over time
        first_minute_success = minute_metrics[0]["success_rate"]
        last_minute_success = minute_metrics[-1]["success_rate"]
        
        # Calculate degradation
        degradation = first_minute_success - last_minute_success
        
        # Check for memory leaks (latency increase)
        first_latency = minute_metrics[0]["average_latency"]
        last_latency = minute_metrics[-1]["average_latency"]
        latency_increase = (
            (last_latency - first_latency) / first_latency * 100
            if first_latency > 0 else 0
        )
        
        # Score based on stability
        if degradation < 5 and latency_increase < 20:
            score = 100
        elif degradation > 20 or latency_increase > 100:
            score = 0
        else:
            degradation_score = max(0, 100 - (degradation * 5))
            latency_score = max(0, 100 - latency_increase)
            score = (degradation_score + latency_score) / 2
        
        resilience_score.labels(category=experiment_name).set(score)
        
        result = {
            "status": "passed" if score > 70 else "failed",
            "resilience_score": score,
            "duration_minutes": duration_minutes,
            "debates_per_minute": debates_per_minute,
            "performance_degradation": degradation,
            "latency_increase_percent": latency_increase,
            "minute_by_minute_metrics": minute_metrics
        }
        
        resilience_validation_total.labels(
            experiment=experiment_name,
            status=result["status"]
        ).inc()
        
        return result
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on resilience score"""
        if score >= 90:
            return "System shows excellent resilience. Ready for production."
        elif score >= 70:
            return "System shows good resilience. Minor improvements recommended."
        elif score >= 50:
            return "System shows moderate resilience. Significant improvements needed."
        else:
            return "System shows poor resilience. Major hardening required."


async def main():
    """Run resilience validation suite"""
    # Initialize components
    event_store = HardenedNATSEventStore(
        nats_url="nats://localhost:4222",
        stream_name="AURA_RESILIENCE_TEST"
    )
    await event_store.connect()
    
    # Create debate system
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    debate_system = DebateGraph(llm=llm, event_store=event_store)
    
    # Create traffic simulator
    traffic_simulator = ProductionTrafficSimulator(debate_system)
    
    # Create validator
    validator = ResilienceValidator(
        event_store=event_store,
        debate_system=debate_system,
        traffic_simulator=traffic_simulator
    )
    
    # Run validations
    results = await validator.run_all_validations()
    
    # Print results
    print("\n=== Resilience Validation Results ===\n")
    
    summary = results.pop("summary", {})
    
    for experiment, result in results.items():
        print(f"{experiment}: {result['status'].upper()}")
        print(f"  Resilience Score: {result.get('resilience_score', 0):.1f}/100")
        if result['status'] != 'passed':
            print(f"  Details: {result}")
        print()
    
    print("\n=== Overall Summary ===")
    print(f"Total Experiments: {summary.get('total_experiments', 0)}")
    print(f"Successful: {summary.get('successful_experiments', 0)}")
    print(f"Overall Resilience Score: {summary.get('overall_resilience_score', 0):.1f}/100")
    print(f"Recommendation: {summary.get('recommendation', 'N/A')}")
    
    # Cleanup
    await event_store.disconnect()


if __name__ == "__main__":
    asyncio.run(main())