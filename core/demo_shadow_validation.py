#!/usr/bin/env python3
"""
🚀 AURA Intelligence Shadow Mode Validation Demo

This script demonstrates shadow mode validation with sample events.
It shows how to test the system without affecting production traffic.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field

# Mock imports for demo (replace with actual imports in production)
try:
    from aura_intelligence.config import AURASettings
    from aura_intelligence.utils import get_logger, timer
except ImportError:
    print("Note: Running in demo mode without full imports")
    
    # Simple mock implementations for demo
    class AURASettings:
        @classmethod
        def from_env(cls):
            return cls()
        
        def print_configuration_summary(self):
            print("🔧 AURA Intelligence Configuration (Demo Mode)")
            print("=" * 50)
            print("Environment: shadow")
            print("Deployment Mode: shadow")
            print("Shadow Traffic: 100%")
            print("=" * 50)
    
    def get_logger(name):
        class Logger:
            def info(self, msg, **kwargs):
                print(f"[INFO] {msg}", kwargs if kwargs else "")
            def warning(self, msg, **kwargs):
                print(f"[WARN] {msg}", kwargs if kwargs else "")
            def error(self, msg, **kwargs):
                print(f"[ERROR] {msg}", kwargs if kwargs else "")
        return Logger()
    
    from contextlib import contextmanager
    @contextmanager
    def timer(name):
        start = time.time()
        yield
        print(f"⏱️  {name} took {time.time() - start:.2f}s")


logger = get_logger(__name__)


class ShadowEvent(BaseModel):
    """Event structure for shadow mode testing."""
    event_id: str = Field(description="Unique event ID")
    type: str = Field(description="Event type")
    content: Dict[str, Any] = Field(description="Event content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShadowValidationResult(BaseModel):
    """Results from shadow mode validation."""
    passed: bool = Field(description="Whether validation passed")
    events_processed: int = Field(description="Number of events processed")
    events_succeeded: int = Field(description="Number of successful events")
    events_failed: int = Field(description="Number of failed events")
    latency_p50: float = Field(description="50th percentile latency (ms)")
    latency_p95: float = Field(description="95th percentile latency (ms)")
    latency_p99: float = Field(description="99th percentile latency (ms)")
    errors: List[str] = Field(default_factory=list)
    comparison_results: Dict[str, Any] = Field(default_factory=dict)


class ShadowValidator:
    """Shadow mode validator for AURA Intelligence."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.latencies: List[float] = []
        self.errors: List[str] = []
        
    async def process_event(self, event: ShadowEvent) -> tuple[bool, float]:
        """
        Process a single event in shadow mode.
        
        Returns:
            Tuple of (success, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Simulate event processing based on type
            match event.type:
                case "user_query":
                    await self._process_user_query(event)
                case "system_event":
                    await self._process_system_event(event)
                case "error_event":
                    await self._process_error_event(event)
                case _:
                    await self._process_generic_event(event)
            
            latency_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"✅ Processed {event.type} event",
                event_id=event.event_id,
                latency_ms=f"{latency_ms:.2f}"
            )
            return True, latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"Failed to process {event.type}: {str(e)}"
            self.logger.error(error_msg, event_id=event.event_id)
            self.errors.append(error_msg)
            return False, latency_ms
    
    async def _process_user_query(self, event: ShadowEvent):
        """Process user query event."""
        # Simulate AI processing delay
        await asyncio.sleep(0.1 + (hash(event.event_id) % 100) / 1000)
        
        # Simulate occasional errors for testing
        if hash(event.event_id) % 20 == 0:
            raise Exception("Simulated processing error")
    
    async def _process_system_event(self, event: ShadowEvent):
        """Process system event."""
        # Fast processing for system events
        await asyncio.sleep(0.01)
    
    async def _process_error_event(self, event: ShadowEvent):
        """Process error event."""
        # Simulate error analysis
        await asyncio.sleep(0.05)
    
    async def _process_generic_event(self, event: ShadowEvent):
        """Process generic event."""
        await asyncio.sleep(0.02)
    
    def calculate_percentile(self, percentile: float) -> float:
        """Calculate latency percentile."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    async def run_validation(
        self,
        test_events: List[Dict[str, Any]],
        compare_with_production: bool = True
    ) -> ShadowValidationResult:
        """
        Run shadow mode validation with test events.
        
        Args:
            test_events: List of test events to process
            compare_with_production: Whether to compare with production results
            
        Returns:
            Validation results
        """
        self.logger.info(f"🚀 Starting shadow validation with {len(test_events)} events")
        
        events_succeeded = 0
        events_failed = 0
        
        # Process events concurrently
        tasks = []
        for i, event_data in enumerate(test_events):
            event = ShadowEvent(
                event_id=f"shadow-{i}-{int(time.time())}",
                type=event_data.get("type", "unknown"),
                content=event_data
            )
            tasks.append(self.process_event(event))
        
        # Wait for all events to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                events_failed += 1
                self.errors.append(str(result))
            else:
                success, latency = result
                self.latencies.append(latency)
                if success:
                    events_succeeded += 1
                else:
                    events_failed += 1
        
        # Calculate metrics
        result = ShadowValidationResult(
            passed=events_failed == 0,
            events_processed=len(test_events),
            events_succeeded=events_succeeded,
            events_failed=events_failed,
            latency_p50=self.calculate_percentile(50),
            latency_p95=self.calculate_percentile(95),
            latency_p99=self.calculate_percentile(99),
            errors=self.errors
        )
        
        # Compare with production if enabled
        if compare_with_production:
            result.comparison_results = await self._compare_with_production(result)
        
        return result
    
    async def _compare_with_production(self, shadow_result: ShadowValidationResult) -> Dict[str, Any]:
        """Compare shadow results with production baseline."""
        # In a real implementation, this would fetch production metrics
        # For demo, we'll use mock baseline values
        production_baseline = {
            "latency_p50": 50.0,
            "latency_p95": 200.0,
            "latency_p99": 500.0,
            "error_rate": 0.01
        }
        
        shadow_error_rate = shadow_result.events_failed / shadow_result.events_processed if shadow_result.events_processed > 0 else 0
        
        comparison = {
            "latency_p50_diff": shadow_result.latency_p50 - production_baseline["latency_p50"],
            "latency_p95_diff": shadow_result.latency_p95 - production_baseline["latency_p95"],
            "latency_p99_diff": shadow_result.latency_p99 - production_baseline["latency_p99"],
            "error_rate_diff": shadow_error_rate - production_baseline["error_rate"],
            "performance_acceptable": (
                shadow_result.latency_p95 < production_baseline["latency_p95"] * 1.2 and
                shadow_error_rate <= production_baseline["error_rate"] * 1.5
            )
        }
        
        return comparison


async def main():
    """Run the shadow mode validation demo."""
    print("🎯 AURA Intelligence Shadow Mode Validation Demo")
    print("=" * 60)
    
    # Load configuration
    try:
        settings = AURASettings.from_env()
        settings.print_configuration_summary()
    except Exception as e:
        print(f"Note: Using demo configuration ({e})")
    
    # Create test events
    test_events = [
        # User queries
        {"type": "user_query", "content": "What is the weather today?"},
        {"type": "user_query", "content": "Analyze system performance"},
        {"type": "user_query", "content": "Show me recent errors"},
        
        # System events
        {"type": "system_event", "action": "health_check"},
        {"type": "system_event", "action": "metric_collection"},
        {"type": "system_event", "action": "cache_refresh"},
        
        # Error events
        {"type": "error_event", "severity": "warning", "message": "High memory usage"},
        {"type": "error_event", "severity": "critical", "message": "Database connection failed"},
        
        # Mixed events to test variety
        {"type": "audit_event", "action": "user_login", "user": "test_user"},
        {"type": "performance_event", "metric": "response_time", "value": 250},
    ]
    
    # Run validation
    validator = ShadowValidator()
    
    with timer("Shadow Validation"):
        results = await validator.run_validation(test_events)
    
    # Print results
    print("\n📊 Shadow Validation Results")
    print("=" * 60)
    print(f"Status: {'✅ PASSED' if results.passed else '❌ FAILED'}")
    print(f"Events Processed: {results.events_processed}")
    print(f"Events Succeeded: {results.events_succeeded}")
    print(f"Events Failed: {results.events_failed}")
    print(f"\nLatency Metrics:")
    print(f"  P50: {results.latency_p50:.2f}ms")
    print(f"  P95: {results.latency_p95:.2f}ms")
    print(f"  P99: {results.latency_p99:.2f}ms")
    
    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)}):")
        for error in results.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    if results.comparison_results:
        print("\n🔄 Production Comparison:")
        comp = results.comparison_results
        print(f"  Latency P95 Diff: {comp['latency_p95_diff']:+.2f}ms")
        print(f"  Error Rate Diff: {comp['error_rate_diff']:+.2%}")
        print(f"  Performance Acceptable: {'✅ Yes' if comp['performance_acceptable'] else '❌ No'}")
    
    print("\n✨ Shadow validation complete!")
    
    # Return exit code based on results
    return 0 if results.passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)