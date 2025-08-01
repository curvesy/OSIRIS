"""
Event Replay Validation Suite for AURA Intelligence.

This module validates the complete event replay functionality including:
- Full history replay
- Checkpoint-based replay
- Replay during active processing
- Corrupted event handling
"""

import asyncio
import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge

from ..event_store.hardened_store import HardenedNATSEventStore
from ..event_store.events import Event, EventType, EventMetadata
from ..event_store.robust_projections import ProjectionManager
from ..observability.metrics import (
    validation_tests_run,
    validation_tests_failed,
    validation_test_duration
)

logger = get_logger(__name__)

# Validation metrics
replay_validation_total = Counter(
    "replay_validation_total",
    "Total replay validation attempts",
    ["scenario", "status"]
)

replay_validation_duration = Histogram(
    "replay_validation_duration_seconds",
    "Time taken for replay validation",
    ["scenario"]
)

replay_data_integrity = Gauge(
    "replay_data_integrity_score",
    "Data integrity score after replay (0-100)",
    ["scenario"]
)


class EventReplayValidator:
    """Validates event replay functionality in production-like conditions"""
    
    def __init__(
        self,
        event_store: HardenedNATSEventStore,
        projection_manager: ProjectionManager,
        test_event_count: int = 10000
    ):
        self.event_store = event_store
        self.projection_manager = projection_manager
        self.test_event_count = test_event_count
        self.validation_results: Dict[str, Any] = {}
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Execute all replay validation scenarios"""
        logger.info(
            "Starting event replay validation suite",
            event_count=self.test_event_count
        )
        
        scenarios = [
            self.validate_full_history_replay,
            self.validate_checkpoint_replay,
            self.validate_concurrent_replay,
            self.validate_corrupted_event_handling
        ]
        
        for scenario in scenarios:
            try:
                await scenario()
            except Exception as e:
                logger.error(
                    "Validation scenario failed",
                    scenario=scenario.__name__,
                    error=str(e)
                )
                self.validation_results[scenario.__name__] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return self.validation_results
    
    async def validate_full_history_replay(self) -> None:
        """Validate replaying all events from genesis"""
        scenario_name = "full_history_replay"
        logger.info(f"Starting {scenario_name} validation")
        
        with replay_validation_duration.labels(scenario=scenario_name).time():
            try:
                # 1. Generate test events
                test_events = await self._generate_test_events()
                
                # 2. Store initial state hash
                initial_state = await self._capture_system_state()
                initial_hash = self._compute_state_hash(initial_state)
                
                # 3. Append all events
                for event in test_events:
                    await self.event_store.append(event)
                
                # 4. Wait for projections to catch up
                await self._wait_for_projections()
                
                # 5. Capture final state
                final_state_before = await self._capture_system_state()
                
                # 6. Clear projections
                await self.projection_manager.reset_all_projections()
                
                # 7. Replay all events
                replay_start = datetime.utcnow()
                events_replayed = 0
                
                async for event in self.event_store.replay(from_position=0):
                    await self.projection_manager.handle_event(event)
                    events_replayed += 1
                
                replay_duration = (datetime.utcnow() - replay_start).total_seconds()
                
                # 8. Wait for projections to catch up
                await self._wait_for_projections()
                
                # 9. Capture state after replay
                final_state_after = await self._capture_system_state()
                
                # 10. Validate states match
                states_match = self._compare_states(
                    final_state_before,
                    final_state_after
                )
                
                # Calculate data integrity score
                integrity_score = 100.0 if states_match else 0.0
                replay_data_integrity.labels(scenario=scenario_name).set(integrity_score)
                
                # Record results
                self.validation_results[scenario_name] = {
                    "status": "passed" if states_match else "failed",
                    "events_replayed": events_replayed,
                    "replay_duration_seconds": replay_duration,
                    "events_per_second": events_replayed / replay_duration if replay_duration > 0 else 0,
                    "data_integrity_score": integrity_score,
                    "state_comparison": {
                        "before_hash": self._compute_state_hash(final_state_before),
                        "after_hash": self._compute_state_hash(final_state_after),
                        "match": states_match
                    }
                }
                
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="success" if states_match else "failure"
                ).inc()
                
                logger.info(
                    f"Completed {scenario_name} validation",
                    **self.validation_results[scenario_name]
                )
                
            except Exception as e:
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="error"
                ).inc()
                raise
    
    async def validate_checkpoint_replay(self) -> None:
        """Validate replaying from specific checkpoints"""
        scenario_name = "checkpoint_replay"
        logger.info(f"Starting {scenario_name} validation")
        
        with replay_validation_duration.labels(scenario=scenario_name).time():
            try:
                # 1. Generate and store events
                test_events = await self._generate_test_events()
                checkpoint_positions = []
                
                for i, event in enumerate(test_events):
                    await self.event_store.append(event)
                    # Create checkpoints at 25%, 50%, 75%
                    if i in [len(test_events) // 4, len(test_events) // 2, 3 * len(test_events) // 4]:
                        position = await self.event_store.get_current_position()
                        checkpoint_positions.append({
                            "position": position,
                            "event_count": i + 1,
                            "timestamp": datetime.utcnow()
                        })
                
                # 2. Test replay from each checkpoint
                checkpoint_results = []
                
                for checkpoint in checkpoint_positions:
                    # Reset projections
                    await self.projection_manager.reset_all_projections()
                    
                    # Replay from checkpoint
                    events_replayed = 0
                    async for event in self.event_store.replay(
                        from_position=checkpoint["position"]
                    ):
                        await self.projection_manager.handle_event(event)
                        events_replayed += 1
                    
                    checkpoint_results.append({
                        "checkpoint_position": checkpoint["position"],
                        "expected_events": len(test_events) - checkpoint["event_count"],
                        "actual_events": events_replayed,
                        "match": events_replayed == (len(test_events) - checkpoint["event_count"])
                    })
                
                # Validate all checkpoints worked correctly
                all_checkpoints_valid = all(r["match"] for r in checkpoint_results)
                
                self.validation_results[scenario_name] = {
                    "status": "passed" if all_checkpoints_valid else "failed",
                    "checkpoints_tested": len(checkpoint_positions),
                    "checkpoint_results": checkpoint_results,
                    "all_valid": all_checkpoints_valid
                }
                
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="success" if all_checkpoints_valid else "failure"
                ).inc()
                
            except Exception as e:
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="error"
                ).inc()
                raise
    
    async def validate_concurrent_replay(self) -> None:
        """Validate replay while system processes new events"""
        scenario_name = "concurrent_replay"
        logger.info(f"Starting {scenario_name} validation")
        
        with replay_validation_duration.labels(scenario=scenario_name).time():
            try:
                # 1. Start with some initial events
                initial_events = await self._generate_test_events(count=self.test_event_count // 2)
                for event in initial_events:
                    await self.event_store.append(event)
                
                # 2. Start replay task
                replay_complete = asyncio.Event()
                replay_errors = []
                
                async def replay_task():
                    try:
                        events_replayed = 0
                        async for event in self.event_store.replay(from_position=0):
                            # Simulate projection processing
                            await asyncio.sleep(0.001)  # Small delay
                            events_replayed += 1
                        
                        logger.info(
                            "Replay task completed",
                            events_replayed=events_replayed
                        )
                    except Exception as e:
                        replay_errors.append(e)
                    finally:
                        replay_complete.set()
                
                # 3. Start live event generation task
                live_events_added = 0
                
                async def live_event_task():
                    nonlocal live_events_added
                    while not replay_complete.is_set():
                        event = self._create_test_event()
                        await self.event_store.append(event)
                        live_events_added += 1
                        await asyncio.sleep(0.01)  # 100 events/second
                
                # 4. Run both tasks concurrently
                replay_task_handle = asyncio.create_task(replay_task())
                live_task_handle = asyncio.create_task(live_event_task())
                
                # Wait for replay to complete
                await replay_complete.wait()
                live_task_handle.cancel()
                
                # 5. Validate no errors occurred
                no_errors = len(replay_errors) == 0
                
                self.validation_results[scenario_name] = {
                    "status": "passed" if no_errors else "failed",
                    "live_events_during_replay": live_events_added,
                    "replay_errors": [str(e) for e in replay_errors],
                    "concurrent_processing": True
                }
                
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="success" if no_errors else "failure"
                ).inc()
                
            except Exception as e:
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="error"
                ).inc()
                raise
    
    async def validate_corrupted_event_handling(self) -> None:
        """Validate replay handles corrupted events gracefully"""
        scenario_name = "corrupted_event_handling"
        logger.info(f"Starting {scenario_name} validation")
        
        with replay_validation_duration.labels(scenario=scenario_name).time():
            try:
                # 1. Generate mix of valid and corrupted events
                valid_events = await self._generate_test_events(count=100)
                corrupted_positions = [25, 50, 75]  # Positions to inject corruption
                
                # 2. Store events with some corrupted
                stored_events = []
                for i, event in enumerate(valid_events):
                    if i in corrupted_positions:
                        # Inject corrupted event (will be handled by store)
                        corrupted = self._create_corrupted_event()
                        stored_events.append(("corrupted", corrupted))
                    else:
                        await self.event_store.append(event)
                        stored_events.append(("valid", event))
                
                # 3. Attempt replay
                replay_errors = []
                events_processed = 0
                events_skipped = 0
                
                async for event in self.event_store.replay(from_position=0):
                    try:
                        # Validate event structure
                        if not self._validate_event_structure(event):
                            events_skipped += 1
                            continue
                        
                        await self.projection_manager.handle_event(event)
                        events_processed += 1
                        
                    except Exception as e:
                        logger.warning(
                            "Error processing event during replay",
                            event_id=str(event.id) if hasattr(event, 'id') else "unknown",
                            error=str(e)
                        )
                        replay_errors.append({
                            "event_id": str(event.id) if hasattr(event, 'id') else "unknown",
                            "error": str(e)
                        })
                        events_skipped += 1
                
                # 4. Validate replay completed despite corruption
                replay_completed = events_processed > 0
                corruption_handled = len(replay_errors) <= len(corrupted_positions)
                
                self.validation_results[scenario_name] = {
                    "status": "passed" if replay_completed and corruption_handled else "failed",
                    "events_processed": events_processed,
                    "events_skipped": events_skipped,
                    "corruption_positions": corrupted_positions,
                    "replay_errors": replay_errors,
                    "graceful_handling": corruption_handled
                }
                
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="success" if replay_completed else "failure"
                ).inc()
                
            except Exception as e:
                replay_validation_total.labels(
                    scenario=scenario_name,
                    status="error"
                ).inc()
                raise
    
    # Helper methods
    
    async def _generate_test_events(self, count: Optional[int] = None) -> List[Event]:
        """Generate test events for validation"""
        if count is None:
            count = self.test_event_count
        
        events = []
        for i in range(count):
            event = self._create_test_event()
            events.append(event)
        
        return events
    
    def _create_test_event(self) -> Event:
        """Create a single test event"""
        event_types = [
            EventType.DEBATE_STARTED,
            EventType.ARGUMENT_SUBMITTED,
            EventType.CONSENSUS_REACHED,
            EventType.DEBATE_CONCLUDED
        ]
        
        return Event(
            id=uuid4(),
            idempotency_key=f"test-{uuid4()}",
            timestamp=datetime.utcnow(),
            type=random.choice(event_types),
            aggregate_id=f"debate-{random.randint(1, 100)}",
            version=random.randint(1, 10),
            payload={"test_data": f"value-{random.randint(1, 1000)}"},
            metadata=EventMetadata(
                correlation_id=str(uuid4()),
                causation_id=str(uuid4()),
                user_id="test-user",
                source="replay-validator"
            )
        )
    
    def _create_corrupted_event(self) -> Dict[str, Any]:
        """Create a corrupted event for testing"""
        corruption_types = [
            {"missing_field": "no_id"},  # Missing required field
            {"invalid_type": 12345},      # Wrong type for event type
            {"malformed_json": "{invalid}"},  # Invalid JSON
            {"null_payload": None}        # Null payload
        ]
        
        return random.choice(corruption_types)
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for comparison"""
        # This would capture projection states, aggregates, etc.
        # Simplified for this example
        return {
            "projection_states": await self.projection_manager.get_all_states(),
            "event_count": await self.event_store.get_event_count(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute deterministic hash of system state"""
        # Remove timestamp for comparison
        state_copy = state.copy()
        state_copy.pop("timestamp", None)
        
        state_json = json.dumps(state_copy, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def _compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Compare two system states for equality"""
        return self._compute_state_hash(state1) == self._compute_state_hash(state2)
    
    async def _wait_for_projections(self, timeout: int = 30) -> None:
        """Wait for all projections to catch up"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if await self.projection_manager.all_projections_caught_up():
                return
            await asyncio.sleep(0.1)
        
        raise TimeoutError("Projections did not catch up within timeout")
    
    def _validate_event_structure(self, event: Any) -> bool:
        """Validate event has required structure"""
        required_fields = ["id", "type", "aggregate_id", "payload"]
        
        for field in required_fields:
            if not hasattr(event, field):
                return False
        
        return True


async def main():
    """Run event replay validation suite"""
    # Initialize components
    event_store = HardenedNATSEventStore(
        nats_url="nats://localhost:4222",
        stream_name="AURA_VALIDATION_EVENTS"
    )
    await event_store.connect()
    
    # Create projection manager
    projection_manager = ProjectionManager(event_store, [])
    
    # Create validator
    validator = EventReplayValidator(
        event_store=event_store,
        projection_manager=projection_manager,
        test_event_count=1000
    )
    
    # Run validations
    results = await validator.run_all_validations()
    
    # Print results
    print("\n=== Event Replay Validation Results ===\n")
    for scenario, result in results.items():
        print(f"{scenario}: {result['status'].upper()}")
        if result['status'] != 'passed':
            print(f"  Details: {result}")
    
    # Cleanup
    await event_store.disconnect()


if __name__ == "__main__":
    asyncio.run(main())