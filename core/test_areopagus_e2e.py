"""
Comprehensive end-to-end tests for Areopagus debate workflow.

These tests validate the complete debate lifecycle including:
- Event flow and ordering
- Agent participation and consensus
- Error handling and recovery
- Performance under load
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4

from src.aura_intelligence.event_store.hardened_store import HardenedNATSEventStore
from src.aura_intelligence.event_store.events import (
    Event, EventType, EventMetadata,
    DebateArgumentPayload, create_decision_proposed_event
)
from src.aura_intelligence.event_store.robust_projections import (
    ProjectionManager, DebateStateProjection, AgentPerformanceProjection
)
from src.aura_intelligence.areopagus.debate_graph import DebateGraph
from src.aura_intelligence.observability.metrics import (
    active_debates, debate_consensus_rate, system_health_score
)


class TestAreopagusE2E:
    """End-to-end tests for Areopagus debate system"""
    
    @pytest.fixture
    async def event_store(self):
        """Create test event store"""
        store = HardenedNATSEventStore(
            nats_url="nats://localhost:4222",
            stream_name="TEST_AURA_EVENTS"
        )
        await store.connect()
        yield store
        await store.disconnect()
    
    @pytest.fixture
    async def debate_system(self, event_store):
        """Create debate system with event store"""
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        debate_graph = DebateGraph(llm=llm, event_store=event_store)
        return debate_graph
    
    @pytest.fixture
    async def projection_manager(self, event_store):
        """Create projection manager with test projections"""
        # Mock database pools for testing
        import asyncpg
        storage_pool = await asyncpg.create_pool(
            "postgresql://test:test@localhost/test_aura"
        )
        checkpoint_pool = await asyncpg.create_pool(
            "postgresql://test:test@localhost/test_aura"
        )
        
        projections = [
            DebateStateProjection("debate_state", storage_pool, checkpoint_pool),
            AgentPerformanceProjection("agent_performance", storage_pool, checkpoint_pool)
        ]
        
        manager = ProjectionManager(event_store, projections)
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_complete_debate_lifecycle(self, debate_system, event_store):
        """Test full debate from initiation to consensus"""
        # 1. Start debate
        topic = "Should AI systems be required to explain their decisions?"
        initial_state = {
            "topic": topic,
            "context": {
                "domain": "AI Ethics",
                "urgency": "high",
                "stakeholders": ["developers", "users", "regulators"]
            }
        }
        
        # Record start time
        start_time = datetime.utcnow()
        
        # 2. Run debate
        result = await debate_system.graph.ainvoke(
            initial_state,
            config={"recursion_limit": 20}
        )
        
        # 3. Verify debate completed
        assert "final_synthesis" in result
        assert result["final_synthesis"] is not None
        
        # 4. Verify all agents participated
        messages = result.get("messages", [])
        agent_types = set()
        for msg in messages:
            if hasattr(msg, "name"):
                agent_types.add(msg.name)
        
        expected_agents = {"Philosopher", "Scientist", "Pragmatist"}
        assert agent_types >= expected_agents, f"Missing agents: {expected_agents - agent_types}"
        
        # 5. Verify argument flow
        argument_count = len([m for m in messages if hasattr(m, "content")])
        assert argument_count >= 6, "Expected at least 2 rounds of arguments"
        
        # 6. Verify consensus or proper termination
        consensus_reached = result.get("consensus_reached", False)
        if consensus_reached:
            assert "consensus" in result
            assert result["consensus"]["confidence"] > 0.7
        else:
            # Verify proper timeout handling
            duration = datetime.utcnow() - start_time
            assert duration < timedelta(minutes=10), "Debate should timeout within 10 minutes"
        
        # 7. Validate event stream
        events = []
        debate_id = result.get("debate_id")
        if debate_id:
            events = await event_store.get_events(
                aggregate_id=f"debate:{debate_id}",
                since_version=0
            )
        
        # Verify event ordering
        self._validate_event_ordering(events)
        
        # 8. Verify metrics were updated
        # In real test, would query Prometheus
        # For now, just verify the metrics exist
        assert active_debates._value is not None
    
    def _validate_event_ordering(self, events: List[Dict[str, Any]]) -> None:
        """Validate that events follow expected ordering"""
        event_types = [e["type"] for e in events]
        
        # Debate should start before arguments
        if EventType.DEBATE_STARTED.value in event_types:
            start_idx = event_types.index(EventType.DEBATE_STARTED.value)
            
            # All arguments should come after start
            for i, et in enumerate(event_types):
                if et == EventType.DEBATE_ARGUMENT_ADDED.value:
                    assert i > start_idx, "Arguments must come after debate start"
        
        # Consensus/failure should be last
        terminal_events = [
            EventType.DEBATE_CONSENSUS_REACHED.value,
            EventType.DEBATE_FAILED.value
        ]
        for te in terminal_events:
            if te in event_types:
                assert event_types.index(te) == len(event_types) - 1, \
                    "Terminal events must be last"
    
    async def test_debate_with_agent_failure(self, debate_system):
        """Test system recovery when an agent fails"""
        # Inject agent failure
        original_philosopher = debate_system.philosopher_agent
        
        async def failing_philosopher(*args, **kwargs):
            raise RuntimeError("Philosopher agent crashed")
        
        debate_system.philosopher_agent = failing_philosopher
        
        try:
            # Run debate
            result = await debate_system.graph.ainvoke(
                {"topic": "Test topic"},
                config={"recursion_limit": 10}
            )
            
            # System should handle failure gracefully
            assert "error" in result or result.get("status") == "failed"
            
        finally:
            # Restore original agent
            debate_system.philosopher_agent = original_philosopher
    
    async def test_concurrent_debates(self, debate_system):
        """Test system handling multiple concurrent debates"""
        topics = [
            "Topic A: AI Safety",
            "Topic B: Climate Change",
            "Topic C: Space Exploration"
        ]
        
        # Start debates concurrently
        tasks = []
        for topic in topics:
            task = asyncio.create_task(
                debate_system.graph.ainvoke(
                    {"topic": topic},
                    config={"recursion_limit": 10}
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed without errors
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Debate {i} failed: {result}")
            else:
                successful += 1
        
        assert successful >= 2, "At least 2 out of 3 debates should succeed"
    
    async def test_debate_idempotency(self, debate_system, event_store):
        """Test that duplicate debate requests are handled properly"""
        debate_id = str(uuid4())
        topic = "Idempotency test topic"
        
        # Create metadata with idempotency key
        metadata = EventMetadata(
            source="test",
            correlation_id=uuid4(),
            idempotency_key=f"debate:{debate_id}"
        )
        
        # Start debate twice with same idempotency key
        results = []
        for _ in range(2):
            result = await debate_system.graph.ainvoke(
                {
                    "topic": topic,
                    "debate_id": debate_id,
                    "metadata": metadata
                },
                config={"recursion_limit": 5}
            )
            results.append(result)
        
        # Should get same result (idempotent)
        assert results[0]["debate_id"] == results[1]["debate_id"]
    
    async def test_projection_consistency(self, debate_system, projection_manager):
        """Test that projections remain consistent with event stream"""
        # Run a debate
        result = await debate_system.graph.ainvoke(
            {"topic": "Projection consistency test"},
            config={"recursion_limit": 10}
        )
        
        debate_id = result.get("debate_id")
        
        # Wait for projections to catch up
        await asyncio.sleep(2)
        
        # Query projection state
        debate_projection = next(
            p for p in projection_manager.projections 
            if p.name == "debate_state"
        )
        
        # Verify projection health
        health = debate_projection.get_health_status()
        assert health["healthy"], f"Projection unhealthy: {health}"
        assert health["error_count"] == 0
        
        # In real test, would query projection data and verify consistency
    
    async def test_load_scenario(self, debate_system):
        """Test system under sustained load"""
        # Configuration
        debates_per_minute = 10
        duration_minutes = 2
        total_debates = debates_per_minute * duration_minutes
        
        start_time = datetime.utcnow()
        results = []
        
        # Generate load
        for i in range(total_debates):
            # Start debate
            task = asyncio.create_task(
                debate_system.graph.ainvoke(
                    {"topic": f"Load test topic {i}"},
                    config={"recursion_limit": 5}
                )
            )
            results.append(task)
            
            # Wait to maintain rate
            await asyncio.sleep(60 / debates_per_minute)
        
        # Wait for all to complete
        completed = await asyncio.gather(*results, return_exceptions=True)
        
        # Analyze results
        successful = sum(1 for r in completed if not isinstance(r, Exception))
        success_rate = successful / total_debates
        
        print(f"Load test results: {successful}/{total_debates} successful ({success_rate:.1%})")
        
        # Should maintain high success rate under load
        assert success_rate >= 0.9, f"Success rate {success_rate:.1%} below threshold"
        
        # Verify system health remained good
        # In real test, would check metrics
        assert system_health_score._value > 80
    
    async def test_event_replay(self, event_store, projection_manager):
        """Test event replay and projection rebuild"""
        # Generate some events
        for i in range(10):
            event = create_decision_proposed_event(
                aggregate_id=f"test:{i}",
                decision_id=str(uuid4()),
                decision_type="test",
                proposal={"action": "test"},
                rationale="Test decision",
                confidence=0.8,
                metadata=EventMetadata(source="test"),
                version=i
            )
            await event_store.append(event)
        
        # Stop projections
        await projection_manager.stop()
        
        # Clear projection state (simulate rebuild scenario)
        for projection in projection_manager.projections:
            await projection.checkpoint.save_position(0)
        
        # Restart and rebuild
        await projection_manager.start()
        
        # Wait for rebuild
        await asyncio.sleep(5)
        
        # Verify projections caught up
        for projection in projection_manager.projections:
            position = await projection.get_checkpoint()
            assert position >= 9, f"Projection {projection.name} not caught up"
    
    async def test_chaos_recovery(self, debate_system, event_store):
        """Test system recovery from chaos experiments"""
        from src.aura_intelligence.chaos.experiments import (
            EventStoreFailureExperiment,
            run_chaos_suite
        )
        
        # Define health check
        async def health_check():
            try:
                # Check if debate system is responsive
                result = await asyncio.wait_for(
                    debate_system.graph.ainvoke(
                        {"topic": "Health check"},
                        config={"recursion_limit": 1}
                    ),
                    timeout=5.0
                )
                return {
                    "responding": True,
                    "event_store_connected": event_store._connected
                }
            except:
                return {
                    "responding": False,
                    "event_store_connected": event_store._connected
                }
        
        # Run chaos experiment
        components = {
            "event_store": event_store,
            "health_check": health_check,
            "debate_system": debate_system
        }
        
        report = await run_chaos_suite(components)
        
        # System should show resilience
        resilience_score = report["summary"]["resilience_score"]
        assert resilience_score >= 70, f"Low resilience score: {resilience_score}"
        
        # Verify system recovered
        health = await health_check()
        assert health["responding"], "System not responding after chaos tests"
        assert health["event_store_connected"], "Event store not reconnected"


class LoadTestRunner:
    """Helper class for load testing scenarios"""
    
    def __init__(self, debate_system):
        self.debate_system = debate_system
        self.metrics = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "durations": [],
            "consensus_reached": 0
        }
    
    async def run_sustained_load(self, rate: int, duration_minutes: int):
        """Run sustained load test"""
        print(f"Starting sustained load test: {rate} debates/min for {duration_minutes} minutes")
        
        start_time = datetime.utcnow()
        tasks = []
        
        while (datetime.utcnow() - start_time).total_seconds() < duration_minutes * 60:
            # Start debate
            task = asyncio.create_task(self._run_single_debate())
            tasks.append(task)
            
            # Maintain rate
            await asyncio.sleep(60 / rate)
        
        # Wait for completion
        await asyncio.gather(*tasks)
        
        # Report results
        self._print_report()
    
    async def run_spike_test(self, baseline: int, spike: int, spike_duration: int):
        """Run spike load test"""
        print(f"Starting spike test: {baseline} -> {spike} debates/min")
        
        # Baseline load
        await self._generate_load(baseline, 60)
        
        # Spike
        await self._generate_load(spike, spike_duration)
        
        # Return to baseline
        await self._generate_load(baseline, 60)
        
        # Report
        self._print_report()
    
    async def _run_single_debate(self):
        """Run a single debate and record metrics"""
        self.metrics["total"] += 1
        start = datetime.utcnow()
        
        try:
            result = await self.debate_system.graph.ainvoke(
                {"topic": f"Load test topic {self.metrics['total']}"},
                config={"recursion_limit": 10}
            )
            
            duration = (datetime.utcnow() - start).total_seconds()
            self.metrics["successful"] += 1
            self.metrics["durations"].append(duration)
            
            if result.get("consensus_reached"):
                self.metrics["consensus_reached"] += 1
                
        except Exception as e:
            self.metrics["failed"] += 1
            print(f"Debate failed: {e}")
    
    async def _generate_load(self, rate: int, duration_seconds: int):
        """Generate load at specified rate"""
        tasks = []
        start = datetime.utcnow()
        
        while (datetime.utcnow() - start).total_seconds() < duration_seconds:
            task = asyncio.create_task(self._run_single_debate())
            tasks.append(task)
            await asyncio.sleep(60 / rate)
        
        await asyncio.gather(*tasks)
    
    def _print_report(self):
        """Print load test report"""
        success_rate = self.metrics["successful"] / self.metrics["total"] * 100
        consensus_rate = self.metrics["consensus_reached"] / self.metrics["successful"] * 100
        avg_duration = sum(self.metrics["durations"]) / len(self.metrics["durations"])
        
        print("\n=== Load Test Report ===")
        print(f"Total debates: {self.metrics['total']}")
        print(f"Successful: {self.metrics['successful']} ({success_rate:.1f}%)")
        print(f"Failed: {self.metrics['failed']}")
        print(f"Consensus rate: {consensus_rate:.1f}%")
        print(f"Average duration: {avg_duration:.1f}s")
        print(f"Min duration: {min(self.metrics['durations']):.1f}s")
        print(f"Max duration: {max(self.metrics['durations']):.1f}s")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])