"""
Examples of selective consensus usage in AURA Intelligence.

Demonstrates when to use consensus vs. simpler coordination mechanisms.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from .types import ConsensusRequest, DecisionType
from .manager import ConsensusManager, ConsensusConfig
from .raft import RaftConfig
from .byzantine import BFTConfig


async def example_resource_allocation():
    """
    Example: GPU resource allocation requires consensus.
    
    Multiple agents competing for limited GPU resources.
    """
    # Initialize consensus manager
    config = ConsensusConfig(
        node_id="coordinator-1",
        enable_raft=True,
        enable_bft=True,
        enable_multi_raft=False,
        raft_peers=["coordinator-1", "coordinator-2", "coordinator-3"],
        bft_validators=["validator-1", "validator-2", "validator-3", "validator-4"]
    )
    
    consensus_manager = ConsensusManager(config)
    await consensus_manager.start()
    
    # GPU allocation request (needs consensus)
    gpu_request = ConsensusRequest(
        request_id="gpu-alloc-123",
        decision_type=DecisionType.RESOURCE_ALLOCATION,
        proposal={
            "action": "allocate_gpu",
            "agent_id": "research-agent-42",
            "gpu_count": 4,
            "duration_hours": 24,
            "purpose": "model_training"
        },
        timeout=timedelta(seconds=5),
        requester="research-agent-42"
    )
    
    # This will use Raft consensus (operational decision)
    result = await consensus_manager.propose(gpu_request)
    
    if result.status == ConsensusState.ACCEPTED:
        print(f"GPU allocation approved: {result.decision}")
        # Actually allocate the GPUs
    else:
        print(f"GPU allocation rejected: {result.reason}")
    
    await consensus_manager.stop()


async def example_model_update():
    """
    Example: Critical model update requires Byzantine consensus.
    
    Updating a model that affects all agents needs strong consensus.
    """
    config = ConsensusConfig(
        node_id="validator-1",
        enable_bft=True,
        bft_validators=["validator-1", "validator-2", "validator-3", "validator-4"]
    )
    
    consensus_manager = ConsensusManager(config)
    await consensus_manager.start()
    
    # Critical model update (needs BFT consensus)
    model_update = ConsensusRequest(
        request_id="model-update-v2.5",
        decision_type=DecisionType.STRATEGIC,
        proposal={
            "action": "update_core_model",
            "model_id": "aura-core-v2.5",
            "changes": {
                "architecture": "transformer-xl",
                "parameters": "175B",
                "training_data": "2024-Q4"
            },
            "rollout_strategy": "canary",
            "impact": "all_agents"
        },
        timeout=timedelta(seconds=30),
        requester="ml-platform"
    )
    
    # This will use Byzantine consensus (strategic decision)
    result = await consensus_manager.propose(model_update)
    
    if result.status == ConsensusState.ACCEPTED:
        print(f"Model update approved with proof: {result.consensus_proof}")
        # Initiate safe rollout
    
    await consensus_manager.stop()


async def example_no_consensus_needed():
    """
    Example: Individual agent decisions don't need consensus.
    
    These use simpler patterns like event sourcing or optimistic locking.
    """
    from ..events import EventProducer
    
    # Simple event publishing for agent decisions
    producer = EventProducer("localhost:9092")
    await producer.start()
    
    # Agent makes independent decision
    agent_decision = {
        "agent_id": "analyst-7",
        "decision_type": "analyze_document",
        "document_id": "doc-456",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Just publish to Kafka, no consensus needed
    await producer.send_event("agent.decisions", agent_decision)
    
    # For cache updates - use optimistic locking
    cache_update = {
        "key": "user_preferences_123",
        "value": {"theme": "dark", "language": "en"},
        "version": 42  # Optimistic lock version
    }
    
    # This would use Redis SET with NX or version check
    # No consensus needed
    
    await producer.stop()


async def example_distributed_lock():
    """
    Example: Quick coordination uses distributed locks, not consensus.
    
    For short-lived exclusive access.
    """
    import aioredis
    
    # Redis for distributed locking
    redis = await aioredis.create_redis_pool('redis://localhost')
    
    # Acquire lock for quick operation
    lock_key = "feature_flag:new_ui"
    lock_token = "unique-token-123"
    
    # Try to acquire lock (SET NX EX)
    acquired = await redis.set(
        lock_key, 
        lock_token,
        expire=5,  # 5 second timeout
        exist=False  # Only if not exists
    )
    
    if acquired:
        try:
            # Do exclusive operation
            print("Updating feature flag...")
            await asyncio.sleep(0.1)
            
            # Update complete
        finally:
            # Release lock (only if we still hold it)
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await redis.eval(lua_script, keys=[lock_key], args=[lock_token])
    
    redis.close()
    await redis.wait_closed()


async def example_decision_router():
    """
    Example: Smart routing based on decision criticality.
    """
    class SmartDecisionRouter:
        def __init__(self):
            self.consensus_manager = None
            self.event_producer = None
            self.redis = None
        
        async def route_decision(self, decision_type: str, payload: Dict[str, Any]):
            """Route decision to appropriate mechanism."""
            
            # Critical decisions need consensus
            if decision_type in ["resource_allocation", "model_update", "security_policy"]:
                request = ConsensusRequest(
                    request_id=f"{decision_type}-{datetime.utcnow().timestamp()}",
                    decision_type=DecisionType.OPERATIONAL,
                    proposal=payload,
                    timeout=timedelta(seconds=5)
                )
                return await self.consensus_manager.propose(request)
            
            # Coordinated decisions need locks
            elif decision_type in ["feature_toggle", "rate_limit_update"]:
                lock_key = f"lock:{decision_type}"
                async with self.redis_lock(lock_key, timeout=2):
                    # Process with exclusive access
                    return {"status": "processed", "mode": "locked"}
            
            # Everything else just publishes events
            else:
                await self.event_producer.send_event(
                    f"decisions.{decision_type}",
                    payload
                )
                return {"status": "published", "mode": "eventual"}
    
    router = SmartDecisionRouter()
    
    # Examples of routing
    await router.route_decision("gpu_allocation", {"gpu_count": 8})  # Consensus
    await router.route_decision("cache_update", {"key": "val"})      # Event only
    await router.route_decision("feature_toggle", {"feature": "x"})  # Lock


# Performance comparison
async def performance_comparison():
    """
    Compare latency of different coordination mechanisms.
    """
    import time
    
    results = {}
    
    # 1. Event publishing (fastest)
    start = time.time()
    producer = EventProducer("localhost:9092")
    await producer.start()
    await producer.send_event("test", {"data": "test"})
    await producer.stop()
    results["event_only"] = (time.time() - start) * 1000  # ~1-5ms
    
    # 2. Distributed lock
    start = time.time()
    # Redis lock acquire/release
    # Simulated: await redis.set() + await redis.del()
    await asyncio.sleep(0.005)  # Simulate Redis RTT
    results["distributed_lock"] = (time.time() - start) * 1000  # ~5-10ms
    
    # 3. Raft consensus
    start = time.time()
    # Raft: propose -> replicate -> commit
    # Simulated: 2 * network RTT + processing
    await asyncio.sleep(0.050)  # Simulate consensus
    results["raft_consensus"] = (time.time() - start) * 1000  # ~50-100ms
    
    # 4. Byzantine consensus
    start = time.time()
    # BFT: 3 phases with broadcasts
    # Simulated: 3 * network RTT + crypto
    await asyncio.sleep(0.150)  # Simulate BFT
    results["bft_consensus"] = (time.time() - start) * 1000  # ~150-300ms
    
    print("Coordination Mechanism Latencies:")
    for mechanism, latency in results.items():
        print(f"  {mechanism}: {latency:.1f}ms")
    
    print("\nRecommendations:")
    print("- Use events for: logging, metrics, non-critical updates")
    print("- Use locks for: quick exclusive access, feature flags")
    print("- Use Raft for: resource allocation, leader election")
    print("- Use BFT for: critical updates, financial transactions")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_resource_allocation())
    asyncio.run(example_no_consensus_needed())
    asyncio.run(performance_comparison())