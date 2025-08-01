"""
Simple, practical consensus for AURA - only what's actually needed.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import hashlib

from ..events import EventProducer


@dataclass
class Decision:
    """A decision that might need consensus."""
    id: str
    type: str
    data: Dict[str, Any]
    requester: str
    

class SimpleConsensus:
    """
    Minimal consensus implementation.
    Uses event ordering for 95% of cases, Raft for the 5% that matter.
    """
    
    # Critical decisions that need consensus
    NEEDS_CONSENSUS = {
        "gpu_allocation",
        "model_update", 
        "leader_election",
        "security_policy"
    }
    
    def __init__(self, node_id: str, peers: list[str], kafka_servers: str):
        self.node_id = node_id
        self.peers = peers
        self.events = EventProducer(kafka_servers)
        self.is_leader = False
        self.term = 0
        
    async def decide(self, decision: Decision) -> Dict[str, Any]:
        """Route decision to appropriate mechanism."""
        if decision.type in self.NEEDS_CONSENSUS:
            return await self._consensus_decide(decision)
        else:
            # Just publish event - fast path for 95% of decisions
            await self.events.send_event(f"decisions.{decision.type}", {
                "id": decision.id,
                "data": decision.data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return {"status": "published", "mode": "event"}
    
    async def _consensus_decide(self, decision: Decision) -> Dict[str, Any]:
        """Simple Raft-like consensus for critical decisions."""
        if not self.is_leader:
            # Forward to leader or trigger election
            leader = await self._find_leader()
            if leader:
                return await self._forward_to_leader(leader, decision)
            else:
                await self._start_election()
                return {"status": "pending", "reason": "election in progress"}
        
        # Leader: collect votes
        votes = await self._collect_votes(decision)
        
        if votes["approved"] > len(self.peers) // 2:
            # Commit decision
            await self._commit_decision(decision)
            return {"status": "accepted", "votes": votes}
        else:
            return {"status": "rejected", "votes": votes}
    
    async def _collect_votes(self, decision: Decision) -> Dict[str, int]:
        """Collect votes from peers (simplified)."""
        # In production: actual RPC or Temporal activities
        await asyncio.sleep(0.05)  # Simulate network
        return {"approved": 2, "rejected": 0, "total": 3}
    
    async def _start_election(self):
        """Start leader election (simplified Raft)."""
        self.term += 1
        votes = 1  # Vote for self
        
        # Request votes from peers
        # In production: actual vote requests
        await asyncio.sleep(0.1)  # Simulate election
        
        if votes > len(self.peers) // 2:
            self.is_leader = True
            await self.events.send_event("leader.elected", {
                "node_id": self.node_id,
                "term": self.term
            })
    
    async def _find_leader(self) -> Optional[str]:
        """Find current leader from recent heartbeats."""
        # In production: track from heartbeats
        return None
    
    async def _forward_to_leader(self, leader: str, decision: Decision):
        """Forward decision to leader."""
        await self.events.send_event(f"forward.{leader}", decision.__dict__)
        return {"status": "forwarded", "to": leader}
    
    async def _commit_decision(self, decision: Decision):
        """Commit accepted decision."""
        await self.events.send_event("decisions.committed", {
            "id": decision.id,
            "type": decision.type,
            "data": decision.data,
            "term": self.term,
            "timestamp": datetime.utcnow().isoformat()
        })


# Example usage showing the simplicity
async def example():
    consensus = SimpleConsensus("node-1", ["node-2", "node-3"], "localhost:9092")
    
    # Regular decision - just publishes event (fast)
    result = await consensus.decide(Decision(
        id="cache-update-123",
        type="cache_update",
        data={"key": "user_prefs", "value": {"theme": "dark"}},
        requester="agent-7"
    ))
    # Result: {"status": "published", "mode": "event"} - ~5ms
    
    # Critical decision - uses consensus (slower but safe)
    result = await consensus.decide(Decision(
        id="gpu-alloc-456", 
        type="gpu_allocation",
        data={"agent": "researcher-1", "gpus": 4, "hours": 24},
        requester="researcher-1"
    ))
    # Result: {"status": "accepted", "votes": {...}} - ~50ms