"""
Raft Consensus Implementation for AURA Intelligence.

Used selectively for critical decisions:
- Resource allocation (GPUs, API quotas)
- Agent group leader election
- Critical workflow triggers
- Compliance-required decisions
"""

from typing import Dict, Any, Optional, List, Set, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import random
import structlog
from abc import ABC, abstractmethod

from opentelemetry import trace, metrics

from .types import (
    ConsensusRequest, ConsensusResult, ConsensusState, ConsensusProof,
    Vote, VoteType, RaftState, LogEntry,
    RaftVoteRequest, RaftVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse
)
from ..events import EventProducer
from ..agents.temporal import execute_workflow

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class RaftMetrics:
    """Centralized metrics for Raft consensus."""
    
    def __init__(self):
        self.state_changes = meter.create_counter(
            name="raft.state.changes",
            description="Number of Raft state changes"
        )
        self.elections = meter.create_counter(
            name="raft.elections",
            description="Number of Raft elections"
        )
        self.log_size = meter.create_gauge(
            name="raft.log.size",
            description="Size of Raft log"
        )


class RaftRPC(Protocol):
    """Protocol for Raft RPC communication."""
    
    async def send_vote_request(self, target: str, request: RaftVoteRequest) -> Optional[RaftVoteResponse]:
        ...
    
    async def send_append_entries(self, target: str, request: AppendEntriesRequest) -> Optional[AppendEntriesResponse]:
        ...


class TemporalRaftRPC:
    """Temporal-based RPC implementation for reliability."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
    
    async def send_vote_request(self, target: str, request: RaftVoteRequest) -> Optional[RaftVoteResponse]:
        return await execute_workflow(
            "RaftRPCWorkflow",
            {"from": self.node_id, "to": target, "type": "vote", "request": request},
            id=f"raft-vote-{self.node_id}-{target}-{datetime.utcnow().timestamp()}"
        )
    
    async def send_append_entries(self, target: str, request: AppendEntriesRequest) -> Optional[AppendEntriesResponse]:
        return await execute_workflow(
            "RaftRPCWorkflow",
            {"from": self.node_id, "to": target, "type": "append", "request": request},
            id=f"raft-append-{self.node_id}-{target}-{datetime.utcnow().timestamp()}"
        )


class RaftLog:
    """Manages the Raft log with efficient operations."""
    
    def __init__(self, max_size: int = 10000):
        self.entries: List[LogEntry] = []
        self.max_size = max_size
        self.snapshot_index = -1
        
    def append(self, entry: LogEntry) -> int:
        """Append entry and return its index."""
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self._create_snapshot()
        return len(self.entries) - 1
    
    def get(self, index: int) -> Optional[LogEntry]:
        """Get entry at index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None
    
    def last(self) -> Optional[LogEntry]:
        """Get last entry."""
        return self.entries[-1] if self.entries else None
    
    def truncate(self, index: int):
        """Remove entries after index."""
        self.entries = self.entries[:index]
    
    def _create_snapshot(self):
        """Create snapshot and compact log."""
        # In production: persist snapshot to disk
        self.snapshot_index = len(self.entries) // 2
        self.entries = self.entries[self.snapshot_index:]


class RaftTimer:
    """Manages Raft timers with jitter."""
    
    def __init__(self, config: 'RaftConfig'):
        self.config = config
        self._tasks: Dict[str, asyncio.Task] = {}
    
    def election_timeout(self) -> timedelta:
        """Generate random election timeout."""
        base = self.config.election_timeout_ms
        jitter = random.randint(0, base)
        return timedelta(milliseconds=base + jitter)
    
    async def reset_election_timer(self, callback: Callable):
        """Reset election timer."""
        self.cancel("election")
        timeout = self.election_timeout()
        self._tasks["election"] = asyncio.create_task(
            self._timer("election", timeout, callback)
        )
    
    async def start_heartbeat_timer(self, callback: Callable):
        """Start heartbeat timer."""
        self.cancel("heartbeat")
        interval = timedelta(milliseconds=self.config.heartbeat_interval_ms)
        self._tasks["heartbeat"] = asyncio.create_task(
            self._repeating_timer("heartbeat", interval, callback)
        )
    
    def cancel(self, name: str):
        """Cancel named timer."""
        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]
    
    def cancel_all(self):
        """Cancel all timers."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
    
    async def _timer(self, name: str, timeout: timedelta, callback: Callable):
        """Single-shot timer."""
        try:
            await asyncio.sleep(timeout.total_seconds())
            await callback()
        except asyncio.CancelledError:
            pass
    
    async def _repeating_timer(self, name: str, interval: timedelta, callback: Callable):
        """Repeating timer."""
        try:
            while True:
                await asyncio.sleep(interval.total_seconds())
                await callback()
        except asyncio.CancelledError:
            pass


class RaftStateMachine:
    """Manages Raft state transitions."""
    
    def __init__(self, node_id: str, metrics: RaftMetrics):
        self.node_id = node_id
        self.metrics = metrics
        self.state = RaftState.FOLLOWER
        self.term = 0
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        self.last_heartbeat = datetime.utcnow()
    
    def transition_to(self, new_state: RaftState):
        """Transition to new state."""
        if self.state != new_state:
            logger.info(
                f"State transition",
                node_id=self.node_id,
                old_state=self.state.value,
                new_state=new_state.value,
                term=self.term
            )
            self.state = new_state
            self.metrics.state_changes.add(1, {
                "node_id": self.node_id,
                "state": new_state.value
            })
    
    def start_new_term(self, term: int):
        """Start new term."""
        self.term = term
        self.voted_for = None
        self.leader_id = None
    
    def record_vote(self, term: int, candidate: str):
        """Record vote for term."""
        if term > self.term:
            self.start_new_term(term)
        self.voted_for = candidate
    
    def update_heartbeat(self, leader_id: str):
        """Update heartbeat from leader."""
        self.last_heartbeat = datetime.utcnow()
        self.leader_id = leader_id


class RaftCore:
    """Core Raft consensus logic."""
    
    def __init__(
        self,
        config: 'RaftConfig',
        rpc: RaftRPC,
        event_producer: EventProducer,
        metrics: RaftMetrics
    ):
        self.config = config
        self.rpc = rpc
        self.event_producer = event_producer
        self.metrics = metrics
        
        # State
        self.state_machine = RaftStateMachine(config.node_id, metrics)
        self.log = RaftLog(config.max_log_size)
        self.timer = RaftTimer(config)
        
        # Consensus tracking
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Candidate state  
        self.votes_received: Set[str] = set()
        
        # Request tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.batch_buffer: List[ConsensusRequest] = []
        self.batch_lock = asyncio.Lock()
    
    async def start(self):
        """Start Raft node."""
        await self.event_producer.start()
        await self.timer.reset_election_timer(self._on_election_timeout)
        await self._publish_state_change()
    
    async def stop(self):
        """Stop Raft node."""
        self.timer.cancel_all()
        await self.event_producer.stop()
    
    async def propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Propose value for consensus."""
        with tracer.start_as_current_span("raft.propose") as span:
            span.set_attributes({
                "node_id": self.config.node_id,
                "state": self.state_machine.state.value,
                "request_id": request.request_id
            })
            
            if self.state_machine.state != RaftState.LEADER:
                return await self._handle_non_leader_proposal(request)
            
            if self.config.pipeline_enabled:
                return await self._batch_propose(request)
            else:
                return await self._direct_propose(request)
    
    async def _handle_non_leader_proposal(self, request: ConsensusRequest) -> ConsensusResult:
        """Handle proposal when not leader."""
        if self.state_machine.leader_id:
            # Forward to leader
            await self.event_producer.send_event(
                f"raft.forward.{self.state_machine.leader_id}",
                request.__dict__
            )
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.FORWARDED,
                metadata={"leader": self.state_machine.leader_id}
            )
        else:
            # No leader, trigger election
            await self._start_election()
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.PENDING,
                reason="No leader, election in progress"
            )
    
    async def _direct_propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Direct proposal without batching."""
        entry = LogEntry(
            term=self.state_machine.term,
            index=len(self.log.entries),
            command=request.proposal,
            request_id=request.request_id
        )
        
        index = self.log.append(entry)
        self.metrics.log_size.set(len(self.log.entries))
        
        future = asyncio.Future()
        self.pending_requests[request.request_id] = future
        
        await self._replicate_entry(entry)
        
        try:
            await asyncio.wait_for(future, timeout=request.timeout.total_seconds())
            return future.result()
        except asyncio.TimeoutError:
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.TIMEOUT,
                reason="Consensus timeout"
            )
    
    async def _batch_propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Batch proposal for efficiency."""
        async with self.batch_lock:
            self.batch_buffer.append(request)
            future = asyncio.Future()
            self.pending_requests[request.request_id] = future
            
            if len(self.batch_buffer) >= self.config.batch_size:
                await self._process_batch()
        
        try:
            await asyncio.wait_for(future, timeout=request.timeout.total_seconds())
            return future.result()
        except asyncio.TimeoutError:
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.TIMEOUT,
                reason="Batch timeout"
            )
    
    async def _process_batch(self):
        """Process batched requests."""
        if not self.batch_buffer:
            return
        
        batch_entry = LogEntry(
            term=self.state_machine.term,
            index=len(self.log.entries),
            command={
                "batch": [req.proposal for req in self.batch_buffer],
                "ids": [req.request_id for req in self.batch_buffer]
            },
            request_id=f"batch-{datetime.utcnow().timestamp()}"
        )
        
        self.batch_buffer.clear()
        self.log.append(batch_entry)
        await self._replicate_entry(batch_entry)
    
    async def _replicate_entry(self, entry: LogEntry):
        """Replicate to followers in parallel."""
        tasks = [
            self._send_append_entries_to_peer(peer, [entry])
            for peer in self.config.peers
            if peer != self.config.node_id
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_append_entries_to_peer(self, peer: str, entries: List[LogEntry]) -> bool:
        """Send AppendEntries to single peer."""
        prev_index = self.next_index.get(peer, 0) - 1
        prev_term = 0
        
        if prev_index >= 0:
            prev_entry = self.log.get(prev_index)
            if prev_entry:
                prev_term = prev_entry.term
        
        request = AppendEntriesRequest(
            term=self.state_machine.term,
            leader_id=self.config.node_id,
            prev_log_index=prev_index,
            prev_log_term=prev_term,
            entries=entries,
            leader_commit=self.commit_index
        )
        
        response = await self.rpc.send_append_entries(peer, request)
        
        if response and response.success:
            if entries:
                self.next_index[peer] = entries[-1].index + 1
                self.match_index[peer] = entries[-1].index
            await self._check_commit()
            return True
        
        return False
    
    async def _check_commit(self):
        """Check if entries can be committed."""
        for i in range(len(self.log.entries) - 1, self.commit_index, -1):
            entry = self.log.get(i)
            if entry and entry.term == self.state_machine.term:
                replicated = sum(
                    1 for peer in self.config.peers
                    if peer == self.config.node_id or self.match_index.get(peer, 0) >= i
                )
                
                if replicated > len(self.config.peers) // 2:
                    self.commit_index = i
                    await self._apply_committed()
                    break
    
    async def _apply_committed(self):
        """Apply committed entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log.get(self.last_applied)
            if not entry:
                continue
            
            if "batch" in entry.command:
                for req_id in entry.command["ids"]:
                    await self._complete_request(req_id, entry.command)
            else:
                await self._complete_request(entry.request_id, entry.command)
    
    async def _complete_request(self, request_id: str, decision: Any):
        """Complete pending request."""
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_result(ConsensusResult(
                request_id=request_id,
                status=ConsensusState.ACCEPTED,
                decision=decision,
                consensus_type="raft"
            ))
    
    async def _on_election_timeout(self):
        """Handle election timeout."""
        if self.state_machine.state != RaftState.LEADER:
            time_since = datetime.utcnow() - self.state_machine.last_heartbeat
            if time_since > self.timer.election_timeout():
                await self._start_election()
    
    async def _start_election(self):
        """Start leader election."""
        self.state_machine.start_new_term(self.state_machine.term + 1)
        self.state_machine.transition_to(RaftState.CANDIDATE)
        self.state_machine.record_vote(self.state_machine.term, self.config.node_id)
        self.votes_received = {self.config.node_id}
        
        self.metrics.elections.add(1, {"node_id": self.config.node_id})
        
        # Request votes in parallel
        vote_tasks = [
            self._request_vote_from_peer(peer)
            for peer in self.config.peers
            if peer != self.config.node_id
        ]
        
        if vote_tasks:
            await asyncio.gather(*vote_tasks, return_exceptions=True)
        
        if len(self.votes_received) > len(self.config.peers) // 2:
            await self._become_leader()
    
    async def _request_vote_from_peer(self, peer: str) -> bool:
        """Request vote from single peer."""
        last_entry = self.log.last()
        
        request = RaftVoteRequest(
            term=self.state_machine.term,
            candidate_id=self.config.node_id,
            last_log_index=len(self.log.entries) - 1,
            last_log_term=last_entry.term if last_entry else 0
        )
        
        response = await self.rpc.send_vote_request(peer, request)
        
        if response and response.vote_granted:
            self.votes_received.add(response.voter_id)
            return True
        
        return False
    
    async def _become_leader(self):
        """Become leader."""
        self.state_machine.transition_to(RaftState.LEADER)
        
        # Initialize leader state
        for peer in self.config.peers:
            self.next_index[peer] = len(self.log.entries)
            self.match_index[peer] = 0
        
        # Start heartbeat
        await self.timer.start_heartbeat_timer(self._send_heartbeats)
        await self._publish_state_change()
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers."""
        tasks = [
            self._send_append_entries_to_peer(peer, [])
            for peer in self.config.peers
            if peer != self.config.node_id
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _publish_state_change(self):
        """Publish state change event."""
        await self.event_producer.send_event("raft.state.changes", {
            "node_id": self.config.node_id,
            "state": self.state_machine.state.value,
            "term": self.state_machine.term,
            "leader_id": self.state_machine.leader_id,
            "log_size": len(self.log.entries),
            "commit_index": self.commit_index
        })


@dataclass
class RaftConfig:
    """Raft configuration."""
    node_id: str
    peers: List[str]
    election_timeout_ms: int = 150
    heartbeat_interval_ms: int = 50
    rpc_timeout_ms: int = 30
    max_log_size: int = 10000
    snapshot_interval: int = 1000
    batch_size: int = 100
    pipeline_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    use_temporal_for_rpc: bool = True


class RaftConsensus:
    """High-level Raft consensus interface."""
    
    def __init__(self, config: RaftConfig):
        self.config = config
        self.metrics = RaftMetrics()
        self.event_producer = EventProducer(config.kafka_bootstrap_servers)
        
        rpc = TemporalRaftRPC(config.node_id) if config.use_temporal_for_rpc else None
        self.core = RaftCore(config, rpc, self.event_producer, self.metrics)
    
    async def start(self):
        """Start Raft consensus."""
        await self.core.start()
    
    async def stop(self):
        """Stop Raft consensus."""
        await self.core.stop()
    
    async def propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Propose value for consensus."""
        return await self.core.propose(request)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get consensus status."""
        return {
            "node_id": self.config.node_id,
            "state": self.core.state_machine.state.value,
            "term": self.core.state_machine.term,
            "leader": self.core.state_machine.leader_id,
            "log_size": len(self.core.log.entries),
            "commit_index": self.core.commit_index,
            "last_applied": self.core.last_applied,
            "peers": self.config.peers
        }