"""
LNN Consensus Integration for AURA Intelligence.

This module integrates Liquid Neural Networks with Byzantine consensus protocols
to enable distributed, fault-tolerant AI decision-making. Based on the latest
research in decentralized AI coordination (2025).

Key Features:
- Distributed LNN inference with consensus
- Byzantine fault tolerance for adversarial environments
- Weighted voting based on model confidence
- Edge-optimized consensus for tactical deployment
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
import numpy as np
import structlog
from opentelemetry import trace

from .lnn import LiquidNeuralNetwork, EdgeLNN, LNNConfig
from ..consensus import (
    ConsensusManager,
    ConsensusLevel,
    Decision,
    ConsensusResult,
    ByzantineConsensus,
    RaftConsensus
)
from ..events import EventProducer, EventConsumer
from ..observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("lnn_consensus")


@dataclass
class LNNDecision:
    """Decision made by an LNN with confidence metrics."""
    model_id: str
    output: torch.Tensor
    confidence: float
    inference_time_ms: float
    adaptation_count: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ConsensusLNNConfig:
    """Configuration for consensus-based LNN."""
    lnn_config: LNNConfig
    consensus_threshold: float = 0.67  # 2/3 majority
    min_participants: int = 3
    max_inference_time_ms: float = 1000.0
    use_weighted_voting: bool = True
    edge_deployment: bool = False
    enable_hedged_requests: bool = True
    byzantine_tolerance: int = 1  # f in 3f+1


class DistributedLNN:
    """
    Distributed Liquid Neural Network with consensus.
    
    Enables multiple LNN instances to collaborate on decisions
    with Byzantine fault tolerance.
    """
    
    def __init__(
        self,
        node_id: str,
        config: ConsensusLNNConfig,
        consensus_manager: ConsensusManager,
        event_producer: Optional[EventProducer] = None
    ):
        self.node_id = node_id
        self.config = config
        self.consensus_manager = consensus_manager
        self.event_producer = event_producer
        
        # Create local LNN
        if config.edge_deployment:
            self.lnn = EdgeLNN(config.lnn_config)
        else:
            self.lnn = LiquidNeuralNetwork(config.lnn_config)
        
        # Peer tracking
        self.peers: Dict[str, float] = {}  # node_id -> trust_score
        self.decision_history: List[LNNDecision] = []
        
        # Metrics
        self.consensus_successes = 0
        self.consensus_failures = 0
        self.total_decisions = 0
    
    @tracer.start_as_current_span("distributed_lnn_inference")
    async def distributed_inference(
        self,
        input_data: torch.Tensor,
        decision_id: str,
        timeout: Optional[timedelta] = None
    ) -> Tuple[torch.Tensor, ConsensusResult]:
        """
        Perform distributed inference with consensus.
        
        Args:
            input_data: Input tensor for LNN
            decision_id: Unique identifier for this decision
            timeout: Maximum time to wait for consensus
            
        Returns:
            (consensus_output, consensus_result)
        """
        timeout = timeout or timedelta(milliseconds=self.config.max_inference_time_ms)
        
        # Local inference
        local_decision = await self._local_inference(input_data, decision_id)
        
        # Create consensus decision
        consensus_decision = Decision(
            id=decision_id,
            type="lnn_inference",
            data={
                "output": local_decision.output.tolist(),
                "confidence": local_decision.confidence,
                "model_id": local_decision.model_id,
                "metadata": local_decision.metadata
            },
            proposer=self.node_id,
            timestamp=datetime.utcnow()
        )
        
        # Submit to consensus
        if self.config.byzantine_tolerance > 0:
            # Use Byzantine consensus for adversarial environments
            result = await self._byzantine_consensus(consensus_decision, timeout)
        else:
            # Use Raft for benign failures only
            result = await self.consensus_manager.propose(
                consensus_decision,
                ConsensusLevel.STRONG
            )
        
        # Process consensus result
        if result.accepted:
            consensus_output = await self._aggregate_decisions(result)
            self.consensus_successes += 1
        else:
            # Fallback to local decision
            consensus_output = local_decision.output
            self.consensus_failures += 1
            logger.warning(
                "Consensus failed, using local decision",
                decision_id=decision_id,
                reason=result.metadata.get("reason")
            )
        
        self.total_decisions += 1
        
        # Adapt based on consensus feedback
        if result.accepted and self.config.lnn_config.consensus_enabled:
            await self._adapt_from_consensus(local_decision, result)
        
        return consensus_output, result
    
    async def _local_inference(
        self,
        input_data: torch.Tensor,
        decision_id: str
    ) -> LNNDecision:
        """Perform local LNN inference."""
        start_time = datetime.utcnow()
        
        # Run inference
        with torch.no_grad():
            output = self.lnn(input_data)
        
        # Calculate confidence (using output entropy)
        if output.dim() > 1:
            probs = torch.softmax(output, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            confidence = 1.0 - entropy.mean().item() / np.log(output.size(-1))
        else:
            confidence = torch.sigmoid(output).mean().item()
        
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Get LNN metrics
        if hasattr(self.lnn, 'get_metrics'):
            metrics = self.lnn.get_metrics()
        else:
            metrics = {}
        
        decision = LNNDecision(
            model_id=self.node_id,
            output=output,
            confidence=float(confidence),
            inference_time_ms=inference_time,
            adaptation_count=metrics.get('adaptation_count', 0),
            timestamp=datetime.utcnow(),
            metadata=metrics
        )
        
        self.decision_history.append(decision)
        
        # Publish to event stream if available
        if self.event_producer:
            await self._publish_decision_event(decision, decision_id)
        
        return decision
    
    async def _byzantine_consensus(
        self,
        decision: Decision,
        timeout: timedelta
    ) -> ConsensusResult:
        """Run Byzantine fault tolerant consensus."""
        # Create Byzantine consensus instance
        bft_config = {
            "node_id": self.node_id,
            "f": self.config.byzantine_tolerance,
            "view_timeout_ms": int(timeout.total_seconds() * 1000)
        }
        
        bft = ByzantineConsensus(
            node_id=self.node_id,
            peers=list(self.peers.keys()),
            **bft_config
        )
        
        # Run consensus
        return await bft.propose(decision)
    
    async def _aggregate_decisions(
        self,
        consensus_result: ConsensusResult
    ) -> torch.Tensor:
        """Aggregate decisions from multiple nodes."""
        decisions = consensus_result.metadata.get("decisions", [])
        
        if not decisions:
            raise ValueError("No decisions to aggregate")
        
        # Extract outputs and weights
        outputs = []
        weights = []
        
        for d in decisions:
            output = torch.tensor(d["output"])
            outputs.append(output)
            
            if self.config.use_weighted_voting:
                # Weight by confidence and trust score
                confidence = d.get("confidence", 1.0)
                trust = self.peers.get(d["model_id"], 1.0)
                weight = confidence * trust
            else:
                weight = 1.0
            
            weights.append(weight)
        
        # Stack and aggregate
        outputs = torch.stack(outputs)
        weights = torch.tensor(weights).unsqueeze(-1)
        weights = weights / weights.sum()
        
        # Weighted average
        aggregated = (outputs * weights).sum(dim=0)
        
        return aggregated
    
    async def _adapt_from_consensus(
        self,
        local_decision: LNNDecision,
        consensus_result: ConsensusResult
    ):
        """Adapt LNN based on consensus feedback."""
        # Calculate error between local and consensus
        consensus_output = torch.tensor(
            consensus_result.metadata.get("consensus_output", [])
        )
        
        if consensus_output.numel() == 0:
            return
        
        error = consensus_output - local_decision.output
        
        # Adapt if error is significant
        if error.abs().mean() > 0.1:
            self.lnn.adapt(error)
            logger.info(
                "Adapted LNN from consensus feedback",
                error_magnitude=float(error.abs().mean())
            )
    
    async def _publish_decision_event(
        self,
        decision: LNNDecision,
        decision_id: str
    ):
        """Publish decision to event stream."""
        event = {
            "type": "lnn_decision",
            "decision_id": decision_id,
            "model_id": decision.model_id,
            "confidence": decision.confidence,
            "inference_time_ms": decision.inference_time_ms,
            "timestamp": decision.timestamp.isoformat()
        }
        
        await self.event_producer.send_event(
            topic="lnn-decisions",
            key=decision_id,
            value=event
        )
    
    def update_peer_trust(self, peer_id: str, performance: float):
        """Update trust score for a peer based on performance."""
        if peer_id not in self.peers:
            self.peers[peer_id] = 1.0
        
        # Exponential moving average
        alpha = 0.1
        self.peers[peer_id] = (
            alpha * performance + (1 - alpha) * self.peers[peer_id]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get distributed LNN metrics."""
        base_metrics = self.lnn.get_metrics() if hasattr(self.lnn, 'get_metrics') else {}
        
        return {
            **base_metrics,
            "consensus_success_rate": (
                self.consensus_successes / self.total_decisions
                if self.total_decisions > 0 else 0.0
            ),
            "total_decisions": self.total_decisions,
            "peer_count": len(self.peers),
            "average_peer_trust": (
                np.mean(list(self.peers.values()))
                if self.peers else 1.0
            )
        }


class LNNConsensusOrchestrator:
    """
    Orchestrates multiple distributed LNN nodes.
    
    Manages the lifecycle of distributed LNN inference
    including node discovery, load balancing, and failover.
    """
    
    def __init__(
        self,
        config: ConsensusLNNConfig,
        kafka_servers: Optional[List[str]] = None
    ):
        self.config = config
        self.nodes: Dict[str, DistributedLNN] = {}
        self.kafka_servers = kafka_servers
        self.consensus_manager = ConsensusManager()
        
        # Load balancing state
        self.node_loads: Dict[str, float] = {}
        self.failed_nodes: set = set()
    
    async def add_node(
        self,
        node_id: str,
        lnn_config: Optional[LNNConfig] = None
    ) -> DistributedLNN:
        """Add a new LNN node to the cluster."""
        config = ConsensusLNNConfig(
            lnn_config=lnn_config or self.config.lnn_config,
            **{k: v for k, v in self.config.__dict__.items() if k != 'lnn_config'}
        )
        
        # Create event producer if Kafka available
        event_producer = None
        if self.kafka_servers:
            event_producer = EventProducer(
                bootstrap_servers=self.kafka_servers,
                client_id=f"lnn-{node_id}"
            )
            await event_producer.start()
        
        # Create distributed LNN
        node = DistributedLNN(
            node_id=node_id,
            config=config,
            consensus_manager=self.consensus_manager,
            event_producer=event_producer
        )
        
        # Update peer lists
        for existing_id, existing_node in self.nodes.items():
            existing_node.peers[node_id] = 1.0
            node.peers[existing_id] = 1.0
        
        self.nodes[node_id] = node
        self.node_loads[node_id] = 0.0
        
        logger.info(f"Added LNN node: {node_id}")
        return node
    
    async def remove_node(self, node_id: str):
        """Remove a node from the cluster."""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            
            # Clean up event producer
            if node.event_producer:
                await node.event_producer.stop()
            
            # Remove from peer lists
            for other_node in self.nodes.values():
                other_node.peers.pop(node_id, None)
            
            self.node_loads.pop(node_id, None)
            self.failed_nodes.discard(node_id)
            
            logger.info(f"Removed LNN node: {node_id}")
    
    async def distributed_inference(
        self,
        input_data: torch.Tensor,
        decision_id: str,
        required_nodes: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform distributed inference across multiple nodes.
        
        Args:
            input_data: Input for LNN inference
            decision_id: Unique decision identifier
            required_nodes: Number of nodes to use (default: based on Byzantine tolerance)
            
        Returns:
            (consensus_output, metadata)
        """
        required_nodes = required_nodes or (3 * self.config.byzantine_tolerance + 1)
        
        # Select nodes based on load
        selected_nodes = await self._select_nodes(required_nodes)
        
        if len(selected_nodes) < self.config.min_participants:
            raise ValueError(
                f"Not enough healthy nodes: {len(selected_nodes)} < {self.config.min_participants}"
            )
        
        # Run distributed inference
        tasks = []
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            task = asyncio.create_task(
                self._node_inference_with_timeout(node, input_data, decision_id)
            )
            tasks.append((node_id, task))
        
        # Collect results
        results = []
        failed_count = 0
        
        for node_id, task in tasks:
            try:
                output, result = await task
                results.append({
                    "node_id": node_id,
                    "output": output,
                    "result": result
                })
            except asyncio.TimeoutError:
                logger.warning(f"Node {node_id} timed out")
                self._mark_node_failed(node_id)
                failed_count += 1
            except Exception as e:
                logger.error(f"Node {node_id} failed: {e}")
                self._mark_node_failed(node_id)
                failed_count += 1
        
        # Aggregate successful results
        if len(results) >= self.config.min_participants:
            consensus_output = await self._final_aggregation(results)
            
            metadata = {
                "successful_nodes": len(results),
                "failed_nodes": failed_count,
                "selected_nodes": selected_nodes,
                "decision_id": decision_id
            }
            
            return consensus_output, metadata
        else:
            raise RuntimeError(
                f"Insufficient successful nodes: {len(results)} < {self.config.min_participants}"
            )
    
    async def _select_nodes(self, count: int) -> List[str]:
        """Select nodes based on load and health."""
        # Filter healthy nodes
        healthy_nodes = [
            node_id for node_id in self.nodes
            if node_id not in self.failed_nodes
        ]
        
        # Sort by load
        sorted_nodes = sorted(
            healthy_nodes,
            key=lambda n: self.node_loads.get(n, 0.0)
        )
        
        return sorted_nodes[:count]
    
    async def _node_inference_with_timeout(
        self,
        node: DistributedLNN,
        input_data: torch.Tensor,
        decision_id: str
    ) -> Tuple[torch.Tensor, ConsensusResult]:
        """Run node inference with timeout."""
        timeout = self.config.max_inference_time_ms / 1000.0
        
        # Update load
        self.node_loads[node.node_id] += 1
        
        try:
            result = await asyncio.wait_for(
                node.distributed_inference(input_data, decision_id),
                timeout=timeout
            )
            return result
        finally:
            # Decrease load
            self.node_loads[node.node_id] = max(
                0, self.node_loads[node.node_id] - 1
            )
    
    async def _final_aggregation(
        self,
        results: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Final aggregation of all node results."""
        outputs = []
        weights = []
        
        for r in results:
            output = r["output"]
            result = r["result"]
            
            # Weight by consensus quality
            weight = result.metadata.get("confidence", 1.0)
            
            outputs.append(output)
            weights.append(weight)
        
        # Convert to tensors
        outputs = torch.stack([o if isinstance(o, torch.Tensor) else torch.tensor(o) for o in outputs])
        weights = torch.tensor(weights).unsqueeze(-1)
        weights = weights / weights.sum()
        
        # Weighted average
        return (outputs * weights).sum(dim=0)
    
    def _mark_node_failed(self, node_id: str):
        """Mark a node as failed."""
        self.failed_nodes.add(node_id)
        
        # Schedule recovery check
        asyncio.create_task(self._check_node_recovery(node_id))
    
    async def _check_node_recovery(self, node_id: str):
        """Check if a failed node has recovered."""
        await asyncio.sleep(60)  # Wait 1 minute
        
        if node_id in self.nodes and node_id in self.failed_nodes:
            # Try a health check
            try:
                node = self.nodes[node_id]
                test_input = torch.randn(1, node.config.lnn_config.input_size)
                await asyncio.wait_for(
                    node.distributed_inference(test_input, "health_check"),
                    timeout=5.0
                )
                
                # Node recovered
                self.failed_nodes.remove(node_id)
                logger.info(f"Node {node_id} recovered")
                
            except Exception:
                # Still failed, check again later
                asyncio.create_task(self._check_node_recovery(node_id))
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire cluster."""
        node_metrics = {}
        
        for node_id, node in self.nodes.items():
            node_metrics[node_id] = node.get_metrics()
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len(self.nodes) - len(self.failed_nodes),
            "failed_nodes": list(self.failed_nodes),
            "average_load": np.mean(list(self.node_loads.values())) if self.node_loads else 0.0,
            "node_metrics": node_metrics
        }


# Example usage functions
async def create_lnn_cluster(
    num_nodes: int = 5,
    input_size: int = 128,
    hidden_size: int = 256,
    output_size: int = 10,
    edge_deployment: bool = False
) -> LNNConsensusOrchestrator:
    """Create a cluster of distributed LNN nodes."""
    # Configure LNN
    lnn_config = LNNConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=3,
        sparsity=0.8 if edge_deployment else 0.5,
        consensus_enabled=True,
        use_cuda=not edge_deployment
    )
    
    # Configure consensus
    consensus_config = ConsensusLNNConfig(
        lnn_config=lnn_config,
        byzantine_tolerance=1,  # Tolerate 1 Byzantine node
        min_participants=3,
        edge_deployment=edge_deployment,
        use_weighted_voting=True
    )
    
    # Create orchestrator
    orchestrator = LNNConsensusOrchestrator(consensus_config)
    
    # Add nodes
    for i in range(num_nodes):
        await orchestrator.add_node(f"lnn-node-{i}")
    
    return orchestrator


async def run_distributed_lnn_inference(
    orchestrator: LNNConsensusOrchestrator,
    input_data: torch.Tensor
) -> torch.Tensor:
    """Run a distributed inference example."""
    decision_id = f"decision-{datetime.utcnow().timestamp()}"
    
    output, metadata = await orchestrator.distributed_inference(
        input_data,
        decision_id
    )
    
    logger.info(
        "Distributed inference complete",
        decision_id=decision_id,
        **metadata
    )
    
    return output