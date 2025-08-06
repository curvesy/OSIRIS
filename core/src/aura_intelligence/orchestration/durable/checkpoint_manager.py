"""
💾 Checkpoint Manager

Manages workflow state checkpointing and recovery for durable orchestration.
Provides automatic checkpointing, recovery strategies, and TDA-aware state
management for fault-tolerant multi-agent workflows.

Key Features:
- Automatic workflow state checkpointing
- Multiple recovery strategies (rollback, forward recovery)
- TDA context preservation across checkpoints
- Efficient checkpoint storage and retrieval

TDA Integration:
- Preserves TDA context in checkpoints
- Uses TDA patterns for recovery strategy selection
- Correlates checkpoint performance with TDA metrics
- Implements TDA-aware checkpoint optimization
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import pickle
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

# TDA integration
try:
    from aura_intelligence.observability.tracing import get_tracer
    from ..semantic.tda_integration import TDAContextIntegration
    from ..semantic.base_interfaces import TDAContext
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None
    TDAContextIntegration = None
    TDAContext = None

class CheckpointStatus(Enum):
    """Status of workflow checkpoints"""
    CREATED = "created"
    ACTIVE = "active"
    RECOVERED = "recovered"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"

class RecoveryStrategy(Enum):
    """Recovery strategies for checkpoint restoration"""
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    FORWARD_RECOVERY = "forward_recovery"
    HYBRID_RECOVERY = "hybrid_recovery"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class WorkflowCheckpoint:
    """Workflow state checkpoint"""
    checkpoint_id: str
    workflow_id: str
    step_index: int
    step_name: str
    workflow_state: Dict[str, Any]
    tda_context: Optional[Dict[str, Any]]
    timestamp: datetime
    status: CheckpointStatus = CheckpointStatus.CREATED
    metadata: Dict[str, Any] = None
    checksum: Optional[str] = None
    size_bytes: int = 0

class CheckpointManager:
    """
    Manages workflow checkpoints and recovery operations
    """
    
    def __init__(
        self,
        storage_backend: Optional[str] = "memory",
        tda_integration: Optional[TDAContextIntegration] = None,
        checkpoint_retention_hours: int = 24
    ):
        self.storage_backend = storage_backend
        self.tda_integration = tda_integration or TDAContextIntegration() if TDAContextIntegration else None
        self.checkpoint_retention_hours = checkpoint_retention_hours
        
        # In-memory storage (replace with persistent storage in production)
        self.checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self.workflow_checkpoints: Dict[str, List[str]] = {}  # workflow_id -> checkpoint_ids
        
        # Recovery statistics
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "recovery_times": [],
            "strategy_usage": {strategy.value: 0 for strategy in RecoveryStrategy}
        }
    
    async def create_checkpoint(
        self,
        workflow_id: str,
        step_index: int,
        step_name: str,
        workflow_state: Dict[str, Any],
        tda_correlation_id: Optional[str] = None
    ) -> WorkflowCheckpoint:
        """
        Create a new workflow checkpoint
        """
        if tracer:
            with tracer.start_as_current_span("create_checkpoint") as span:
                span.set_attributes({
                    "workflow.id": workflow_id,
                    "checkpoint.step_index": step_index,
                    "checkpoint.step_name": step_name,
                    "tda.correlation_id": tda_correlation_id or "none"
                })
        
        checkpoint_id = f"{workflow_id}_checkpoint_{step_index}_{uuid.uuid4().hex[:8]}"
        
        # Get TDA context for checkpoint
        tda_context = None
        if self.tda_integration and tda_correlation_id:
            tda_context = await self.tda_integration.get_context(tda_correlation_id)
        
        # Create checkpoint
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            step_index=step_index,
            step_name=step_name,
            workflow_state=workflow_state,
            tda_context=asdict(tda_context) if tda_context else None,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "tda_correlation_id": tda_correlation_id,
                "creation_method": "automatic",
                "storage_backend": self.storage_backend
            }
        )
        
        # Calculate checksum for integrity verification
        checkpoint.checksum = self._calculate_checksum(checkpoint)
        checkpoint.size_bytes = len(json.dumps(asdict(checkpoint), default=str))
        
        # Store checkpoint
        await self._store_checkpoint(checkpoint)
        
        # Update workflow checkpoint tracking
        if workflow_id not in self.workflow_checkpoints:
            self.workflow_checkpoints[workflow_id] = []
        self.workflow_checkpoints[workflow_id].append(checkpoint_id)
        
        # Send checkpoint creation to TDA
        if self.tda_integration and tda_correlation_id:
            await self.tda_integration.send_orchestration_result(
                {
                    "event_type": "checkpoint_created",
                    "checkpoint_id": checkpoint_id,
                    "workflow_id": workflow_id,
                    "step_index": step_index,
                    "size_bytes": checkpoint.size_bytes
                },
                tda_correlation_id
            )
        
        return checkpoint
    
    async def recover_from_checkpoint(
        self,
        checkpoint_id: str,
        recovery_strategy: Optional[RecoveryStrategy] = None,
        tda_correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recover workflow from a specific checkpoint
        """
        if tracer:
            with tracer.start_as_current_span("recover_from_checkpoint") as span:
                span.set_attributes({
                    "checkpoint.id": checkpoint_id,
                    "recovery.strategy": recovery_strategy.value,
                    "tda.correlation_id": tda_correlation_id or "none"
                })
        
        start_time = datetime.now(timezone.utc)
        self.recovery_stats["total_recoveries"] += 1
        
        # Determine recovery strategy if not specified
        if not recovery_strategy:
            recovery_strategy = await self._determine_optimal_recovery_strategy(
                checkpoint_id, tda_correlation_id
            )
        
        self.recovery_stats["strategy_usage"][recovery_strategy.value] += 1
        
        try:
            # Retrieve checkpoint
            checkpoint = await self._retrieve_checkpoint(checkpoint_id)
            if not checkpoint:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            # Verify checkpoint integrity
            if not self._verify_checkpoint_integrity(checkpoint):
                checkpoint.status = CheckpointStatus.CORRUPTED
                raise ValueError(f"Checkpoint {checkpoint_id} is corrupted")
            
            # Execute recovery strategy
            recovery_result = await self._execute_recovery_strategy(
                checkpoint, recovery_strategy, tda_correlation_id
            )
            
            # Update checkpoint status
            checkpoint.status = CheckpointStatus.RECOVERED
            await self._store_checkpoint(checkpoint)
            
            # Record successful recovery
            recovery_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.recovery_stats["successful_recoveries"] += 1
            self.recovery_stats["recovery_times"].append(recovery_time)
            
            # Send recovery success to TDA
            if self.tda_integration and tda_correlation_id:
                await self.tda_integration.send_orchestration_result(
                    {
                        "event_type": "checkpoint_recovery_success",
                        "checkpoint_id": checkpoint_id,
                        "recovery_strategy": recovery_strategy.value,
                        "recovery_time": recovery_time,
                        "workflow_id": checkpoint.workflow_id
                    },
                    tda_correlation_id
                )
            
            return {
                "status": "success",
                "checkpoint_id": checkpoint_id,
                "recovery_strategy": recovery_strategy.value,
                "recovery_time": recovery_time,
                "workflow_state": recovery_result["workflow_state"],
                "tda_context": recovery_result.get("tda_context")
            }
            
        except Exception as e:
            # Record failed recovery
            recovery_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Send recovery failure to TDA
            if self.tda_integration and tda_correlation_id:
                await self.tda_integration.send_orchestration_result(
                    {
                        "event_type": "checkpoint_recovery_failure",
                        "checkpoint_id": checkpoint_id,
                        "recovery_strategy": recovery_strategy.value,
                        "error": str(e),
                        "recovery_time": recovery_time
                    },
                    tda_correlation_id
                )
            
            return {
                "status": "failed",
                "checkpoint_id": checkpoint_id,
                "error": str(e),
                "recovery_time": recovery_time
            }
    
    async def _execute_recovery_strategy(
        self,
        checkpoint: WorkflowCheckpoint,
        strategy: RecoveryStrategy,
        tda_correlation_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute the specified recovery strategy
        """
        if strategy == RecoveryStrategy.ROLLBACK_TO_CHECKPOINT:
            return await self._rollback_recovery(checkpoint, tda_correlation_id)
        elif strategy == RecoveryStrategy.FORWARD_RECOVERY:
            return await self._forward_recovery(checkpoint, tda_correlation_id)
        elif strategy == RecoveryStrategy.HYBRID_RECOVERY:
            return await self._hybrid_recovery(checkpoint, tda_correlation_id)
        else:
            return await self._manual_intervention_recovery(checkpoint, tda_correlation_id)
    
    async def _rollback_recovery(
        self,
        checkpoint: WorkflowCheckpoint,
        tda_correlation_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Rollback to checkpoint state
        """
        # Restore workflow state exactly as it was at checkpoint
        return {
            "workflow_state": checkpoint.workflow_state,
            "tda_context": checkpoint.tda_context,
            "recovery_method": "rollback",
            "step_index": checkpoint.step_index,
            "step_name": checkpoint.step_name
        }
    
    async def _forward_recovery(
        self,
        checkpoint: WorkflowCheckpoint,
        tda_correlation_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Forward recovery with TDA context enhancement
        """
        # Get current TDA context for forward recovery
        current_tda_context = None
        if self.tda_integration and tda_correlation_id:
            current_tda_context = await self.tda_integration.get_context(tda_correlation_id)
        
        # Merge checkpoint state with current context
        enhanced_state = checkpoint.workflow_state.copy()
        if current_tda_context:
            enhanced_state["tda_context"] = asdict(current_tda_context)
            enhanced_state["recovery_enhancement"] = {
                "original_tda_context": checkpoint.tda_context,
                "current_tda_context": asdict(current_tda_context),
                "enhancement_timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "workflow_state": enhanced_state,
            "tda_context": asdict(current_tda_context) if current_tda_context else checkpoint.tda_context,
            "recovery_method": "forward_recovery",
            "step_index": checkpoint.step_index,
            "step_name": checkpoint.step_name
        }
    
    async def _hybrid_recovery(
        self,
        checkpoint: WorkflowCheckpoint,
        tda_correlation_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Hybrid recovery combining rollback and forward strategies
        """
        # Start with rollback state
        rollback_result = await self._rollback_recovery(checkpoint, tda_correlation_id)
        
        # Enhance with forward recovery elements
        forward_result = await self._forward_recovery(checkpoint, tda_correlation_id)
        
        # Combine strategies intelligently
        hybrid_state = rollback_result["workflow_state"].copy()
        hybrid_state.update({
            "hybrid_recovery": {
                "rollback_state": rollback_result["workflow_state"],
                "forward_enhancements": forward_result["workflow_state"].get("recovery_enhancement", {}),
                "strategy": "hybrid"
            }
        })
        
        return {
            "workflow_state": hybrid_state,
            "tda_context": forward_result["tda_context"],
            "recovery_method": "hybrid",
            "step_index": checkpoint.step_index,
            "step_name": checkpoint.step_name
        }
    
    async def _manual_intervention_recovery(
        self,
        checkpoint: WorkflowCheckpoint,
        tda_correlation_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Manual intervention recovery (placeholder for human intervention)
        """
        return {
            "workflow_state": checkpoint.workflow_state,
            "tda_context": checkpoint.tda_context,
            "recovery_method": "manual_intervention",
            "step_index": checkpoint.step_index,
            "step_name": checkpoint.step_name,
            "requires_manual_action": True,
            "intervention_instructions": [
                "Review checkpoint state for consistency",
                "Verify TDA context is still valid",
                "Manually resolve any state conflicts",
                "Resume workflow from appropriate step"
            ]
        }
    
    async def _store_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """
        Store checkpoint using configured backend
        """
        if self.storage_backend == "memory":
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        else:
            # Placeholder for other storage backends (Redis, S3, etc.)
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
    
    async def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Retrieve checkpoint from storage
        """
        return self.checkpoints.get(checkpoint_id)
    
    def _calculate_checksum(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Calculate checksum for checkpoint integrity verification
        """
        # Create a deterministic representation of the checkpoint
        checkpoint_data = {
            "workflow_id": checkpoint.workflow_id,
            "step_index": checkpoint.step_index,
            "workflow_state": checkpoint.workflow_state,
            "tda_context": checkpoint.tda_context
        }
        
        checkpoint_json = json.dumps(checkpoint_data, sort_keys=True, default=str)
        return hashlib.sha256(checkpoint_json.encode()).hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint: WorkflowCheckpoint) -> bool:
        """
        Verify checkpoint integrity using checksum
        """
        if not checkpoint.checksum:
            return False
        
        calculated_checksum = self._calculate_checksum(checkpoint)
        return calculated_checksum == checkpoint.checksum
    
    async def cleanup_expired_checkpoints(self):
        """
        Clean up expired checkpoints based on retention policy
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.checkpoint_retention_hours)
        expired_checkpoints = []
        
        for checkpoint_id, checkpoint in self.checkpoints.items():
            if checkpoint.timestamp < cutoff_time:
                expired_checkpoints.append(checkpoint_id)
                checkpoint.status = CheckpointStatus.EXPIRED
        
        # Remove expired checkpoints
        for checkpoint_id in expired_checkpoints:
            del self.checkpoints[checkpoint_id]
            
            # Update workflow checkpoint tracking
            for workflow_id, checkpoint_ids in self.workflow_checkpoints.items():
                if checkpoint_id in checkpoint_ids:
                    checkpoint_ids.remove(checkpoint_id)
        
        return len(expired_checkpoints)
    
    def get_workflow_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        """
        Get all checkpoints for a specific workflow
        """
        checkpoint_ids = self.workflow_checkpoints.get(workflow_id, [])
        return [self.checkpoints[cid] for cid in checkpoint_ids if cid in self.checkpoints]
    
    def get_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Get the latest checkpoint for a workflow
        """
        checkpoints = self.get_workflow_checkpoints(workflow_id)
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda c: c.timestamp)
    
    async def _determine_optimal_recovery_strategy(
        self,
        checkpoint_id: str,
        tda_correlation_id: Optional[str]
    ) -> RecoveryStrategy:
        """
        Intelligently determine the optimal recovery strategy based on context
        """
        # Get TDA context for intelligent decision making
        tda_context = None
        if self.tda_integration and tda_correlation_id:
            try:
                tda_context = await self.tda_integration.get_context(tda_correlation_id)
            except Exception:
                pass  # Continue without TDA context
        
        # Get checkpoint to analyze
        checkpoint = await self._retrieve_checkpoint(checkpoint_id)
        if not checkpoint:
            return RecoveryStrategy.ROLLBACK_TO_CHECKPOINT  # Safe default
        
        # Analyze checkpoint age
        checkpoint_age = (datetime.now(timezone.utc) - checkpoint.timestamp).total_seconds()
        
        # Decision logic based on multiple factors
        if tda_context:
            # High urgency scenarios prefer forward recovery
            if hasattr(tda_context, 'urgency_level') and tda_context.urgency_level == 'critical':
                return RecoveryStrategy.FORWARD_RECOVERY
            
            # High anomaly severity suggests hybrid approach
            if tda_context.anomaly_severity > 0.8:
                return RecoveryStrategy.HYBRID_RECOVERY
            
            # Complex scenarios benefit from hybrid recovery
            if hasattr(tda_context, 'complexity_score') and tda_context.complexity_score > 0.7:
                return RecoveryStrategy.HYBRID_RECOVERY
        
        # Time-based decisions
        if checkpoint_age < 300:  # Less than 5 minutes old
            return RecoveryStrategy.ROLLBACK_TO_CHECKPOINT  # Recent, safe to rollback
        elif checkpoint_age < 1800:  # Less than 30 minutes old
            return RecoveryStrategy.FORWARD_RECOVERY  # Moderate age, try forward recovery
        else:
            return RecoveryStrategy.HYBRID_RECOVERY  # Old checkpoint, use hybrid approach
    
    def get_checkpoint_metrics(self) -> Dict[str, Any]:
        """
        Get checkpoint management metrics
        """
        total_checkpoints = len(self.checkpoints)
        active_checkpoints = sum(1 for c in self.checkpoints.values() if c.status == CheckpointStatus.ACTIVE)
        corrupted_checkpoints = sum(1 for c in self.checkpoints.values() if c.status == CheckpointStatus.CORRUPTED)
        
        avg_recovery_time = 0.0
        if self.recovery_stats["recovery_times"]:
            avg_recovery_time = sum(self.recovery_stats["recovery_times"]) / len(self.recovery_stats["recovery_times"])
        
        success_rate = 0.0
        if self.recovery_stats["total_recoveries"] > 0:
            success_rate = self.recovery_stats["successful_recoveries"] / self.recovery_stats["total_recoveries"]
        
        return {
            "total_checkpoints": total_checkpoints,
            "active_checkpoints": active_checkpoints,
            "corrupted_checkpoints": corrupted_checkpoints,
            "total_recoveries": self.recovery_stats["total_recoveries"],
            "successful_recoveries": self.recovery_stats["successful_recoveries"],
            "recovery_success_rate": success_rate,
            "average_recovery_time": avg_recovery_time,
            "strategy_usage": self.recovery_stats["strategy_usage"]
        }