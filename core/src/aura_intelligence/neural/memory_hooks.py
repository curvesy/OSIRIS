"""
Memory Hooks for LNN Integration.

Background hooks for indexing LNN decisions and adaptations into the
memory system for future retrieval and analysis.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
import asyncio
import logging
from dataclasses import dataclass
from collections import deque
import json

from ..observability import create_tracer
from ..resilience import resilient, ResilienceLevel

logger = logging.getLogger(__name__)
tracer = create_tracer("lnn_memory_hooks")


@dataclass
class MemoryEvent:
    """Event to be processed by memory hooks."""
    event_type: str
    agent_id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime


class LNNMemoryHooks:
    """Background hooks for indexing LNN decisions."""
    
    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        retention_days: int = 30
    ):
        self.memory = memory_manager
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.retention_days = retention_days
        
        # Event queue for batching
        self._event_queue: deque = deque(maxlen=1000)
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self._events_processed = 0
        self._events_failed = 0
        self._last_flush = datetime.now(timezone.utc)
        
    async def start(self):
        """Start background processing."""
        if self._running:
            return
            
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("LNN memory hooks started")
        
    async def stop(self):
        """Stop background processing."""
        self._running = False
        
        # Flush remaining events
        await self._flush_events()
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"LNN memory hooks stopped. Processed: {self._events_processed}, Failed: {self._events_failed}")
        
    @resilient(criticality=ResilienceLevel.STANDARD)
    async def on_inference_complete(self, result: Dict[str, Any]):
        """Index inference results for future retrieval."""
        with tracer.start_as_current_span("on_inference_complete") as span:
            span.set_attribute("result.confidence", result.get("confidence", 0.0))
            
            event = MemoryEvent(
                event_type="inference",
                agent_id="lnn",
                content={
                    "predictions": result.get("predictions"),
                    "confidence": result.get("confidence", 0.0),
                    "context_influence": result.get("context_influence", 0.0),
                    "similar_patterns": result.get("similar_patterns", [])
                },
                metadata={
                    "timestamp": datetime.now(timezone.utc),
                    "model_version": result.get("inference_metadata", {}).get("model_version", "1.0"),
                    "latency_ms": result.get("inference_metadata", {}).get("latency_ms", 0),
                    "context_used": result.get("inference_metadata", {}).get("context_used", False)
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            await self._enqueue_event(event)
            
    @resilient(criticality=ResilienceLevel.STANDARD)
    async def on_adaptation(self, adaptation_event: Dict[str, Any]):
        """Track model adaptations for analysis."""
        with tracer.start_as_current_span("on_adaptation") as span:
            span.set_attribute("adaptation.trigger", adaptation_event.get("trigger", "unknown"))
            
            event = MemoryEvent(
                event_type="adaptation",
                agent_id="lnn",
                content={
                    "parameters_changed": adaptation_event.get("parameters_changed", []),
                    "performance_before": adaptation_event.get("performance_before", {}),
                    "performance_after": adaptation_event.get("performance_after", {}),
                    "trigger": adaptation_event.get("trigger", "unknown")
                },
                metadata={
                    "timestamp": datetime.now(timezone.utc),
                    "adaptation_rate": adaptation_event.get("adaptation_rate", 0.01),
                    "performance_delta": adaptation_event.get("performance_delta", 0.0)
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            await self._enqueue_event(event)
            
    @resilient(criticality=ResilienceLevel.STANDARD)
    async def on_pattern_detected(self, pattern: Dict[str, Any]):
        """Index detected patterns for future similarity search."""
        with tracer.start_as_current_span("on_pattern_detected") as span:
            span.set_attribute("pattern.type", pattern.get("type", "unknown"))
            
            event = MemoryEvent(
                event_type="pattern",
                agent_id="lnn",
                content={
                    "pattern_type": pattern.get("type"),
                    "features": pattern.get("features", []),
                    "confidence": pattern.get("confidence", 0.0),
                    "context": pattern.get("context", {})
                },
                metadata={
                    "timestamp": datetime.now(timezone.utc),
                    "source": pattern.get("source", "lnn"),
                    "importance": pattern.get("importance", 0.5)
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            await self._enqueue_event(event)
            
    async def search_similar_decisions(
        self,
        query_features: List[float],
        limit: int = 10,
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Search for similar past decisions."""
        if not self.memory:
            return []
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        results = await self.memory.search_memories(
            agent_id="lnn",
            query_embedding=query_features,
            memory_types=["inference"],
            min_timestamp=cutoff_time,
            limit=limit
        )
        
        return results
        
    async def get_adaptation_history(
        self,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent adaptation history."""
        if not self.memory:
            return []
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        adaptations = await self.memory.get_memories(
            agent_id="lnn",
            memory_type="adaptation",
            min_timestamp=cutoff_time
        )
        
        return sorted(adaptations, key=lambda x: x["timestamp"], reverse=True)
        
    async def _enqueue_event(self, event: MemoryEvent):
        """Add event to processing queue."""
        self._event_queue.append(event)
        
        # Trigger immediate flush if queue is full
        if len(self._event_queue) >= self.batch_size:
            asyncio.create_task(self._flush_events())
            
    async def _flush_loop(self):
        """Background loop for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except Exception as e:
                logger.error(f"Flush loop error: {e}")
                
    async def _flush_events(self):
        """Flush queued events to memory store."""
        if not self.memory or not self._event_queue:
            return
            
        # Get events to process
        events_to_process = []
        while self._event_queue and len(events_to_process) < self.batch_size:
            events_to_process.append(self._event_queue.popleft())
            
        if not events_to_process:
            return
            
        with tracer.start_as_current_span("flush_events") as span:
            span.set_attribute("event.count", len(events_to_process))
            
            # Group by event type for batch processing
            by_type: Dict[str, List[MemoryEvent]] = {}
            for event in events_to_process:
                by_type.setdefault(event.event_type, []).append(event)
                
            # Process each type
            for event_type, events in by_type.items():
                try:
                    memories = []
                    for event in events:
                        memory = {
                            "agent_id": event.agent_id,
                            "memory_type": event.event_type,
                            "content": event.content,
                            "metadata": event.metadata,
                            "timestamp": event.timestamp,
                            "embedding": self._compute_embedding(event)
                        }
                        memories.append(memory)
                        
                    # Batch insert
                    await self.memory.add_memories_batch(memories)
                    self._events_processed += len(memories)
                    
                except Exception as e:
                    logger.error(f"Failed to process {event_type} events: {e}")
                    self._events_failed += len(events)
                    
                    # Re-queue failed events if queue has space
                    for event in events:
                        if len(self._event_queue) < self._event_queue.maxlen:
                            self._event_queue.append(event)
                            
            self._last_flush = datetime.now(timezone.utc)
            
    def _compute_embedding(self, event: MemoryEvent) -> List[float]:
        """Compute embedding for similarity search."""
        # Simplified embedding - in production, use proper encoder
        embedding = []
        
        if event.event_type == "inference":
            # Use predictions and confidence
            predictions = event.content.get("predictions", [])
            if isinstance(predictions, list):
                embedding.extend(predictions[:10])  # First 10 values
            else:
                embedding.append(float(predictions))
            embedding.append(event.content.get("confidence", 0.0))
            embedding.append(event.content.get("context_influence", 0.0))
            
        elif event.event_type == "adaptation":
            # Use performance delta
            embedding.append(event.metadata.get("performance_delta", 0.0))
            embedding.append(event.metadata.get("adaptation_rate", 0.01))
            
        elif event.event_type == "pattern":
            # Use pattern features
            features = event.content.get("features", [])
            embedding.extend(features[:20])  # First 20 features
            
        # Pad to fixed size
        target_size = 32
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
            
        return embedding
        
    async def cleanup_old_memories(self):
        """Remove memories older than retention period."""
        if not self.memory:
            return
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        
        deleted = await self.memory.delete_memories(
            agent_id="lnn",
            before_timestamp=cutoff_time
        )
        
        logger.info(f"Cleaned up {deleted} old LNN memories")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get hook metrics."""
        return {
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "queue_size": len(self._event_queue),
            "last_flush": self._last_flush.isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._last_flush).total_seconds()
        }