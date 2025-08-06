"""
Event Bus Adapter for Shape-Aware Memory V2
==========================================

This module integrates Shape Memory V2 with the Event Bus nervous system,
enabling real-time memory updates, invalidations, and synchronization
across the distributed AURA platform.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
import numpy as np

from ..orchestration.bus_protocol import EventBus, Event
from ..tda.models import TDAResult, BettiNumbers
from ..observability.metrics import metrics_collector
from .shape_memory_v2 import ShapeAwareMemoryV2, ShapeMemoryV2Config
from .shape_aware_memory import TopologicalSignature


class MemoryBusAdapter:
    """
    Adapter that connects Shape Memory V2 to the Event Bus.
    
    Handles:
    - Memory store events from TDA pipeline
    - Memory retrieval requests from agents
    - Cache invalidation broadcasts
    - Index update notifications
    - Memory migration events
    """
    
    def __init__(
        self,
        memory_system: ShapeAwareMemoryV2,
        event_bus: EventBus,
        topic_prefix: str = "memory"
    ):
        self.memory_system = memory_system
        self.event_bus = event_bus
        self.topic_prefix = topic_prefix
        
        # Event handlers
        self._handlers: Dict[str, Callable] = {
            f"{topic_prefix}:store": self._handle_store_event,
            f"{topic_prefix}:retrieve": self._handle_retrieve_event,
            f"{topic_prefix}:invalidate": self._handle_invalidate_event,
            f"{topic_prefix}:find_anomalies": self._handle_find_anomalies_event,
            f"{topic_prefix}:bulk_store": self._handle_bulk_store_event,
            f"{topic_prefix}:reindex": self._handle_reindex_event,
            f"{topic_prefix}:tier": self._handle_tier_event
        }
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the adapter and subscribe to events."""
        if self._running:
            return
        
        # Subscribe to all memory events
        for topic, handler in self._handlers.items():
            await self.event_bus.subscribe(topic, handler)
        
        # Subscribe to TDA completion events
        await self.event_bus.subscribe(
            "tda:analysis_complete",
            self._handle_tda_complete
        )
        
        # Subscribe to agent memory requests
        await self.event_bus.subscribe(
            "agent:memory_request",
            self._handle_agent_request
        )
        
        self._running = True
        metrics_collector.memory_bus_adapter_started.inc()
        
        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._periodic_stats())
        )
    
    async def stop(self) -> None:
        """Stop the adapter."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _handle_store_event(self, event: Event) -> None:
        """Handle memory store request."""
        try:
            data = event.data
            
            # Extract TDA result
            tda_result = TDAResult(
                betti_numbers=BettiNumbers(
                    b0=data["betti_numbers"]["b0"],
                    b1=data["betti_numbers"]["b1"],
                    b2=data["betti_numbers"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence_diagram"]),
                topological_features=data.get("topological_features", {})
            )
            
            # Store memory
            memory = await self.memory_system.store(
                content=data["content"],
                tda_result=tda_result,
                context_type=data.get("context_type", "general"),
                metadata=data.get("metadata", {})
            )
            
            # Publish success event
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:stored",
                data={
                    "memory_id": memory.memory_id,
                    "request_id": event.id
                }
            ))
            
            metrics_collector.memory_bus_store_success.inc()
            
        except Exception as e:
            # Publish error event
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:store_error",
                data={
                    "request_id": event.id,
                    "error": str(e)
                }
            ))
            
            metrics_collector.memory_bus_store_errors.inc()
    
    async def _handle_retrieve_event(self, event: Event) -> None:
        """Handle memory retrieval request."""
        try:
            data = event.data
            
            # Create query signature
            query_signature = TopologicalSignature(
                betti_numbers=BettiNumbers(
                    b0=data["betti_numbers"]["b0"],
                    b1=data["betti_numbers"]["b1"],
                    b2=data["betti_numbers"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence_diagram"])
            )
            
            # Retrieve memories
            memories = await self.memory_system.retrieve(
                query_signature=query_signature,
                k=data.get("k", 10),
                context_filter=data.get("context_filter"),
                time_filter=data.get("time_filter")
            )
            
            # Convert to serializable format
            results = []
            for memory in memories:
                results.append({
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "context_type": memory.context_type,
                    "similarity_score": memory.similarity_score,
                    "metadata": memory.metadata
                })
            
            # Publish results
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:retrieved",
                data={
                    "request_id": event.id,
                    "results": results,
                    "count": len(results)
                }
            ))
            
            metrics_collector.memory_bus_retrieve_success.inc()
            
        except Exception as e:
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:retrieve_error",
                data={
                    "request_id": event.id,
                    "error": str(e)
                }
            ))
            
            metrics_collector.memory_bus_retrieve_errors.inc()
    
    async def _handle_invalidate_event(self, event: Event) -> None:
        """Handle cache invalidation request."""
        memory_ids = event.data.get("memory_ids", [])
        
        for memory_id in memory_ids:
            # Invalidate in memory system cache
            if memory_id in self.memory_system._memory_cache:
                del self.memory_system._memory_cache[memory_id]
        
        # Broadcast invalidation complete
        await self.event_bus.publish(Event(
            topic=f"{self.topic_prefix}:invalidated",
            data={
                "memory_ids": memory_ids,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ))
    
    async def _handle_find_anomalies_event(self, event: Event) -> None:
        """Handle anomaly search request."""
        try:
            data = event.data
            
            # Create anomaly signature
            anomaly_signature = TopologicalSignature(
                betti_numbers=BettiNumbers(
                    b0=data["betti_numbers"]["b0"],
                    b1=data["betti_numbers"]["b1"],
                    b2=data["betti_numbers"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence_diagram"])
            )
            
            # Find anomalies
            anomaly_patterns = await self.memory_system.find_anomalies(
                anomaly_signature=anomaly_signature,
                similarity_threshold=data.get("similarity_threshold", 0.8),
                time_window=data.get("time_window")
            )
            
            # Convert results
            results = []
            for memory, similarity in anomaly_patterns:
                results.append({
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "similarity": similarity,
                    "created_at": memory.created_at.isoformat()
                })
            
            # Publish results
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:anomalies_found",
                data={
                    "request_id": event.id,
                    "anomalies": results,
                    "count": len(results)
                }
            ))
            
        except Exception as e:
            await self.event_bus.publish(Event(
                topic=f"{self.topic_prefix}:anomaly_error",
                data={
                    "request_id": event.id,
                    "error": str(e)
                }
            ))
    
    async def _handle_bulk_store_event(self, event: Event) -> None:
        """Handle bulk memory storage."""
        memories_data = event.data.get("memories", [])
        stored_ids = []
        
        for mem_data in memories_data:
            try:
                tda_result = TDAResult(
                    betti_numbers=BettiNumbers(
                        b0=mem_data["betti_numbers"]["b0"],
                        b1=mem_data["betti_numbers"]["b1"],
                        b2=mem_data["betti_numbers"]["b2"]
                    ),
                    persistence_diagram=np.array(mem_data["persistence_diagram"]),
                    topological_features=mem_data.get("topological_features", {})
                )
                
                memory = await self.memory_system.store(
                    content=mem_data["content"],
                    tda_result=tda_result,
                    context_type=mem_data.get("context_type", "general"),
                    metadata=mem_data.get("metadata", {})
                )
                
                stored_ids.append(memory.memory_id)
                
            except Exception as e:
                print(f"Error storing bulk memory: {e}")
        
        # Publish completion
        await self.event_bus.publish(Event(
            topic=f"{self.topic_prefix}:bulk_stored",
            data={
                "request_id": event.id,
                "stored_ids": stored_ids,
                "count": len(stored_ids),
                "total_requested": len(memories_data)
            }
        ))
    
    async def _handle_reindex_event(self, event: Event) -> None:
        """Handle reindex request."""
        # Trigger index rebuild
        await self.memory_system._rebuild_index()
        
        # Publish completion
        await self.event_bus.publish(Event(
            topic=f"{self.topic_prefix}:reindexed",
            data={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_memories": self.memory_system._total_memories
            }
        ))
    
    async def _handle_tier_event(self, event: Event) -> None:
        """Handle memory tiering request."""
        # Run tiering process
        await self.memory_system.tier_memories()
        
        # Publish completion
        await self.event_bus.publish(Event(
            topic=f"{self.topic_prefix}:tiered",
            data={
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ))
    
    async def _handle_tda_complete(self, event: Event) -> None:
        """Handle TDA analysis completion - auto-store memory."""
        data = event.data
        
        # Auto-store TDA results as memories
        if data.get("auto_store", True):
            tda_result = TDAResult(
                betti_numbers=BettiNumbers(
                    b0=data["betti_numbers"]["b0"],
                    b1=data["betti_numbers"]["b1"],
                    b2=data["betti_numbers"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence_diagram"]),
                topological_features=data.get("topological_features", {})
            )
            
            # Determine context type from features
            context_type = "general"
            if data.get("is_anomaly", False):
                context_type = "anomaly"
            elif data.get("pattern_type"):
                context_type = data["pattern_type"]
            
            memory = await self.memory_system.store(
                content={
                    "source": "tda_pipeline",
                    "data": data.get("raw_data", {}),
                    "features": data.get("topological_features", {})
                },
                tda_result=tda_result,
                context_type=context_type,
                metadata={
                    "tda_id": data.get("analysis_id"),
                    "timestamp": data.get("timestamp")
                }
            )
            
            # Notify completion
            await self.event_bus.publish(Event(
                topic="tda:memory_stored",
                data={
                    "analysis_id": data.get("analysis_id"),
                    "memory_id": memory.memory_id
                }
            ))
    
    async def _handle_agent_request(self, event: Event) -> None:
        """Handle memory request from agents."""
        data = event.data
        agent_id = data.get("agent_id")
        
        # Convert agent context to topological signature
        if "topological_context" in data:
            topo_ctx = data["topological_context"]
            query_signature = TopologicalSignature(
                betti_numbers=BettiNumbers(
                    b0=topo_ctx["b0"],
                    b1=topo_ctx["b1"],
                    b2=topo_ctx["b2"]
                ),
                persistence_diagram=np.array(topo_ctx.get("persistence", []))
            )
            
            # Retrieve relevant memories
            try:
                memories = await self.memory_system.retrieve(
                    query_signature=query_signature,
                    k=data.get("k", 5),
                    context_filter=data.get("context_filter")
                )
            except RuntimeError as e:
                logger.error(f"Memory retrieval failed for agent {agent_id}: {e}")
                # Send empty results
                memories = []
            
            # Send to agent
            await self.event_bus.publish(Event(
                topic=f"agent:{agent_id}:memory_response",
                data={
                    "request_id": event.id,
                    "memories": [
                        {
                            "content": m.content,
                            "context_type": m.context_type,
                            "similarity": m.similarity_score
                        }
                        for m in memories
                    ]
                }
            ))
    
    async def _periodic_stats(self) -> None:
        """Publish periodic memory statistics."""
        while self._running:
            try:
                # Gather stats
                stats = {
                    "total_memories": self.memory_system._total_memories,
                    "cache_size": len(self.memory_system._memory_cache),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Publish stats
                await self.event_bus.publish(Event(
                    topic=f"{self.topic_prefix}:stats",
                    data=stats
                ))
                
                # Wait 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error publishing stats: {e}")
                await asyncio.sleep(60)


# Integration example
async def demo_memory_bus_integration():
    """Demonstrate Memory Bus integration."""
    
    # Initialize components
    config = ShapeMemoryV2Config(
        event_bus_enabled=True
    )
    
    memory_system = ShapeAwareMemoryV2(config)
    await memory_system.initialize()
    
    event_bus = EventBus(redis_url="redis://localhost:6379")
    await event_bus.initialize()
    
    # Create adapter
    adapter = MemoryBusAdapter(memory_system, event_bus)
    await adapter.start()
    
    print("Memory Bus Adapter Demo")
    print("=" * 50)
    
    # Simulate TDA completion event
    await event_bus.publish(Event(
        topic="tda:analysis_complete",
        data={
            "analysis_id": "tda_123",
            "betti_numbers": {"b0": 1, "b1": 2, "b2": 0},
            "persistence_diagram": [[0.1, 0.5], [0.2, 0.8]],
            "topological_features": {"holes": 2},
            "is_anomaly": True,
            "auto_store": True
        }
    ))
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Simulate agent memory request
    await event_bus.publish(Event(
        topic="agent:memory_request",
        data={
            "agent_id": "agent_456",
            "topological_context": {
                "b0": 1,
                "b1": 2,
                "b2": 0,
                "persistence": [[0.1, 0.6], [0.2, 0.7]]
            },
            "k": 5,
            "context_filter": "anomaly"
        }
    ))
    
    # Wait for response
    await asyncio.sleep(1)
    
    print("\nMemory Bus integration complete!")
    
    # Cleanup
    await adapter.stop()
    await memory_system.cleanup()
    await event_bus.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_memory_bus_integration())