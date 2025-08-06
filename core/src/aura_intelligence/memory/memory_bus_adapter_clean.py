"""
Memory Bus Adapter - Clean Implementation
========================================

Connects Shape Memory V2 to the Event Bus for real-time synchronization.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from ..orchestration.bus_protocol import EventBus, Event
from ..tda.models import TDAResult, BettiNumbers
from .shape_memory_v2_clean import ShapeMemoryV2, ShapeMemoryConfig

logger = logging.getLogger(__name__)


class MemoryBusAdapter:
    """
    Lightweight adapter connecting memory system to event bus.
    
    Handles:
    - Memory store/retrieve events
    - TDA pipeline integration
    - Agent memory requests
    """
    
    def __init__(
        self,
        memory_system: ShapeMemoryV2,
        event_bus: EventBus,
        topic_prefix: str = "memory"
    ):
        self.memory = memory_system
        self.bus = event_bus
        self.topic_prefix = topic_prefix
        self._running = False
        
    async def start(self) -> None:
        """Start listening to events."""
        if self._running:
            return
            
        # Subscribe to relevant events
        await self.bus.subscribe(f"{self.topic_prefix}:store", self._handle_store)
        await self.bus.subscribe(f"{self.topic_prefix}:retrieve", self._handle_retrieve)
        await self.bus.subscribe("tda:complete", self._handle_tda_complete)
        await self.bus.subscribe("agent:memory_request", self._handle_agent_request)
        
        self._running = True
        logger.info("Memory bus adapter started")
        
    async def stop(self) -> None:
        """Stop the adapter."""
        self._running = False
        
    async def _handle_store(self, event: Event) -> None:
        """Handle memory store request."""
        try:
            data = event.data
            
            # Create TDA result
            tda_result = TDAResult(
                betti_numbers=BettiNumbers(
                    b0=data["betti"]["b0"],
                    b1=data["betti"]["b1"],
                    b2=data["betti"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence"]),
                topological_features=data.get("features", {})
            )
            
            # Store memory
            memory_id = self.memory.store(
                content=data["content"],
                tda_result=tda_result,
                context_type=data.get("context_type", "general")
            )
            
            # Publish success
            await self.bus.publish(Event(
                topic=f"{self.topic_prefix}:stored",
                data={"memory_id": memory_id, "request_id": event.id}
            ))
            
        except Exception as e:
            logger.error(f"Store error: {e}")
            await self.bus.publish(Event(
                topic=f"{self.topic_prefix}:error",
                data={"request_id": event.id, "error": str(e)}
            ))
            
    async def _handle_retrieve(self, event: Event) -> None:
        """Handle memory retrieval request."""
        try:
            data = event.data
            
            # Create query TDA
            query_tda = TDAResult(
                betti_numbers=BettiNumbers(
                    b0=data["betti"]["b0"],
                    b1=data["betti"]["b1"],
                    b2=data["betti"]["b2"]
                ),
                persistence_diagram=np.array(data["persistence"]),
                topological_features={}
            )
            
            # Retrieve memories
            results = self.memory.retrieve(
                query_tda=query_tda,
                k=data.get("k", 10),
                context_filter=data.get("context_filter")
            )
            
            # Format results
            memories = []
            for entry, similarity in results:
                memories.append({
                    "id": entry.id,
                    "content": entry.content,
                    "similarity": similarity,
                    "context_type": entry.context_type
                })
            
            # Publish results
            await self.bus.publish(Event(
                topic=f"{self.topic_prefix}:retrieved",
                data={
                    "request_id": event.id,
                    "memories": memories,
                    "count": len(memories)
                }
            ))
            
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            await self.bus.publish(Event(
                topic=f"{self.topic_prefix}:error",
                data={"request_id": event.id, "error": str(e)}
            ))
            
    async def _handle_tda_complete(self, event: Event) -> None:
        """Auto-store TDA results as memories."""
        if not event.data.get("auto_store", True):
            return
            
        data = event.data
        
        # Create TDA result
        tda_result = TDAResult(
            betti_numbers=BettiNumbers(
                b0=data["betti"]["b0"],
                b1=data["betti"]["b1"],
                b2=data["betti"]["b2"]
            ),
            persistence_diagram=np.array(data["persistence"]),
            topological_features=data.get("features", {})
        )
        
        # Determine context
        context_type = "anomaly" if data.get("is_anomaly") else "general"
        
        # Store
        memory_id = self.memory.store(
            content={
                "source": "tda_pipeline",
                "data": data.get("raw_data", {}),
                "timestamp": data.get("timestamp")
            },
            tda_result=tda_result,
            context_type=context_type
        )
        
        logger.debug(f"Auto-stored TDA result as {memory_id}")
        
    async def _handle_agent_request(self, event: Event) -> None:
        """Handle memory request from agents."""
        data = event.data
        agent_id = data.get("agent_id")
        
        if "topological_context" in data:
            ctx = data["topological_context"]
            
            # Create query
            query_tda = TDAResult(
                betti_numbers=BettiNumbers(
                    b0=ctx["b0"],
                    b1=ctx["b1"],
                    b2=ctx["b2"]
                ),
                persistence_diagram=np.array(ctx.get("persistence", [])),
                topological_features={}
            )
            
            # Retrieve
            try:
                results = self.memory.retrieve(
                    query_tda=query_tda,
                    k=data.get("k", 5)
                )
            except RuntimeError as e:
                logger.error(f"Memory retrieval failed: {e}")
                # Return empty results to agent
                results = []
            
            # Format for agent
            memories = []
            for entry, similarity in results:
                memories.append({
                    "content": entry.content,
                    "similarity": similarity
                })
            
            # Send to agent
            await self.bus.publish(Event(
                topic=f"agent:{agent_id}:memory_response",
                data={
                    "request_id": event.id,
                    "memories": memories
                }
            ))