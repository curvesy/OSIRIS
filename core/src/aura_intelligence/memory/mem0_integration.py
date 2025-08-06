"""
Mem0 Integration for AURA Intelligence
"""

from typing import Dict, Any, List, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Memory entry structure"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    relevance: float = 1.0


class Mem0Manager:
    """
    Manager for Mem0 memory integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memories: Dict[str, List[Memory]] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize Mem0 connection."""
        logger.info("Initializing Mem0Manager")
        self._initialized = True
        
    async def store_memory(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory for an agent.
        
        Args:
            agent_id: Agent identifier
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        if not self._initialized:
            await self.initialize()
            
        memory_id = f"{agent_id}_{len(self.memories.get(agent_id, []))}"
        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            timestamp=str(time.time())
        )
        
        if agent_id not in self.memories:
            self.memories[agent_id] = []
        self.memories[agent_id].append(memory)
        
        return memory_id
        
    async def retrieve_memories(
        self,
        agent_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Retrieve memories for an agent.
        
        Args:
            agent_id: Agent identifier
            query: Optional search query
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        if not self._initialized:
            await self.initialize()
            
        agent_memories = self.memories.get(agent_id, [])
        
        # Simple relevance filtering if query provided
        if query and agent_memories:
            # Sort by simple string matching (in production, use embeddings)
            sorted_memories = sorted(
                agent_memories,
                key=lambda m: query.lower() in m.content.lower(),
                reverse=True
            )
            return sorted_memories[:limit]
            
        return agent_memories[-limit:]
        
    async def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Update an existing memory."""
        # Find and update memory
        for agent_memories in self.memories.values():
            for memory in agent_memories:
                if memory.id == memory_id:
                    memory.content = content
                    if metadata:
                        memory.metadata.update(metadata)
                    return
                    
    async def delete_memory(self, memory_id: str):
        """Delete a memory."""
        # Find and remove memory
        for agent_id, agent_memories in self.memories.items():
            self.memories[agent_id] = [m for m in agent_memories if m.id != memory_id]
            
    async def clear_agent_memories(self, agent_id: str):
        """Clear all memories for an agent."""
        if agent_id in self.memories:
            self.memories[agent_id].clear()
            
    async def shutdown(self):
        """Shutdown Mem0 connection."""
        logger.info("Shutting down Mem0Manager")
        self._initialized = False