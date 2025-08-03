"""
Agent Registry

Registry for managing and discovering LNN Council Agents.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
import asyncio
from collections import defaultdict

from .contracts import AgentCapability
from .interfaces import ICouncilAgent


class AgentRegistry:
    """Registry for managing council agents."""
    
    def __init__(self):
        self._agents: Dict[str, ICouncilAgent] = {}
        self._capabilities: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: ICouncilAgent) -> bool:
        """Register an agent in the registry."""
        async with self._lock:
            agent_id = agent.agent_id
            
            if agent_id in self._agents:
                return False  # Agent already registered
            
            # Register agent
            self._agents[agent_id] = agent
            
            # Register capabilities
            capabilities = await agent.get_capabilities()
            for cap_str in capabilities:
                try:
                    capability = AgentCapability(cap_str)
                    self._capabilities[capability].add(agent_id)
                except ValueError:
                    # Skip unknown capabilities
                    pass
            
            return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry."""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            
            # Remove from capabilities
            capabilities = await agent.get_capabilities()
            for cap_str in capabilities:
                try:
                    capability = AgentCapability(cap_str)
                    self._capabilities[capability].discard(agent_id)
                except ValueError:
                    pass
            
            # Remove agent
            del self._agents[agent_id]
            return True
    
    async def find_agents_by_capability(
        self, 
        capability: AgentCapability
    ) -> List[ICouncilAgent]:
        """Find agents with specific capability."""
        async with self._lock:
            agent_ids = self._capabilities.get(capability, set())
            return [self._agents[agent_id] for agent_id in agent_ids]
    
    async def find_agents_by_capabilities(
        self, 
        capabilities: List[AgentCapability],
        require_all: bool = True
    ) -> List[ICouncilAgent]:
        """Find agents with multiple capabilities."""
        async with self._lock:
            if not capabilities:
                return list(self._agents.values())
            
            if require_all:
                # Agent must have ALL capabilities
                agent_sets = [self._capabilities.get(cap, set()) for cap in capabilities]
                if not agent_sets:
                    return []
                
                common_agents = set.intersection(*agent_sets)
                return [self._agents[agent_id] for agent_id in common_agents]
            else:
                # Agent must have ANY capability
                agent_ids = set()
                for cap in capabilities:
                    agent_ids.update(self._capabilities.get(cap, set()))
                
                return [self._agents[agent_id] for agent_id in agent_ids]
    
    async def get_agent(self, agent_id: str) -> Optional[ICouncilAgent]:
        """Get agent by ID."""
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def list_agents(self) -> List[ICouncilAgent]:
        """List all registered agents."""
        async with self._lock:
            return list(self._agents.values())
    
    async def get_registry_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        async with self._lock:
            return {
                "total_agents": len(self._agents),
                "capabilities": {
                    cap.value: len(agent_ids) 
                    for cap, agent_ids in self._capabilities.items()
                },
                "agent_ids": list(self._agents.keys())
            }


# Global registry instance
_global_registry = AgentRegistry()


def get_global_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_registry