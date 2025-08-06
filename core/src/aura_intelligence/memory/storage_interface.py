"""
Storage Interface - Clean Abstraction
====================================

Defines the contract for memory storage backends.
This allows us to swap between Redis, in-memory, or other stores.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class MemoryStorage(ABC):
    """Abstract interface for memory storage backends."""
    
    @abstractmethod
    def add(
        self,
        memory_id: str,
        embedding: np.ndarray,
        content: Dict[str, Any],
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a memory with its embedding."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar memories."""
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check storage health."""
        pass


class InMemoryStorage(MemoryStorage):
    """Simple in-memory storage for testing."""
    
    def __init__(self):
        self.memories = {}
        self.embeddings = {}
    
    def add(
        self,
        memory_id: str,
        embedding: np.ndarray,
        content: Dict[str, Any],
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store in memory."""
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "context_type": context_type,
            "metadata": metadata or {}
        }
        self.embeddings[memory_id] = embedding
        return True
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Brute force search."""
        results = []
        
        for mem_id, embedding in self.embeddings.items():
            memory = self.memories[mem_id]
            
            # Apply filter
            if context_filter and memory["context_type"] != context_filter:
                continue
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity >= score_threshold:
                results.append((memory, float(similarity)))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID."""
        return self.memories.get(memory_id)
    
    def delete(self, memory_id: str) -> bool:
        """Delete from memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            del self.embeddings[memory_id]
            return True
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """Always healthy for in-memory."""
        return {
            "status": "healthy",
            "total_memories": len(self.memories)
        }