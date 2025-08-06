"""
Vector Search Module for AURA Intelligence

Provides vector similarity search capabilities for embeddings,
memory retrieval, and semantic matching.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search operation."""
    id: str
    score: float
    vector: np.ndarray
    metadata: Dict[str, Any]


class VectorIndex:
    """
    In-memory vector index for similarity search.
    
    In production, this would use FAISS, Annoy, or pgvector.
    """
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    def add(self, id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Add a vector to the index."""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} != index dimension {self.dimension}")
            
        self.vectors[id] = vector
        self.metadata[id] = metadata or {}
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[VectorSearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of search results sorted by similarity
        """
        if not self.vectors:
            return []
            
        scores = []
        for id, vector in self.vectors.items():
            if self.metric == "cosine":
                score = self._cosine_similarity(query_vector, vector)
            elif self.metric == "euclidean":
                score = -np.linalg.norm(query_vector - vector)  # Negative for sorting
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
            scores.append((id, score, vector))
            
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for id, score, vector in scores[:k]:
            results.append(VectorSearchResult(
                id=id,
                score=score,
                vector=vector,
                metadata=self.metadata.get(id, {})
            ))
            
        return results
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    def remove(self, id: str):
        """Remove a vector from the index."""
        self.vectors.pop(id, None)
        self.metadata.pop(id, None)
        
    def clear(self):
        """Clear all vectors from the index."""
        self.vectors.clear()
        self.metadata.clear()
        
    def size(self) -> int:
        """Get number of vectors in index."""
        return len(self.vectors)


class VectorSearchEngine:
    """
    Main vector search engine with multiple indices.
    """
    
    def __init__(self):
        self.indices: Dict[str, VectorIndex] = {}
        
    def create_index(self, name: str, dimension: int, metric: str = "cosine") -> VectorIndex:
        """Create a new vector index."""
        index = VectorIndex(dimension, metric)
        self.indices[name] = index
        return index
        
    def get_index(self, name: str) -> Optional[VectorIndex]:
        """Get an existing index."""
        return self.indices.get(name)
        
    def delete_index(self, name: str):
        """Delete an index."""
        self.indices.pop(name, None)
        
    async def hybrid_search(
        self,
        index_name: str,
        query_vector: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search with vector similarity and metadata filters.
        
        Args:
            index_name: Name of the index to search
            query_vector: Query vector
            filters: Metadata filters to apply
            k: Number of results
            
        Returns:
            Filtered search results
        """
        index = self.get_index(index_name)
        if not index:
            return []
            
        # Get all results
        results = index.search(query_vector, k=k*2)  # Get extra for filtering
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    if result.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results
            
        return results[:k]


# Global engine instance
vector_search_engine = VectorSearchEngine()


class LlamaIndexClient:
    """
    Client for LlamaIndex integration.
    
    In production, this would integrate with LlamaIndex for advanced RAG.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.index = VectorIndex(dimension=384)
        
    async def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query the index."""
        query_vector = create_embedding(query)
        results = self.index.search(query_vector, k=k)
        return [{"text": r.metadata.get("text", ""), "score": r.score} for r in results]
        
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents."""
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            vector = create_embedding(text)
            self.index.add(f"doc_{i}", vector, doc)


def create_embedding(text: str, model: str = "simple") -> np.ndarray:
    """
    Create embedding from text.
    
    In production, this would use OpenAI, Sentence Transformers, etc.
    """
    if model == "simple":
        # Simple hash-based embedding for testing
        hash_val = hash(text)
        np.random.seed(abs(hash_val) % (2**32))
        return np.random.randn(384)  # Standard embedding dimension
    else:
        raise ValueError(f"Unknown embedding model: {model}")


async def semantic_search(
    query: str,
    corpus: List[Dict[str, Any]],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search over a corpus.
    
    Args:
        query: Search query
        corpus: List of documents with 'text' field
        k: Number of results
        
    Returns:
        Top k most similar documents
    """
    # Create temporary index
    index = VectorIndex(dimension=384)
    
    # Index corpus
    for i, doc in enumerate(corpus):
        text = doc.get("text", "")
        vector = create_embedding(text)
        index.add(str(i), vector, doc)
        
    # Search
    query_vector = create_embedding(query)
    results = index.search(query_vector, k=k)
    
    # Return documents
    return [r.metadata for r in results]