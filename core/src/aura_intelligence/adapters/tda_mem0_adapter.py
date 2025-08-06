"""
TDA to Mem0 Adapter - Production Ready
Stores topological features in episodic memory for agent context enrichment
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from mem0 import Memory
# MemoryConfig is no longer needed in newer mem0 versions

from ..tda.models import PersistenceDiagram, TDAResult
from ..utils.logger import get_logger
from ..observability.metrics import metrics_collector
from ..observability.tracing import trace_span

logger = get_logger(__name__)


@dataclass
class TDAMemoryEntry:
    """Structured TDA memory entry for Mem0"""
    entry_id: str
    algorithm: str
    timestamp: datetime
    betti_numbers: List[int]
    persistence_features: Dict[str, float]
    anomaly_indicators: Dict[str, Any]
    context_tags: List[str]
    embedding_vector: Optional[List[float]] = None


class TDAMem0Adapter:
    """
    Production adapter for storing TDA results in Mem0 episodic memory.
    
    Features:
    - Semantic search over topological features
    - Temporal memory with decay
    - Context-aware retrieval
    - Agent-friendly format
    - Vector embeddings for similarity
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory = Memory()
        self._initialized = False
        
    async def initialize(self):
        """Initialize Mem0 connection and collections"""
        try:
            # Initialize memory system
            await self.memory.initialize()
            
            # Create TDA-specific collection
            await self.memory.create_collection(
                name="tda_features",
                schema={
                    "entry_id": "string",
                    "algorithm": "string",
                    "timestamp": "datetime",
                    "betti_numbers": "list",
                    "persistence_features": "dict",
                    "anomaly_indicators": "dict",
                    "context_tags": "list",
                    "embedding_vector": "vector[768]"
                }
            )
            
            self._initialized = True
            logger.info("TDA Mem0 adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TDA Mem0 adapter: {e}")
            raise
            
    @trace_span("store_tda_memory")
    async def store_tda_memory(
        self,
        result: TDAResult,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = 86400  # 24 hours default
    ) -> str:
        """
        Store TDA result as episodic memory.
        
        Args:
            result: TDA computation result
            agent_id: ID of agent storing the memory
            context: Additional context for the memory
            ttl: Time to live in seconds
            
        Returns:
            memory_id of stored entry
        """
        if not self._initialized:
            await self.initialize()
            
        entry_id = f"tda_mem_{result.algorithm}_{datetime.now(timezone.utc).timestamp()}"
        
        # Extract persistence features
        persistence_features = self._extract_persistence_features(result)
        
        # Extract anomaly indicators
        anomaly_indicators = {
            "anomaly_score": result.anomaly_score or 0.0,
            "anomalous_dimensions": self._find_anomalous_dimensions(result),
            "persistence_outliers": self._find_persistence_outliers(result)
        }
        
        # Generate context tags
        context_tags = self._generate_context_tags(result, context)
        
        # Create memory entry
        memory_entry = TDAMemoryEntry(
            entry_id=entry_id,
            algorithm=result.algorithm,
            timestamp=datetime.now(timezone.utc),
            betti_numbers=result.betti_numbers.to_list() if result.betti_numbers else [],
            persistence_features=persistence_features,
            anomaly_indicators=anomaly_indicators,
            context_tags=context_tags
        )
        
        # Generate embedding for semantic search
        embedding = await self._generate_embedding(memory_entry)
        memory_entry.embedding_vector = embedding
        
        try:
            # Store in Mem0
            memory_data = {
                **asdict(memory_entry),
                "agent_id": agent_id,
                "metadata": {
                    "trace_id": result.trace_id,
                    "computation_time": result.computation_time,
                    "num_points": result.num_points,
                    "context": context or {}
                }
            }
            
            memory_id = await self.memory.add(
                data=memory_data,
                user_id=agent_id,
                metadata={
                    "type": "tda_analysis",
                    "algorithm": result.algorithm,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                ttl=ttl
            )
            
            # Update metrics
            metrics_collector.increment("tda.memories.stored", tags={"algorithm": result.algorithm})
            logger.info(f"Stored TDA memory {entry_id} for agent {agent_id}")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store TDA memory: {e}")
            metrics_collector.increment("tda.memory.errors", tags={"error": type(e).__name__})
            raise
            
    def _extract_persistence_features(self, result: TDAResult) -> Dict[str, float]:
        """Extract key persistence features for memory"""
        features = {
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "avg_persistence": 0.0,
            "persistence_entropy": 0.0,
            "num_components": 0,
            "num_loops": 0,
            "num_voids": 0
        }
        
        all_persistences = []
        
        for dim, diagram in enumerate(result.persistence_diagrams):
            persistences = [death - birth for birth, death in diagram.pairs]
            
            if persistences:
                features["total_persistence"] += sum(persistences)
                features["max_persistence"] = max(features["max_persistence"], max(persistences))
                all_persistences.extend(persistences)
                
            # Count features by dimension
            if dim == 0:
                features["num_components"] = len(diagram.pairs)
            elif dim == 1:
                features["num_loops"] = len(diagram.pairs)
            elif dim == 2:
                features["num_voids"] = len(diagram.pairs)
                
        if all_persistences:
            features["avg_persistence"] = np.mean(all_persistences)
            # Calculate persistence entropy
            p = np.array(all_persistences) / sum(all_persistences)
            features["persistence_entropy"] = -np.sum(p * np.log(p + 1e-10))
            
        return features
        
    def _find_anomalous_dimensions(self, result: TDAResult) -> List[int]:
        """Identify dimensions with anomalous topology"""
        anomalous_dims = []
        
        for dim, diagram in enumerate(result.persistence_diagrams):
            if len(diagram.pairs) > 10:  # Unusually many features
                anomalous_dims.append(dim)
            elif diagram.max_persistence() > 2.0:  # Unusually persistent feature
                anomalous_dims.append(dim)
                
        return anomalous_dims
        
    def _find_persistence_outliers(self, result: TDAResult) -> List[Dict[str, Any]]:
        """Find outlier persistence features"""
        outliers = []
        
        for dim, diagram in enumerate(result.persistence_diagrams):
            persistences = [death - birth for birth, death in diagram.pairs]
            
            if persistences:
                mean_p = np.mean(persistences)
                std_p = np.std(persistences)
                
                for i, (birth, death) in enumerate(diagram.pairs):
                    persistence = death - birth
                    if persistence > mean_p + 2 * std_p:  # 2 sigma outlier
                        outliers.append({
                            "dimension": dim,
                            "birth": birth,
                            "death": death,
                            "persistence": persistence,
                            "z_score": (persistence - mean_p) / (std_p + 1e-10)
                        })
                        
        return outliers
        
    def _generate_context_tags(self, result: TDAResult, context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate searchable context tags"""
        tags = [
            f"algorithm:{result.algorithm}",
            f"dimension:{result.max_dimension}",
            f"points:{result.num_points}"
        ]
        
        # Add anomaly tags
        if result.anomaly_score and result.anomaly_score > 0.7:
            tags.append("anomaly:high")
        elif result.anomaly_score and result.anomaly_score > 0.4:
            tags.append("anomaly:medium")
            
        # Add context tags
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    tags.append(f"{key}:{value}")
                    
        return tags
        
    async def _generate_embedding(self, entry: TDAMemoryEntry) -> List[float]:
        """Generate embedding vector for semantic search"""
        # In production, use a proper embedding model
        # This is a placeholder that creates a feature vector
        features = []
        
        # Add betti numbers
        features.extend(entry.betti_numbers[:10])  # Pad/truncate to 10
        features.extend([0] * max(0, 10 - len(entry.betti_numbers)))
        
        # Add persistence features
        for key in ["total_persistence", "max_persistence", "avg_persistence", 
                   "persistence_entropy", "num_components", "num_loops", "num_voids"]:
            features.append(entry.persistence_features.get(key, 0.0))
            
        # Add anomaly score
        features.append(entry.anomaly_indicators.get("anomaly_score", 0.0))
        
        # Normalize and pad to 768 dimensions (standard embedding size)
        features = np.array(features)
        features = features / (np.linalg.norm(features) + 1e-10)
        features = np.pad(features, (0, 768 - len(features)), mode='constant')
        
        return features.tolist()
        
    @trace_span("recall_tda_memories")
    async def recall_tda_memories(
        self,
        agent_id: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        time_window: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recall TDA memories for an agent.
        
        Args:
            agent_id: ID of agent recalling memories
            query: Semantic search query
            filters: Additional filters (algorithm, anomaly_score, etc.)
            limit: Maximum memories to return
            time_window: Time range filter
            
        Returns:
            List of relevant TDA memories
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Build search parameters
            search_params = {
                "user_id": agent_id,
                "limit": limit,
                "metadata_filter": {"type": "tda_analysis"}
            }
            
            if query:
                search_params["query"] = query
                
            if filters:
                if "algorithm" in filters:
                    search_params["metadata_filter"]["algorithm"] = filters["algorithm"]
                    
            if time_window:
                start, end = time_window
                search_params["time_filter"] = {
                    "start": start.isoformat(),
                    "end": end.isoformat()
                }
                
            # Search memories
            memories = await self.memory.search(**search_params)
            
            # Post-process and filter
            processed_memories = []
            for memory in memories:
                # Apply additional filters
                if filters and "anomaly_threshold" in filters:
                    anomaly_score = memory.get("anomaly_indicators", {}).get("anomaly_score", 0)
                    if anomaly_score < filters["anomaly_threshold"]:
                        continue
                        
                processed_memories.append({
                    "memory_id": memory["id"],
                    "entry_id": memory["entry_id"],
                    "algorithm": memory["algorithm"],
                    "timestamp": memory["timestamp"],
                    "betti_numbers": memory["betti_numbers"],
                    "persistence_features": memory["persistence_features"],
                    "anomaly_indicators": memory["anomaly_indicators"],
                    "context_tags": memory["context_tags"],
                    "relevance_score": memory.get("score", 1.0),
                    "metadata": memory.get("metadata", {})
                })
                
            return processed_memories
            
        except Exception as e:
            logger.error(f"Failed to recall TDA memories: {e}")
            raise
            
    @trace_span("get_tda_context_for_agent")
    async def get_tda_context_for_agent(
        self,
        agent_id: str,
        current_data_id: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get TDA context for agent decision making.
        
        Args:
            agent_id: ID of agent requesting context
            current_data_id: Current data being analyzed
            lookback_hours: Hours to look back for context
            
        Returns:
            TDA context summary for agent
        """
        if not self._initialized:
            await self.initialize()
            
        # Define time window
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Recall recent memories
        memories = await self.recall_tda_memories(
            agent_id=agent_id,
            time_window=(start_time, end_time),
            limit=50
        )
        
        # Build context summary
        context = {
            "agent_id": agent_id,
            "current_data_id": current_data_id,
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "topological_summary": {
                "total_analyses": len(memories),
                "algorithms_used": set(),
                "avg_anomaly_score": 0.0,
                "max_anomaly_score": 0.0,
                "persistent_features": [],
                "recent_anomalies": []
            },
            "memories": memories[:10]  # Include top 10 most relevant
        }
        
        # Aggregate statistics
        anomaly_scores = []
        for memory in memories:
            context["topological_summary"]["algorithms_used"].add(memory["algorithm"])
            
            anomaly_score = memory["anomaly_indicators"].get("anomaly_score", 0)
            anomaly_scores.append(anomaly_score)
            
            if anomaly_score > 0.7:
                context["topological_summary"]["recent_anomalies"].append({
                    "timestamp": memory["timestamp"],
                    "algorithm": memory["algorithm"],
                    "score": anomaly_score,
                    "features": memory["persistence_features"]
                })
                
        if anomaly_scores:
            context["topological_summary"]["avg_anomaly_score"] = np.mean(anomaly_scores)
            context["topological_summary"]["max_anomaly_score"] = max(anomaly_scores)
            
        context["topological_summary"]["algorithms_used"] = list(
            context["topological_summary"]["algorithms_used"]
        )
        
        return context