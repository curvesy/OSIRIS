"""
🏆 Memory Ranking & Decay Service

TTL policies, scoring algorithms, and decay functions for semantic long-term memory.
Implements intelligent memory retention based on access patterns and relevance.

Based on partab.md: "TTL, scoring, decay policies" specification.
"""

import asyncio
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

from aura_intelligence.utils.logger import get_logger


class MemoryImportance(Enum):
    """Memory importance levels for TTL policies."""
    CRITICAL = "critical"      # 90 days TTL
    HIGH = "high"             # 30 days TTL  
    MEDIUM = "medium"         # 7 days TTL
    LOW = "low"               # 1 day TTL
    EPHEMERAL = "ephemeral"   # 1 hour TTL


@dataclass
class MemoryScore:
    """Memory scoring with decay factors."""
    base_score: float
    access_frequency: int
    recency_score: float
    relevance_score: float
    decay_factor: float
    final_score: float
    importance_level: MemoryImportance
    ttl_seconds: int


@dataclass
class MemoryAccessPattern:
    """Memory access pattern tracking."""
    signature_hash: str
    access_count: int
    last_access_time: datetime
    first_access_time: datetime
    access_velocity: float  # accesses per hour
    context_relevance: float


class MemoryRankingService:
    """
    🏆 Memory Ranking & Decay Service
    
    Features:
    - Intelligent TTL policies based on importance
    - Access pattern analysis and scoring
    - Exponential decay functions for aging memories
    - Context-aware relevance scoring
    - Automated cleanup of low-value memories
    - Performance monitoring and optimization
    """
    
    def __init__(self, redis_url: str):
        """Initialize memory ranking service."""
        
        self.redis_url = redis_url
        self.redis_client = None
        
        # TTL configuration (seconds)
        self.ttl_policies = {
            MemoryImportance.CRITICAL: 86400 * 90,    # 90 days
            MemoryImportance.HIGH: 86400 * 30,        # 30 days
            MemoryImportance.MEDIUM: 86400 * 7,       # 7 days
            MemoryImportance.LOW: 86400 * 1,          # 1 day
            MemoryImportance.EPHEMERAL: 3600 * 1      # 1 hour
        }
        
        # Scoring parameters
        self.decay_rate = 0.1  # Exponential decay rate per day
        self.access_weight = 0.3
        self.recency_weight = 0.4
        self.relevance_weight = 0.3
        
        # Performance tracking
        self.ranking_operations = 0
        self.cleanup_operations = 0
        self.total_ranking_time = 0.0
        
        # Background cleanup
        self.cleanup_task = None
        self.is_running = False
        
        self.logger = get_logger(__name__)
        self.logger.info("🏆 Memory Ranking Service initialized")
    
    async def initialize(self):
        """Initialize Redis connection."""
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            await self.redis_client.ping()
            self.logger.info("✅ Redis connection established for ranking service")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Redis initialization failed: {e}")
            return False
    
    async def start_background_cleanup(self, interval_hours: int = 6):
        """Start background cleanup process."""
        
        if self.is_running:
            self.logger.warning("⚠️ Background cleanup already running")
            return
        
        if not self.redis_client:
            await self.initialize()
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(
            self._background_cleanup_loop(interval_hours)
        )
        
        self.logger.info(f"🧹 Background cleanup started (interval: {interval_hours}h)")
    
    async def stop_background_cleanup(self):
        """Stop background cleanup process."""
        
        if not self.is_running:
            return
        
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("⏹️ Background cleanup stopped")
    
    async def _background_cleanup_loop(self, interval_hours: int):
        """Background loop for memory cleanup."""
        
        while self.is_running:
            try:
                await self.cleanup_expired_memories()
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Background cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def score_memory(self, signature_hash: str, 
                          context_data: Dict[str, Any] = None) -> MemoryScore:
        """
        Calculate comprehensive memory score with decay factors.
        
        Args:
            signature_hash: Signature to score
            context_data: Additional context for relevance scoring
            
        Returns:
            MemoryScore with all scoring components
        """
        
        try:
            start_time = time.time()
            
            # Get access pattern
            access_pattern = await self._get_access_pattern(signature_hash)
            
            # Calculate component scores
            base_score = await self._calculate_base_score(signature_hash)
            recency_score = self._calculate_recency_score(access_pattern)
            relevance_score = await self._calculate_relevance_score(signature_hash, context_data)
            decay_factor = self._calculate_decay_factor(access_pattern)
            
            # Weighted final score
            final_score = (
                base_score * (1 - self.access_weight - self.recency_weight - self.relevance_weight) +
                access_pattern.access_velocity * self.access_weight +
                recency_score * self.recency_weight +
                relevance_score * self.relevance_weight
            ) * decay_factor
            
            # Determine importance level
            importance_level = self._determine_importance_level(final_score, access_pattern)
            ttl_seconds = self.ttl_policies[importance_level]
            
            # Create memory score
            memory_score = MemoryScore(
                base_score=base_score,
                access_frequency=access_pattern.access_count,
                recency_score=recency_score,
                relevance_score=relevance_score,
                decay_factor=decay_factor,
                final_score=final_score,
                importance_level=importance_level,
                ttl_seconds=ttl_seconds
            )
            
            # Update Redis with score and TTL
            await self._update_memory_score(signature_hash, memory_score)
            
            # Performance tracking
            ranking_time = time.time() - start_time
            self.ranking_operations += 1
            self.total_ranking_time += ranking_time
            
            return memory_score
            
        except Exception as e:
            self.logger.error(f"❌ Memory scoring failed for {signature_hash[:8]}...: {e}")
            # Return default low score
            return MemoryScore(
                base_score=0.1,
                access_frequency=0,
                recency_score=0.0,
                relevance_score=0.0,
                decay_factor=0.5,
                final_score=0.05,
                importance_level=MemoryImportance.LOW,
                ttl_seconds=self.ttl_policies[MemoryImportance.LOW]
            )
    
    async def _get_access_pattern(self, signature_hash: str) -> MemoryAccessPattern:
        """Get or create access pattern for signature."""
        
        pattern_key = f"access:pattern:{signature_hash}"
        
        try:
            pattern_data = await self.redis_client.hgetall(pattern_key)
            
            if pattern_data:
                # Load existing pattern
                first_access = datetime.fromisoformat(pattern_data["first_access_time"])
                last_access = datetime.fromisoformat(pattern_data["last_access_time"])
                access_count = int(pattern_data["access_count"])
                
                # Calculate access velocity
                time_span_hours = max((last_access - first_access).total_seconds() / 3600, 1)
                access_velocity = access_count / time_span_hours
                
                return MemoryAccessPattern(
                    signature_hash=signature_hash,
                    access_count=access_count,
                    last_access_time=last_access,
                    first_access_time=first_access,
                    access_velocity=access_velocity,
                    context_relevance=float(pattern_data.get("context_relevance", 0.5))
                )
            else:
                # Create new pattern
                now = datetime.now()
                return MemoryAccessPattern(
                    signature_hash=signature_hash,
                    access_count=1,
                    last_access_time=now,
                    first_access_time=now,
                    access_velocity=1.0,
                    context_relevance=0.5
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get access pattern: {e}")
            # Return default pattern
            now = datetime.now()
            return MemoryAccessPattern(
                signature_hash=signature_hash,
                access_count=1,
                last_access_time=now,
                first_access_time=now,
                access_velocity=1.0,
                context_relevance=0.5
            )
    
    async def _calculate_base_score(self, signature_hash: str) -> float:
        """Calculate base score from signature properties."""
        
        try:
            # Get signature data from hot or semantic tier
            signature_key = f"signature:{signature_hash}"
            signature_data = await self.redis_client.hgetall(signature_key)
            
            if not signature_data:
                return 0.5  # Default score
            
            # Score based on Betti numbers (topological complexity)
            betti_0 = int(signature_data.get("betti_0", 0))
            betti_1 = int(signature_data.get("betti_1", 0))
            betti_2 = int(signature_data.get("betti_2", 0))
            
            # Higher topological complexity = higher base score
            complexity_score = min((betti_0 + betti_1 * 2 + betti_2 * 3) / 20.0, 1.0)
            
            # Anomaly score contribution
            anomaly_score = float(signature_data.get("anomaly_score", 0.0))
            
            # Combined base score
            base_score = (complexity_score * 0.7 + anomaly_score * 0.3)
            
            return max(min(base_score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"❌ Base score calculation failed: {e}")
            return 0.5
    
    def _calculate_recency_score(self, access_pattern: MemoryAccessPattern) -> float:
        """Calculate recency score based on last access time."""
        
        hours_since_access = (datetime.now() - access_pattern.last_access_time).total_seconds() / 3600
        
        # Exponential decay: score = e^(-decay_rate * hours)
        recency_score = math.exp(-self.decay_rate * hours_since_access / 24)  # Decay per day
        
        return max(min(recency_score, 1.0), 0.0)
    
    async def _calculate_relevance_score(self, signature_hash: str, 
                                       context_data: Dict[str, Any] = None) -> float:
        """Calculate context-aware relevance score."""
        
        if not context_data:
            return 0.5  # Default relevance
        
        try:
            # Get signature context
            context_key = f"context:{signature_hash}"
            stored_context = await self.redis_client.hgetall(context_key)
            
            if not stored_context:
                return 0.5
            
            # Calculate relevance based on context similarity
            # This is a simplified implementation - in production use semantic similarity
            
            relevance_factors = []
            
            # Agent relevance
            if "agent_id" in context_data and "agent_id" in stored_context:
                if context_data["agent_id"] == stored_context["agent_id"]:
                    relevance_factors.append(0.8)
                else:
                    relevance_factors.append(0.2)
            
            # Event type relevance
            if "event_type" in context_data and "event_type" in stored_context:
                if context_data["event_type"] == stored_context["event_type"]:
                    relevance_factors.append(0.7)
                else:
                    relevance_factors.append(0.3)
            
            # Time relevance
            if "timestamp" in context_data and "timestamp" in stored_context:
                time_diff_hours = abs(
                    (datetime.fromisoformat(context_data["timestamp"]) - 
                     datetime.fromisoformat(stored_context["timestamp"])).total_seconds() / 3600
                )
                time_relevance = max(0.1, 1.0 - time_diff_hours / 168)  # Decay over week
                relevance_factors.append(time_relevance)
            
            # Average relevance
            if relevance_factors:
                return sum(relevance_factors) / len(relevance_factors)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"❌ Relevance calculation failed: {e}")
            return 0.5
    
    def _calculate_decay_factor(self, access_pattern: MemoryAccessPattern) -> float:
        """Calculate decay factor based on age and access patterns."""
        
        # Age-based decay
        age_days = (datetime.now() - access_pattern.first_access_time).total_seconds() / 86400
        age_decay = math.exp(-self.decay_rate * age_days)
        
        # Access frequency boost
        frequency_boost = min(math.log(access_pattern.access_count + 1) / 5, 0.5)
        
        # Combined decay factor
        decay_factor = age_decay + frequency_boost
        
        return max(min(decay_factor, 1.0), 0.1)  # Keep minimum 0.1
    
    def _determine_importance_level(self, final_score: float, 
                                  access_pattern: MemoryAccessPattern) -> MemoryImportance:
        """Determine importance level based on score and access patterns."""
        
        # High access frequency = higher importance
        if access_pattern.access_count > 100 or access_pattern.access_velocity > 10:
            if final_score > 0.8:
                return MemoryImportance.CRITICAL
            elif final_score > 0.6:
                return MemoryImportance.HIGH
        
        # Score-based classification
        if final_score > 0.8:
            return MemoryImportance.HIGH
        elif final_score > 0.6:
            return MemoryImportance.MEDIUM
        elif final_score > 0.3:
            return MemoryImportance.LOW
        else:
            return MemoryImportance.EPHEMERAL
    
    async def _update_memory_score(self, signature_hash: str, memory_score: MemoryScore):
        """Update Redis with memory score and TTL."""
        
        try:
            score_key = f"score:{signature_hash}"
            
            score_data = {
                "final_score": memory_score.final_score,
                "importance_level": memory_score.importance_level.value,
                "last_scored": datetime.now().isoformat(),
                "access_frequency": memory_score.access_frequency,
                "decay_factor": memory_score.decay_factor
            }
            
            # Update score with TTL
            pipe = self.redis_client.pipeline()
            pipe.hset(score_key, mapping=score_data)
            pipe.expire(score_key, memory_score.ttl_seconds)
            
            # Update main signature TTL
            signature_key = f"signature:{signature_hash}"
            pipe.expire(signature_key, memory_score.ttl_seconds)
            
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update memory score: {e}")
    
    async def cleanup_expired_memories(self) -> Dict[str, Any]:
        """Clean up expired and low-value memories."""
        
        try:
            start_time = time.time()
            
            # Get all memory scores
            score_keys = await self.redis_client.keys("score:*")
            
            expired_count = 0
            low_value_count = 0
            
            for score_key in score_keys:
                signature_hash = score_key.split(":")[-1]
                
                # Check if expired
                ttl = await self.redis_client.ttl(score_key)
                if ttl <= 0:
                    await self._cleanup_signature(signature_hash)
                    expired_count += 1
                    continue
                
                # Check if low value
                score_data = await self.redis_client.hgetall(score_key)
                if score_data:
                    final_score = float(score_data.get("final_score", 0))
                    if final_score < 0.1:  # Very low value threshold
                        await self._cleanup_signature(signature_hash)
                        low_value_count += 1
            
            cleanup_time = time.time() - start_time
            self.cleanup_operations += 1
            
            result = {
                "status": "success",
                "expired_cleaned": expired_count,
                "low_value_cleaned": low_value_count,
                "total_cleaned": expired_count + low_value_count,
                "cleanup_time_seconds": cleanup_time
            }
            
            self.logger.info(f"🧹 Cleaned up {expired_count + low_value_count} memories in {cleanup_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Memory cleanup failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _cleanup_signature(self, signature_hash: str):
        """Clean up all data associated with a signature."""
        
        keys_to_delete = [
            f"signature:{signature_hash}",
            f"score:{signature_hash}",
            f"access:pattern:{signature_hash}",
            f"context:{signature_hash}"
        ]
        
        await self.redis_client.delete(*keys_to_delete)
    
    def get_ranking_metrics(self) -> Dict[str, Any]:
        """Get ranking service performance metrics."""
        
        avg_ranking_time = self.total_ranking_time / max(self.ranking_operations, 1)
        
        return {
            "ranking_operations": self.ranking_operations,
            "cleanup_operations": self.cleanup_operations,
            "total_ranking_time_seconds": self.total_ranking_time,
            "avg_ranking_time_seconds": avg_ranking_time,
            "ttl_policies": {level.value: ttl for level, ttl in self.ttl_policies.items()},
            "decay_rate": self.decay_rate,
            "is_running": self.is_running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on ranking service."""
        
        try:
            # Check Redis connectivity
            redis_healthy = False
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    redis_healthy = True
                except Exception:
                    pass
            
            # Get memory statistics
            memory_stats = {}
            if redis_healthy:
                try:
                    score_count = len(await self.redis_client.keys("score:*"))
                    pattern_count = len(await self.redis_client.keys("access:pattern:*"))
                    memory_stats = {
                        "scored_memories": score_count,
                        "tracked_patterns": pattern_count
                    }
                except Exception:
                    pass
            
            return {
                "status": "healthy" if redis_healthy else "unhealthy",
                "redis_healthy": redis_healthy,
                "background_running": self.is_running,
                "memory_stats": memory_stats,
                "metrics": self.get_ranking_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "redis_healthy": False
            }
