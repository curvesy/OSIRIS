"""
🧠 Production-Grade Semantic Memory Consolidation Pipeline

This module implements the "Cold to Wise" pipeline that transforms archived
historical data into semantic wisdom using enterprise-grade patterns:

- High-water mark coordination for incremental processing
- HDBSCAN clustering for pattern discovery
- Vector similarity threshold-based semantic grouping
- Batch processing with 30-day windows for pattern stability
- Redis vector index population with server-side search
- Comprehensive monitoring and error handling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Production-grade imports
from prometheus_client import Counter, Histogram, Gauge
import redis
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from aura_intelligence.enterprise.mem0_semantic.sync import SemanticMemorySync
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.utils.logger import get_logger


# Production-Grade Prometheus Metrics
CONSOLIDATION_JOB_DURATION = Histogram(
    'semantic_consolidation_job_duration_seconds',
    'Duration of semantic consolidation job execution',
    ['status']
)

CONSOLIDATION_JOB_SUCCESS = Counter(
    'semantic_consolidation_job_success_total',
    'Total number of successful consolidation jobs'
)

CONSOLIDATION_JOB_FAILURES = Counter(
    'semantic_consolidation_job_failures_total',
    'Total number of failed consolidation jobs',
    ['error_type']
)

SEMANTIC_PATTERNS_DISCOVERED = Counter(
    'semantic_patterns_discovered_total',
    'Total number of semantic patterns discovered',
    ['pattern_type']
)

SEMANTIC_CLUSTERING_DURATION = Histogram(
    'semantic_clustering_duration_seconds',
    'Duration of semantic clustering operations',
    ['algorithm']
)

SEMANTIC_RECORDS_PROCESSED = Counter(
    'semantic_records_processed_total',
    'Total number of records processed for semantic consolidation'
)


@dataclass
class SemanticPattern:
    """Represents a discovered semantic pattern."""
    
    pattern_id: str
    cluster_label: int
    centroid_vector: np.ndarray
    member_signatures: List[str]
    frequency: int
    confidence_score: float
    pattern_type: str
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'pattern_id': self.pattern_id,
            'cluster_label': self.cluster_label,
            'centroid_vector': self.centroid_vector.tolist(),
            'member_signatures': self.member_signatures,
            'frequency': self.frequency,
            'confidence_score': self.confidence_score,
            'pattern_type': self.pattern_type,
            'discovered_at': self.discovered_at.isoformat()
        }


@dataclass
class HighWaterMark:
    """High-water mark for incremental processing coordination."""
    
    last_processed_timestamp: datetime
    last_processed_partition: str
    records_processed: int
    patterns_discovered: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'last_processed_timestamp': self.last_processed_timestamp.isoformat(),
            'last_processed_partition': self.last_processed_partition,
            'records_processed': self.records_processed,
            'patterns_discovered': self.patterns_discovered
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HighWaterMark':
        """Create from dictionary."""
        return cls(
            last_processed_timestamp=datetime.fromisoformat(data['last_processed_timestamp']),
            last_processed_partition=data['last_processed_partition'],
            records_processed=data['records_processed'],
            patterns_discovered=data['patterns_discovered']
        )


class SemanticConsolidationPipeline:
    """
    🧠 Production-Grade Semantic Memory Consolidation Pipeline
    
    Implements the "Cold to Wise" transformation with:
    - High-water mark coordination for incremental processing
    - HDBSCAN clustering for density-based pattern discovery
    - Vector similarity threshold (0.85) for semantic clustering
    - Batch processing with 30-day windows for pattern stability
    - Redis vector index population with server-side search
    - Comprehensive monitoring and error handling
    """
    
    def __init__(
        self,
        semantic_memory: SemanticMemorySync,
        vectorizer: SignatureVectorizer,
        s3_bucket: str,
        s3_prefix: str = "aura-intelligence/hot-memory-archive",
        similarity_threshold: float = 0.85,
        batch_window_days: int = 30
    ):
        self.semantic_memory = semantic_memory
        self.vectorizer = vectorizer
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.similarity_threshold = similarity_threshold
        self.batch_window_days = batch_window_days
        self.logger = get_logger(__name__)
        
        # Initialize clustering algorithm
        self.clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='cosine',
            cluster_selection_epsilon=1 - similarity_threshold
        )
        
        # Initialize scaler for vector normalization
        self.scaler = StandardScaler()
        
        # High-water mark for incremental processing
        self.high_water_mark_key = "semantic:consolidation:high_water_mark"
        
        self.logger.info("🧠 Semantic Consolidation Pipeline initialized")
    
    async def consolidate_archived_data(self) -> Dict[str, Any]:
        """
        Main consolidation process - transforms archived data into semantic wisdom.
        
        This is the main entry point for scheduled consolidation jobs.
        
        Returns:
            Comprehensive consolidation statistics and metrics
        """
        
        start_time = time.time()
        
        try:
            with CONSOLIDATION_JOB_DURATION.labels(status='success').time():
                self.logger.info("🧠 Starting semantic consolidation pipeline...")
                
                # Phase 1: Load high-water mark for incremental processing
                high_water_mark = await self._load_high_water_mark()
                
                # Phase 2: Discover new archived data since last processing
                archived_data = await self._discover_new_archived_data(high_water_mark)
                
                if archived_data.empty:
                    self.logger.info("✅ No new archived data to consolidate")
                    CONSOLIDATION_JOB_SUCCESS.inc()
                    return {
                        "status": "success",
                        "patterns_discovered": 0,
                        "records_processed": 0,
                        "duration_seconds": time.time() - start_time
                    }
                
                # Phase 3: Perform semantic clustering and pattern discovery
                patterns = await self._discover_semantic_patterns(archived_data)
                
                # Phase 4: Populate semantic memory with discovered patterns
                populated_count = await self._populate_semantic_memory(patterns)
                
                # Phase 5: Update high-water mark
                await self._update_high_water_mark(high_water_mark, archived_data, patterns)
                
                # Final statistics
                consolidation_stats = {
                    "status": "success",
                    "patterns_discovered": len(patterns),
                    "records_processed": len(archived_data),
                    "semantic_memories_populated": populated_count,
                    "duration_seconds": time.time() - start_time,
                    "high_water_mark": high_water_mark.to_dict()
                }
                
                # Update metrics
                CONSOLIDATION_JOB_SUCCESS.inc()
                SEMANTIC_RECORDS_PROCESSED.inc(len(archived_data))
                SEMANTIC_PATTERNS_DISCOVERED.labels(pattern_type='clustered').inc(len(patterns))
                
                self.logger.info(f"✅ Semantic consolidation completed: {consolidation_stats}")
                return consolidation_stats
                
        except Exception as e:
            duration = time.time() - start_time
            CONSOLIDATION_JOB_DURATION.labels(status='failure').observe(duration)
            CONSOLIDATION_JOB_FAILURES.labels(error_type='system_failure').inc()
            
            self.logger.error(f"❌ Semantic consolidation failed after {duration:.2f}s: {e}")
            
            return {
                "status": "failure",
                "error": str(e),
                "duration_seconds": duration,
                "patterns_discovered": 0,
                "records_processed": 0
            }
    
    async def _load_high_water_mark(self) -> HighWaterMark:
        """Load high-water mark for incremental processing coordination."""
        
        try:
            # Try to load existing high-water mark from Redis
            redis_client = self.semantic_memory.redis_client
            hwm_data = redis_client.get(self.high_water_mark_key)
            
            if hwm_data:
                hwm_dict = json.loads(hwm_data.decode('utf-8'))
                high_water_mark = HighWaterMark.from_dict(hwm_dict)
                self.logger.info(f"📊 Loaded high-water mark: {high_water_mark.last_processed_timestamp}")
                return high_water_mark
            else:
                # Initialize new high-water mark (start from 30 days ago)
                initial_timestamp = datetime.now() - timedelta(days=self.batch_window_days)
                high_water_mark = HighWaterMark(
                    last_processed_timestamp=initial_timestamp,
                    last_processed_partition="",
                    records_processed=0,
                    patterns_discovered=0
                )
                self.logger.info(f"📊 Initialized new high-water mark: {initial_timestamp}")
                return high_water_mark
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load high-water mark: {e}")
            # Fallback to safe default
            return HighWaterMark(
                last_processed_timestamp=datetime.now() - timedelta(days=self.batch_window_days),
                last_processed_partition="",
                records_processed=0,
                patterns_discovered=0
            )

    async def _discover_new_archived_data(self, high_water_mark: HighWaterMark) -> pd.DataFrame:
        """
        Discover new archived data since last processing using S3 scanning.

        Implements incremental processing by scanning S3 for new Parquet files
        since the last high-water mark timestamp.
        """

        try:
            import boto3
            from botocore.exceptions import ClientError

            # Initialize S3 client
            s3_client = boto3.client('s3')

            # Scan S3 for archived Parquet files since high-water mark
            cutoff_time = high_water_mark.last_processed_timestamp

            # List objects in S3 bucket with prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )

            new_files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Parse Hive-style partition from key
                        key = obj['Key']
                        if key.endswith('.parquet') and self._is_newer_than_hwm(key, cutoff_time):
                            new_files.append({
                                'key': key,
                                'last_modified': obj['LastModified'],
                                'size': obj['Size']
                            })

            if not new_files:
                self.logger.info("📊 No new archived files found")
                return pd.DataFrame()

            # Load and combine data from new files
            combined_data = []
            for file_info in new_files[:50]:  # Limit batch size
                try:
                    # Download and read Parquet file
                    response = s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=file_info['key']
                    )

                    # Read Parquet data
                    import io
                    parquet_data = pd.read_parquet(io.BytesIO(response['Body'].read()))
                    combined_data.append(parquet_data)

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to read {file_info['key']}: {e}")
                    continue

            if combined_data:
                result = pd.concat(combined_data, ignore_index=True)
                self.logger.info(f"📊 Discovered {len(result)} records from {len(combined_data)} files")
                return result
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Failed to discover archived data: {e}")
            return pd.DataFrame()

    def _is_newer_than_hwm(self, s3_key: str, cutoff_time: datetime) -> bool:
        """Check if S3 key represents data newer than high-water mark."""

        try:
            # Parse Hive-style partition from S3 key
            # Format: .../year=2024/month=12/day=25/hour=14/...
            parts = s3_key.split('/')

            year = month = day = hour = None
            for part in parts:
                if part.startswith('year='):
                    year = int(part.split('=')[1])
                elif part.startswith('month='):
                    month = int(part.split('=')[1])
                elif part.startswith('day='):
                    day = int(part.split('=')[1])
                elif part.startswith('hour='):
                    hour = int(part.split('=')[1])

            if all(x is not None for x in [year, month, day, hour]):
                file_timestamp = datetime(year, month, day, hour)
                return file_timestamp > cutoff_time

            return False

        except Exception:
            return False

    async def _discover_semantic_patterns(self, data: pd.DataFrame) -> List[SemanticPattern]:
        """
        Discover semantic patterns using HDBSCAN clustering.

        Performs density-based clustering on signature vectors to identify
        recurring patterns and semantic relationships.
        """

        try:
            with SEMANTIC_CLUSTERING_DURATION.labels(algorithm='hdbscan').time():
                self.logger.info(f"🔍 Starting pattern discovery on {len(data)} records")

                # Vectorize signatures
                vectors = []
                signature_hashes = []

                for _, row in data.iterrows():
                    try:
                        # Create signature object for vectorization
                        from aura_intelligence.enterprise.mem0_hot.schema import TopologicalSignatureModel

                        signature = TopologicalSignatureModel(
                            hash=row['signature_hash'],
                            betti_0=row['betti_0'],
                            betti_1=row['betti_1'],
                            betti_2=row['betti_2'],
                            anomaly_score=row.get('anomaly_score', 0.0),
                            timestamp=row['timestamp']
                        )

                        vector = self.vectorizer.vectorize_signature(signature)
                        vectors.append(vector)
                        signature_hashes.append(row['signature_hash'])

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to vectorize signature {row['signature_hash']}: {e}")
                        continue

                if len(vectors) < 10:
                    self.logger.warning("⚠️ Insufficient vectors for clustering")
                    return []

                # Normalize vectors
                vectors_array = np.array(vectors)
                normalized_vectors = self.scaler.fit_transform(vectors_array)

                # Perform HDBSCAN clustering
                cluster_labels = self.clusterer.fit_predict(normalized_vectors)

                # Extract patterns from clusters
                patterns = []
                unique_labels = set(cluster_labels)

                for label in unique_labels:
                    if label == -1:  # Skip noise points
                        continue

                    # Get cluster members
                    cluster_mask = cluster_labels == label
                    cluster_vectors = normalized_vectors[cluster_mask]
                    cluster_signatures = [signature_hashes[i] for i in range(len(signature_hashes)) if cluster_mask[i]]

                    # Calculate cluster centroid
                    centroid = np.mean(cluster_vectors, axis=0)

                    # Calculate confidence score based on cluster density
                    confidence_score = self._calculate_cluster_confidence(cluster_vectors, centroid)

                    # Create semantic pattern
                    pattern = SemanticPattern(
                        pattern_id=f"pattern_{int(time.time())}_{label}",
                        cluster_label=label,
                        centroid_vector=centroid,
                        member_signatures=cluster_signatures,
                        frequency=len(cluster_signatures),
                        confidence_score=confidence_score,
                        pattern_type="clustered_topology",
                        discovered_at=datetime.now()
                    )

                    patterns.append(pattern)

                self.logger.info(f"🔍 Discovered {len(patterns)} semantic patterns")
                return patterns

        except Exception as e:
            self.logger.error(f"❌ Pattern discovery failed: {e}")
            CONSOLIDATION_JOB_FAILURES.labels(error_type='clustering_failure').inc()
            return []

    def _calculate_cluster_confidence(self, cluster_vectors: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate confidence score for a cluster based on internal cohesion."""

        try:
            # Calculate average distance from centroid
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            avg_distance = np.mean(distances)

            # Convert to confidence score (lower distance = higher confidence)
            confidence = max(0.0, 1.0 - avg_distance)
            return min(1.0, confidence)

        except Exception:
            return 0.5  # Default confidence

    async def _populate_semantic_memory(self, patterns: List[SemanticPattern]) -> int:
        """
        Populate semantic memory with discovered patterns using production-grade sync.

        Converts patterns to consolidated memories and syncs to Redis vector index.
        """

        try:
            if not patterns:
                return 0

            # Convert patterns to consolidated memories format
            consolidated_memories = []

            for pattern in patterns:
                memory = {
                    'hash': pattern.pattern_id,
                    'embedding': pattern.centroid_vector.tolist(),
                    'betti_numbers': [0, 0, 0],  # Patterns don't have specific Betti numbers
                    'agent_id': 'semantic_consolidation_pipeline',
                    'event_type': 'semantic_pattern',
                    'timestamp': pattern.discovered_at.isoformat(),
                    'metadata': {
                        'pattern_type': pattern.pattern_type,
                        'frequency': pattern.frequency,
                        'confidence_score': pattern.confidence_score,
                        'member_count': len(pattern.member_signatures),
                        'cluster_label': pattern.cluster_label
                    }
                }

                consolidated_memories.append(memory)

            # Use production-grade semantic sync
            operations = await self.semantic_memory.sync_consolidated_memories(consolidated_memories)

            self.logger.info(f"✅ Populated {len(consolidated_memories)} semantic patterns ({operations} operations)")
            return len(consolidated_memories)

        except Exception as e:
            self.logger.error(f"❌ Failed to populate semantic memory: {e}")
            CONSOLIDATION_JOB_FAILURES.labels(error_type='memory_population_failure').inc()
            return 0

    async def _update_high_water_mark(
        self,
        high_water_mark: HighWaterMark,
        processed_data: pd.DataFrame,
        patterns: List[SemanticPattern]
    ):
        """Update high-water mark after successful processing."""

        try:
            if not processed_data.empty:
                # Update high-water mark with latest processed timestamp
                latest_timestamp = processed_data['timestamp'].max()

                high_water_mark.last_processed_timestamp = latest_timestamp
                high_water_mark.records_processed += len(processed_data)
                high_water_mark.patterns_discovered += len(patterns)

                # Store updated high-water mark in Redis
                redis_client = self.semantic_memory.redis_client
                hwm_json = json.dumps(high_water_mark.to_dict())
                redis_client.set(self.high_water_mark_key, hwm_json)

                self.logger.info(f"📊 Updated high-water mark: {latest_timestamp}")

        except Exception as e:
            self.logger.error(f"❌ Failed to update high-water mark: {e}")
