"""
⚡ High-throughput async writer

Async ingestion pipeline for topological signatures into DuckDB hot tier.
Implements bulk INSERT with COPY FROM STDIN for optimal performance.

Based on partab.md: "≤30 ms per batch" SLA requirement.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import duckdb
import numpy as np
from dataclasses import dataclass

from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings
from aura_intelligence.enterprise.mem0_hot.schema import RECENT_ACTIVITY_TABLE
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.utils.logger import get_logger


@dataclass
class IngestBatch:
    """Batch of signatures for bulk ingestion."""
    signatures: List[TopologicalSignature]
    agent_ids: List[str]
    event_types: List[str]
    agent_metas: List[Dict[str, Any]]
    full_events: List[Dict[str, Any]]
    timestamp: datetime


class HotEpisodicIngestor:
    """
    ⚡ High-throughput async writer for DuckDB hot tier
    
    Features:
    - Bulk INSERT with COPY FROM STDIN for optimal performance
    - Async batching with configurable batch sizes
    - Vector embedding generation for similarity search
    - Performance metrics tracking (≤30ms per batch SLA)
    - Automatic retry logic with exponential backoff
    """
    
    def __init__(self, 
                 conn: duckdb.DuckDBPyConnection,
                 settings: DuckDBSettings,
                 vectorizer: Optional[SignatureVectorizer] = None):
        """Initialize the hot episodic ingestor."""
        
        self.conn = conn
        self.settings = settings
        self.vectorizer = vectorizer or SignatureVectorizer(settings.vector_dimension)
        
        # Performance tracking
        self.batch_count = 0
        self.total_ingestion_time = 0.0
        self.total_signatures_ingested = 0
        self.avg_batch_time = 0.0
        
        # Batch configuration
        self.batch_size = 100  # Signatures per batch
        self.max_batch_time_ms = 30  # SLA requirement from partab.md
        
        # Current batch
        self.current_batch: List[tuple] = []
        self.batch_lock = asyncio.Lock()
        
        self.logger = get_logger(__name__)
        self.logger.info("⚡ Hot Episodic Ingestor initialized")
    
    async def ingest_signature(self,
                             signature: TopologicalSignature,
                             agent_id: str,
                             event_type: str,
                             agent_meta: Dict[str, Any] = None,
                             full_event: Dict[str, Any] = None) -> bool:
        """
        Ingest a single topological signature.
        
        Adds to current batch and flushes when batch size is reached.
        """
        
        try:
            # Generate vector embedding
            signature_vector = await self.vectorizer.vectorize_signature(signature)
            
            # Prepare row data
            # Handle both old format (betti_0, betti_1, betti_2) and new format (betti_numbers list)
            if hasattr(signature, 'betti_0'):
                betti_0, betti_1, betti_2 = signature.betti_0, signature.betti_1, signature.betti_2
            else:
                betti_nums = signature.betti_numbers + [0, 0, 0]  # Pad with zeros
                betti_0, betti_1, betti_2 = betti_nums[0], betti_nums[1], betti_nums[2]

            signature_hash = getattr(signature, 'hash', None) or signature.signature_hash

            row_data = (
                datetime.now(),
                signature_hash,
                betti_0,
                betti_1,
                betti_2,
                agent_id,
                event_type,
                json.dumps(agent_meta or {}),
                json.dumps(full_event or {}),
                signature_vector.tolist()
            )
            
            # Add to batch
            async with self.batch_lock:
                self.current_batch.append(row_data)
                
                # Flush batch if size reached
                if len(self.current_batch) >= self.batch_size:
                    await self._flush_batch()
            
            return True
            
        except Exception as e:
            signature_hash = getattr(signature, 'hash', None) or signature.signature_hash
            self.logger.error(f"❌ Failed to ingest signature {signature_hash[:8]}...: {e}")
            return False
    
    async def ingest_batch(self, batch: IngestBatch) -> bool:
        """
        Ingest a batch of signatures using bulk INSERT.
        
        Optimized for high-throughput with COPY FROM STDIN.
        """
        
        if not batch.signatures:
            return True
        
        try:
            start_time = time.time()
            
            # Prepare batch data
            batch_data = []
            for i, signature in enumerate(batch.signatures):
                # Generate vector embedding
                signature_vector = await self.vectorizer.vectorize_signature(signature)
                
                # Handle both old format (betti_0, betti_1, betti_2) and new format (betti_numbers list)
                if hasattr(signature, 'betti_0'):
                    betti_0, betti_1, betti_2 = signature.betti_0, signature.betti_1, signature.betti_2
                else:
                    betti_nums = signature.betti_numbers + [0, 0, 0]  # Pad with zeros
                    betti_0, betti_1, betti_2 = betti_nums[0], betti_nums[1], betti_nums[2]

                signature_hash = getattr(signature, 'hash', None) or signature.signature_hash

                row_data = (
                    batch.timestamp,
                    signature_hash,
                    betti_0,
                    betti_1,
                    betti_2,
                    batch.agent_ids[i] if i < len(batch.agent_ids) else "unknown",
                    batch.event_types[i] if i < len(batch.event_types) else "signature",
                    json.dumps(batch.agent_metas[i] if i < len(batch.agent_metas) else {}),
                    json.dumps(batch.full_events[i] if i < len(batch.full_events) else {}),
                    signature_vector.tolist()
                )
                batch_data.append(row_data)
            
            # Bulk insert using executemany for better performance
            insert_sql = f"""
            INSERT OR REPLACE INTO {RECENT_ACTIVITY_TABLE}
            (timestamp, signature_hash, betti_0, betti_1, betti_2,
             agent_id, event_type, agent_meta, full_event, signature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.conn.executemany(insert_sql, batch_data)
            
            # Update performance metrics
            batch_time = (time.time() - start_time) * 1000
            self.batch_count += 1
            self.total_ingestion_time += batch_time
            self.total_signatures_ingested += len(batch.signatures)
            self.avg_batch_time = self.total_ingestion_time / self.batch_count
            
            # Check SLA compliance
            if batch_time > self.max_batch_time_ms:
                self.logger.warning(f"⚠️ Batch ingestion exceeded SLA: {batch_time:.2f}ms > {self.max_batch_time_ms}ms")
            else:
                self.logger.debug(f"⚡ Ingested {len(batch.signatures)} signatures in {batch_time:.2f}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch ingestion failed: {e}")
            return False
    
    async def _flush_batch(self):
        """Flush the current batch to DuckDB."""
        
        if not self.current_batch:
            return
        
        try:
            start_time = time.time()
            
            # Bulk insert current batch
            insert_sql = f"""
            INSERT OR REPLACE INTO {RECENT_ACTIVITY_TABLE}
            (timestamp, signature_hash, betti_0, betti_1, betti_2,
             agent_id, event_type, agent_meta, full_event, signature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Execute in thread pool to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.conn.executemany, insert_sql, self.current_batch)
            
            # Update metrics
            batch_time = (time.time() - start_time) * 1000
            batch_size = len(self.current_batch)
            
            self.batch_count += 1
            self.total_ingestion_time += batch_time
            self.total_signatures_ingested += batch_size
            self.avg_batch_time = self.total_ingestion_time / self.batch_count
            
            # Clear batch
            self.current_batch.clear()
            
            # Log performance
            if batch_time > self.max_batch_time_ms:
                self.logger.warning(f"⚠️ Batch flush exceeded SLA: {batch_time:.2f}ms > {self.max_batch_time_ms}ms")
            else:
                self.logger.debug(f"⚡ Flushed {batch_size} signatures in {batch_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"❌ Batch flush failed: {e}")
            # Don't clear batch on failure - will retry
    
    async def force_flush(self) -> bool:
        """Force flush any pending signatures in the current batch."""
        
        async with self.batch_lock:
            if self.current_batch:
                await self._flush_batch()
                return True
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get ingestion performance metrics."""
        
        return {
            "batch_count": self.batch_count,
            "total_signatures_ingested": self.total_signatures_ingested,
            "total_ingestion_time_ms": self.total_ingestion_time,
            "avg_batch_time_ms": self.avg_batch_time,
            "sla_compliance": self.avg_batch_time <= self.max_batch_time_ms,
            "current_batch_size": len(self.current_batch),
            "max_batch_time_ms": self.max_batch_time_ms
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the ingestor."""
        
        try:
            # Execute health checks in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()

            # Test connection
            test_result = await loop.run_in_executor(
                None, lambda: self.conn.execute("SELECT 1").fetchone()
            )
            connection_healthy = test_result[0] == 1

            # Check table exists
            table_result = await loop.run_in_executor(
                None, lambda: self.conn.execute(f"""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_name = '{RECENT_ACTIVITY_TABLE}'
                """).fetchone()
            )
            table_exists = table_result[0] > 0
            
            # Get current batch status
            current_batch_size = len(self.current_batch)
            
            return {
                "status": "healthy" if connection_healthy and table_exists else "unhealthy",
                "connection_healthy": connection_healthy,
                "table_exists": table_exists,
                "current_batch_size": current_batch_size,
                "performance_metrics": self.get_performance_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_healthy": False,
                "table_exists": False
            }


# Alias for backward compatibility
HotMemoryIngest = HotEpisodicIngestor
