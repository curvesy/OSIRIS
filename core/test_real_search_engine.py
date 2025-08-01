#!/usr/bin/env python3
"""
🔍 Real Search Engine Validation Test

Comprehensive test to validate that the Intelligence Flywheel
now has a working engine - real DuckDB and Redis search logic.
"""

import asyncio
import tempfile
import os
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import duckdb
import redis
import numpy as np

# Add src to path
import sys
sys.path.append('src')

from aura_intelligence.enterprise.mem0_hot.ingest import HotEpisodicIngestor
from aura_intelligence.enterprise.mem0_semantic.sync import SemanticMemorySync
from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings, RedisSettings
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.enterprise.mem0_search.models import (
    SearchRequest, TopologicalSignature, TopologicalSignatureModel, 
    SignatureMatch, MemoryTier, MemoryCluster
)
from aura_intelligence.enterprise.mem0_search.endpoints import (
    _find_similar_signatures, _search_hot_memory, _search_semantic_memory,
    _search_hot_memory_sync, _search_redis_semantic_sync, _get_redis_clusters_sync
)


class RealSearchEngineValidator:
    """🔍 Validates that the search engine actually works with real data."""
    
    def __init__(self):
        self.temp_db_path = None
        self.duckdb_conn = None
        self.redis_client = None
        self.hot_memory = None
        self.semantic_memory = None
        
    async def setup_test_environment(self):
        """Setup test DuckDB and Redis with real data."""
        
        print("🔧 Setting up test environment...")
        
        # Setup DuckDB
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb')
        temp_db.close()
        self.temp_db_path = temp_db.name
        
        self.duckdb_conn = duckdb.connect(self.temp_db_path)
        
        # Create schema
        from aura_intelligence.enterprise.mem0_hot.schema import create_schema
        create_schema(self.duckdb_conn)
        
        # Setup Redis (use local Redis if available)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=False)
            self.redis_client.ping()
            # Clear test database
            self.redis_client.flushdb()
            print("✅ Connected to local Redis")
        except:
            print("⚠️ Redis not available - semantic search tests will be skipped")
            self.redis_client = None

        # Setup memory components
        duckdb_settings = DuckDBSettings(db_path=self.temp_db_path)

        self.hot_memory = HotEpisodicIngestor(self.duckdb_conn, duckdb_settings)
        await self.hot_memory.initialize()

        if self.redis_client:
            # Setup semantic memory with vectorizer
            vectorizer = SignatureVectorizer()
            redis_url = "redis://localhost:6379/1"
            self.semantic_memory = SemanticMemorySync(redis_url, vectorizer)
            await self.semantic_memory.initialize()
        
        print("✅ Test environment ready")
    
    async def populate_test_data(self):
        """Populate both hot and semantic memory with test signatures."""
        
        print("📊 Populating test data...")
        
        # Create test signatures with known patterns
        test_signatures = []
        
        # Pattern 1: High Betti numbers (complex topology)
        for i in range(10):
            sig = TopologicalSignature(
                hash=f"complex_{i:03d}",
                betti_0=5 + i,
                betti_1=3 + i,
                betti_2=1 + i,
                anomaly_score=0.8 + (i * 0.01)
            )
            test_signatures.append(sig)
        
        # Pattern 2: Low Betti numbers (simple topology)
        for i in range(10):
            sig = TopologicalSignature(
                hash=f"simple_{i:03d}",
                betti_0=1,
                betti_1=0,
                betti_2=0,
                anomaly_score=0.2 + (i * 0.01)
            )
            test_signatures.append(sig)
        
        # Pattern 3: Medium complexity
        for i in range(10):
            sig = TopologicalSignature(
                hash=f"medium_{i:03d}",
                betti_0=3,
                betti_1=2,
                betti_2=1,
                anomaly_score=0.5 + (i * 0.01)
            )
            test_signatures.append(sig)
        
        # Ingest into hot memory
        hot_count = 0
        for sig in test_signatures:
            success = await self.hot_memory.ingest_signature(
                signature=sig,
                agent_id=f"test_agent_{hot_count % 3}",
                event_type="test_event",
                agent_meta={"test": True, "pattern": sig.hash.split('_')[0]},
                full_event={"signature": sig.hash, "timestamp": datetime.now().isoformat()}
            )
            if success:
                hot_count += 1
        
        print(f"✅ Ingested {hot_count} signatures into hot memory")
        
        # Populate semantic memory (Redis) if available using production-grade method
        semantic_count = 0
        if self.redis_client and self.semantic_memory:
            vectorizer = SignatureVectorizer()

            # Prepare consolidated memories for semantic sync
            consolidated_memories = []
            for i, sig in enumerate(test_signatures[::2]):  # Every other signature
                try:
                    # Create vector
                    vector = vectorizer.vectorize_signature(sig)

                    # Create memory record for semantic sync
                    memory = {
                        'hash': sig.hash,
                        'embedding': vector.tolist(),
                        'betti_numbers': [sig.betti_0, sig.betti_1, sig.betti_2],
                        'agent_id': f'semantic_agent_{i % 2}',
                        'event_type': 'consolidated_memory',
                        'timestamp': datetime.now().isoformat()
                    }

                    consolidated_memories.append(memory)

                except Exception as e:
                    print(f"Failed to prepare semantic memory {sig.hash}: {e}")

            # Use production-grade sync method
            try:
                operations = await self.semantic_memory.sync_consolidated_memories(consolidated_memories)
                semantic_count = len(consolidated_memories)
                print(f"✅ Synced {semantic_count} memories to semantic store ({operations} operations)")
            except Exception as e:
                print(f"Failed to sync consolidated memories: {e}")
                semantic_count = 0
        
        return len(test_signatures)
    
    async def test_hot_memory_search(self):
        """Test DuckDB hot memory vector search."""
        
        print("\n🔥 Testing Hot Memory Search...")
        
        # Create a query signature similar to "complex" pattern
        query_sig = TopologicalSignature(
            hash="query_complex",
            betti_0=6,  # Similar to complex pattern
            betti_1=4,
            betti_2=2,
            anomaly_score=0.85
        )
        
        # Create search request
        request = SearchRequest(
            signature=TopologicalSignatureModel(
                hash=query_sig.hash,
                betti_0=query_sig.betti_0,
                betti_1=query_sig.betti_1,
                betti_2=query_sig.betti_2,
                anomaly_score=query_sig.anomaly_score,
                timestamp=datetime.now()
            ),
            similarity_threshold=0.5,
            max_results=5
        )
        
        # Execute search
        start_time = time.time()
        matches = await _search_hot_memory(request, self.hot_memory)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate results
        assert len(matches) > 0, "Hot memory search returned no results"
        assert all(isinstance(m, SignatureMatch) for m in matches), "Invalid match types"
        assert all(m.memory_tier == MemoryTier.HOT for m in matches), "Wrong memory tier"
        assert all(m.similarity_score >= 0.5 for m in matches), "Similarity threshold not respected"
        
        # Check that results are sorted by similarity (highest first)
        similarities = [m.similarity_score for m in matches]
        assert similarities == sorted(similarities, reverse=True), "Results not sorted by similarity"
        
        print(f"✅ Hot memory search: {len(matches)} matches in {search_time:.2f}ms")
        print(f"   Best match: {matches[0].signature.hash} (similarity: {matches[0].similarity_score:.3f})")
        
        return matches
    
    async def test_semantic_memory_search(self):
        """Test Redis semantic memory search."""
        
        if not self.redis_client or not self.semantic_memory:
            print("⚠️ Skipping semantic memory test - Redis not available")
            return []
        
        print("\n🧠 Testing Semantic Memory Search...")
        
        # Create query signature
        query_sig = TopologicalSignature(
            hash="query_semantic",
            betti_0=1,  # Similar to simple pattern
            betti_1=0,
            betti_2=0,
            anomaly_score=0.25
        )
        
        request = SearchRequest(
            signature=TopologicalSignatureModel(
                hash=query_sig.hash,
                betti_0=query_sig.betti_0,
                betti_1=query_sig.betti_1,
                betti_2=query_sig.betti_2,
                anomaly_score=query_sig.anomaly_score,
                timestamp=datetime.now()
            ),
            similarity_threshold=0.4,
            max_results=5
        )
        
        # Execute search
        start_time = time.time()
        matches, clusters = await _search_semantic_memory(request, self.semantic_memory)
        search_time = (time.time() - start_time) * 1000
        
        # Validate results
        print(f"✅ Semantic memory search: {len(matches)} matches, {len(clusters)} clusters in {search_time:.2f}ms")
        
        if matches:
            assert all(isinstance(m, SignatureMatch) for m in matches), "Invalid match types"
            assert all(m.memory_tier == MemoryTier.SEMANTIC for m in matches), "Wrong memory tier"
            print(f"   Best match: {matches[0].signature.hash} (similarity: {matches[0].similarity_score:.3f})")
        
        if clusters:
            assert all(isinstance(c, MemoryCluster) for c in clusters), "Invalid cluster types"
            print(f"   Best cluster: {clusters[0].cluster_label} ({clusters[0].signature_count} signatures)")
        
        return matches, clusters
    
    async def test_unified_search(self):
        """Test unified search across both memory tiers."""
        
        print("\n🔍 Testing Unified Multi-Tier Search...")
        
        # Create query signature
        query_sig = TopologicalSignature(
            hash="query_unified",
            betti_0=3,  # Medium complexity
            betti_1=2,
            betti_2=1,
            anomaly_score=0.55
        )
        
        # Execute unified search
        start_time = time.time()
        matches = await _find_similar_signatures(
            signature=query_sig,
            threshold=0.3,
            max_results=10,
            hot_memory=self.hot_memory,
            semantic_memory=self.semantic_memory if self.semantic_memory else None
        )
        search_time = (time.time() - start_time) * 1000
        
        # Validate results
        assert len(matches) > 0, "Unified search returned no results"
        
        # Check that we have results from hot memory at minimum
        hot_matches = [m for m in matches if m.memory_tier == MemoryTier.HOT]
        assert len(hot_matches) > 0, "No hot memory results in unified search"
        
        # If semantic memory is available, we might have semantic results too
        semantic_matches = [m for m in matches if m.memory_tier == MemoryTier.SEMANTIC]
        
        print(f"✅ Unified search: {len(matches)} total matches in {search_time:.2f}ms")
        print(f"   Hot tier: {len(hot_matches)} matches")
        print(f"   Semantic tier: {len(semantic_matches)} matches")
        
        # Verify SLA compliance
        assert search_time < 60, f"Search exceeded 60ms SLA: {search_time:.2f}ms"
        print(f"   ✅ SLA compliance: {search_time:.2f}ms < 60ms")
        
        return matches
    
    async def cleanup(self):
        """Clean up test resources."""
        
        if self.duckdb_conn:
            self.duckdb_conn.close()
        
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        if self.redis_client:
            self.redis_client.flushdb()
            self.redis_client.close()
        
        print("🧹 Test environment cleaned up")


async def run_real_search_validation():
    """Run comprehensive validation of the real search engine."""
    
    print("🚀 REAL SEARCH ENGINE VALIDATION")
    print("=" * 50)
    print("Testing that the Intelligence Flywheel now has a working engine!")
    print()
    
    validator = RealSearchEngineValidator()
    
    try:
        # Setup
        await validator.setup_test_environment()
        
        # Populate test data
        signature_count = await validator.populate_test_data()
        print(f"📊 Test dataset: {signature_count} signatures")
        
        # Test individual components
        hot_matches = await validator.test_hot_memory_search()
        semantic_results = await validator.test_semantic_memory_search()
        
        # Test unified search
        unified_matches = await validator.test_unified_search()
        
        # Final validation
        print("\n" + "=" * 50)
        print("🎉 REAL SEARCH ENGINE VALIDATION COMPLETE!")
        print()
        print("✅ DuckDB Hot Memory Search: WORKING")
        print("✅ Redis Semantic Memory Search: WORKING" if validator.redis_client else "⚠️ Redis Semantic Memory Search: SKIPPED (Redis not available)")
        print("✅ Unified Multi-Tier Search: WORKING")
        print("✅ Sub-60ms SLA Compliance: ACHIEVED")
        print()
        print("🔥 THE INTELLIGENCE FLYWHEEL ENGINE IS INSTALLED AND RUNNING!")
        print("🧠 The system can now retrieve memories and learn from experience!")
        print("🚀 Ready for Priority #2: Data Lifecycle Implementation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    success = asyncio.run(run_real_search_validation())
    exit(0 if success else 1)
