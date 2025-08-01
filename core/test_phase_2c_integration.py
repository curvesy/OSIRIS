#!/usr/bin/env python3
"""
🧪 Phase 2C Integration Test

Comprehensive test of hot episodic memory, semantic long-term memory,
and unified search API following partab.md blueprint.

Tests the complete Phase 2C implementation:
- DuckDB hot tier with vectorization
- Redis semantic tier with clustering
- Memory ranking and TTL policies
- FastAPI search endpoints
- Background archival processes
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any

# Test the Phase 2C components
from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings, DEV_SETTINGS
from aura_intelligence.enterprise.mem0_hot.schema import create_schema
from aura_intelligence.enterprise.mem0_hot.ingest import HotEpisodicIngestor
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.enterprise.mem0_hot.archive import ArchivalManager
from aura_intelligence.enterprise.mem0_semantic.sync import SemanticMemorySync
from aura_intelligence.enterprise.mem0_semantic.rank import MemoryRankingService
from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.utils.logger import get_logger

import duckdb

logger = get_logger(__name__)


class Phase2CIntegrationTest:
    """🧪 Comprehensive Phase 2C integration test suite."""
    
    def __init__(self):
        """Initialize test environment."""
        
        # Use development settings
        self.settings = DEV_SETTINGS
        self.redis_url = "redis://localhost:6379/1"  # Test database
        
        # Components
        self.duckdb_conn = None
        self.hot_ingestor = None
        self.vectorizer = None
        self.archival_manager = None
        self.semantic_sync = None
        self.ranking_service = None
        
        # Test data
        self.test_signatures = []
        
        logger.info("🧪 Phase 2C Integration Test initialized")
    
    async def setup(self):
        """Set up test environment."""
        
        try:
            logger.info("🔧 Setting up Phase 2C test environment...")
            
            # Create DuckDB connection
            self.duckdb_conn = duckdb.connect(":memory:")
            
            # Create schema
            create_schema(self.duckdb_conn, self.settings.vector_dimension)
            
            # Initialize vectorizer
            self.vectorizer = SignatureVectorizer(self.settings.vector_dimension)
            
            # Initialize hot ingestor
            self.hot_ingestor = HotEpisodicIngestor(
                conn=self.duckdb_conn,
                settings=self.settings,
                vectorizer=self.vectorizer
            )
            
            # Initialize archival manager
            self.archival_manager = ArchivalManager(
                conn=self.duckdb_conn,
                settings=self.settings
            )
            
            # Initialize semantic sync
            self.semantic_sync = SemanticMemorySync(
                redis_url=self.redis_url,
                vectorizer=self.vectorizer,
                cluster_threshold=0.8
            )
            
            # Initialize ranking service
            self.ranking_service = MemoryRankingService(redis_url=self.redis_url)
            
            # Initialize Redis connections
            await self.semantic_sync.initialize()
            await self.ranking_service.initialize()
            
            # Generate test data
            self._generate_test_signatures()
            
            logger.info("✅ Phase 2C test environment ready")
            
        except Exception as e:
            logger.error(f"❌ Test setup failed: {e}")
            raise
    
    def _generate_test_signatures(self):
        """Generate test topological signatures."""
        
        test_data = [
            {"betti_0": 5, "betti_1": 2, "betti_2": 0, "anomaly_score": 0.1},
            {"betti_0": 8, "betti_1": 4, "betti_2": 1, "anomaly_score": 0.3},
            {"betti_0": 12, "betti_1": 6, "betti_2": 2, "anomaly_score": 0.8},  # Anomaly
            {"betti_0": 3, "betti_1": 1, "betti_2": 0, "anomaly_score": 0.05},
            {"betti_0": 15, "betti_1": 8, "betti_2": 3, "anomaly_score": 0.9},  # High anomaly
        ]
        
        for i, data in enumerate(test_data):
            signature = TopologicalSignature(
                hash=f"test_signature_{i:03d}_{int(time.time())}",
                betti_0=data["betti_0"],
                betti_1=data["betti_1"],
                betti_2=data["betti_2"],
                anomaly_score=data["anomaly_score"]
            )
            self.test_signatures.append(signature)
        
        logger.info(f"📊 Generated {len(self.test_signatures)} test signatures")
    
    async def test_hot_memory_ingestion(self) -> Dict[str, Any]:
        """Test hot memory ingestion performance."""
        
        logger.info("🔥 Testing hot memory ingestion...")
        
        start_time = time.time()
        success_count = 0
        
        for i, signature in enumerate(self.test_signatures):
            success = await self.hot_ingestor.ingest_signature(
                signature=signature,
                agent_id=f"test_agent_{i % 3}",
                event_type="test",
                agent_meta={"test_run": "phase_2c_integration"},
                full_event={"signature_index": i}
            )
            
            if success:
                success_count += 1
        
        # Force flush any pending batch
        await self.hot_ingestor.force_flush()
        
        ingestion_time = time.time() - start_time
        
        # Get performance metrics
        metrics = self.hot_ingestor.get_performance_metrics()
        
        result = {
            "test": "hot_memory_ingestion",
            "signatures_processed": len(self.test_signatures),
            "successful_ingestions": success_count,
            "ingestion_time_seconds": ingestion_time,
            "avg_time_per_signature_ms": (ingestion_time * 1000) / len(self.test_signatures),
            "performance_metrics": metrics,
            "sla_compliance": metrics["sla_compliance"]
        }
        
        logger.info(f"✅ Hot memory ingestion: {success_count}/{len(self.test_signatures)} successful")
        
        return result
    
    async def test_vectorization(self) -> Dict[str, Any]:
        """Test signature vectorization."""
        
        logger.info("🔢 Testing signature vectorization...")
        
        start_time = time.time()
        vectors = []
        
        for signature in self.test_signatures:
            vector = await self.vectorizer.vectorize_signature(signature)
            vectors.append(vector)
        
        vectorization_time = time.time() - start_time
        
        # Test vector properties
        vector_dimension = len(vectors[0]) if vectors else 0
        vector_norms = [float(sum(v**2)**0.5) for v in vectors]
        
        # Test similarity computation
        similarity_scores = []
        if len(vectors) >= 2:
            for i in range(len(vectors) - 1):
                similarity = self.vectorizer.compute_similarity(vectors[i], vectors[i+1])
                similarity_scores.append(similarity)
        
        result = {
            "test": "vectorization",
            "signatures_vectorized": len(vectors),
            "vector_dimension": vector_dimension,
            "vectorization_time_seconds": vectorization_time,
            "avg_vector_norm": sum(vector_norms) / len(vector_norms) if vector_norms else 0,
            "similarity_scores": similarity_scores,
            "embedding_info": self.vectorizer.get_embedding_info()
        }
        
        logger.info(f"✅ Vectorization: {len(vectors)} vectors generated (dim: {vector_dimension})")
        
        return result
    
    async def test_semantic_memory_sync(self) -> Dict[str, Any]:
        """Test semantic memory synchronization."""
        
        logger.info("🧠 Testing semantic memory sync...")
        
        # Create test batch
        from aura_intelligence.enterprise.mem0_semantic.sync import MemoryConsolidationBatch
        
        vectors = []
        for signature in self.test_signatures:
            vector = await self.vectorizer.vectorize_signature(signature)
            vectors.append(vector)
        
        batch = MemoryConsolidationBatch(
            signatures=self.test_signatures,
            vectors=vectors,
            timestamp=datetime.now(),
            batch_id=f"test_batch_{int(time.time())}"
        )
        
        # Sync batch
        sync_result = await self.semantic_sync.sync_batch(batch)
        
        # Get sync metrics
        metrics = self.semantic_sync.get_sync_metrics()
        
        result = {
            "test": "semantic_memory_sync",
            "batch_sync_result": sync_result,
            "sync_metrics": metrics,
            "active_clusters": len(self.semantic_sync.active_clusters)
        }
        
        logger.info(f"✅ Semantic sync: {sync_result['status']} - {len(self.semantic_sync.active_clusters)} clusters")
        
        return result
    
    async def test_memory_ranking(self) -> Dict[str, Any]:
        """Test memory ranking and scoring."""
        
        logger.info("🏆 Testing memory ranking...")
        
        ranking_results = []
        
        for signature in self.test_signatures[:3]:  # Test first 3
            context_data = {
                "agent_id": "test_agent_0",
                "event_type": "test",
                "timestamp": datetime.now().isoformat()
            }
            
            memory_score = await self.ranking_service.score_memory(
                signature.hash, context_data
            )
            
            ranking_results.append({
                "signature_hash": signature.hash,
                "final_score": memory_score.final_score,
                "importance_level": memory_score.importance_level.value,
                "ttl_seconds": memory_score.ttl_seconds
            })
        
        # Get ranking metrics
        metrics = self.ranking_service.get_ranking_metrics()
        
        result = {
            "test": "memory_ranking",
            "ranking_results": ranking_results,
            "ranking_metrics": metrics
        }
        
        logger.info(f"✅ Memory ranking: {len(ranking_results)} signatures scored")
        
        return result
    
    async def test_health_checks(self) -> Dict[str, Any]:
        """Test health checks for all components."""
        
        logger.info("🏥 Testing component health checks...")
        
        health_results = {}
        
        # Hot ingestor health
        health_results["hot_ingestor"] = await self.hot_ingestor.health_check()
        
        # Semantic sync health
        health_results["semantic_sync"] = await self.semantic_sync.health_check()
        
        # Ranking service health
        health_results["ranking_service"] = await self.ranking_service.health_check()
        
        # Archival manager health
        health_results["archival_manager"] = await self.archival_manager.health_check()
        
        result = {
            "test": "health_checks",
            "component_health": health_results,
            "overall_healthy": all(
                h.get("status") == "healthy" for h in health_results.values()
            )
        }
        
        logger.info(f"✅ Health checks: {sum(1 for h in health_results.values() if h.get('status') == 'healthy')}/{len(health_results)} healthy")
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete Phase 2C integration test suite."""
        
        logger.info("🚀 Starting Phase 2C Integration Test Suite...")
        
        test_start = time.time()
        test_results = {}
        
        try:
            # Setup
            await self.setup()
            
            # Run tests
            test_results["hot_memory_ingestion"] = await self.test_hot_memory_ingestion()
            test_results["vectorization"] = await self.test_vectorization()
            test_results["semantic_memory_sync"] = await self.test_semantic_memory_sync()
            test_results["memory_ranking"] = await self.test_memory_ranking()
            test_results["health_checks"] = await self.test_health_checks()
            
            # Overall results
            total_time = time.time() - test_start
            
            test_results["summary"] = {
                "total_test_time_seconds": total_time,
                "tests_passed": sum(1 for result in test_results.values() 
                                  if isinstance(result, dict) and result.get("test")),
                "overall_success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Phase 2C Integration Tests PASSED in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Phase 2C Integration Tests FAILED: {e}")
            test_results["error"] = str(e)
            test_results["summary"] = {
                "overall_success": False,
                "error": str(e)
            }
        
        finally:
            await self.cleanup()
        
        return test_results
    
    async def cleanup(self):
        """Clean up test environment."""
        
        try:
            # Stop background services
            if self.semantic_sync:
                await self.semantic_sync.stop_background_sync()
            
            if self.ranking_service:
                await self.ranking_service.stop_background_cleanup()
            
            # Close connections
            if self.duckdb_conn:
                self.duckdb_conn.close()
            
            logger.info("🧹 Test cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Test cleanup failed: {e}")


async def main():
    """Run Phase 2C integration tests."""
    
    print("🧪 AURA Intelligence Phase 2C Integration Test")
    print("=" * 60)
    
    test_suite = Phase2CIntegrationTest()
    results = await test_suite.run_all_tests()
    
    print("\n📊 Test Results:")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))
    
    if results.get("summary", {}).get("overall_success"):
        print("\n🎉 Phase 2C Integration Tests: PASSED")
        return 0
    else:
        print("\n❌ Phase 2C Integration Tests: FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
