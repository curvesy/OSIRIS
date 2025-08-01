"""
🧪 End-to-End Search Validation Tests for Phase 2C

Tests the complete Intelligence Flywheel:
1. Ingest topological signatures
2. Call /search endpoint 
3. Validate context retrieval
4. Measure p95 latency under 60ms SLA

This validates that the real search logic implementation works correctly.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from aura_intelligence.core.topology import TopologicalSignature
from aura_intelligence.enterprise.mem0_search.endpoints import create_search_router
from aura_intelligence.enterprise.mem0_search.schemas import (
    AnalyzeRequest, SearchRequest, TopologicalSignatureModel
)
from aura_intelligence.enterprise.mem0_hot import (
    HotEpisodicIngestor, DuckDBSettings, create_schema
)
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer


class TestPhase2CSearchValidation:
    """End-to-end validation of Phase 2C search functionality."""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Set up test environment with real DuckDB and mock services."""
        
        import duckdb
        
        # Create in-memory DuckDB for testing
        conn = duckdb.connect(":memory:")
        
        # Install VSS extension for vector similarity
        try:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
        except:
            pass  # Extension might already be loaded
        
        # Create schema
        schema_created = create_schema(conn, vector_dimension=128)
        assert schema_created, "Failed to create test schema"
        
        # Set up test settings
        test_settings = DuckDBSettings(
            memory_limit_gb=1,
            threads=2,
            retention_hours=1,
            vector_dimension=128
        )
        
        # Create hot memory ingestor
        vectorizer = SignatureVectorizer()
        hot_memory = HotEpisodicIngestor(conn, test_settings, vectorizer)
        
        # Mock semantic memory and ranking service
        semantic_memory = AsyncMock()
        ranking_service = AsyncMock()
        
        # Create FastAPI router
        router = create_search_router()
        
        return {
            "conn": conn,
            "hot_memory": hot_memory,
            "semantic_memory": semantic_memory,
            "ranking_service": ranking_service,
            "router": router,
            "vectorizer": vectorizer
        }
    
    def create_test_signature(self, 
                            betti_0: int = 5, 
                            betti_1: int = 3, 
                            betti_2: int = 1,
                            suffix: str = "") -> TopologicalSignature:
        """Create a test topological signature."""
        
        return TopologicalSignature(
            hash=f"test_signature_{betti_0}_{betti_1}_{betti_2}{suffix}",
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            persistence_diagram={
                "birth_death_pairs": [[0.0, 1.0], [0.5, 2.0]],
                "dimension": [0, 1]
            },
            algorithm="mojo_tda",
            computation_time_ms=15.5
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_pipeline(self, setup_test_environment):
        """
        Test complete search pipeline:
        1. Ingest signatures
        2. Search for similar signatures
        3. Validate results
        4. Measure latency
        """
        
        env = await setup_test_environment
        hot_memory = env["hot_memory"]
        vectorizer = env["vectorizer"]
        
        # Step 1: Ingest test signatures
        test_signatures = [
            self.create_test_signature(5, 3, 1, "_base"),
            self.create_test_signature(5, 3, 2, "_similar1"),  # Similar to base
            self.create_test_signature(5, 4, 1, "_similar2"),  # Similar to base
            self.create_test_signature(10, 8, 5, "_different"), # Different from base
        ]
        
        # Ingest all signatures
        for i, signature in enumerate(test_signatures):
            success = await hot_memory.ingest_signature(
                signature=signature,
                agent_id=f"test_agent_{i}",
                event_type="test_event",
                agent_meta={"test_id": i},
                full_event={"test_data": f"event_{i}"}
            )
            assert success, f"Failed to ingest signature {i}"
        
        # Force flush to ensure all data is written
        await hot_memory.force_flush()
        
        # Step 2: Test vector similarity search
        base_signature = test_signatures[0]
        query_vector = vectorizer.vectorize_signature(base_signature)
        
        # Import the sync search function
        from aura_intelligence.enterprise.mem0_search.endpoints import _search_hot_memory_sync
        
        # Step 3: Measure search latency
        latencies = []
        num_iterations = 10
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            # Execute search
            matches = _search_hot_memory_sync(
                conn=env["conn"],
                query_vector=query_vector.tolist(),
                threshold=0.5,
                max_results=5
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Step 4: Validate results
        assert len(matches) > 0, "No matches found"
        
        # Should find the base signature itself with highest similarity
        base_match = next((m for m in matches if m.signature.hash == base_signature.hash), None)
        assert base_match is not None, "Base signature not found in results"
        assert base_match.similarity_score > 0.95, f"Base signature similarity too low: {base_match.similarity_score}"
        
        # Should find similar signatures
        similar_matches = [m for m in matches if m.similarity_score > 0.7 and m.signature.hash != base_signature.hash]
        assert len(similar_matches) >= 1, "No similar signatures found"
        
        # Step 5: Validate SLA compliance
        p95_latency = statistics.quantile(latencies, 0.95)
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        print(f"\n📊 Search Performance Metrics:")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   P95 latency: {p95_latency:.2f}ms")
        print(f"   Max latency: {max_latency:.2f}ms")
        print(f"   Total matches found: {len(matches)}")
        
        # SLA validation (60ms p95 latency)
        assert p95_latency < 60.0, f"P95 latency {p95_latency:.2f}ms exceeds 60ms SLA"
        assert avg_latency < 30.0, f"Average latency {avg_latency:.2f}ms exceeds 30ms target"
        
        print("✅ All search validation tests passed!")
    
    @pytest.mark.asyncio
    async def test_search_api_endpoint_integration(self, setup_test_environment):
        """Test the FastAPI search endpoint integration."""
        
        env = await setup_test_environment
        
        # This would test the actual FastAPI endpoint
        # For now, we'll test the core search logic
        
        # Create test request
        test_signature = TopologicalSignatureModel(
            hash="test_api_signature",
            betti_numbers=[3, 2, 1],
            persistence_diagram={},
            algorithm="mojo_tda",
            computation_time_ms=10.0
        )
        
        # Test analyze endpoint logic
        from aura_intelligence.enterprise.mem0_search.endpoints import _find_similar_signatures
        
        # Convert to TopologicalSignature for search
        signature_obj = TopologicalSignature(
            hash=test_signature.hash,
            betti_0=test_signature.betti_numbers[0],
            betti_1=test_signature.betti_numbers[1],
            betti_2=test_signature.betti_numbers[2],
            persistence_diagram=test_signature.persistence_diagram,
            algorithm=test_signature.algorithm,
            computation_time_ms=test_signature.computation_time_ms
        )
        
        # Test similarity search
        matches = await _find_similar_signatures(
            signature=signature_obj,
            threshold=0.7,
            max_results=5,
            hot_memory=env["hot_memory"],
            semantic_memory=env["semantic_memory"]
        )
        
        # Should return empty list for new signature (no similar signatures ingested)
        assert isinstance(matches, list), "Matches should be a list"
        
        print("✅ API endpoint integration test passed!")
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, setup_test_environment):
        """Test search performance under concurrent load."""
        
        env = await setup_test_environment
        hot_memory = env["hot_memory"]
        vectorizer = env["vectorizer"]
        
        # Ingest multiple signatures for testing
        signatures = [self.create_test_signature(i, i+1, i+2, f"_concurrent_{i}") for i in range(20)]
        
        for signature in signatures:
            await hot_memory.ingest_signature(
                signature=signature,
                agent_id="concurrent_test_agent",
                event_type="concurrent_test",
                agent_meta={"concurrent": True},
                full_event={"test": "concurrent_load"}
            )
        
        await hot_memory.force_flush()
        
        # Test concurrent searches
        async def perform_search(signature_idx: int) -> float:
            """Perform a single search and return latency."""
            
            query_signature = signatures[signature_idx % len(signatures)]
            query_vector = vectorizer.vectorize_signature(query_signature)
            
            start_time = time.time()
            
            from aura_intelligence.enterprise.mem0_search.endpoints import _search_hot_memory_sync
            matches = _search_hot_memory_sync(
                conn=env["conn"],
                query_vector=query_vector.tolist(),
                threshold=0.5,
                max_results=10
            )
            
            latency = (time.time() - start_time) * 1000
            return latency
        
        # Run concurrent searches
        concurrent_tasks = [perform_search(i) for i in range(50)]
        latencies = await asyncio.gather(*concurrent_tasks)
        
        # Validate concurrent performance
        p95_latency = statistics.quantile(latencies, 0.95)
        avg_latency = statistics.mean(latencies)
        
        print(f"\n📊 Concurrent Search Performance:")
        print(f"   Concurrent requests: {len(latencies)}")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   P95 latency: {p95_latency:.2f}ms")
        
        # More lenient SLA for concurrent load
        assert p95_latency < 100.0, f"Concurrent P95 latency {p95_latency:.2f}ms exceeds 100ms"
        assert avg_latency < 50.0, f"Concurrent average latency {avg_latency:.2f}ms exceeds 50ms"
        
        print("✅ Concurrent search performance test passed!")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
