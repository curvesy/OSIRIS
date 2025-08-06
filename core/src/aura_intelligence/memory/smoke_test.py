#!/usr/bin/env python3
"""
Smoke Test for Shape Memory V2
==============================

Quick end-to-end test to verify the system is working correctly.
"""

import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import sys
import os
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Direct imports to avoid loading entire system
import sys
original_modules = set(sys.modules.keys())

# Import only what we need
from aura_intelligence.memory.shape_memory_v2_prod import ShapeMemoryV2, ShapeMemoryConfig
from aura_intelligence.memory.redis_store import RedisVectorStore, RedisConfig
from aura_intelligence.tda.models import TDAResult, BettiNumbers

# Clean up any extra imports
for module in list(sys.modules.keys()):
    if module not in original_modules and 'aura_intelligence' in module:
        if 'memory' not in module and 'tda' not in module:
            del sys.modules[module]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmokeTest:
    """Smoke test for Shape Memory V2."""
    
    def __init__(self, use_redis: bool = False, seed: int = 42):
        """Initialize smoke test with fixed seed for reproducibility."""
        self.use_redis = use_redis
        self.seed = seed
        np.random.seed(seed)  # Fixed seed for deterministic behavior
        
        self.config = ShapeMemoryConfig(
            storage_backend="redis" if use_redis else "memory",
            redis_url="redis://localhost:6379",
            enable_fusion_scoring=True
        )
        self.memory = ShapeMemoryV2(self.config)
        self.results = {
            "store_latencies": [],
            "retrieve_latencies": [],
            "recalls": [],
            "errors": []
        }
        
        # Track what we store for accurate recall calculation
        self.stored_items = {}  # memory_id -> (content, tda_result, context)
    
    def generate_deterministic_tda(self, item_id: str, category: str) -> TDAResult:
        """Generate deterministic TDA result based on item ID and category."""
        # Use hash of ID for deterministic but varied values
        hash_val = int(hashlib.md5(f"{item_id}_{category}".encode()).hexdigest()[:8], 16)
        
        # Use modulo to get deterministic values in range
        if category == "normal":
            b0 = 1 + (hash_val % 4)  # 1-4
            b1 = hash_val % 3        # 0-2
            b2 = hash_val % 2        # 0-1
        elif category == "anomaly":
            b0 = 5 + (hash_val % 5)  # 5-9
            b1 = 3 + (hash_val % 5)  # 3-7
            b2 = 2 + (hash_val % 3)  # 2-4
        elif category == "danger":
            b0 = 10 + (hash_val % 10) # 10-19
            b1 = 8 + (hash_val % 7)   # 8-14
            b2 = 5 + (hash_val % 5)   # 5-9
        else:
            raise ValueError(f"Unknown category: {category}")
        
        # Generate deterministic persistence diagram
        num_features = b0 + b1 + b2
        
        # Use deterministic generation based on hash
        rng = np.random.RandomState(hash_val)
        births = np.sort(rng.uniform(0, 0.5, num_features))
        lifetimes = rng.exponential(0.2, num_features)
        deaths = births + lifetimes
        persistence_diagram = np.column_stack([births, deaths])
        
        return TDAResult(
            betti_numbers=BettiNumbers(b0=b0, b1=b1, b2=b2),
            persistence_diagram=persistence_diagram,
            confidence=0.8 + (hash_val % 20) / 100.0  # 0.8-0.99
        )
    
    def generate_test_data(self, num_samples: int):
        """Generate test topological data with deterministic patterns."""
        data = []
        
        for i in range(num_samples):
            # Deterministic category assignment
            if i % 10 == 0:  # 10% anomalies
                context = "anomaly"
            elif i % 20 == 0:  # 5% danger (subset of anomalies)
                context = "danger"
            else:
                context = "normal"
            
            item_id = f"test_{i:06d}"  # Padded for consistent ordering
            
            # Generate deterministic TDA
            tda_result = self.generate_deterministic_tda(item_id, context)
            
            content = {
                "id": item_id,
                "timestamp": 1700000000 + i,  # Fixed base timestamp
                "category": context,  # Store category for verification
                "metrics": {
                    "value": float(i % 100) / 100.0  # Deterministic value
                }
            }
            
            data.append((content, tda_result, context))
        
        return data
    
    def test_store_performance(self, data):
        """Test store operation performance."""
        logger.info(f"Testing store performance with {len(data)} items...")
        
        for content, tda_result, context in data:
            start = time.perf_counter()
            try:
                memory_id = self.memory.store(content, tda_result, context)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                self.results["store_latencies"].append(elapsed)
                
                # Track what we stored
                if memory_id:
                    self.stored_items[memory_id] = (content, tda_result, context)
                    
            except Exception as e:
                self.results["errors"].append(f"Store error: {e}")
                logger.error(f"Store failed: {e}")
    
    def test_retrieve_performance(self, test_queries, expected_contexts):
        """Test retrieve operation performance with accurate recall calculation."""
        logger.info(f"Testing retrieve performance with {len(test_queries)} queries...")
        
        for i, (query_tda, expected_context) in enumerate(zip(test_queries, expected_contexts)):
            start = time.perf_counter()
            try:
                results = self.memory.retrieve(query_tda, k=5)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                self.results["retrieve_latencies"].append(elapsed)
                
                # Calculate accurate recall
                if results:
                    # Check if we found items with matching context
                    found_contexts = []
                    for memory_data, score in results:
                        # Get the actual stored context
                        memory_id = memory_data.get("id", "")
                        if memory_id in self.stored_items:
                            _, _, stored_context = self.stored_items[memory_id]
                            found_contexts.append(stored_context)
                        else:
                            # Fallback to metadata context
                            found_contexts.append(memory_data.get("context_type", ""))
                    
                    # Calculate recall@5
                    correct_matches = found_contexts.count(expected_context)
                    recall = correct_matches / min(5, len(found_contexts))
                    self.results["recalls"].append(recall)
                    
                    # Log for debugging
                    if recall < 1.0:
                        logger.debug(f"Query {i}: Expected '{expected_context}', found {found_contexts}")
                
            except Exception as e:
                self.results["errors"].append(f"Retrieve error: {e}")
                logger.error(f"Retrieve failed: {e}")
    
    def test_concurrent_load(self, num_concurrent: int = 10):
        """Test concurrent operations."""
        logger.info(f"Testing concurrent load with {num_concurrent} threads...")
        
        # Generate test data with fixed seed for this test
        rng = np.random.RandomState(self.seed + 1000)
        data = self.generate_test_data(100)
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit concurrent queries
            futures = []
            
            for i in range(num_concurrent):
                # Use deterministic selection
                idx = i % len(data)
                _, tda_result, _ = data[idx]
                future = executor.submit(self.memory.retrieve, tda_result, 5)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    self.results["errors"].append(f"Concurrent error: {e}")
    
    def run_smoke_test(self, num_samples: int = 1000, num_queries: int = 100):
        """Run the complete smoke test."""
        logger.info("Starting Shape Memory V2 smoke test...")
        logger.info(f"Using seed: {self.seed} for reproducibility")
        
        # Generate test data
        data = self.generate_test_data(num_samples)
        
        # Test 1: Store performance
        self.test_store_performance(data)
        
        # Test 2: Retrieve performance
        # Use deterministic query selection
        rng = np.random.RandomState(self.seed)
        query_indices = rng.choice(len(data), num_queries, replace=True)
        test_queries = [data[i][1] for i in query_indices]
        expected_contexts = [data[i][2] for i in query_indices]
        self.test_retrieve_performance(test_queries, expected_contexts)
        
        # Test 3: Concurrent load
        self.test_concurrent_load()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate smoke test report."""
        print("\n" + "="*60)
        print("SHAPE MEMORY V2 SMOKE TEST REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Backend: {self.config.storage_backend}")
        print(f"  Fusion Scoring: {self.config.enable_fusion_scoring}")
        print(f"  Random Seed: {self.seed}")
        
        if self.results["store_latencies"]:
            print(f"\nStore Performance:")
            print(f"  Count: {len(self.results['store_latencies'])}")
            print(f"  P50: {statistics.median(self.results['store_latencies']):.2f} ms")
            print(f"  P95: {np.percentile(self.results['store_latencies'], 95):.2f} ms")
            print(f"  P99: {np.percentile(self.results['store_latencies'], 99):.2f} ms")
        
        if self.results["retrieve_latencies"]:
            print(f"\nRetrieve Performance:")
            print(f"  Count: {len(self.results['retrieve_latencies'])}")
            print(f"  P50: {statistics.median(self.results['retrieve_latencies']):.2f} ms")
            print(f"  P95: {np.percentile(self.results['retrieve_latencies'], 95):.2f} ms")
            print(f"  P99: {np.percentile(self.results['retrieve_latencies'], 99):.2f} ms")
        
        if self.results["recalls"]:
            print(f"\nAccuracy:")
            print(f"  Mean Recall@5: {np.mean(self.results['recalls']):.3f}")
            print(f"  Min Recall@5: {np.min(self.results['recalls']):.3f}")
            print(f"  Max Recall@5: {np.max(self.results['recalls']):.3f}")
        
        print(f"\nErrors: {len(self.results['errors'])}")
        if self.results["errors"]:
            print("  First 5 errors:")
            for error in self.results["errors"][:5]:
                print(f"    - {error}")
        
        # Check against requirements
        print(f"\nProduction Readiness:")
        
        p99_latency = np.percentile(self.results['retrieve_latencies'], 99) if self.results['retrieve_latencies'] else float('inf')
        recall = np.mean(self.results['recalls']) if self.results['recalls'] else 0
        
        if p99_latency < 5.0:
            print("  PASS: P99 Latency < 5ms")
        else:
            print(f"  FAIL: P99 Latency {p99_latency:.2f}ms > 5ms target")
        
        if recall > 0.95:
            print("  PASS: Recall@5 > 0.95")
        else:
            print(f"  FAIL: Recall@5 {recall:.3f} < 0.95 target")
        
        if len(self.results["errors"]) == 0:
            print("  PASS: No errors")
        else:
            print(f"  FAIL: {len(self.results['errors'])} errors occurred")
        
        print("="*60)


def main():
    """Run smoke test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shape Memory V2 Smoke Test")
    parser.add_argument("--redis", action="store_true", help="Use Redis backend")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to store")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    test = SmokeTest(use_redis=args.redis, seed=args.seed)
    
    try:
        test.run_smoke_test(num_samples=args.samples, num_queries=args.queries)
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        sys.exit(1)
    
    # Check if we met requirements
    if test.results["retrieve_latencies"]:
        p99 = np.percentile(test.results["retrieve_latencies"], 99)
        recall = np.mean(test.results["recalls"]) if test.results["recalls"] else 0
        
        if p99 > 5.0 or recall < 0.95 or len(test.results["errors"]) > 0:
            sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()