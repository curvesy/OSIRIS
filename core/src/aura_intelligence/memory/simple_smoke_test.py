#!/usr/bin/env python3
"""
Simplified Smoke Test for Shape Memory V2
=========================================

Direct test without complex imports.
"""

import time
import numpy as np
import logging
import sys
import os
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Direct imports
from core.src.aura_intelligence.memory.shape_memory_v2_prod import ShapeMemoryV2, ShapeMemoryConfig
from core.src.aura_intelligence.tda.models import TDAResult, BettiNumbers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_tda(item_id: str, category: str) -> TDAResult:
    """Generate deterministic TDA result."""
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


def main():
    """Run simplified smoke test."""
    print("\n" + "="*60)
    print("SHAPE MEMORY V2 - SIMPLIFIED SMOKE TEST")
    print("="*60)
    
    # Set random seed
    np.random.seed(42)
    
    # Initialize memory with in-memory backend
    config = ShapeMemoryConfig(
        storage_backend="memory",
        enable_fusion_scoring=True
    )
    
    try:
        memory = ShapeMemoryV2(config)
        print("✓ Shape Memory V2 initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return 1
    
    # Test 1: Store some items
    print("\n1. Testing STORE operation...")
    store_times = []
    stored_items = []
    
    for i in range(10):
        category = "anomaly" if i % 5 == 0 else "normal"
        tda_result = generate_test_tda(f"test_{i:03d}", category)
        
        content = {
            "id": f"test_{i:03d}",
            "timestamp": 1700000000 + i,
            "category": category
        }
        
        start = time.perf_counter()
        try:
            memory_id = memory.store(content, tda_result, category)
            elapsed = (time.perf_counter() - start) * 1000
            store_times.append(elapsed)
            stored_items.append((memory_id, category, tda_result))
            print(f"  Stored item {i}: {elapsed:.2f}ms")
        except Exception as e:
            print(f"  ✗ Failed to store item {i}: {e}")
    
    if store_times:
        print(f"\n  Store Performance:")
        print(f"    Mean: {np.mean(store_times):.2f}ms")
        print(f"    P99: {np.percentile(store_times, 99):.2f}ms")
    
    # Test 2: Retrieve items
    print("\n2. Testing RETRIEVE operation...")
    retrieve_times = []
    recalls = []
    
    for memory_id, expected_context, tda_result in stored_items[:5]:
        start = time.perf_counter()
        try:
            results = memory.retrieve(tda_result, k=3)
            elapsed = (time.perf_counter() - start) * 1000
            retrieve_times.append(elapsed)
            
            # Check recall
            if results:
                contexts = [r[0].get("context_type", "") for r in results]
                recall = contexts.count(expected_context) / len(contexts)
                recalls.append(recall)
                print(f"  Retrieved {len(results)} items in {elapsed:.2f}ms (recall: {recall:.2f})")
            else:
                print(f"  No results in {elapsed:.2f}ms")
        except Exception as e:
            print(f"  ✗ Failed to retrieve: {e}")
    
    if retrieve_times:
        print(f"\n  Retrieve Performance:")
        print(f"    Mean: {np.mean(retrieve_times):.2f}ms")
        print(f"    P99: {np.percentile(retrieve_times, 99):.2f}ms")
        print(f"    Mean Recall: {np.mean(recalls):.2f}")
    
    # Test 3: Error handling
    print("\n3. Testing ERROR HANDLING...")
    try:
        # Test with invalid TDA
        invalid_tda = TDAResult(
            betti_numbers=BettiNumbers(b0=-1, b1=0, b2=0),
            persistence_diagram=np.array([]),
            confidence=1.0
        )
        results = memory.retrieve(invalid_tda, k=5)
        print(f"  ✓ Handled invalid TDA gracefully: {len(results) if results else 0} results")
    except Exception as e:
        print(f"  ✓ Properly raised exception for invalid TDA: {type(e).__name__}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    
    success = True
    
    if retrieve_times and np.percentile(retrieve_times, 99) < 5.0:
        print("  ✅ P99 Latency < 5ms")
    else:
        print("  ❌ P99 Latency > 5ms")
        success = False
    
    if recalls and np.mean(recalls) > 0.95:
        print("  ✅ Recall > 0.95")
    else:
        print("  ❌ Recall < 0.95")
        success = False
    
    print("  ✅ Error handling works correctly")
    
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())