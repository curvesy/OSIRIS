#!/usr/bin/env python3
"""
Test MetricsUpdater thread management
"""

import threading
import time
import gc
from unittest.mock import Mock, MagicMock

# Mock the dependencies
class MockRedisStore:
    def __init__(self):
        self.redis = MagicMock()
        self.redis.ft.return_value.info.return_value = MagicMock(num_docs=100)

def test_metrics_updater():
    """Test that MetricsUpdater uses a single thread and stops cleanly."""
    
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from redis_store import MetricsUpdater
    
    print("Testing MetricsUpdater thread management...")
    print("="*50)
    
    # Get initial thread count
    initial_threads = threading.active_count()
    print(f"Initial thread count: {initial_threads}")
    
    # Create mock store
    mock_store = MockRedisStore()
    
    # Test 1: Single thread creation
    print("\nTest 1: Creating MetricsUpdater...")
    updater = MetricsUpdater(mock_store, interval=1)
    time.sleep(0.1)  # Let thread start
    
    threads_after_create = threading.active_count()
    print(f"Threads after creation: {threads_after_create}")
    print(f"Thread created: {threads_after_create - initial_threads == 1}")
    
    # List all threads
    print("\nActive threads:")
    for thread in threading.enumerate():
        print(f"  - {thread.name} (daemon: {thread.daemon})")
    
    # Test 2: Thread runs multiple updates
    print("\nTest 2: Letting updater run for 3 seconds...")
    time.sleep(3.1)  # Should trigger 3 updates
    
    # Check thread count hasn't grown
    threads_after_run = threading.active_count()
    print(f"Threads after 3s: {threads_after_run}")
    print(f"No thread leak: {threads_after_run == threads_after_create}")
    
    # Test 3: Clean shutdown
    print("\nTest 3: Stopping updater...")
    updater.stop()
    time.sleep(0.5)  # Give thread time to stop
    
    threads_after_stop = threading.active_count()
    print(f"Threads after stop: {threads_after_stop}")
    print(f"Thread stopped: {threads_after_stop == initial_threads}")
    
    # Test 4: Multiple instances don't leak
    print("\nTest 4: Creating and stopping multiple updaters...")
    updaters = []
    for i in range(5):
        updater = MetricsUpdater(mock_store, interval=1)
        updaters.append(updater)
        time.sleep(0.1)
    
    threads_with_multiple = threading.active_count()
    print(f"Threads with 5 updaters: {threads_with_multiple}")
    print(f"Expected threads: {initial_threads + 5}")
    
    # Stop all
    for updater in updaters:
        updater.stop()
    time.sleep(0.5)
    
    threads_after_stop_all = threading.active_count()
    print(f"Threads after stopping all: {threads_after_stop_all}")
    print(f"All threads cleaned up: {threads_after_stop_all == initial_threads}")
    
    # Force garbage collection
    updaters.clear()
    gc.collect()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    if threads_after_stop_all == initial_threads:
        print("✅ MetricsUpdater properly manages threads - no leaks!")
    else:
        print("❌ Thread leak detected!")
        print(f"   Expected: {initial_threads} threads")
        print(f"   Actual: {threads_after_stop_all} threads")

if __name__ == "__main__":
    # Mock the update_vector_count function
    import sys
    sys.modules['observability'] = MagicMock()
    
    test_metrics_updater()