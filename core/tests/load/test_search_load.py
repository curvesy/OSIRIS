"""
🚀 Load Testing Suite - Production-Grade Performance Validation

Realistic load testing for the Intelligence Flywheel search system:
- Simulates real agent search patterns (80% hot, 20% semantic)
- Validates latency SLAs under concurrent load
- Tests system behavior under stress conditions
- Measures throughput and resource utilization
"""

import os
import time
import json
import random
from typing import List, Dict, Any
import numpy as np

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
from loguru import logger


class SearchLoadTest(HttpUser):
    """
    🔍 Search Load Test User
    
    Simulates realistic agent search behavior with:
    - 80% hot tier searches (recent data, fast queries)
    - 20% semantic tier searches (pattern matching, complex queries)
    - Realistic wait times between requests
    - Proper error handling and metrics collection
    """
    
    # Realistic wait time between requests (agent thinking time)
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize test data when user starts."""
        
        # Generate realistic test vectors for consistent testing
        self.test_vectors = self._generate_test_vectors(100)
        self.search_queries = self._generate_search_queries(50)
        
        # Track user metrics
        self.request_count = 0
        self.error_count = 0
        self.latency_samples = []
        
        logger.info(f"🚀 Load test user started with {len(self.test_vectors)} test vectors")
    
    def _generate_test_vectors(self, count: int) -> List[List[float]]:
        """Generate realistic topological signature vectors."""
        
        vectors = []
        for i in range(count):
            # Generate vector with some structure (not purely random)
            base_vector = np.random.rand(768).astype(np.float32)
            
            # Add some clustering structure
            if i % 10 == 0:
                # Create cluster centers
                base_vector *= 2.0
            else:
                # Add noise around cluster centers
                cluster_center = i // 10
                noise = np.random.normal(0, 0.1, 768).astype(np.float32)
                base_vector = self.test_vectors[cluster_center * 10] + noise if cluster_center * 10 < len(vectors) else base_vector
            
            # Normalize
            base_vector = base_vector / np.linalg.norm(base_vector)
            vectors.append(base_vector.tolist())
        
        return vectors
    
    def _generate_search_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic search query patterns."""
        
        queries = []
        query_types = [
            {"tier": "hot", "limit": 10, "threshold": 0.8},
            {"tier": "semantic", "limit": 20, "threshold": 0.7},
            {"tier": "auto", "limit": 15, "threshold": 0.75},
        ]
        
        for i in range(count):
            query_type = random.choice(query_types)
            queries.append({
                "query_id": f"load_test_{i}",
                "vector": random.choice(self.test_vectors),
                **query_type
            })
        
        return queries
    
    @task(80)
    def search_hot_tier(self):
        """
        🔥 Hot tier search (80% of traffic)
        
        Simulates agent searching recent activity data.
        Expected: <60ms P95 latency, high success rate
        """
        
        vector = random.choice(self.test_vectors)
        
        start_time = time.time()
        
        with self.client.post(
            "/search",
            json={
                "query": f"hot_search_{self.request_count}",
                "tier": "hot",
                "limit": 10,
                "threshold": 0.8,
                "agent_id": f"load_test_agent_{self.user_id}",
                "context": {
                    "search_type": "hot_tier",
                    "load_test": True
                }
            },
            catch_response=True,
            name="search_hot_tier"
        ) as response:
            
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Validate response structure
                    assert "results" in result
                    assert "search_time_ms" in result
                    assert "tier_used" in result
                    
                    # Validate performance SLA
                    if latency_ms > 60:  # 60ms SLA
                        response.failure(f"Hot tier search too slow: {latency_ms:.1f}ms > 60ms SLA")
                    
                    # Validate result quality
                    if len(result["results"]) == 0:
                        response.failure("No results returned for hot tier search")
                    
                    self.request_count += 1
                    
                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    response.failure(f"Invalid response format: {e}")
                    self.error_count += 1
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                self.error_count += 1
    
    @task(20)
    def search_semantic_tier(self):
        """
        🧠 Semantic tier search (20% of traffic)
        
        Simulates agent searching for semantic patterns.
        Expected: <200ms P95 latency, good recall
        """
        
        vector = random.choice(self.test_vectors)
        
        start_time = time.time()
        
        with self.client.post(
            "/search",
            json={
                "query": f"semantic_search_{self.request_count}",
                "tier": "semantic",
                "limit": 20,
                "threshold": 0.7,
                "agent_id": f"load_test_agent_{self.user_id}",
                "context": {
                    "search_type": "semantic_tier",
                    "load_test": True
                }
            },
            catch_response=True,
            name="search_semantic_tier"
        ) as response:
            
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Validate response structure
                    assert "results" in result
                    assert "search_time_ms" in result
                    assert "tier_used" in result
                    
                    # Validate performance SLA (more lenient for semantic search)
                    if latency_ms > 200:  # 200ms SLA for semantic
                        response.failure(f"Semantic search too slow: {latency_ms:.1f}ms > 200ms SLA")
                    
                    # Validate semantic search worked
                    if result["tier_used"] != "semantic":
                        response.failure(f"Expected semantic tier, got {result['tier_used']}")
                    
                    self.request_count += 1
                    
                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    response.failure(f"Invalid response format: {e}")
                    self.error_count += 1
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                self.error_count += 1
    
    @task(5)
    def search_hybrid(self):
        """
        🔄 Hybrid search (5% of traffic)
        
        Simulates complex agent queries that search across multiple tiers.
        Expected: <300ms P95 latency, comprehensive results
        """
        
        query = random.choice(self.search_queries)
        
        start_time = time.time()
        
        with self.client.post(
            "/search",
            json={
                "query": f"hybrid_search_{self.request_count}",
                "tier": "auto",
                "limit": query["limit"],
                "threshold": query["threshold"],
                "agent_id": f"load_test_agent_{self.user_id}",
                "context": {
                    "search_type": "hybrid",
                    "load_test": True,
                    "complexity": "high"
                }
            },
            catch_response=True,
            name="search_hybrid"
        ) as response:
            
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Validate response structure
                    assert "results" in result
                    assert "search_time_ms" in result
                    
                    # Validate performance SLA (most lenient for hybrid)
                    if latency_ms > 300:  # 300ms SLA for hybrid
                        response.failure(f"Hybrid search too slow: {latency_ms:.1f}ms > 300ms SLA")
                    
                    # Validate hybrid search provides good results
                    if len(result["results"]) < query["limit"] // 2:
                        response.failure(f"Hybrid search returned too few results: {len(result['results'])}")
                    
                    self.request_count += 1
                    
                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    response.failure(f"Invalid response format: {e}")
                    self.error_count += 1
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                self.error_count += 1
    
    @task(1)
    def health_check(self):
        """
        ❤️ Health check (1% of traffic)
        
        Monitors system health during load testing.
        """
        
        with self.client.get(
            "/health",
            catch_response=True,
            name="health_check"
        ) as response:
            
            if response.status_code == 200:
                try:
                    health = response.json()
                    
                    # Check system health
                    if health.get("status") != "healthy":
                        response.failure(f"System unhealthy: {health.get('status')}")
                    
                    # Check component health
                    components = health.get("components", {})
                    unhealthy_components = [
                        comp for comp, status in components.items() 
                        if status not in ["healthy", "not_initialized"]
                    ]
                    
                    if unhealthy_components:
                        response.failure(f"Unhealthy components: {unhealthy_components}")
                
                except (json.JSONDecodeError, KeyError) as e:
                    response.failure(f"Invalid health response: {e}")
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")
    
    def on_stop(self):
        """Cleanup and report metrics when user stops."""
        
        if self.latency_samples:
            p50 = np.percentile(self.latency_samples, 50)
            p95 = np.percentile(self.latency_samples, 95)
            p99 = np.percentile(self.latency_samples, 99)
            
            logger.info(f"🏁 User {self.user_id} finished:")
            logger.info(f"   Requests: {self.request_count}, Errors: {self.error_count}")
            logger.info(f"   Latency P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")
            logger.info(f"   Error Rate: {self.error_count/max(self.request_count, 1)*100:.1f}%")


class StressTestUser(SearchLoadTest):
    """
    💥 Stress Test User
    
    More aggressive testing to find system breaking points:
    - Higher request rates
    - Larger payloads
    - Edge case scenarios
    """
    
    # More aggressive wait times for stress testing
    wait_time = between(0.01, 0.1)
    
    @task(50)
    def stress_search_burst(self):
        """Send burst of search requests to test system limits."""
        
        # Send multiple requests in quick succession
        for i in range(5):
            vector = random.choice(self.test_vectors)
            
            self.client.post(
                "/search",
                json={
                    "query": f"stress_burst_{i}",
                    "tier": "hot",
                    "limit": 50,  # Larger limit
                    "threshold": 0.5,  # Lower threshold
                    "agent_id": f"stress_agent_{self.user_id}",
                    "context": {"stress_test": True, "burst_id": i}
                },
                name="stress_search_burst"
            )
            
            # Very short wait between burst requests
            time.sleep(0.01)


# ============================================================================
# Event Handlers for Metrics Collection
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Collect detailed metrics for each request."""
    
    # Log slow requests
    if response_time > 1000:  # > 1 second
        logger.warning(f"🐌 Slow request: {name} took {response_time:.0f}ms")
    
    # Log errors
    if exception:
        logger.error(f"❌ Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment and logging."""
    
    logger.info("🚀 Load test starting...")
    logger.info(f"   Target host: {environment.host}")
    logger.info(f"   Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'Unknown'}")
    
    # Create results directory
    os.makedirs("test-results", exist_ok=True)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate final test report."""
    
    stats = environment.stats
    
    # Calculate overall metrics
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    error_rate = (total_failures / max(total_requests, 1)) * 100
    
    # Generate summary report
    report = {
        "test_summary": {
            "total_requests": total_requests,
            "total_failures": total_failures,
            "error_rate_percent": error_rate,
            "average_response_time": stats.total.avg_response_time,
            "min_response_time": stats.total.min_response_time,
            "max_response_time": stats.total.max_response_time,
            "requests_per_second": stats.total.current_rps,
            "test_duration": time.time() - environment.runner.start_time
        },
        "endpoint_breakdown": {}
    }
    
    # Add per-endpoint metrics
    for name, entry in stats.entries.items():
        if entry.num_requests > 0:
            report["endpoint_breakdown"][name] = {
                "requests": entry.num_requests,
                "failures": entry.num_failures,
                "error_rate_percent": (entry.num_failures / entry.num_requests) * 100,
                "avg_response_time": entry.avg_response_time,
                "min_response_time": entry.min_response_time,
                "max_response_time": entry.max_response_time,
                "requests_per_second": entry.current_rps
            }
    
    # Save report
    report_file = f"test-results/load_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("🏁 Load test completed!")
    logger.info(f"   Total Requests: {total_requests}")
    logger.info(f"   Error Rate: {error_rate:.1f}%")
    logger.info(f"   Avg Response Time: {stats.total.avg_response_time:.1f}ms")
    logger.info(f"   RPS: {stats.total.current_rps:.1f}")
    logger.info(f"   Report saved: {report_file}")


if __name__ == "__main__":
    # Run load test directly
    import subprocess
    import sys
    
    # Default load test parameters
    users = int(os.getenv("LOAD_TEST_USERS", "50"))
    spawn_rate = int(os.getenv("LOAD_TEST_SPAWN_RATE", "5"))
    duration = os.getenv("LOAD_TEST_DURATION", "2m")
    host = os.getenv("LOAD_TEST_HOST", "http://localhost:8000")
    
    cmd = [
        "locust",
        "-f", __file__,
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", duration,
        "--headless",
        "--html", "test-results/load_test_report.html"
    ]
    
    logger.info(f"🚀 Running load test: {' '.join(cmd)}")
    subprocess.run(cmd)
