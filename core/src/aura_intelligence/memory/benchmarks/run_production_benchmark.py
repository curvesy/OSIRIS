#!/usr/bin/env python3
"""
Production Benchmark Runner for Shape Memory V2
==============================================

Runs comprehensive benchmarks with realistic topological data
to validate production readiness.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from aura_intelligence.memory.config import get_benchmark_settings, ShapeMemorySettings
from aura_intelligence.memory.redis_store import RedisVectorStore, RedisConfig
from aura_intelligence.memory.shape_memory_v2_prod import ShapeMemoryV2, ShapeMemoryConfig
from aura_intelligence.memory.async_shape_memory import AsyncShapeMemoryV2, AsyncRedisAdapter
from aura_intelligence.memory.observability import PROMETHEUS_REGISTRY
from aura_intelligence.tda.models import TDAResult, BettiNumbers
from prometheus_client import generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopologyDataGenerator:
    """Generate realistic topological data for benchmarking."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_tda_result(self, category: str = "normal") -> TDAResult:
        """Generate a realistic TDA result based on category."""
        if category == "normal":
            # Normal patterns have lower Betti numbers
            b0 = np.random.randint(1, 5)
            b1 = np.random.randint(0, 3)
            b2 = np.random.randint(0, 2)
            num_features = b0 + b1 + b2 + np.random.randint(0, 5)
        elif category == "anomaly":
            # Anomalies have higher Betti numbers
            b0 = np.random.randint(5, 10)
            b1 = np.random.randint(3, 8)
            b2 = np.random.randint(2, 5)
            num_features = b0 + b1 + b2 + np.random.randint(5, 10)
        elif category == "danger":
            # Danger patterns have very high complexity
            b0 = np.random.randint(10, 20)
            b1 = np.random.randint(8, 15)
            b2 = np.random.randint(5, 10)
            num_features = b0 + b1 + b2 + np.random.randint(10, 20)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        # Generate persistence diagram
        births = np.random.uniform(0, 0.5, num_features)
        lifetimes = np.random.exponential(0.2, num_features)
        deaths = births + lifetimes
        persistence_diagram = np.column_stack([births, deaths])
        
        return TDAResult(
            betti_numbers=BettiNumbers(b0=b0, b1=b1, b2=b2),
            persistence_diagram=persistence_diagram,
            confidence=np.random.uniform(0.8, 1.0)
        )
    
    def generate_dataset(
        self,
        num_normal: int,
        num_anomaly: int,
        num_danger: int
    ) -> List[Tuple[Dict[str, Any], TDAResult, str]]:
        """Generate a full dataset for benchmarking."""
        dataset = []
        
        # Generate normal patterns
        for i in range(num_normal):
            content = {
                "id": f"normal_{i}",
                "type": "system_state",
                "timestamp": time.time() - np.random.uniform(0, 86400),
                "metrics": {
                    "cpu": np.random.uniform(20, 60),
                    "memory": np.random.uniform(30, 70),
                    "network": np.random.uniform(10, 50)
                }
            }
            tda = self.generate_tda_result("normal")
            dataset.append((content, tda, "normal"))
        
        # Generate anomalies
        for i in range(num_anomaly):
            content = {
                "id": f"anomaly_{i}",
                "type": "system_anomaly",
                "timestamp": time.time() - np.random.uniform(0, 3600),
                "metrics": {
                    "cpu": np.random.uniform(70, 90),
                    "memory": np.random.uniform(80, 95),
                    "network": np.random.uniform(60, 90)
                }
            }
            tda = self.generate_tda_result("anomaly")
            dataset.append((content, tda, "anomaly"))
        
        # Generate danger patterns
        for i in range(num_danger):
            content = {
                "id": f"danger_{i}",
                "type": "critical_failure",
                "timestamp": time.time() - np.random.uniform(0, 300),
                "metrics": {
                    "cpu": np.random.uniform(90, 100),
                    "memory": np.random.uniform(95, 100),
                    "network": np.random.uniform(90, 100)
                }
            }
            tda = self.generate_tda_result("danger")
            dataset.append((content, tda, "danger"))
        
        return dataset


class BenchmarkRunner:
    """Run comprehensive benchmarks."""
    
    def __init__(self):
        self.settings = get_benchmark_settings()
        self.results = {}
        
    async def run_sync_benchmark(
        self,
        memory: ShapeMemoryV2,
        dataset: List[Tuple[Dict[str, Any], TDAResult, str]],
        num_queries: int
    ) -> Dict[str, Any]:
        """Benchmark synchronous implementation."""
        logger.info("Running synchronous benchmark...")
        
        # Store phase
        store_latencies = []
        store_start = time.time()
        
        for content, tda, context in dataset:
            start = time.perf_counter()
            memory_id = memory.store(content, tda, context)
            elapsed = time.perf_counter() - start
            store_latencies.append(elapsed)
            
            if len(store_latencies) % 1000 == 0:
                logger.info(f"Stored {len(store_latencies)} memories...")
        
        store_duration = time.time() - store_start
        store_throughput = len(dataset) / store_duration
        
        # Query phase
        query_latencies = []
        query_recalls = []
        query_start = time.time()
        
        # Select random queries
        query_indices = np.random.choice(len(dataset), num_queries, replace=True)
        
        for idx in query_indices:
            _, query_tda, expected_context = dataset[idx]
            
            start = time.perf_counter()
            results = memory.retrieve(query_tda, k=10)
            elapsed = time.perf_counter() - start
            query_latencies.append(elapsed)
            
            # Calculate recall
            contexts = [r[0].get("context_type") for r in results[:5]]
            recall = contexts.count(expected_context) / 5.0
            query_recalls.append(recall)
        
        query_duration = time.time() - query_start
        query_throughput = num_queries / query_duration
        
        # Calculate metrics
        store_latencies = np.array(store_latencies) * 1000  # Convert to ms
        query_latencies = np.array(query_latencies) * 1000
        
        return {
            "type": "synchronous",
            "store": {
                "count": len(dataset),
                "duration_s": store_duration,
                "throughput_ops": store_throughput,
                "latency_p50_ms": np.percentile(store_latencies, 50),
                "latency_p95_ms": np.percentile(store_latencies, 95),
                "latency_p99_ms": np.percentile(store_latencies, 99)
            },
            "query": {
                "count": num_queries,
                "duration_s": query_duration,
                "throughput_qps": query_throughput,
                "latency_p50_ms": np.percentile(query_latencies, 50),
                "latency_p95_ms": np.percentile(query_latencies, 95),
                "latency_p99_ms": np.percentile(query_latencies, 99),
                "recall_at_5": np.mean(query_recalls)
            }
        }
    
    async def run_async_benchmark(
        self,
        memory: AsyncShapeMemoryV2,
        dataset: List[Tuple[Dict[str, Any], TDAResult, str]],
        num_queries: int
    ) -> Dict[str, Any]:
        """Benchmark asynchronous implementation."""
        logger.info("Running asynchronous benchmark...")
        
        # Store phase - batch processing
        store_start = time.time()
        batch_size = 100
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            items = [(c, t, ctx, None) for c, t, ctx in batch]
            await memory.batch_store(items)
            
            if i % 1000 == 0:
                logger.info(f"Stored {i} memories asynchronously...")
        
        store_duration = time.time() - store_start
        store_throughput = len(dataset) / store_duration
        
        # Query phase - concurrent queries
        query_latencies = []
        query_recalls = []
        query_start = time.time()
        
        # Run queries in batches
        query_indices = np.random.choice(len(dataset), num_queries, replace=True)
        query_batch_size = 50
        
        for i in range(0, num_queries, query_batch_size):
            batch_indices = query_indices[i:i+query_batch_size]
            
            # Create query tasks
            tasks = []
            for idx in batch_indices:
                _, query_tda, expected_context = dataset[idx]
                
                async def query_task(tda, context):
                    start = time.perf_counter()
                    results = await memory.retrieve(tda, k=10)
                    elapsed = time.perf_counter() - start
                    
                    contexts = [r[0].get("context_type") for r in results[:5]]
                    recall = contexts.count(context) / 5.0
                    
                    return elapsed, recall
                
                tasks.append(query_task(query_tda, expected_context))
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            
            for elapsed, recall in batch_results:
                query_latencies.append(elapsed)
                query_recalls.append(recall)
        
        query_duration = time.time() - query_start
        query_throughput = num_queries / query_duration
        
        # Calculate metrics
        query_latencies = np.array(query_latencies) * 1000  # Convert to ms
        
        return {
            "type": "asynchronous",
            "store": {
                "count": len(dataset),
                "duration_s": store_duration,
                "throughput_ops": store_throughput,
                "batch_size": batch_size
            },
            "query": {
                "count": num_queries,
                "duration_s": query_duration,
                "throughput_qps": query_throughput,
                "latency_p50_ms": np.percentile(query_latencies, 50),
                "latency_p95_ms": np.percentile(query_latencies, 95),
                "latency_p99_ms": np.percentile(query_latencies, 99),
                "recall_at_5": np.mean(query_recalls),
                "batch_size": query_batch_size
            }
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().isoformat()
        
        report = f"""# Shape Memory V2 Production Benchmark Report

Generated: {timestamp}

## Configuration
- Vectors: {self.settings.num_vectors:,}
- Queries: {self.settings.num_queries:,}
- Dimensions: {self.settings.vector_dim}

## Results Summary

### Synchronous Implementation
"""
        
        if "sync" in results:
            sync = results["sync"]
            report += f"""
**Store Performance:**
- Throughput: {sync['store']['throughput_ops']:.1f} ops/sec
- P50 Latency: {sync['store']['latency_p50_ms']:.2f} ms
- P95 Latency: {sync['store']['latency_p95_ms']:.2f} ms
- P99 Latency: {sync['store']['latency_p99_ms']:.2f} ms

**Query Performance:**
- Throughput: {sync['query']['throughput_qps']:.1f} qps
- P50 Latency: {sync['query']['latency_p50_ms']:.2f} ms
- P95 Latency: {sync['query']['latency_p95_ms']:.2f} ms
- P99 Latency: {sync['query']['latency_p99_ms']:.2f} ms
- Recall@5: {sync['query']['recall_at_5']:.3f}
"""
        
        report += "\n### Asynchronous Implementation\n"
        
        if "async" in results:
            async_res = results["async"]
            report += f"""
**Store Performance:**
- Throughput: {async_res['store']['throughput_ops']:.1f} ops/sec
- Batch Size: {async_res['store']['batch_size']}
- Speedup: {async_res['store']['throughput_ops'] / sync['store']['throughput_ops']:.1f}x

**Query Performance:**
- Throughput: {async_res['query']['throughput_qps']:.1f} qps
- P50 Latency: {async_res['query']['latency_p50_ms']:.2f} ms
- P95 Latency: {async_res['query']['latency_p95_ms']:.2f} ms
- P99 Latency: {async_res['query']['latency_p99_ms']:.2f} ms
- Recall@5: {async_res['query']['recall_at_5']:.3f}
- Speedup: {async_res['query']['throughput_qps'] / sync['query']['throughput_qps']:.1f}x
"""
        
        # Production readiness assessment
        report += "\n## Production Readiness Assessment\n\n"
        
        p99_target = 5.0  # 5ms target
        recall_target = 0.95
        
        sync_ready = (
            sync['query']['latency_p99_ms'] <= p99_target and
            sync['query']['recall_at_5'] >= recall_target
        )
        
        async_ready = (
            async_res['query']['latency_p99_ms'] <= p99_target and
            async_res['query']['recall_at_5'] >= recall_target
        )
        
        if sync_ready:
            report += "✅ **Synchronous implementation meets production requirements**\n"
        else:
            report += "❌ **Synchronous implementation does not meet requirements**\n"
            
        if async_ready:
            report += "✅ **Asynchronous implementation meets production requirements**\n"
        else:
            report += "❌ **Asynchronous implementation does not meet requirements**\n"
        
        # Prometheus metrics
        report += "\n## Prometheus Metrics\n\n```\n"
        report += generate_latest(PROMETHEUS_REGISTRY).decode('utf-8')
        report += "```\n"
        
        return report
    
    def plot_results(self, results: Dict[str, Any], output_dir: Path):
        """Generate visualization plots."""
        # Latency comparison plot
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        categories = ['P50', 'P95', 'P99']
        sync_latencies = [
            results['sync']['query']['latency_p50_ms'],
            results['sync']['query']['latency_p95_ms'],
            results['sync']['query']['latency_p99_ms']
        ]
        async_latencies = [
            results['async']['query']['latency_p50_ms'],
            results['async']['query']['latency_p95_ms'],
            results['async']['query']['latency_p99_ms']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, sync_latencies, width, label='Synchronous')
        plt.bar(x + width/2, async_latencies, width, label='Asynchronous')
        
        plt.xlabel('Percentile')
        plt.ylabel('Latency (ms)')
        plt.title('Query Latency Comparison')
        plt.xticks(x, categories)
        plt.legend()
        plt.axhline(y=5.0, color='r', linestyle='--', label='Target (5ms)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_comparison.png')
        plt.close()
        
        # Throughput comparison
        plt.figure(figsize=(10, 6))
        
        metrics = ['Store Throughput\n(ops/sec)', 'Query Throughput\n(qps)']
        sync_values = [
            results['sync']['store']['throughput_ops'],
            results['sync']['query']['throughput_qps']
        ]
        async_values = [
            results['async']['store']['throughput_ops'],
            results['async']['query']['throughput_qps']
        ]
        
        x = np.arange(len(metrics))
        plt.bar(x - width/2, sync_values, width, label='Synchronous')
        plt.bar(x + width/2, async_values, width, label='Asynchronous')
        
        plt.ylabel('Throughput')
        plt.title('Throughput Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_comparison.png')
        plt.close()


async def main():
    """Run the production benchmark."""
    parser = argparse.ArgumentParser(description="Production benchmark for Shape Memory V2")
    parser.add_argument("--vectors", type=int, default=100000, help="Number of vectors")
    parser.add_argument("--queries", type=int, default=10000, help="Number of queries")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"), help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing benchmark...")
    
    # Generate dataset
    generator = TopologyDataGenerator()
    dataset = generator.generate_dataset(
        num_normal=int(args.vectors * 0.8),
        num_anomaly=int(args.vectors * 0.15),
        num_danger=int(args.vectors * 0.05)
    )
    
    logger.info(f"Generated {len(dataset)} topological patterns")
    
    # Initialize storage
    config = ShapeMemoryConfig(
        storage_backend="redis",
        enable_fusion_scoring=True
    )
    
    # Run synchronous benchmark
    sync_memory = ShapeMemoryV2(config)
    runner = BenchmarkRunner()
    
    sync_results = await runner.run_sync_benchmark(
        sync_memory,
        dataset[:args.vectors],
        args.queries
    )
    
    # Run asynchronous benchmark
    redis_store = RedisVectorStore(RedisConfig())
    async_store = AsyncRedisAdapter(redis_store)
    async_memory = AsyncShapeMemoryV2(async_store)
    
    async_results = await runner.run_async_benchmark(
        async_memory,
        dataset[:args.vectors],
        args.queries
    )
    
    # Combine results
    results = {
        "sync": sync_results,
        "async": async_results
    }
    
    # Generate report
    report = runner.generate_report(results)
    report_path = args.output / "benchmark_report.md"
    report_path.write_text(report)
    logger.info(f"Report saved to {report_path}")
    
    # Save raw results
    results_path = args.output / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    runner.plot_results(results, args.output)
    logger.info(f"Plots saved to {args.output}")
    
    # Clean up
    sync_memory.close()
    await async_memory.close()
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())