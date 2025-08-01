"""
⚡ Workflow Performance Benchmarks
Continuous benchmarking for workflow nodes and TDA algorithms.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from aura_common.logging import get_logger
from ..orchestration.workflows.state import create_initial_state
from ..orchestration.workflows.nodes import (
    create_observer_node,
    create_supervisor_node,
    create_analyst_node
)
from ..tda.unified_engine import create_unified_tda_engine

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    duration_ms: float
    memory_mb: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    name: str
    runs: int
    success_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_avg: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowBenchmark:
    """Benchmark suite for workflow components."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmark suite."""
        self.output_dir = output_dir or Path("benchmarks/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    async def run_all(self, iterations: int = 100) -> Dict[str, BenchmarkSummary]:
        """Run all benchmarks."""
        logger.info(f"Starting workflow benchmarks with {iterations} iterations")
        
        summaries = {}
        
        # Node benchmarks
        summaries["observer"] = await self.benchmark_observer_node(iterations)
        summaries["supervisor"] = await self.benchmark_supervisor_node(iterations)
        summaries["analyst"] = await self.benchmark_analyst_node(iterations)
        
        # TDA benchmarks
        summaries["tda_small"] = await self.benchmark_tda_small(iterations)
        summaries["tda_medium"] = await self.benchmark_tda_medium(iterations)
        
        # Save results
        self.save_results(summaries)
        
        return summaries
    
    async def benchmark_observer_node(self, iterations: int) -> BenchmarkSummary:
        """Benchmark observer node."""
        results = []
        node = create_observer_node()
        
        for i in range(iterations):
            state = create_initial_state(f"bench-{i}", f"thread-{i}")
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                await node(state)
                duration_ms = (time.time() - start_time) * 1000
                memory_mb = self._get_memory_usage() - start_memory
                
                results.append(BenchmarkResult(
                    name="observer_node",
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    success=True
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    name="observer_node",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                ))
        
        return self._summarize_results("observer_node", results)
    
    async def benchmark_supervisor_node(self, iterations: int) -> BenchmarkSummary:
        """Benchmark supervisor node."""
        results = []
        node = create_supervisor_node()
        
        for i in range(iterations):
            # Create state with evidence
            state = create_initial_state(f"bench-{i}", f"thread-{i}")
            state["evidence_log"] = [
                {"type": "test", "data": {"value": i}}
            ]
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                await node(state)
                duration_ms = (time.time() - start_time) * 1000
                memory_mb = self._get_memory_usage() - start_memory
                
                results.append(BenchmarkResult(
                    name="supervisor_node",
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    success=True
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    name="supervisor_node",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                ))
        
        return self._summarize_results("supervisor_node", results)
    
    async def benchmark_analyst_node(self, iterations: int) -> BenchmarkSummary:
        """Benchmark analyst node."""
        results = []
        node = create_analyst_node()
        
        for i in range(iterations):
            # Create state with evidence
            state = create_initial_state(f"bench-{i}", f"thread-{i}")
            state["evidence_log"] = [
                {
                    "type": "metrics",
                    "metrics": {
                        "cpu_usage": 0.5,
                        "memory_usage": 0.6,
                        "error_rate": 0.01
                    }
                }
            ]
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                await node(state)
                duration_ms = (time.time() - start_time) * 1000
                memory_mb = self._get_memory_usage() - start_memory
                
                results.append(BenchmarkResult(
                    name="analyst_node",
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    success=True
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    name="analyst_node",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                ))
        
        return self._summarize_results("analyst_node", results)
    
    async def benchmark_tda_small(self, iterations: int) -> BenchmarkSummary:
        """Benchmark TDA with small dataset."""
        results = []
        engine = create_unified_tda_engine()
        
        for i in range(iterations):
            # Small dataset (100 points)
            data = np.random.rand(100, 3)
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                response = await engine.analyze(data)
                duration_ms = response.computation_time_ms
                memory_mb = self._get_memory_usage() - start_memory
                
                results.append(BenchmarkResult(
                    name="tda_small",
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    success=True,
                    metadata={
                        "algorithm": response.algorithm_used,
                        "speedup": response.performance_metrics.get("speedup", 1.0)
                    }
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    name="tda_small",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                ))
        
        return self._summarize_results("tda_small", results)
    
    async def benchmark_tda_medium(self, iterations: int) -> BenchmarkSummary:
        """Benchmark TDA with medium dataset."""
        results = []
        engine = create_unified_tda_engine()
        
        for i in range(iterations):
            # Medium dataset (1000 points)
            data = np.random.rand(1000, 3)
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                response = await engine.analyze(data)
                duration_ms = response.computation_time_ms
                memory_mb = self._get_memory_usage() - start_memory
                
                results.append(BenchmarkResult(
                    name="tda_medium",
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    success=True,
                    metadata={
                        "algorithm": response.algorithm_used,
                        "speedup": response.performance_metrics.get("speedup", 1.0)
                    }
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    name="tda_medium",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=str(e)
                ))
        
        return self._summarize_results("tda_medium", results)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _summarize_results(
        self,
        name: str,
        results: List[BenchmarkResult]
    ) -> BenchmarkSummary:
        """Summarize benchmark results."""
        successful = [r for r in results if r.success]
        
        if not successful:
            return BenchmarkSummary(
                name=name,
                runs=len(results),
                success_rate=0.0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
                memory_avg=0.0
            )
        
        latencies = sorted([r.duration_ms for r in successful])
        memories = [r.memory_mb for r in successful]
        
        return BenchmarkSummary(
            name=name,
            runs=len(results),
            success_rate=len(successful) / len(results),
            latency_p50=self._percentile(latencies, 50),
            latency_p95=self._percentile(latencies, 95),
            latency_p99=self._percentile(latencies, 99),
            memory_avg=statistics.mean(memories) if memories else 0.0,
            metadata={
                "algorithms": list(set(
                    r.metadata.get("algorithm", "")
                    for r in successful
                    if r.metadata.get("algorithm")
                ))
            }
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        index = int(len(data) * percentile / 100)
        return data[min(index, len(data) - 1)]
    
    def save_results(self, summaries: Dict[str, BenchmarkSummary]):
        """Save benchmark results to file."""
        timestamp = datetime.utcnow().isoformat()
        
        # Convert to JSON-serializable format
        data = {
            "timestamp": timestamp,
            "summaries": {
                name: {
                    "runs": summary.runs,
                    "success_rate": summary.success_rate,
                    "latency_p50": summary.latency_p50,
                    "latency_p95": summary.latency_p95,
                    "latency_p99": summary.latency_p99,
                    "memory_avg": summary.memory_avg,
                    "metadata": summary.metadata
                }
                for name, summary in summaries.items()
            }
        }
        
        # Save to timestamped file
        filename = f"benchmark_{timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        # Also save as latest
        latest_path = self.output_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def print_summary(self, summaries: Dict[str, BenchmarkSummary]):
        """Print benchmark summary."""
        print("\n📊 BENCHMARK RESULTS")
        print("=" * 60)
        
        for name, summary in summaries.items():
            print(f"\n{name}:")
            print(f"  Runs: {summary.runs}")
            print(f"  Success Rate: {summary.success_rate:.1%}")
            print(f"  Latency P50: {summary.latency_p50:.2f}ms")
            print(f"  Latency P95: {summary.latency_p95:.2f}ms")
            print(f"  Latency P99: {summary.latency_p99:.2f}ms")
            print(f"  Memory Avg: {summary.memory_avg:.2f}MB")
            
            if summary.metadata.get("algorithms"):
                print(f"  Algorithms: {', '.join(summary.metadata['algorithms'])}")


async def run_benchmarks(iterations: int = 100):
    """Run workflow benchmarks."""
    benchmark = WorkflowBenchmark()
    summaries = await benchmark.run_all(iterations)
    benchmark.print_summary(summaries)
    return summaries


if __name__ == "__main__":
    asyncio.run(run_benchmarks())