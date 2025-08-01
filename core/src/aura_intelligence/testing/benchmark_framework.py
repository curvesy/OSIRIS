"""
📊 Reproducible Benchmarking Framework for Streaming TDA
Ensures consistent performance measurements across CI/CD runs
"""

import asyncio
import json
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import hashlib
import platform
import psutil
import os
from pathlib import Path

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..tda.streaming import StreamingTDAProcessor
from ..tda.streaming.parallel_processor import MultiScaleProcessor, ScaleConfig
from ..infrastructure.kafka_event_mesh import KafkaEventMesh
from .chaos_engineering import RealisticDataGenerator

logger = structlog.get_logger(__name__)

# Benchmark metrics
BENCHMARK_RUNS = Counter('benchmark_runs_total', 'Total benchmark runs', ['suite', 'test'])
BENCHMARK_DURATION = Histogram('benchmark_duration_seconds', 'Benchmark duration', ['suite', 'test'])
BENCHMARK_SCORE = Gauge('benchmark_score', 'Benchmark score', ['suite', 'test', 'metric'])


@dataclass
class BenchmarkConfig:
    """Configuration for reproducible benchmarks"""
    name: str
    seed: int = 42
    warmup_iterations: int = 10
    test_iterations: int = 100
    data_size: int = 10000
    dimensions: int = 3
    timeout_seconds: float = 300.0
    environment_vars: Dict[str, str] = field(default_factory=dict)
    

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    environment: Dict[str, Any]
    metrics: Dict[str, float]
    raw_measurements: List[float]
    percentiles: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks"""
    name: str
    description: str
    benchmarks: List[BenchmarkConfig]
    baseline: Optional[Dict[str, float]] = None
    

class EnvironmentCapture:
    """Captures and validates environment for reproducibility"""
    
    @staticmethod
    def capture() -> Dict[str, Any]:
        """Capture current environment"""
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            },
            'process': {
                'pid': os.getpid(),
                'nice': os.nice(0),
                'affinity': list(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else None,
            },
            'environment': {
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
                'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS'),
            },
            'timestamp': datetime.now().isoformat(),
        }
        
    @staticmethod
    def validate_consistency(env1: Dict[str, Any], env2: Dict[str, Any]) -> List[str]:
        """Check if two environments are consistent"""
        differences = []
        
        # Check CPU
        if env1['hardware']['cpu_count'] != env2['hardware']['cpu_count']:
            differences.append(f"CPU count differs: {env1['hardware']['cpu_count']} vs {env2['hardware']['cpu_count']}")
            
        # Check memory (allow 10% variance)
        mem1 = env1['hardware']['memory_total_gb']
        mem2 = env2['hardware']['memory_total_gb']
        if abs(mem1 - mem2) / mem1 > 0.1:
            differences.append(f"Memory differs significantly: {mem1:.1f}GB vs {mem2:.1f}GB")
            
        # Check Python version
        if env1['platform']['python_version'] != env2['platform']['python_version']:
            differences.append(f"Python version differs: {env1['platform']['python_version']} vs {env2['platform']['python_version']}")
            
        return differences
        

class ReproducibleBenchmark:
    """Base class for reproducible benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.environment = EnvironmentCapture.capture()
        self._setup_reproducibility()
        
    def _setup_reproducibility(self) -> None:
        """Setup for reproducible results"""
        # Set random seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Set environment variables
        for key, value in self.config.environment_vars.items():
            os.environ[key] = value
            
        # Log setup
        logger.info(
            "benchmark_setup",
            name=self.config.name,
            seed=self.config.seed,
            environment=self.environment
        )
        
    async def run(self) -> BenchmarkResult:
        """Run the benchmark"""
        result = BenchmarkResult(
            config=self.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            environment=self.environment,
            metrics={},
            raw_measurements=[],
            percentiles={}
        )
        
        try:
            # Warmup
            logger.info("benchmark_warmup", iterations=self.config.warmup_iterations)
            for _ in range(self.config.warmup_iterations):
                await self._run_single_iteration()
                
            # Actual benchmark
            logger.info("benchmark_start", iterations=self.config.test_iterations)
            measurements = []
            
            for i in range(self.config.test_iterations):
                start = time.perf_counter()
                await self._run_single_iteration()
                elapsed = time.perf_counter() - start
                measurements.append(elapsed)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(
                        "benchmark_progress",
                        iteration=i + 1,
                        total=self.config.test_iterations,
                        last_time=elapsed
                    )
                    
            result.raw_measurements = measurements
            result.end_time = datetime.now()
            
            # Calculate metrics
            result.metrics = self._calculate_metrics(measurements)
            result.percentiles = self._calculate_percentiles(measurements)
            
            # Record in Prometheus
            BENCHMARK_RUNS.labels(suite="default", test=self.config.name).inc()
            BENCHMARK_DURATION.labels(
                suite="default",
                test=self.config.name
            ).observe(result.metrics['mean'])
            
            for metric, value in result.metrics.items():
                BENCHMARK_SCORE.labels(
                    suite="default",
                    test=self.config.name,
                    metric=metric
                ).set(value)
                
        except Exception as e:
            logger.error("benchmark_error", error=str(e))
            result.metadata['error'] = str(e)
            
        return result
        
    async def _run_single_iteration(self) -> None:
        """Override this method in subclasses"""
        raise NotImplementedError
        
    def _calculate_metrics(self, measurements: List[float]) -> Dict[str, float]:
        """Calculate standard metrics"""
        arr = np.array(measurements)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'cv': float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,  # Coefficient of variation
        }
        
    def _calculate_percentiles(self, measurements: List[float]) -> Dict[str, float]:
        """Calculate percentiles"""
        arr = np.array(measurements)
        return {
            'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'p999': float(np.percentile(arr, 99.9)),
        }
        

class StreamingTDABenchmark(ReproducibleBenchmark):
    """Benchmark for streaming TDA operations"""
    
    def __init__(self, config: BenchmarkConfig, processor: StreamingTDAProcessor):
        super().__init__(config)
        self.processor = processor
        self.data_generator = self._create_data_generator()
        
    def _create_data_generator(self) -> np.ndarray:
        """Create reproducible test data"""
        np.random.seed(self.config.seed)
        return np.random.randn(
            self.config.data_size,
            self.config.dimensions
        )
        
    async def _run_single_iteration(self) -> None:
        """Run single benchmark iteration"""
        await self.processor.process_batch(self.data_generator)
        

class MultiScaleBenchmark(ReproducibleBenchmark):
    """Benchmark for multi-scale processing"""
    
    def __init__(
        self,
        config: BenchmarkConfig,
        scales: List[ScaleConfig]
    ):
        super().__init__(config)
        self.processor = MultiScaleProcessor(scales)
        self.data_generator = self._create_data_generator()
        
    def _create_data_generator(self) -> np.ndarray:
        """Create reproducible test data"""
        np.random.seed(self.config.seed)
        return np.random.randn(
            self.config.data_size,
            self.config.dimensions
        )
        
    async def _run_single_iteration(self) -> None:
        """Run single benchmark iteration"""
        await self.processor.add_points(self.data_generator)
        
    async def cleanup(self) -> None:
        """Cleanup after benchmark"""
        await self.processor.shutdown()
        

class BenchmarkRunner:
    """Runs and manages benchmark suites"""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, List[BenchmarkResult]] = {}
        
    async def run_suite(self, suite: BenchmarkSuite) -> Dict[str, BenchmarkResult]:
        """Run a complete benchmark suite"""
        logger.info("running_benchmark_suite", name=suite.name)
        
        suite_results = {}
        
        for config in suite.benchmarks:
            # Create appropriate benchmark
            if "streaming" in config.name:
                processor = StreamingTDAProcessor(
                    window_size=10000,
                    slide_interval=1000
                )
                benchmark = StreamingTDABenchmark(config, processor)
            elif "multiscale" in config.name:
                scales = [
                    ScaleConfig("fast", 1000, 100),
                    ScaleConfig("medium", 5000, 500),
                    ScaleConfig("slow", 10000, 1000)
                ]
                benchmark = MultiScaleBenchmark(config, scales)
            else:
                continue
                
            # Run benchmark
            result = await benchmark.run()
            suite_results[config.name] = result
            
            # Cleanup
            if hasattr(benchmark, 'cleanup'):
                await benchmark.cleanup()
                
            # Save results
            self._save_result(suite.name, result)
            
        # Generate report
        self._generate_report(suite, suite_results)
        
        return suite_results
        
    def _save_result(self, suite_name: str, result: BenchmarkResult) -> None:
        """Save benchmark result to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_{result.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
            
        logger.info("benchmark_saved", file=str(filepath))
        
    def _generate_report(
        self,
        suite: BenchmarkSuite,
        results: Dict[str, BenchmarkResult]
    ) -> None:
        """Generate benchmark report with visualizations"""
        report_path = self.output_dir / f"{suite.name}_report.html"
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Benchmark Suite: {suite.name}', fontsize=16)
        
        # Plot 1: Performance comparison
        ax = axes[0, 0]
        names = list(results.keys())
        means = [r.metrics['mean'] for r in results.values()]
        stds = [r.metrics['std'] for r in results.values()]
        
        ax.bar(names, means, yerr=stds, capsize=5)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Mean Performance')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Percentiles
        ax = axes[0, 1]
        percentiles = ['p50', 'p90', 'p95', 'p99']
        for name, result in results.items():
            values = [result.percentiles[p] for p in percentiles]
            ax.plot(percentiles, values, marker='o', label=name)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Latency Percentiles')
        ax.legend()
        
        # Plot 3: Distribution
        ax = axes[1, 0]
        for name, result in results.items():
            if result.raw_measurements:
                ax.hist(
                    result.raw_measurements,
                    bins=30,
                    alpha=0.5,
                    label=name,
                    density=True
                )
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Density')
        ax.set_title('Performance Distribution')
        ax.legend()
        
        # Plot 4: Baseline comparison
        ax = axes[1, 1]
        if suite.baseline:
            for name, result in results.items():
                if name in suite.baseline:
                    baseline = suite.baseline[name]
                    current = result.metrics['mean']
                    improvement = (baseline - current) / baseline * 100
                    
                    color = 'green' if improvement > 0 else 'red'
                    ax.bar(name, improvement, color=color)
                    
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Performance vs Baseline')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{suite.name}_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        # Generate HTML report
        html_content = self._generate_html_report(suite, results, plot_path)
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        logger.info("benchmark_report_generated", path=str(report_path))
        
    def _generate_html_report(
        self,
        suite: BenchmarkSuite,
        results: Dict[str, BenchmarkResult],
        plot_path: Path
    ) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report: {suite.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-family: monospace; }}
                .improved {{ color: green; }}
                .degraded {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report: {suite.name}</h1>
            <p>{suite.description}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Environment</h2>
            <pre>{json.dumps(results[list(results.keys())[0]].environment, indent=2)}</pre>
            
            <h2>Results Summary</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Mean (s)</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>P50</th>
                    <th>P95</th>
                    <th>P99</th>
                </tr>
        """
        
        for name, result in results.items():
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="metric">{result.metrics['mean']:.4f}</td>
                    <td class="metric">{result.metrics['std']:.4f}</td>
                    <td class="metric">{result.metrics['min']:.4f}</td>
                    <td class="metric">{result.metrics['max']:.4f}</td>
                    <td class="metric">{result.percentiles['p50']:.4f}</td>
                    <td class="metric">{result.percentiles['p95']:.4f}</td>
                    <td class="metric">{result.percentiles['p99']:.4f}</td>
                </tr>
            """
            
        html += f"""
            </table>
            
            <h2>Performance Plots</h2>
            <img src="{plot_path.name}" style="max-width: 100%;">
            
            <h2>Configuration</h2>
            <pre>{json.dumps(asdict(suite.benchmarks[0]), indent=2)}</pre>
        </body>
        </html>
        """
        
        return html
        

class CIBenchmarkValidator:
    """Validates benchmark results for CI/CD pipelines"""
    
    def __init__(
        self,
        tolerance_percent: float = 10.0,
        min_iterations: int = 50
    ):
        self.tolerance_percent = tolerance_percent
        self.min_iterations = min_iterations
        
    def validate_result(
        self,
        result: BenchmarkResult,
        baseline: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Validate benchmark result against baseline"""
        issues = []
        
        # Check iterations
        if len(result.raw_measurements) < self.min_iterations:
            issues.append(
                f"Insufficient iterations: {len(result.raw_measurements)} < {self.min_iterations}"
            )
            
        # Check performance regression
        if 'mean' in baseline:
            current_mean = result.metrics['mean']
            baseline_mean = baseline['mean']
            regression = (current_mean - baseline_mean) / baseline_mean * 100
            
            if regression > self.tolerance_percent:
                issues.append(
                    f"Performance regression: {regression:.1f}% slower than baseline"
                )
                
        # Check variability
        cv = result.metrics.get('cv', 0)
        if cv > 0.2:  # Coefficient of variation > 20%
            issues.append(f"High variability: CV={cv:.2f}")
            
        # Check for outliers
        if result.metrics['max'] > result.metrics['mean'] * 3:
            issues.append("Extreme outliers detected")
            
        return len(issues) == 0, issues
        

# Predefined benchmark suites
STREAMING_TDA_SUITE = BenchmarkSuite(
    name="streaming_tda",
    description="Benchmarks for streaming TDA operations",
    benchmarks=[
        BenchmarkConfig(
            name="streaming_small",
            data_size=1000,
            test_iterations=100
        ),
        BenchmarkConfig(
            name="streaming_medium",
            data_size=10000,
            test_iterations=50
        ),
        BenchmarkConfig(
            name="streaming_large",
            data_size=100000,
            test_iterations=20
        ),
    ],
    baseline={
        "streaming_small": 0.01,
        "streaming_medium": 0.1,
        "streaming_large": 1.0,
    }
)

MULTI_SCALE_SUITE = BenchmarkSuite(
    name="multi_scale",
    description="Benchmarks for multi-scale parallel processing",
    benchmarks=[
        BenchmarkConfig(
            name="multiscale_2scales",
            data_size=10000,
            test_iterations=50
        ),
        BenchmarkConfig(
            name="multiscale_4scales",
            data_size=10000,
            test_iterations=50
        ),
        BenchmarkConfig(
            name="multiscale_8scales",
            data_size=10000,
            test_iterations=50
        ),
    ]
)


# Example usage for CI/CD
async def run_ci_benchmarks() -> bool:
    """Run benchmarks for CI/CD pipeline"""
    runner = BenchmarkRunner()
    validator = CIBenchmarkValidator()
    
    all_passed = True
    
    # Run streaming benchmarks
    streaming_results = await runner.run_suite(STREAMING_TDA_SUITE)
    
    # Validate results
    for name, result in streaming_results.items():
        if name in STREAMING_TDA_SUITE.baseline:
            passed, issues = validator.validate_result(
                result,
                {'mean': STREAMING_TDA_SUITE.baseline[name]}
            )
            
            if not passed:
                logger.error(
                    "benchmark_validation_failed",
                    benchmark=name,
                    issues=issues
                )
                all_passed = False
            else:
                logger.info("benchmark_passed", benchmark=name)
                
    return all_passed


if __name__ == "__main__":
    # Run CI benchmarks
    success = asyncio.run(run_ci_benchmarks())
    exit(0 if success else 1)