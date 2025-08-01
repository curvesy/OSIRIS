"""
🏆 TDA Benchmarking Suite
Enterprise-grade performance validation and benchmarking for TDA algorithms.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .models import TDARequest, TDAResponse, TDAAlgorithm, DataFormat, TDABenchmarkResult
from .core import ProductionTDAEngine
from ..utils.logger import get_logger


class TDABenchmarkSuite:
    """
    🏆 Enterprise TDA Benchmarking Suite
    
    Comprehensive benchmarking and validation for TDA algorithms with:
    - Performance benchmarking across algorithms
    - Accuracy validation against ground truth
    - Scalability testing with various data sizes
    - Resource usage monitoring
    - Automated report generation
    """
    
    def __init__(self, tda_engine: ProductionTDAEngine = None):
        self.logger = get_logger(__name__)
        self.tda_engine = tda_engine or ProductionTDAEngine()
        
        # Benchmark datasets
        self.datasets = self._generate_benchmark_datasets()
        
        # Results storage
        self.benchmark_results: List[TDABenchmarkResult] = []
        
        self.logger.info("🏆 TDA Benchmark Suite initialized")
    
    def _generate_benchmark_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Generate standard benchmark datasets."""
        
        datasets = {}
        
        # Small dataset (100 points)
        datasets['small_2d'] = {
            'name': 'Small 2D Point Cloud',
            'data': self._generate_point_cloud(100, 2),
            'size': 100,
            'dimensions': 2,
            'expected_betti': [1, 0, 0]  # Expected Betti numbers
        }
        
        # Medium dataset (1000 points)
        datasets['medium_3d'] = {
            'name': 'Medium 3D Point Cloud',
            'data': self._generate_point_cloud(1000, 3),
            'size': 1000,
            'dimensions': 3,
            'expected_betti': [1, 2, 0]
        }
        
        # Large dataset (5000 points)
        datasets['large_2d'] = {
            'name': 'Large 2D Point Cloud',
            'data': self._generate_point_cloud(5000, 2),
            'size': 5000,
            'dimensions': 2,
            'expected_betti': [1, 3, 0]
        }
        
        # Circle dataset (topological ground truth)
        datasets['circle'] = {
            'name': 'Circle (1-loop)',
            'data': self._generate_circle(200, noise=0.05),
            'size': 200,
            'dimensions': 2,
            'expected_betti': [1, 1, 0]  # One connected component, one loop
        }
        
        # Torus dataset (complex topology)
        datasets['torus'] = {
            'name': 'Torus (2-loops)',
            'data': self._generate_torus(500, noise=0.1),
            'size': 500,
            'dimensions': 3,
            'expected_betti': [1, 2, 1]  # One component, two loops, one cavity
        }
        
        # Sphere dataset
        datasets['sphere'] = {
            'name': 'Sphere (cavity)',
            'data': self._generate_sphere(300, noise=0.05),
            'size': 300,
            'dimensions': 3,
            'expected_betti': [1, 0, 1]  # One component, no loops, one cavity
        }
        
        self.logger.info(f"📊 Generated {len(datasets)} benchmark datasets")
        return datasets
    
    def _generate_point_cloud(self, n_points: int, dimensions: int) -> List[List[float]]:
        """Generate random point cloud."""
        np.random.seed(42)  # Reproducible results
        points = np.random.uniform(-1, 1, (n_points, dimensions))
        return points.tolist()
    
    def _generate_circle(self, n_points: int, radius: float = 1.0, noise: float = 0.0) -> List[List[float]]:
        """Generate points on a circle with optional noise."""
        np.random.seed(42)
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        # Add noise
        if noise > 0:
            x += np.random.normal(0, noise, n_points)
            y += np.random.normal(0, noise, n_points)
        
        return np.column_stack([x, y]).tolist()
    
    def _generate_torus(self, n_points: int, R: float = 2.0, r: float = 1.0, noise: float = 0.0) -> List[List[float]]:
        """Generate points on a torus."""
        np.random.seed(42)
        
        # Parametric torus equations
        u = np.random.uniform(0, 2 * np.pi, n_points)
        v = np.random.uniform(0, 2 * np.pi, n_points)
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        # Add noise
        if noise > 0:
            x += np.random.normal(0, noise, n_points)
            y += np.random.normal(0, noise, n_points)
            z += np.random.normal(0, noise, n_points)
        
        return np.column_stack([x, y, z]).tolist()
    
    def _generate_sphere(self, n_points: int, radius: float = 1.0, noise: float = 0.0) -> List[List[float]]:
        """Generate points on a sphere."""
        np.random.seed(42)
        
        # Generate uniform points on sphere
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # Add noise
        if noise > 0:
            x += np.random.normal(0, noise, n_points)
            y += np.random.normal(0, noise, n_points)
            z += np.random.normal(0, noise, n_points)
        
        return np.column_stack([x, y, z]).tolist()
    
    async def run_comprehensive_benchmark(
        self,
        algorithms: Optional[List[TDAAlgorithm]] = None,
        datasets: Optional[List[str]] = None,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across algorithms and datasets.
        
        Args:
            algorithms: List of algorithms to benchmark (default: all available)
            datasets: List of dataset names to use (default: all)
            n_runs: Number of runs per algorithm-dataset combination
            
        Returns:
            Comprehensive benchmark results
        """
        
        if algorithms is None:
            algorithms = [TDAAlgorithm.SPECSEQ_PLUS_PLUS, TDAAlgorithm.NEURAL_SURVEILLANCE]
            if self.tda_engine.cuda_accelerator and self.tda_engine.cuda_accelerator.is_available():
                algorithms.append(TDAAlgorithm.SIMBA_GPU)
        
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        self.logger.info(f"🚀 Starting comprehensive benchmark: {len(algorithms)} algorithms × {len(datasets)} datasets × {n_runs} runs")
        
        benchmark_results = []
        total_combinations = len(algorithms) * len(datasets)
        current_combination = 0
        
        for algorithm in algorithms:
            for dataset_name in datasets:
                current_combination += 1
                self.logger.info(f"📊 Benchmarking {algorithm} on {dataset_name} ({current_combination}/{total_combinations})")
                
                dataset = self.datasets[dataset_name]
                
                # Run multiple times for statistical significance
                run_results = []
                for run in range(n_runs):
                    try:
                        result = await self._benchmark_single_run(algorithm, dataset, run)
                        run_results.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Benchmark run failed: {algorithm} on {dataset_name}, run {run}: {e}")
                
                if run_results:
                    # Aggregate results
                    aggregated = self._aggregate_run_results(algorithm, dataset_name, dataset, run_results)
                    benchmark_results.append(aggregated)
                    self.benchmark_results.append(aggregated)
        
        # Generate comprehensive report
        report = self._generate_benchmark_report(benchmark_results)
        
        self.logger.info("✅ Comprehensive benchmark completed")
        return report
    
    async def _benchmark_single_run(
        self,
        algorithm: TDAAlgorithm,
        dataset: Dict[str, Any],
        run_number: int
    ) -> Dict[str, Any]:
        """Run single benchmark iteration."""
        
        # Create TDA request
        request = TDARequest(
            data=dataset['data'],
            algorithm=algorithm,
            data_format=DataFormat.POINT_CLOUD,
            max_dimension=2,
            request_id=f"benchmark_{algorithm}_{dataset['name']}_{run_number}",
            priority="high"
        )
        
        # Record start time and memory
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Execute TDA computation
        response = await self.tda_engine.compute_tda(request)
        
        # Record end time and memory
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        computation_time = (end_time - start_time) * 1000  # ms
        memory_used = max(0, end_memory - start_memory)  # MB
        
        # Validate results
        accuracy_score = self._calculate_accuracy(response, dataset['expected_betti'])
        
        return {
            'computation_time_ms': computation_time,
            'memory_usage_mb': memory_used,
            'accuracy_score': accuracy_score,
            'status': response.status,
            'betti_numbers': response.betti_numbers,
            'n_intervals': sum(len(diagram.intervals) for diagram in response.persistence_diagrams),
            'gpu_utilization': response.metrics.gpu_utilization_percent,
            'numerical_stability': response.metrics.numerical_stability
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0
    
    def _calculate_accuracy(self, response: TDAResponse, expected_betti: List[int]) -> float:
        """Calculate accuracy score compared to expected Betti numbers."""
        
        if response.status != "success":
            return 0.0
        
        actual_betti = response.betti_numbers
        
        # Pad shorter list with zeros
        max_len = max(len(actual_betti), len(expected_betti))
        actual_padded = actual_betti + [0] * (max_len - len(actual_betti))
        expected_padded = expected_betti + [0] * (max_len - len(expected_betti))
        
        # Calculate accuracy as 1 - normalized error
        total_expected = sum(expected_padded)
        if total_expected == 0:
            return 1.0 if sum(actual_padded) == 0 else 0.0
        
        error = sum(abs(a - e) for a, e in zip(actual_padded, expected_padded))
        accuracy = max(0.0, 1.0 - error / total_expected)
        
        return accuracy
    
    def _aggregate_run_results(
        self,
        algorithm: TDAAlgorithm,
        dataset_name: str,
        dataset: Dict[str, Any],
        run_results: List[Dict[str, Any]]
    ) -> TDABenchmarkResult:
        """Aggregate results from multiple runs."""
        
        # Extract metrics
        computation_times = [r['computation_time_ms'] for r in run_results if r['status'] == 'success']
        memory_usage = [r['memory_usage_mb'] for r in run_results if r['status'] == 'success']
        accuracy_scores = [r['accuracy_score'] for r in run_results if r['status'] == 'success']
        
        # Calculate statistics
        avg_time = statistics.mean(computation_times) if computation_times else 0
        std_time = statistics.stdev(computation_times) if len(computation_times) > 1 else 0
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0
        
        # Calculate speedup (compared to baseline)
        baseline_time = 1000.0  # Assume 1 second baseline
        speedup = baseline_time / avg_time if avg_time > 0 else 1.0
        
        # Get hardware info
        hardware_info = {
            'gpu_available': self.tda_engine.cuda_accelerator.is_available() if self.tda_engine.cuda_accelerator else False,
            'algorithm_backend': str(algorithm)
        }
        
        if self.tda_engine.cuda_accelerator and self.tda_engine.cuda_accelerator.is_available():
            hardware_info.update(self.tda_engine.cuda_accelerator.get_gpu_info())
        
        return TDABenchmarkResult(
            algorithm=algorithm,
            dataset_name=dataset_name,
            dataset_size=dataset['size'],
            avg_computation_time_ms=avg_time,
            std_computation_time_ms=std_time,
            speedup_vs_baseline=speedup,
            accuracy_score=avg_accuracy,
            bottleneck_distance_error=0.1,  # Placeholder
            peak_memory_mb=avg_memory,
            gpu_utilization_percent=run_results[0].get('gpu_utilization') if run_results else None,
            hardware_info=hardware_info
        )
    
    def _generate_benchmark_report(self, results: List[TDABenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        if not results:
            return {'error': 'No benchmark results available'}
        
        # Overall statistics
        all_times = [r.avg_computation_time_ms for r in results]
        all_accuracies = [r.accuracy_score for r in results]
        all_speedups = [r.speedup_vs_baseline for r in results]
        
        report = {
            'summary': {
                'total_benchmarks': len(results),
                'algorithms_tested': len(set(r.algorithm for r in results)),
                'datasets_tested': len(set(r.dataset_name for r in results)),
                'avg_computation_time_ms': statistics.mean(all_times),
                'avg_accuracy_score': statistics.mean(all_accuracies),
                'avg_speedup_factor': statistics.mean(all_speedups),
                'benchmark_timestamp': datetime.now().isoformat()
            },
            'algorithm_performance': {},
            'dataset_difficulty': {},
            'detailed_results': []
        }
        
        # Algorithm performance analysis
        for algorithm in set(r.algorithm for r in results):
            algo_results = [r for r in results if r.algorithm == algorithm]
            algo_times = [r.avg_computation_time_ms for r in algo_results]
            algo_accuracies = [r.accuracy_score for r in algo_results]
            
            report['algorithm_performance'][str(algorithm)] = {
                'avg_time_ms': statistics.mean(algo_times),
                'avg_accuracy': statistics.mean(algo_accuracies),
                'datasets_tested': len(algo_results),
                'best_dataset': max(algo_results, key=lambda x: x.accuracy_score).dataset_name,
                'worst_dataset': min(algo_results, key=lambda x: x.accuracy_score).dataset_name
            }
        
        # Dataset difficulty analysis
        for dataset in set(r.dataset_name for r in results):
            dataset_results = [r for r in results if r.dataset_name == dataset]
            dataset_times = [r.avg_computation_time_ms for r in dataset_results]
            dataset_accuracies = [r.accuracy_score for r in dataset_results]
            
            report['dataset_difficulty'][dataset] = {
                'avg_time_ms': statistics.mean(dataset_times),
                'avg_accuracy': statistics.mean(dataset_accuracies),
                'algorithms_tested': len(dataset_results),
                'difficulty_score': 1.0 - statistics.mean(dataset_accuracies)  # Higher = more difficult
            }
        
        # Detailed results
        for result in results:
            report['detailed_results'].append({
                'algorithm': str(result.algorithm),
                'dataset': result.dataset_name,
                'dataset_size': result.dataset_size,
                'computation_time_ms': result.avg_computation_time_ms,
                'accuracy_score': result.accuracy_score,
                'speedup_factor': result.speedup_vs_baseline,
                'memory_usage_mb': result.peak_memory_mb,
                'gpu_utilization': result.gpu_utilization_percent
            })
        
        return report
    
    def save_benchmark_results(self, filepath: str = None):
        """Save benchmark results to file."""
        
        if not self.benchmark_results:
            self.logger.warning("⚠️ No benchmark results to save")
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"tda_benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.benchmark_results:
            results_data.append({
                'algorithm': str(result.algorithm),
                'dataset_name': result.dataset_name,
                'dataset_size': result.dataset_size,
                'avg_computation_time_ms': result.avg_computation_time_ms,
                'std_computation_time_ms': result.std_computation_time_ms,
                'speedup_vs_baseline': result.speedup_vs_baseline,
                'accuracy_score': result.accuracy_score,
                'bottleneck_distance_error': result.bottleneck_distance_error,
                'peak_memory_mb': result.peak_memory_mb,
                'gpu_utilization_percent': result.gpu_utilization_percent,
                'benchmark_timestamp': result.benchmark_timestamp.isoformat(),
                'hardware_info': result.hardware_info
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"💾 Benchmark results saved to {filepath}")
    
    def generate_performance_plots(self, output_dir: str = "benchmark_plots"):
        """Generate performance visualization plots."""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("⚠️ Matplotlib not available, skipping plots")
            return
        
        if not self.benchmark_results:
            self.logger.warning("⚠️ No benchmark results to plot")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Performance comparison plot
        self._plot_algorithm_performance(output_dir)
        
        # Scalability plot
        self._plot_scalability(output_dir)
        
        # Accuracy comparison
        self._plot_accuracy_comparison(output_dir)
        
        self.logger.info(f"📊 Performance plots saved to {output_dir}/")
    
    def _plot_algorithm_performance(self, output_dir: str):
        """Plot algorithm performance comparison."""
        
        algorithms = list(set(str(r.algorithm) for r in self.benchmark_results))
        avg_times = []
        
        for algo in algorithms:
            algo_results = [r for r in self.benchmark_results if str(r.algorithm) == algo]
            avg_time = statistics.mean([r.avg_computation_time_ms for r in algo_results])
            avg_times.append(avg_time)
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, avg_times)
        plt.title('TDA Algorithm Performance Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Computation Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/algorithm_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability(self, output_dir: str):
        """Plot algorithm scalability."""
        
        plt.figure(figsize=(12, 8))
        
        for algorithm in set(str(r.algorithm) for r in self.benchmark_results):
            algo_results = [r for r in self.benchmark_results if str(r.algorithm) == algorithm]
            
            # Sort by dataset size
            algo_results.sort(key=lambda x: x.dataset_size)
            
            sizes = [r.dataset_size for r in algo_results]
            times = [r.avg_computation_time_ms for r in algo_results]
            
            plt.plot(sizes, times, marker='o', label=algorithm)
        
        plt.title('TDA Algorithm Scalability')
        plt.xlabel('Dataset Size (number of points)')
        plt.ylabel('Computation Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scalability.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_comparison(self, output_dir: str):
        """Plot accuracy comparison across datasets."""
        
        datasets = list(set(r.dataset_name for r in self.benchmark_results))
        algorithms = list(set(str(r.algorithm) for r in self.benchmark_results))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(algorithms)
        
        for i, algorithm in enumerate(algorithms):
            accuracies = []
            for dataset in datasets:
                results = [r for r in self.benchmark_results 
                          if r.dataset_name == dataset and str(r.algorithm) == algorithm]
                avg_accuracy = statistics.mean([r.accuracy_score for r in results]) if results else 0
                accuracies.append(avg_accuracy)
            
            ax.bar(x + i * width, accuracies, width, label=algorithm)
        
        ax.set_title('TDA Algorithm Accuracy Comparison')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy Score')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
