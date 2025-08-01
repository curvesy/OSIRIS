"""
🔥 AURA Intelligence TDA Engine - Production Grade
Enterprise-ready Topological Data Analysis with 30x GPU acceleration.
"""

from .core import ProductionTDAEngine
from .models import TDARequest, TDAResponse, TDAMetrics, TDAConfiguration, TDAAlgorithm, DataFormat
from .algorithms import SpecSeqPlusPlus, SimBaGPU, NeuralSurveillance
from .benchmarks import TDABenchmarkSuite
from .cuda_kernels import CUDAAccelerator

__all__ = [
    'ProductionTDAEngine',
    'TDARequest', 
    'TDAResponse',
    'TDAMetrics',
    'TDAConfiguration',
    'TDAAlgorithm',
    'DataFormat',
    'SpecSeqPlusPlus',
    'SimBaGPU', 
    'NeuralSurveillance',
    'TDABenchmarkSuite',
    'CUDAAccelerator'
]

__version__ = "1.0.0"
