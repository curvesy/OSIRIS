"""
Liquid Neural Networks (LNN) for AURA Intelligence.

A revolutionary neural architecture featuring continuous-time dynamics,
adaptive computation, and exceptional efficiency for time-series and
sequential data processing.

Key Features:
- Continuous-time neural dynamics using ODEs
- 10-100x parameter efficiency vs traditional NNs
- Real-time adaptability to changing inputs
- Explainable decision pathways
- Superior performance on temporal data
"""

from .core import (
    LiquidNeuron,
    LiquidLayer,
    LiquidNeuralNetwork,
    LiquidConfig,
    TimeConstants,
    WiringConfig
)

from .dynamics import (
    ODESolver,
    RungeKutta4,
    AdaptiveStepSolver,
    liquid_dynamics,
    compute_gradients
)

from .architectures import (
    LiquidRNN,
    LiquidTransformer,
    LiquidAutoencoder,
    HybridLiquidNet,
    StreamingLNN
)

from .training import (
    LiquidTrainer,
    BackpropThroughTime,
    AdjointSensitivity,
    SparsityRegularizer,
    TemporalLoss
)

from .utils import (
    create_sparse_wiring,
    visualize_dynamics,
    analyze_stability,
    export_to_onnx,
    profile_efficiency
)

__all__ = [
    # Core
    "LiquidNeuron",
    "LiquidLayer", 
    "LiquidNeuralNetwork",
    "LiquidConfig",
    "TimeConstants",
    "WiringConfig",
    
    # Dynamics
    "ODESolver",
    "RungeKutta4",
    "AdaptiveStepSolver",
    "liquid_dynamics",
    "compute_gradients",
    
    # Architectures
    "LiquidRNN",
    "LiquidTransformer",
    "LiquidAutoencoder",
    "HybridLiquidNet",
    "StreamingLNN",
    
    # Training
    "LiquidTrainer",
    "BackpropThroughTime",
    "AdjointSensitivity",
    "SparsityRegularizer",
    "TemporalLoss",
    
    # Utils
    "create_sparse_wiring",
    "visualize_dynamics",
    "analyze_stability",
    "export_to_onnx",
    "profile_efficiency"
]

# Version info
__version__ = "1.0.0"
__author__ = "AURA Intelligence Team"