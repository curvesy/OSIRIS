"""
Neural Network Configuration

Type-safe configuration for neural components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ODESolver(str, Enum):
    """ODE solver types for continuous-time dynamics."""
    EULER = "euler"
    RK4 = "rk4"
    ADAPTIVE = "adaptive"
    SEMI_IMPLICIT = "semi_implicit"


class ActivationType(str, Enum):
    """Activation function types."""
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    LIQUID = "liquid"


class NeuralConfig(BaseModel):
    """Validated neural network configuration."""
    
    # Architecture
    input_size: int = Field(gt=0, description="Input dimension")
    hidden_sizes: List[int] = Field(default=[128, 64], description="Hidden layer sizes")
    output_size: int = Field(gt=0, default=4, description="Output dimension")
    num_layers: int = Field(ge=1, le=10, default=3, description="Number of layers")
    
    # Liquid dynamics
    time_constant: float = Field(gt=0, le=10, default=1.0, description="Time constant for liquid dynamics")
    ode_solver: ODESolver = Field(default=ODESolver.RK4, description="ODE solver type")
    solver_steps: int = Field(ge=1, le=100, default=10, description="ODE solver steps")
    
    # Adaptation
    adaptivity_rate: float = Field(ge=0, le=1, default=0.1, description="Adaptation rate")
    learning_rate: float = Field(gt=0, le=0.1, default=0.001, description="Learning rate")
    
    # Architecture details
    activation: ActivationType = Field(default=ActivationType.TANH, description="Activation function")
    dropout_rate: float = Field(ge=0, lt=1, default=0.1, description="Dropout rate")
    use_batch_norm: bool = Field(default=True, description="Use batch normalization")
    
    # Efficiency
    sparsity: float = Field(ge=0, le=1, default=0.7, description="Connection sparsity")
    quantization_bits: Optional[int] = Field(None, ge=1, le=32, description="Quantization bits")
    
    # Device
    device: str = Field(default="cpu", description="Compute device")
    mixed_precision: bool = Field(default=False, description="Use mixed precision")
    
    @validator('hidden_sizes')
    def validate_hidden_sizes(cls, v):
        if not v:
            raise ValueError("Hidden sizes cannot be empty")
        if any(size <= 0 for size in v):
            raise ValueError("All hidden sizes must be positive")
        return v
    
    @validator('device')
    def validate_device(cls, v):
        import torch
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid device: {v}")
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if v == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return v
    
    class Config:
        use_enum_values = True


@dataclass
class LayerConfig:
    """Configuration for individual layers."""
    input_dim: int
    output_dim: int
    activation: ActivationType = ActivationType.TANH
    use_bias: bool = True
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    
    def __post_init__(self):
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_interval: int = 10
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")