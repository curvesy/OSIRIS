"""
Neural Network Layers

Modular, reusable neural network layers for LNN.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ActivationType, LayerConfig


class LiquidTimeStep(nn.Module):
    """Liquid time step module for continuous-time dynamics."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_constant: float = 1.0,
        activation: ActivationType = ActivationType.TANH
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.activation_type = activation
        
        # Learnable parameters
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size) * time_constant)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: ActivationType) -> nn.Module:
        """Get activation function."""
        activations = {
            ActivationType.TANH: nn.Tanh(),
            ActivationType.RELU: nn.ReLU(),
            ActivationType.GELU: nn.GELU(),
            ActivationType.SILU: nn.SiLU(),
            ActivationType.LIQUID: nn.Tanh()  # Default for liquid
        }
        return activations.get(activation, nn.Tanh())
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_h.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Forward pass with liquid dynamics.
        
        Args:
            x: Input tensor [batch_size, input_size]
            h: Hidden state [batch_size, hidden_size]
            dt: Time step for integration
            
        Returns:
            Updated hidden state
        """
        # Compute derivative
        dx = self.activation(self.W_in(x) + self.W_h(h))
        
        # Apply liquid dynamics with adaptive time constant
        h_new = h + dt * (dx - h) / self.tau.abs()
        
        return h_new
    
    def adapt_time_constant(self, feedback: torch.Tensor):
        """Adapt time constant based on feedback."""
        with torch.no_grad():
            # Simple adaptation: increase tau for stable regions, decrease for dynamic
            stability = torch.std(feedback, dim=0)
            self.tau.data = self.tau.data * (1 + 0.1 * (1 - stability))
            self.tau.data = torch.clamp(self.tau.data, 0.1, 10.0)


class AdaptiveLayer(nn.Module):
    """Adaptive neural layer with dynamic properties."""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        
        # Core linear transformation
        self.linear = nn.Linear(
            config.input_dim,
            config.output_dim,
            bias=config.use_bias
        )
        
        # Optional components
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(config.output_dim) if config.use_batch_norm else None
        
        # Activation
        self.activation = self._get_activation(config.activation)
        
        # Adaptive parameters
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1))
        self.importance_weights = nn.Parameter(torch.ones(config.output_dim))
        
        self._initialize_weights()
    
    def _get_activation(self, activation: ActivationType) -> nn.Module:
        """Get activation function."""
        activations = {
            ActivationType.TANH: nn.Tanh(),
            ActivationType.RELU: nn.ReLU(),
            ActivationType.GELU: nn.GELU(),
            ActivationType.SILU: nn.SiLU(),
            ActivationType.LIQUID: nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Linear transformation
        out = self.linear(x)
        
        # Batch normalization
        if self.batch_norm is not None and out.shape[0] > 1:
            out = self.batch_norm(out)
        
        # Apply importance weights
        out = out * self.importance_weights
        
        # Activation
        out = self.activation(out)
        
        # Dropout
        if self.dropout is not None and self.training:
            out = self.dropout(out)
        
        return out
    
    def adapt(self, gradient: torch.Tensor):
        """Adapt layer based on gradient information."""
        with torch.no_grad():
            # Update importance weights based on gradient magnitude
            grad_importance = torch.abs(gradient).mean(dim=0)
            self.importance_weights.data = (
                (1 - self.adaptation_rate) * self.importance_weights.data +
                self.adaptation_rate * grad_importance / (grad_importance.mean() + 1e-8)
            )
            self.importance_weights.data = torch.clamp(self.importance_weights.data, 0.1, 10.0)


class SparseConnection(nn.Module):
    """Sparse connection layer for efficiency."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sparsity: float = 0.9,
        trainable_mask: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        
        # Weight matrix
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))
        
        # Sparsity mask
        if trainable_mask:
            self.mask = nn.Parameter(torch.ones(output_size, input_size))
        else:
            self.register_buffer('mask', self._create_sparse_mask())
    
    def _create_sparse_mask(self) -> torch.Tensor:
        """Create sparse connectivity mask."""
        mask = torch.rand(self.output_size, self.input_size)
        threshold = torch.quantile(mask.flatten(), self.sparsity)
        return (mask > threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse connections."""
        # Apply mask to weights
        masked_weight = self.weight * self.mask
        
        # Linear transformation
        return F.linear(x, masked_weight, self.bias)
    
    def prune_connections(self, threshold: float = 0.01):
        """Prune weak connections."""
        with torch.no_grad():
            # Update mask based on weight magnitude
            weight_magnitude = torch.abs(self.weight)
            self.mask.data = (weight_magnitude > threshold).float()


class AttentionLayer(nn.Module):
    """Self-attention layer for context integration."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        B, L, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        
        # Final projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out