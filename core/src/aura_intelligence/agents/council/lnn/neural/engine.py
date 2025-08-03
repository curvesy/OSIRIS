"""
Liquid Neural Engine

Main neural network engine implementation.
"""

from typing import Dict, Any, Optional, List
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from ..interfaces import INeuralEngine
from .config import NeuralConfig, ODESolver
from .layers import LiquidTimeStep, AdaptiveLayer, SparseConnection, AttentionLayer


class LiquidNeuralEngine(nn.Module, INeuralEngine):
    """Production-grade liquid neural network engine."""
    
    def __init__(self, config: Optional[NeuralConfig] = None):
        super().__init__()
        self.config = config or NeuralConfig()
        self.device = torch.device(self.config.device)
        self.is_initialized = False
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Metrics
        self.forward_count = 0
        self.total_forward_time = 0.0
        self.adaptation_count = 0
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the neural engine."""
        # Update config if provided
        if config:
            self.config = NeuralConfig(**config)
            self.device = torch.device(self.config.device)
        
        # Build network architecture
        self._build_network()
        
        # Move to device
        self.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        self.is_initialized = True
    
    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input attention layer (optional)
        if self.config.num_layers > 2:
            self.input_attention = AttentionLayer(
                dim=self.config.input_size,
                num_heads=4,
                dropout=self.config.dropout_rate
            )
        else:
            self.input_attention = None
        
        # Build liquid layers
        self.liquid_layers = nn.ModuleList()
        
        # First liquid layer
        self.liquid_layers.append(
            LiquidTimeStep(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_sizes[0],
                time_constant=self.config.time_constant,
                activation=self.config.activation
            )
        )
        
        # Hidden liquid layers
        for i in range(1, len(self.config.hidden_sizes)):
            if self.config.sparsity < 1.0:
                # Use sparse connections for efficiency
                layer = SparseConnection(
                    input_size=self.config.hidden_sizes[i-1],
                    output_size=self.config.hidden_sizes[i],
                    sparsity=self.config.sparsity
                )
            else:
                # Regular adaptive layer
                layer = AdaptiveLayer(
                    LayerConfig(
                        input_dim=self.config.hidden_sizes[i-1],
                        output_dim=self.config.hidden_sizes[i],
                        activation=self.config.activation,
                        dropout_rate=self.config.dropout_rate,
                        use_batch_norm=self.config.use_batch_norm
                    )
                )
            self.liquid_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(
            self.config.hidden_sizes[-1],
            self.config.output_size
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper parameter groups."""
        # Separate parameters by type
        liquid_params = []
        regular_params = []
        
        for name, param in self.named_parameters():
            if 'tau' in name or 'adaptation' in name:
                liquid_params.append(param)
            else:
                regular_params.append(param)
        
        # Different learning rates for different parameter types
        param_groups = [
            {'params': regular_params, 'lr': self.config.learning_rate},
            {'params': liquid_params, 'lr': self.config.learning_rate * 0.1}
        ]
        
        return optim.AdamW(param_groups, weight_decay=1e-5)
    
    async def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if not self.is_initialized:
            raise RuntimeError("Neural engine not initialized")
        
        start_time = time.time()
        
        # Move to device
        x = features.to(self.device)
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Apply input attention if available
        if self.input_attention is not None:
            x = x.unsqueeze(1)  # Add sequence dimension
            x = self.input_attention(x)
            x = x.squeeze(1)  # Remove sequence dimension
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.config.hidden_sizes[0], device=self.device)
        
        # Process through liquid layers
        for i, layer in enumerate(self.liquid_layers):
            if i == 0:
                # First layer is liquid time step
                h = self._solve_ode(layer, x, h)
            else:
                # Other layers
                if isinstance(layer, (AdaptiveLayer, SparseConnection)):
                    h = layer(h)
                else:
                    h = layer(h, h)  # For any other liquid layers
            
            # Apply dropout between layers
            if self.training and i < len(self.liquid_layers) - 1:
                h = self.dropout(h)
        
        # Output projection
        output = self.output_layer(h)
        
        # Update metrics
        self.forward_count += 1
        self.total_forward_time += time.time() - start_time
        
        return output
    
    def _solve_ode(
        self,
        layer: LiquidTimeStep,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Solve ODE using specified solver."""
        if self.config.ode_solver == ODESolver.EULER:
            return self._euler_solver(layer, x, h)
        elif self.config.ode_solver == ODESolver.RK4:
            return self._rk4_solver(layer, x, h)
        elif self.config.ode_solver == ODESolver.ADAPTIVE:
            return self._adaptive_solver(layer, x, h)
        else:
            # Default to semi-implicit for stability
            return self._semi_implicit_solver(layer, x, h)
    
    def _euler_solver(
        self,
        layer: LiquidTimeStep,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Simple Euler solver."""
        dt = 1.0 / self.config.solver_steps
        
        for _ in range(self.config.solver_steps):
            h = layer(x, h, dt)
        
        return h
    
    def _rk4_solver(
        self,
        layer: LiquidTimeStep,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """4th order Runge-Kutta solver."""
        dt = 1.0 / self.config.solver_steps
        
        for _ in range(self.config.solver_steps):
            k1 = layer(x, h, dt) - h
            k2 = layer(x, h + 0.5 * k1, dt) - (h + 0.5 * k1)
            k3 = layer(x, h + 0.5 * k2, dt) - (h + 0.5 * k2)
            k4 = layer(x, h + k3, dt) - (h + k3)
            
            h = h + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return h
    
    def _semi_implicit_solver(
        self,
        layer: LiquidTimeStep,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Semi-implicit solver for better stability."""
        dt = 1.0 / self.config.solver_steps
        
        for _ in range(self.config.solver_steps):
            # Predict
            h_pred = layer(x, h, dt)
            # Correct
            h = 0.5 * (h + layer(x, h_pred, dt))
        
        return h
    
    def _adaptive_solver(
        self,
        layer: LiquidTimeStep,
        x: torch.Tensor,
        h: torch.Tensor,
        tol: float = 1e-5
    ) -> torch.Tensor:
        """Adaptive step size solver."""
        t = 0.0
        dt = 0.1
        
        while t < 1.0:
            # Try a step
            h1 = layer(x, h, dt)
            
            # Try two half steps
            h_half = layer(x, h, dt/2)
            h2 = layer(x, h_half, dt/2)
            
            # Estimate error
            error = torch.abs(h1 - h2).max().item()
            
            if error < tol:
                # Accept step
                h = h2
                t += dt
                # Increase step size
                dt = min(dt * 1.5, 1.0 - t)
            else:
                # Reject step, decrease step size
                dt = dt * 0.5
            
            # Safety check
            if dt < 1e-6:
                break
        
        return h
    
    async def adapt(self, feedback: Dict[str, Any]) -> None:
        """Adapt network based on feedback."""
        if not self.is_initialized:
            return
        
        self.adaptation_count += 1
        
        # Extract feedback components
        loss = feedback.get('loss')
        gradients = feedback.get('gradients')
        performance = feedback.get('performance', {})
        
        # Adapt liquid time constants
        if 'stability' in performance:
            stability = torch.tensor(performance['stability'], device=self.device)
            for layer in self.liquid_layers:
                if isinstance(layer, LiquidTimeStep):
                    layer.adapt_time_constant(stability)
        
        # Adapt importance weights
        if gradients is not None:
            for i, layer in enumerate(self.liquid_layers):
                if isinstance(layer, AdaptiveLayer) and i < len(gradients):
                    layer.adapt(gradients[i])
        
        # Prune sparse connections if needed
        if self.adaptation_count % 100 == 0:
            for layer in self.liquid_layers:
                if isinstance(layer, SparseConnection):
                    layer.prune_connections()
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get current network state."""
        return {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'config': self.config.dict(),
            'metrics': {
                'forward_count': self.forward_count,
                'total_forward_time': self.total_forward_time,
                'adaptation_count': self.adaptation_count
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load network state."""
        # Load model state
        if 'model_state' in state:
            super().load_state_dict(state['model_state'])
        
        # Load optimizer state
        if 'optimizer_state' in state and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(state['optimizer_state'])
        
        # Load config
        if 'config' in state:
            self.config = NeuralConfig(**state['config'])
        
        # Load metrics
        if 'metrics' in state:
            metrics = state['metrics']
            self.forward_count = metrics.get('forward_count', 0)
            self.total_forward_time = metrics.get('total_forward_time', 0.0)
            self.adaptation_count = metrics.get('adaptation_count', 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_forward_time = (
            self.total_forward_time / self.forward_count
            if self.forward_count > 0 else 0.0
        )
        
        return {
            'forward_count': self.forward_count,
            'average_forward_time_ms': avg_forward_time * 1000,
            'adaptation_count': self.adaptation_count,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'device': str(self.device),
            'mixed_precision': self.config.mixed_precision
        }