"""
Neural Networks Package for AURA Intelligence.

This package contains advanced neural network implementations including:
- Liquid Neural Networks (LNN) for continuous-time dynamics
- Edge-optimized models for tactical deployment
- Context-aware integration with memory and knowledge systems
- Memory hooks for decision tracking
- Neuro-symbolic integration components
"""

from .lnn import (
    LiquidNeuralNetwork,
    LiquidCell, 
    LiquidLayer,
    ContinuousTimeRNN,
    EdgeLNN,
    LNNConfig,
    LNNMetrics
)

# Context integration - graceful import
try:
    from .context_integration import (
        ContextAwareLNN,
        ContextWindow
    )
    _context_available = True
except ImportError:
    _context_available = False
    ContextAwareLNN = None
    ContextWindow = None

# Memory hooks - graceful import
try:
    from .memory_hooks import (
        LNNMemoryHooks,
        MemoryEvent
    )
    _memory_hooks_available = True
except ImportError:
    _memory_hooks_available = False
    LNNMemoryHooks = None
    MemoryEvent = None

# TDA components - graceful import (not yet implemented)
_tda_available = False
TopologicalDataAnalyzer = None
PersistenceDiagram = None
TDAConfig = None
TDAMetrics = None

__all__ = [
    # LNN components
    "LiquidNeuralNetwork",
    "LiquidCell", 
    "LiquidLayer",
    "ContinuousTimeRNN",
    "EdgeLNN",
    "LNNConfig",
    "LNNMetrics",
]

# Add context components if available
if _context_available:
    __all__.extend([
        "ContextAwareLNN",
        "ContextWindow"
    ])

# Add memory hooks if available
if _memory_hooks_available:
    __all__.extend([
        "LNNMemoryHooks",
        "MemoryEvent"
    ])

# Add TDA components when available
if _tda_available:
    __all__.extend([
        "TopologicalDataAnalyzer",
        "PersistenceDiagram",
        "TDAConfig",
        "TDAMetrics"
    ])