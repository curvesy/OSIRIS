"""
🧠 Neural Observability System - Phase 1, Step 3
Complete sensory awareness for the digital organism using latest 2025 patterns.

Professional modular architecture with separation of concerns:
- Core observability configuration and initialization
- OpenTelemetry integration with AI semantic conventions  
- LangSmith 2.0 streaming traces and evaluation
- Prometheus metrics with AI/LLM specific patterns
- Structured logging with cryptographic signatures
- Knowledge graph event recording for memory consolidation
- Real-time streaming and correlation context

Latest 2025 Features:
✅ OpenTelemetry AI Semantic Conventions (June 2025)
✅ LangSmith 2.0 streaming traces and workflow visualization
✅ Prometheus AI metrics with cost tracking and performance
✅ Structured logging with trace correlation
✅ Knowledge graph integration for learning loops
✅ Bio-inspired organism health monitoring
"""

from .config import (
    ObservabilityConfig,
    create_development_config,
    create_production_config,
)
from .core import NeuralObservabilityCore
try:
    from .opentelemetry_integration import OpenTelemetryManager
    from .tracing import get_tracer
    
    # Create compatibility functions
    def create_tracer(name: str):
        """Create a tracer instance - compatibility function."""
        from opentelemetry import trace
        return trace.get_tracer(name)
    
    def create_meter(name: str):
        """Create a meter instance - compatibility function.""" 
        from opentelemetry import metrics
        return metrics.get_meter(name)
except ImportError:
    # OpenTelemetry is optional
    OpenTelemetryManager = None
from .langsmith_integration import LangSmithManager
from .prometheus_metrics import PrometheusMetricsManager
from .structured_logging import StructuredLoggingManager
from .knowledge_graph import KnowledgeGraphManager
from .health_monitor import OrganismHealthMonitor, HealthMetrics
from .context_managers import (
    ObservabilityContext,
    AgentContext,
    LLMUsageContext,
)
from .metrics import metrics_collector as MetricsCollector
from .tracing import TracingContext
from .layer import ObservabilityLayer
from .neural_metrics import NeuralMetrics

__all__ = [
    # Configuration
    "ObservabilityConfig",
    "create_development_config",
    "create_production_config",

    # Core system
    "NeuralObservabilityCore",

    # Component managers
    "OpenTelemetryManager",
    "LangSmithManager",
    "PrometheusMetricsManager",
    "StructuredLoggingManager",
    "KnowledgeGraphManager",
    "OrganismHealthMonitor",
    "HealthMetrics",

    # Context utilities
    "ObservabilityContext",
    "AgentContext",
    "LLMUsageContext",
    
    # Metrics and tracing
    "MetricsCollector",
    "TracingContext",
    "ObservabilityLayer",
    "NeuralMetrics",
]

# Version info
__version__ = "2025.7.27"
__description__ = "Neural Observability System - Complete sensory awareness for digital organisms"
__version__ = "2025.7.27"
__author__ = "AURA Intelligence - Neural Observability Team"
__description__ = "Complete sensory system for digital organisms"
