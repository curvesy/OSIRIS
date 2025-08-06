"""
Neural Metrics for AURA Intelligence
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class NeuralMetrics:
    """
    Metrics tracking for neural network operations.
    """
    inference_count: int = 0
    total_inference_time: float = 0.0
    error_count: int = 0
    accuracy_scores: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    
    def record_inference(self, duration: float, accuracy: Optional[float] = None):
        """Record an inference operation."""
        self.inference_count += 1
        self.total_inference_time += duration
        self.latencies.append(duration)
        
        if accuracy is not None:
            self.accuracy_scores.append(accuracy)
            
    def record_error(self):
        """Record an error."""
        self.error_count += 1
        
    def get_average_latency(self) -> float:
        """Get average inference latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
        
    def get_average_accuracy(self) -> float:
        """Get average accuracy."""
        if not self.accuracy_scores:
            return 0.0
        return sum(self.accuracy_scores) / len(self.accuracy_scores)
        
    def get_error_rate(self) -> float:
        """Get error rate."""
        total = self.inference_count + self.error_count
        if total == 0:
            return 0.0
        return self.error_count / total
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "error_count": self.error_count,
            "average_latency": self.get_average_latency(),
            "average_accuracy": self.get_average_accuracy(),
            "error_rate": self.get_error_rate()
        }
        
    def reset(self):
        """Reset all metrics."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.error_count = 0
        self.accuracy_scores.clear()
        self.latencies.clear()