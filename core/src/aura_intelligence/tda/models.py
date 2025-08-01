"""
🔥 Production-Grade TDA Models
Enterprise Pydantic models for TDA operations with validation and metrics.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from enum import Enum


class TDAAlgorithm(str, Enum):
    """Supported TDA algorithms."""
    SPECSEQ_PLUS_PLUS = "specseq++"
    SIMBA_GPU = "simba_gpu"
    NEURAL_SURVEILLANCE = "neural_surveillance"
    QUANTUM_TDA = "quantum_tda"
    STREAMING_TDA = "streaming_tda"


class DataFormat(str, Enum):
    """Supported input data formats."""
    POINT_CLOUD = "point_cloud"
    DISTANCE_MATRIX = "distance_matrix"
    SIMPLICIAL_COMPLEX = "simplicial_complex"
    TIME_SERIES = "time_series"
    GRAPH = "graph"


class TDARequest(BaseModel):
    """
    Enterprise-grade TDA computation request.
    
    Validates input data and parameters for production TDA analysis.
    """
    
    # Core request data
    data: Union[List[List[float]], List[List[List[float]]], Dict[str, Any]] = Field(
        ..., 
        description="Input data for TDA analysis"
    )
    
    algorithm: TDAAlgorithm = Field(
        TDAAlgorithm.SPECSEQ_PLUS_PLUS,
        description="TDA algorithm to use"
    )
    
    data_format: DataFormat = Field(
        DataFormat.POINT_CLOUD,
        description="Format of input data"
    )
    
    # Algorithm parameters
    max_dimension: int = Field(
        2, 
        ge=0, 
        le=10,
        description="Maximum homology dimension to compute"
    )
    
    max_edge_length: Optional[float] = Field(
        None,
        gt=0.0,
        description="Maximum edge length for filtration"
    )
    
    resolution: float = Field(
        0.01,
        gt=0.0,
        le=1.0,
        description="Resolution for persistence computation"
    )
    
    # Performance parameters
    use_gpu: bool = Field(
        True,
        description="Enable GPU acceleration if available"
    )
    
    batch_size: int = Field(
        1024,
        ge=1,
        le=10000,
        description="Batch size for GPU processing"
    )
    
    timeout_seconds: int = Field(
        300,
        ge=1,
        le=3600,
        description="Maximum computation time in seconds"
    )
    
    # Metadata
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique request identifier"
    )
    
    priority: Literal["low", "medium", "high", "critical"] = Field(
        "medium",
        description="Request priority level"
    )
    
    client_id: Optional[str] = Field(
        None,
        description="Client identifier for tracking"
    )
    
    @field_validator('data')
    @classmethod
    def validate_data_structure(cls, v, info):
        """Validate data structure based on format."""
        data_format = info.data.get('data_format') if info.data else None
        
        if data_format == DataFormat.POINT_CLOUD:
            if not isinstance(v, list) or not v:
                raise ValueError("Point cloud data must be non-empty list")
            
            # Check if all points have same dimension
            if isinstance(v[0], list):
                dim = len(v[0])
                if not all(isinstance(point, list) and len(point) == dim for point in v):
                    raise ValueError("All points must have same dimension")
        
        elif data_format == DataFormat.DISTANCE_MATRIX:
            if not isinstance(v, list) or not v:
                raise ValueError("Distance matrix must be non-empty list")
            
            # Check if matrix is square
            n = len(v)
            if not all(isinstance(row, list) and len(row) == n for row in v):
                raise ValueError("Distance matrix must be square")
        
        return v
    
    @model_validator(mode='after')
    def validate_algorithm_compatibility(self):
        """Validate algorithm and data format compatibility."""
        # GPU algorithms require compatible data formats
        if self.algorithm == TDAAlgorithm.SIMBA_GPU and not self.use_gpu:
            raise ValueError("SimBa GPU algorithm requires GPU acceleration")
        
        # Streaming algorithms require time series data
        if self.algorithm == TDAAlgorithm.STREAMING_TDA and self.data_format != DataFormat.TIME_SERIES:
            raise ValueError("Streaming TDA requires time series data format")
        
        return self


class PersistenceDiagram(BaseModel):
    """Persistence diagram for a single homology dimension."""
    
    dimension: int = Field(
        ...,
        ge=0,
        description="Homology dimension"
    )
    
    intervals: List[List[float]] = Field(
        ...,
        description="Birth-death intervals as [birth, death] pairs"
    )
    
    @field_validator('intervals')
    @classmethod
    def validate_intervals(cls, v):
        """Validate persistence intervals."""
        for interval in v:
            if len(interval) != 2:
                raise ValueError("Each interval must have exactly 2 values [birth, death]")
            
            birth, death = interval
            if birth > death:
                raise ValueError(f"Birth time {birth} cannot be greater than death time {death}")
        
        return v


class TDAMetrics(BaseModel):
    """Performance and quality metrics for TDA computation."""
    
    # Performance metrics
    computation_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total computation time in milliseconds"
    )
    
    gpu_utilization_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="GPU utilization percentage during computation"
    )
    
    memory_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="Peak memory usage in megabytes"
    )
    
    # Quality metrics
    numerical_stability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical stability score (0-1)"
    )
    
    approximation_error: Optional[float] = Field(
        None,
        ge=0.0,
        description="Approximation error for approximate algorithms"
    )
    
    # Algorithm-specific metrics
    simplices_processed: int = Field(
        ...,
        ge=0,
        description="Number of simplices processed"
    )
    
    filtration_steps: int = Field(
        ...,
        ge=0,
        description="Number of filtration steps"
    )
    
    # Benchmarking
    speedup_factor: Optional[float] = Field(
        None,
        ge=1.0,
        description="Speedup factor compared to baseline"
    )
    
    accuracy_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Accuracy compared to ground truth"
    )


class TDAResponse(BaseModel):
    """
    Enterprise-grade TDA computation response.
    
    Contains results, metrics, and metadata for production use.
    """
    
    # Request metadata
    request_id: str = Field(
        ...,
        description="Original request identifier"
    )
    
    algorithm_used: TDAAlgorithm = Field(
        ...,
        description="Algorithm that was actually used"
    )
    
    # Results
    persistence_diagrams: List[PersistenceDiagram] = Field(
        ...,
        description="Persistence diagrams for each dimension"
    )
    
    betti_numbers: List[int] = Field(
        ...,
        description="Betti numbers for each dimension"
    )
    
    # Advanced results
    persistence_landscapes: Optional[Dict[int, List[List[float]]]] = Field(
        None,
        description="Persistence landscapes for each dimension"
    )
    
    bottleneck_distances: Optional[Dict[str, float]] = Field(
        None,
        description="Bottleneck distances to reference diagrams"
    )
    
    # Metrics and performance
    metrics: TDAMetrics = Field(
        ...,
        description="Performance and quality metrics"
    )
    
    # Status and metadata
    status: Literal["success", "partial", "failed", "timeout"] = Field(
        ...,
        description="Computation status"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if computation failed"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages during computation"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    
    # Enterprise features
    audit_trail: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit trail for compliance"
    )
    
    resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed resource usage statistics"
    )
    
    @model_validator(mode='after')
    def validate_diagrams_consistency(self):
        """Validate consistency between diagrams and betti numbers."""
        if len(self.persistence_diagrams) != len(self.betti_numbers):
            raise ValueError("Number of persistence diagrams must match number of Betti numbers")
        
        for i, (diagram, betti) in enumerate(zip(self.persistence_diagrams, self.betti_numbers)):
            if diagram.dimension != i:
                raise ValueError(f"Diagram dimension {diagram.dimension} doesn't match index {i}")
        
        return self


class TDAConfiguration(BaseModel):
    """Production TDA engine configuration."""
    
    # Engine settings
    default_algorithm: TDAAlgorithm = Field(
        TDAAlgorithm.SPECSEQ_PLUS_PLUS,
        description="Default algorithm to use"
    )
    
    enable_gpu: bool = Field(
        True,
        description="Enable GPU acceleration globally"
    )
    
    max_concurrent_requests: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum concurrent TDA requests"
    )
    
    # Performance settings
    default_timeout_seconds: int = Field(
        300,
        ge=1,
        le=3600,
        description="Default timeout for TDA computations"
    )
    
    memory_limit_gb: float = Field(
        16.0,
        gt=0.0,
        description="Memory limit in gigabytes"
    )
    
    # Quality settings
    min_numerical_stability: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Minimum required numerical stability"
    )
    
    enable_benchmarking: bool = Field(
        True,
        description="Enable automatic benchmarking"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(
        True,
        description="Enable Prometheus metrics collection"
    )
    
    metrics_port: int = Field(
        8080,
        ge=1024,
        le=65535,
        description="Port for metrics endpoint"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Logging level"
    )


class TDABenchmarkResult(BaseModel):
    """Results from TDA algorithm benchmarking."""

    algorithm: TDAAlgorithm = Field(
        ...,
        description="Algorithm that was benchmarked"
    )

    dataset_name: str = Field(
        ...,
        description="Name of benchmark dataset"
    )

    dataset_size: int = Field(
        ...,
        ge=1,
        description="Size of benchmark dataset"
    )

    # Performance results
    avg_computation_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average computation time in milliseconds"
    )

    std_computation_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation of computation time"
    )

    speedup_vs_baseline: float = Field(
        ...,
        ge=0.0,
        description="Speedup factor compared to baseline algorithm"
    )

    # Accuracy results
    accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Accuracy score compared to ground truth"
    )

    bottleneck_distance_error: float = Field(
        ...,
        ge=0.0,
        description="Average bottleneck distance error"
    )

    # Resource usage
    peak_memory_mb: float = Field(
        ...,
        ge=0.0,
        description="Peak memory usage in megabytes"
    )

    gpu_utilization_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Average GPU utilization percentage"
    )

    # Metadata
    benchmark_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When benchmark was run"
    )

    hardware_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Hardware information for reproducibility"
    )
