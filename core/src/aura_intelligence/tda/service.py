"""
TDA Service Module for AURA Intelligence

Provides service interface for TDA operations.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime


class TDAServiceRequest(BaseModel):
    """Request model for TDA service."""
    data: List[List[float]] = Field(..., description="Input data matrix")
    algorithm: str = Field("specseq", description="TDA algorithm to use")
    max_dimension: int = Field(2, description="Maximum homology dimension")
    config: Optional[Dict[str, Any]] = Field(None, description="Algorithm configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "algorithm": "specseq",
                "max_dimension": 2,
                "config": {"resolution": 0.01}
            }
        }


class TDAServiceResponse(BaseModel):
    """Response model for TDA service."""
    request_id: str
    timestamp: datetime
    algorithm: str
    anomaly_score: float = Field(..., ge=0, le=1)
    betti_numbers: List[int]
    computation_time_s: float
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T00:00:00Z",
                "algorithm": "specseq",
                "anomaly_score": 0.75,
                "betti_numbers": [1, 2, 0],
                "computation_time_s": 0.15,
                "metadata": {"n_points": 100, "gpu_used": False}
            }
        }


class TDAService:
    """
    Service interface for TDA operations.
    """
    
    def __init__(self):
        from .unified_engine import UnifiedTDAEngine
        self.engine = UnifiedTDAEngine()
        
    async def analyze(self, request: TDAServiceRequest) -> TDAServiceResponse:
        """
        Analyze data using TDA.
        
        Args:
            request: TDA service request
            
        Returns:
            TDA service response
        """
        import uuid
        
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Run TDA analysis
        result = await self.engine.analyze(
            data,
            algorithm=request.algorithm,
            max_dimension=request.max_dimension,
            config=request.config
        )
        
        # Create response
        return TDAServiceResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            algorithm=request.algorithm,
            anomaly_score=result.get("anomaly_score", 0.0),
            betti_numbers=result.get("betti_numbers", []),
            computation_time_s=result.get("computation_time_s", 0.0),
            metadata=result.get("metadata", {})
        )
        
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "status": "healthy",
            "service": "TDA Service",
            "algorithms_available": ["specseq", "simba", "neural", "deterministic"]
        }
