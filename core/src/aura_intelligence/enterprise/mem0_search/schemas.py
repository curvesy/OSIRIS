"""
📋 Pydantic Models for mem0 Search API

Request/response schemas for /analyze /search /memory endpoints.
Includes Signature, Event, and Memory cluster models.

Based on partab.md: "Pydantic models (Signature, Event…)" specification.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class EventType(str, Enum):
    """Event types for topological signatures."""
    SIGNATURE = "signature"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    CLUSTER = "cluster"
    ANALYSIS = "analysis"


class MemoryTier(str, Enum):
    """Memory tier for search targeting."""
    HOT = "hot"
    SEMANTIC = "semantic"
    BOTH = "both"


class SimilarityMetric(str, Enum):
    """Similarity metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


# Base signature model
class TopologicalSignatureModel(BaseModel):
    """Pydantic model for topological signatures."""
    
    hash: str = Field(..., description="Unique signature hash")
    betti_0: int = Field(..., ge=0, description="0-dimensional Betti number")
    betti_1: int = Field(..., ge=0, description="1-dimensional Betti number")
    betti_2: int = Field(..., ge=0, description="2-dimensional Betti number")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score [0,1]")
    timestamp: datetime = Field(..., description="Signature creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Event model
class EventModel(BaseModel):
    """Pydantic model for events."""
    
    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    agent_id: str = Field(..., description="Agent that generated the event")
    timestamp: datetime = Field(..., description="Event timestamp")
    signature: TopologicalSignatureModel = Field(..., description="Associated signature")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Search and analysis request/response models
class AnalyzeRequest(BaseModel):
    """Request for topological analysis."""
    
    signature: TopologicalSignatureModel = Field(..., description="Signature to analyze")
    agent_id: str = Field(..., description="Requesting agent ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Analysis context")
    include_similar: bool = Field(default=True, description="Include similar signatures")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return")


class SignatureMatch(BaseModel):
    """Signature similarity match result."""
    
    signature: TopologicalSignatureModel = Field(..., description="Matched signature")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    memory_tier: MemoryTier = Field(..., description="Source memory tier")
    last_accessed: datetime = Field(..., description="Last access timestamp")
    access_count: int = Field(..., ge=0, description="Total access count")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalyzeResponse(BaseModel):
    """Response for topological analysis."""
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    input_signature: TopologicalSignatureModel = Field(..., description="Input signature")
    anomaly_detected: bool = Field(..., description="Whether anomaly was detected")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    similar_signatures: List[SignatureMatch] = Field(default_factory=list, description="Similar signatures found")
    memory_stored: bool = Field(..., description="Whether signature was stored in memory")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchRequest(BaseModel):
    """Request for memory search."""
    
    query_signature: Optional[TopologicalSignatureModel] = Field(None, description="Query signature for similarity search")
    query_vector: Optional[List[float]] = Field(None, description="Query vector for direct search")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    event_type: Optional[EventType] = Field(None, description="Filter by event type")
    time_range_start: Optional[datetime] = Field(None, description="Start of time range filter")
    time_range_end: Optional[datetime] = Field(None, description="End of time range filter")
    memory_tier: MemoryTier = Field(default=MemoryTier.BOTH, description="Target memory tier")
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.COSINE, description="Similarity metric")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    include_metadata: bool = Field(default=True, description="Include signature metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryCluster(BaseModel):
    """Memory cluster information."""
    
    cluster_id: str = Field(..., description="Unique cluster identifier")
    centroid_signature: TopologicalSignatureModel = Field(..., description="Cluster centroid signature")
    member_count: int = Field(..., ge=1, description="Number of signatures in cluster")
    cluster_score: float = Field(..., ge=0.0, le=1.0, description="Cluster quality score")
    creation_time: datetime = Field(..., description="Cluster creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResponse(BaseModel):
    """Response for memory search."""
    
    search_id: str = Field(..., description="Unique search identifier")
    query_processed: bool = Field(..., description="Whether query was successfully processed")
    matches: List[SignatureMatch] = Field(default_factory=list, description="Matching signatures")
    clusters: List[MemoryCluster] = Field(default_factory=list, description="Related memory clusters")
    total_matches: int = Field(..., ge=0, description="Total number of matches found")
    search_time_ms: float = Field(..., ge=0.0, description="Search time in milliseconds")
    memory_tiers_searched: List[MemoryTier] = Field(default_factory=list, description="Memory tiers that were searched")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryRequest(BaseModel):
    """Request for memory operations."""
    
    operation: str = Field(..., description="Memory operation (store, retrieve, update, delete)")
    signature: Optional[TopologicalSignatureModel] = Field(None, description="Signature for operation")
    signature_hash: Optional[str] = Field(None, description="Signature hash for retrieval/update/delete")
    agent_id: str = Field(..., description="Agent performing the operation")
    event_type: EventType = Field(default=EventType.SIGNATURE, description="Event type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")
    memory_tier: MemoryTier = Field(default=MemoryTier.HOT, description="Target memory tier")
    force_update: bool = Field(default=False, description="Force update even if exists")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryResponse(BaseModel):
    """Response for memory operations."""
    
    operation_id: str = Field(..., description="Unique operation identifier")
    operation: str = Field(..., description="Performed operation")
    success: bool = Field(..., description="Whether operation was successful")
    signature_hash: Optional[str] = Field(None, description="Affected signature hash")
    memory_tier: MemoryTier = Field(..., description="Memory tier used")
    operation_time_ms: float = Field(..., ge=0.0, description="Operation time in milliseconds")
    message: str = Field(..., description="Operation result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Health check models
class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for individual components."""
    
    component: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: Optional[float] = Field(None, description="Component response time")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Component metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemHealthResponse(BaseModel):
    """System-wide health check response."""
    
    overall_status: HealthStatus = Field(..., description="Overall system health")
    check_timestamp: datetime = Field(..., description="Health check timestamp")
    components: List[ComponentHealth] = Field(default_factory=list, description="Individual component health")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System-wide metrics")
    uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Error response model
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
