"""
🔍 Evidence Models - Typed Evidence Content

Comprehensive evidence collection and management with:
- Typed evidence content using Union types
- Cryptographically signed evidence entries
- Quality metrics and validation
- Global references and lineage
- Temporal consistency

The foundation for trustworthy agent reasoning.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import Field, validator

try:
    from .base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from .enums import EvidenceType, SignatureAlgorithm
    from .crypto import get_crypto_provider
    from .tracecontext import TraceContextMixin
except ImportError:
    # Fallback for direct import (testing/isolation)
    from base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from enums import EvidenceType, SignatureAlgorithm
    from crypto import get_crypto_provider
    from tracecontext import TraceContextMixin


# ============================================================================
# TYPED EVIDENCE CONTENT - Union Types for Different Evidence
# ============================================================================

class LogEvidence(GloballyIdentifiable, MetadataSupport):
    """Evidence from log entries with structured data."""
    
    log_level: str = Field(..., description="Log level (DEBUG, INFO, WARN, ERROR, FATAL)")
    log_text: str = Field(..., description="The actual log message")
    logger_name: str = Field(..., description="Name of the logger that generated this")
    log_timestamp: DateTimeField = Field(..., description="Original log timestamp")
    source_file: Optional[str] = Field(None, description="Source file if available")
    line_number: Optional[int] = Field(None, description="Line number if available")
    thread_id: Optional[str] = Field(None, description="Thread ID if available")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="Structured log data")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'TRACE']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class MetricEvidence(QualityMetrics):
    """Evidence from metrics and measurements."""
    
    metric_name: str = Field(..., description="Name of the metric")
    metric_value: float = Field(..., description="Numeric value of the metric")
    metric_unit: str = Field(..., description="Unit of measurement")
    metric_type: str = Field(..., description="Type of metric (counter, gauge, histogram)")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels/tags")
    measurement_timestamp: DateTimeField = Field(..., description="When the metric was measured")
    aggregation_window: Optional[str] = Field(None, description="Aggregation window if applicable")
    percentile: Optional[float] = Field(None, description="Percentile if applicable")
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        valid_types = ['counter', 'gauge', 'histogram', 'summary']
        if v.lower() not in valid_types:
            raise ValueError(f"Metric type must be one of: {valid_types}")
        return v.lower()


class PatternEvidence(QualityMetrics):
    """Evidence from pattern recognition and analysis."""
    
    pattern_id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of pattern (anomaly, trend, cycle, etc.)")
    pattern_score: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence score")
    features: Dict[str, float] = Field(..., description="Feature vector that defines the pattern")
    model_version: str = Field(..., description="Version of the model that detected the pattern")
    training_data_hash: Optional[str] = Field(None, description="Hash of training data")
    detection_algorithm: str = Field(..., description="Algorithm used for detection")
    false_positive_rate: Optional[float] = Field(None, description="Estimated false positive rate")


class PredictionEvidence(QualityMetrics):
    """Evidence from predictive models and forecasts."""
    
    prediction_id: str = Field(..., description="Unique identifier for the prediction")
    predicted_value: Any = Field(..., description="The predicted value or outcome")
    prediction_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    prediction_horizon: str = Field(..., description="Time horizon for the prediction")
    model_name: str = Field(..., description="Name of the predictive model")
    model_version: str = Field(..., description="Version of the predictive model")
    input_features: Dict[str, Any] = Field(..., description="Input features used for prediction")
    uncertainty_bounds: Optional[Dict[str, float]] = Field(None, description="Uncertainty bounds")
    baseline_comparison: Optional[float] = Field(None, description="Comparison to baseline model")


class CorrelationEvidence(QualityMetrics):
    """Evidence from correlation analysis."""
    
    correlation_id: str = Field(..., description="Unique identifier for the correlation")
    correlation_type: str = Field(..., description="Type of correlation (pearson, spearman, etc.)")
    correlation_coefficient: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient")
    p_value: Optional[float] = Field(None, description="Statistical p-value")
    sample_size: int = Field(..., description="Number of data points in correlation")
    time_window: str = Field(..., description="Time window for correlation analysis")
    variables: List[str] = Field(..., description="Variables involved in correlation")
    causal_direction: Optional[str] = Field(None, description="Suspected causal direction")


class ObservationEvidence(QualityMetrics):
    """Evidence from direct system observations."""
    
    observation_type: str = Field(..., description="Type of observation")
    observed_value: Any = Field(..., description="The observed value or state")
    observation_method: str = Field(..., description="Method used for observation")
    sensor_id: Optional[str] = Field(None, description="ID of sensor or monitoring system")
    measurement_accuracy: Optional[float] = Field(None, description="Measurement accuracy")
    environmental_conditions: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Environmental context"
    )


class ExternalEvidence(QualityMetrics):
    """Evidence from external sources and APIs."""
    
    source_name: str = Field(..., description="Name of the external source")
    source_url: Optional[str] = Field(None, description="URL of the external source")
    api_version: Optional[str] = Field(None, description="API version if applicable")
    data_format: str = Field(..., description="Format of the external data")
    retrieval_method: str = Field(..., description="Method used to retrieve data")
    authentication_method: Optional[str] = Field(None, description="Authentication method used")
    rate_limit_info: Optional[Dict[str, Any]] = Field(None, description="Rate limiting information")
    data_freshness: Optional[str] = Field(None, description="How fresh the external data is")


# Union type for all evidence content
EvidenceContent = Union[
    LogEvidence,
    MetricEvidence,
    PatternEvidence,
    PredictionEvidence,
    CorrelationEvidence,
    ObservationEvidence,
    ExternalEvidence
]


# ============================================================================
# EVIDENCE REFERENCE - Global Lineage Tracking
# ============================================================================

class EvidenceReference(GloballyIdentifiable):
    """Reference to evidence with full lineage for audit trails."""
    
    entry_id: str = Field(..., description="Unique entry identifier")
    agent_id: str = Field(..., description="Agent that created the evidence")
    event_timestamp: DateTimeField = Field(..., description="When the evidence was created")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    
    def __str__(self) -> str:
        return f"Evidence[{self.evidence_type.value}]({self.to_global_id()})"


# ============================================================================
# DOSSIER ENTRY - Main Evidence Container
# ============================================================================

class DossierEntry(
    GloballyIdentifiable,
    MetadataSupport,
    TemporalSupport,
    TraceContextMixin
):
    """
    Cryptographically signed evidence entry with full audit trail.
    
    The main container for all evidence collected by agents with:
    - Typed evidence content
    - Cryptographic signatures
    - Quality metrics
    - Temporal consistency
    - Global lineage tracking
    """
    
    # ========================================================================
    # IDENTITY
    # ========================================================================
    entry_id: str = Field(..., description="Globally unique entry identifier")
    
    # ========================================================================
    # CRYPTOGRAPHIC AUTHENTICATION
    # ========================================================================
    signing_agent_id: str = Field(..., description="Agent that created and signed this evidence")
    agent_public_key: str = Field(..., description="Public key of the signing agent")
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.HMAC_SHA256,
        description="Cryptographic algorithm used for signing"
    )
    content_signature: str = Field(..., description="Cryptographic signature of the evidence content")
    signature_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the evidence was signed"
    )
    
    # ========================================================================
    # SOURCE INFORMATION
    # ========================================================================
    source: str = Field(..., description="Source of the information")
    collection_method: str = Field(..., description="How the evidence was collected")
    source_reliability: float = Field(..., ge=0.0, le=1.0, description="Reliability of the source")
    
    # ========================================================================
    # TYPED EVIDENCE CONTENT
    # ========================================================================
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    content: EvidenceContent = Field(..., description="Typed evidence content")
    summary: str = Field(..., description="Human-readable summary of the evidence")
    
    # ========================================================================
    # QUALITY METRICS
    # ========================================================================
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    freshness: float = Field(..., ge=0.0, le=1.0, description="How recent/relevant the evidence is")
    completeness: float = Field(default=1.0, ge=0.0, le=1.0, description="How complete the evidence is")
    
    # ========================================================================
    # TEMPORAL INFORMATION
    # ========================================================================
    collection_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When this evidence was collected"
    )
    event_timestamp: Optional[DateTimeField] = Field(None, description="When the original event occurred")
    expiry_timestamp: Optional[DateTimeField] = Field(None, description="When this evidence expires")
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    related_entries: List[EvidenceReference] = Field(
        default_factory=list,
        description="References to related evidence entries"
    )
    contradicts: List[EvidenceReference] = Field(
        default_factory=list,
        description="References to contradictory evidence"
    )
    supports: List[EvidenceReference] = Field(
        default_factory=list,
        description="References to supporting evidence"
    )
    derived_from: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence this was derived from"
    )
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    @validator('confidence', 'source_reliability', 'freshness', 'completeness')
    def validate_scores(cls, v):
        return validate_confidence_score(v)
    
    @validator('content_signature')
    def validate_signature_format(cls, v):
        return validate_signature_format(v)
    
    # ========================================================================
    # CRYPTOGRAPHIC METHODS
    # ========================================================================
    def sign_content(self, private_key: str, algorithm: Optional[SignatureAlgorithm] = None) -> str:
        """Generate cryptographic signature of the evidence content."""
        sig_algorithm = algorithm or self.signature_algorithm
        content_bytes = self._get_canonical_content().encode('utf-8')
        provider = get_crypto_provider(sig_algorithm)
        return provider.sign(content_bytes, private_key)
    
    def verify_signature(self, private_key: str) -> bool:
        """Verify the evidence signature."""
        content_bytes = self._get_canonical_content().encode('utf-8')
        provider = get_crypto_provider(self.signature_algorithm)
        return provider.verify(content_bytes, self.content_signature, private_key)
    
    def _get_canonical_content(self) -> str:
        """Get canonical string representation of content for signing."""
        try:
            from .base import datetime_to_iso
        except ImportError:
            from base import datetime_to_iso
        timestamp_iso = datetime_to_iso(self.collection_timestamp)
        return f"{self.evidence_type.value}:{self.content.json()}:{timestamp_iso}"
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    def get_weighted_score(self) -> float:
        """Get a weighted quality score combining all quality metrics."""
        return (
            (self.confidence * 0.4) +
            (self.source_reliability * 0.3) +
            (self.freshness * 0.2) +
            (self.completeness * 0.1)
        )
    
    def is_expired(self) -> bool:
        """Check if the evidence has expired."""
        if not self.expiry_timestamp:
            return False
        return utc_now() > self.expiry_timestamp
    
    def get_reference(self) -> EvidenceReference:
        """Get a reference to this evidence entry."""
        try:
            from .base import datetime_to_iso
        except ImportError:
            from base import datetime_to_iso
        return EvidenceReference(
            workflow_id=self.workflow_id,
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            entry_id=self.entry_id,
            agent_id=self.signing_agent_id,
            event_timestamp=self.collection_timestamp,
            evidence_type=self.evidence_type,
            schema_version=self.schema_version
        )
    
    def __str__(self) -> str:
        return f"Evidence[{self.evidence_type.value}]({self.to_global_id()}) by {self.signing_agent_id}"


# Export public interface
__all__ = [
    # Evidence content types
    'LogEvidence', 'MetricEvidence', 'PatternEvidence', 'PredictionEvidence',
    'CorrelationEvidence', 'ObservationEvidence', 'ExternalEvidence',
    
    # Union type
    'EvidenceContent',
    
    # Main models
    'EvidenceReference', 'DossierEntry'
]
