"""
🏗️ Base Schema Components - Foundation Classes

Core base models, utilities, and common functionality for all schemas:
- DateTime handling and serialization
- Base model configurations
- Common validators
- Utility functions
- Schema versioning support

Foundation for the entire schema system.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Protocol
from pydantic import BaseModel, Field, validator


# ============================================================================
# DATETIME UTILITIES - Consistent Temporal Handling
# ============================================================================

def utc_now() -> datetime:
    """Get current UTC datetime - USE THIS EVERYWHERE."""
    return datetime.now(timezone.utc)


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string for serialization."""
    return dt.isoformat()


def iso_to_datetime(iso_str: str) -> datetime:
    """Convert ISO 8601 string to datetime for deserialization."""
    # Handle both with and without 'Z' suffix
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1] + '+00:00'
    return datetime.fromisoformat(iso_str)


class DateTimeField(datetime):
    """
    Custom datetime field that serializes to ISO 8601.
    
    Ensures consistent datetime handling across all schemas
    with proper timezone awareness and serialization.
    
    Updated for Pydantic v2 compatibility.
    """
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return iso_to_datetime(v)
        raise ValueError('Invalid datetime format')
    
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """
        Pydantic v2 schema customization method.
        Replaces the deprecated __modify_schema__ method.
        """
        json_schema = handler(core_schema)
        json_schema.update({
            'type': 'string',
            'format': 'date-time',
            'example': '2025-01-26T10:30:00+00:00',
            'description': 'ISO 8601 datetime string with timezone information'
        })
        return json_schema


# ============================================================================
# BASE MODEL CONFIGURATIONS
# ============================================================================

class ImmutableBaseModel(BaseModel):
    """
    Base model with immutability enforced.
    
    All state models should inherit from this to ensure
    thread-safety and prevent race conditions.
    """
    
    class Config:
        # Enforce immutability - objects cannot be modified after creation
        allow_mutation = False
        # Validate assignments to catch mutation attempts
        validate_assignment = True
        # Use enum values for serialization
        use_enum_values = True
        # Enable arbitrary types for complex fields
        arbitrary_types_allowed = True


class MutableBaseModel(BaseModel):
    """
    Base model for mutable objects (builders, factories, etc.).
    
    Use sparingly - only for construction and utility classes.
    """
    
    class Config:
        # Allow mutation for builders and factories
        allow_mutation = True
        validate_assignment = True
        use_enum_values = True
        arbitrary_types_allowed = True


# ============================================================================
# SCHEMA VERSIONING SUPPORT
# ============================================================================

class VersionedSchema(ImmutableBaseModel):
    """
    Base class for all versioned schemas.
    
    Provides schema versioning support for long-lived workflows
    and backwards compatibility.
    """
    
    schema_version: str = Field(
        default="2.0",
        description="Schema version for compatibility and migration"
    )
    
    @validator('schema_version')
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        if not v or not isinstance(v, str):
            raise ValueError("Schema version must be a non-empty string")
        
        # Basic semantic version validation
        parts = v.split('.')
        if len(parts) < 2:
            raise ValueError("Schema version must be in format 'major.minor' or 'major.minor.patch'")
        
        try:
            for part in parts:
                int(part)
        except ValueError:
            raise ValueError("Schema version parts must be integers")
        
        return v
    
    def get_major_version(self) -> int:
        """Get major version number."""
        return int(self.schema_version.split('.')[0])
    
    def get_minor_version(self) -> int:
        """Get minor version number."""
        return int(self.schema_version.split('.')[1])
    
    def get_patch_version(self) -> int:
        """Get patch version number (0 if not specified)."""
        parts = self.schema_version.split('.')
        return int(parts[2]) if len(parts) > 2 else 0
    
    def is_compatible_with(self, other_version: str) -> bool:
        """Check if this schema is compatible with another version."""
        other_major = int(other_version.split('.')[0])
        other_minor = int(other_version.split('.')[1])
        
        # Same major version is compatible
        # Higher minor version is backwards compatible
        return (
            self.get_major_version() == other_major and
            self.get_minor_version() >= other_minor
        )


# ============================================================================
# GLOBAL ID SUPPORT
# ============================================================================

class GloballyIdentifiable(VersionedSchema):
    """
    Base class for entities with globally unique composite IDs.
    
    Ensures global uniqueness across distributed systems
    with full lineage tracking.
    """
    
    workflow_id: str = Field(..., description="Workflow that contains this entity")
    task_id: str = Field(..., description="Task that contains this entity")
    correlation_id: str = Field(..., description="End-to-end correlation identifier")
    
    def to_global_id(self) -> str:
        """Generate globally unique identifier with full context."""
        entity_id = getattr(self, 'entry_id', None) or getattr(self, 'action_id', None) or getattr(self, 'decision_id', None)
        if not entity_id:
            raise ValueError("Entity must have entry_id, action_id, or decision_id for global ID")
        return f"{self.workflow_id}:{self.task_id}:{entity_id}"
    
    def get_context_prefix(self) -> str:
        """Get the workflow:task context prefix."""
        return f"{self.workflow_id}:{self.task_id}"


# ============================================================================
# COMMON VALIDATORS
# ============================================================================

def validate_confidence_score(v: float) -> float:
    """Validate confidence scores are between 0.0 and 1.0."""
    if not isinstance(v, (int, float)):
        raise ValueError("Confidence score must be a number")
    if not 0.0 <= v <= 1.0:
        raise ValueError("Confidence score must be between 0.0 and 1.0")
    return float(v)


def validate_signature_format(v: str) -> str:
    """Validate cryptographic signature format."""
    if not v or not isinstance(v, str):
        raise ValueError("Signature must be a non-empty string")
    if len(v) < 32:
        raise ValueError("Signature must be at least 32 characters")
    return v


def validate_non_empty_string(v: str) -> str:
    """Validate string is non-empty."""
    if not v or not isinstance(v, str) or not v.strip():
        raise ValueError("Field must be a non-empty string")
    return v.strip()


def validate_uuid_format(v: str) -> str:
    """Validate UUID format."""
    try:
        uuid.UUID(v)
        return v
    except ValueError:
        raise ValueError("Invalid UUID format")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_entity_id(prefix: str = "entity") -> str:
    """Generate a unique entity ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_correlation_id() -> str:
    """Generate a correlation ID for tracing."""
    return f"corr_{uuid.uuid4().hex}"


def generate_workflow_id(prefix: str = "wf") -> str:
    """Generate a workflow ID."""
    timestamp = int(utc_now().timestamp())
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def generate_task_id(prefix: str = "task") -> str:
    """Generate a task ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def calculate_age_seconds(timestamp: datetime) -> float:
    """Calculate age in seconds from a timestamp."""
    return (utc_now() - timestamp).total_seconds()


def is_recent(timestamp: datetime, max_age_hours: int = 24) -> bool:
    """Check if a timestamp is recent (within max_age_hours)."""
    max_age_seconds = max_age_hours * 3600
    return calculate_age_seconds(timestamp) <= max_age_seconds


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# METADATA SUPPORT
# ============================================================================

class MetadataSupport(VersionedSchema):
    """
    Base class for entities that support metadata and classification.
    
    Provides common metadata fields for categorization,
    retention policies, and compliance.
    """
    
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization and search"
    )
    
    classification: Optional[str] = Field(
        None,
        description="Security classification (public, internal, confidential, restricted)"
    )
    
    retention_policy: Optional[str] = Field(
        None,
        description="Data retention policy identifier"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for extensibility"
    )
    
    @validator('classification')
    def validate_classification(cls, v):
        """Validate security classification."""
        if v is not None:
            valid_classifications = ['public', 'internal', 'confidential', 'restricted']
            if v.lower() not in valid_classifications:
                raise ValueError(f"Classification must be one of: {valid_classifications}")
            return v.lower()
        return v
    
    def add_tag(self, tag: str) -> None:
        """Add a tag (only for mutable instances)."""
        if hasattr(self, '__config__') and not self.__config__.allow_mutation:
            raise ValueError("Cannot modify immutable instance")
        if tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if entity has a specific tag."""
        return tag in self.tags
    
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


# ============================================================================
# TEMPORAL SUPPORT
# ============================================================================

class TemporalSupport(VersionedSchema):
    """
    Base class for entities that require temporal tracking and lifecycle management.
    
    Provides standardized temporal fields for creation, updates, expiration,
    and deadline management across all time-sensitive entities.
    
    Common temporal patterns:
    - Creation and update tracking
    - Event vs collection time distinction  
    - Expiration and deadline management
    - Duration and timeout tracking
    """
    
    # Core temporal tracking
    created_at: DateTimeField = Field(
        default_factory=utc_now,
        description="When this entity was created"
    )
    
    updated_at: DateTimeField = Field(
        default_factory=utc_now,
        description="When this entity was last updated"
    )
    
    # Event timing (when the actual event occurred vs when we recorded it)
    event_timestamp: Optional[DateTimeField] = Field(
        None,
        description="When the original event occurred (if different from creation)"
    )
    
    # Lifecycle management
    expiry_timestamp: Optional[DateTimeField] = Field(
        None,
        description="When this entity expires or becomes invalid"
    )
    
    deadline: Optional[DateTimeField] = Field(
        None,
        description="Deadline for processing or completion"
    )
    
    # Duration tracking
    duration_ms: Optional[float] = Field(
        None,
        description="Duration in milliseconds (for completed operations)",
        ge=0.0
    )
    
    timeout_ms: Optional[float] = Field(
        None,
        description="Timeout in milliseconds (for time-bounded operations)",
        ge=0.0
    )
    
    def is_expired(self) -> bool:
        """Check if the entity has expired."""
        if not self.expiry_timestamp:
            return False
        return utc_now() > self.expiry_timestamp
    
    def is_past_deadline(self) -> bool:
        """Check if the entity is past its deadline."""
        if not self.deadline:
            return False
        return utc_now() > self.deadline
    
    def get_age_seconds(self) -> float:
        """Get the age of this entity in seconds."""
        return calculate_age_seconds(self.created_at)
    
    def get_time_since_update_seconds(self) -> float:
        """Get seconds since last update."""
        return calculate_age_seconds(self.updated_at)
    
    def get_time_to_deadline_seconds(self) -> Optional[float]:
        """Get seconds until deadline (negative if past deadline)."""
        if not self.deadline:
            return None
        return (self.deadline - utc_now()).total_seconds()
    
    def get_time_to_expiry_seconds(self) -> Optional[float]:
        """Get seconds until expiry (negative if expired)."""
        if not self.expiry_timestamp:
            return None
        return (self.expiry_timestamp - utc_now()).total_seconds()
    
    def is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if entity was created recently."""
        return is_recent(self.created_at, max_age_hours)
    
    def is_recently_updated(self, max_age_hours: int = 1) -> bool:
        """Check if entity was updated recently."""
        return is_recent(self.updated_at, max_age_hours)
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get duration in seconds (converted from milliseconds)."""
        if self.duration_ms is None:
            return None
        return self.duration_ms / 1000.0
    
    def get_timeout_seconds(self) -> Optional[float]:
        """Get timeout in seconds (converted from milliseconds)."""
        if self.timeout_ms is None:
            return None
        return self.timeout_ms / 1000.0


# ============================================================================
# QUALITY METRICS SUPPORT
# ============================================================================

class QualityMetrics(GloballyIdentifiable, MetadataSupport):
    """
    Base class for entities that include quality and confidence metrics.
    
    Provides standardized quality assessment fields for evidence,
    decisions, actions, and other measurable entities.
    """
    
    confidence_score: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    
    quality_score: Optional[float] = Field(
        None,
        description="Quality assessment score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    
    reliability_score: Optional[float] = Field(
        None,
        description="Reliability assessment score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    
    source_credibility: Optional[float] = Field(
        None,
        description="Source credibility score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    
    @validator('confidence_score', 'quality_score', 'reliability_score', 'source_credibility')
    def validate_metric_scores(cls, v):
        """Validate all metric scores are within valid range."""
        if v is not None:
            return validate_confidence_score(v)
        return v
    
    def get_overall_quality(self) -> float:
        """Calculate overall quality score from available metrics."""
        scores = [
            self.confidence_score,
            self.quality_score,
            self.reliability_score,
            self.source_credibility
        ]
        
        # Filter out None values
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return self.confidence_score
        
        # Weighted average with confidence score having double weight
        weights = [2.0] + [1.0] * (len(valid_scores) - 1)
        weighted_sum = sum(score * weight for score, weight in zip(valid_scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if entity meets high quality threshold."""
        return self.get_overall_quality() >= threshold
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if entity meets reliability threshold."""
        return (self.reliability_score or self.confidence_score) >= threshold


# ============================================================================
# ERROR HANDLING
# ============================================================================

class SchemaValidationError(ValueError):
    """Custom exception for schema validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, field_value: Any = None):
        self.field_name = field_name
        self.field_value = field_value
        super().__init__(message)


class SchemaVersionError(Exception):
    """Custom exception for schema version compatibility errors."""
    
    def __init__(self, message: str, current_version: str, required_version: str):
        self.current_version = current_version
        self.required_version = required_version
        super().__init__(message)


# ============================================================================
# MIGRATION SUPPORT
# ============================================================================

class MigrationProtocol(Protocol):
    """Protocol for schema migration functions."""
    
    def migrate(self, old_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate data from one schema version to another."""
        ...


def create_migration_registry() -> Dict[str, MigrationProtocol]:
    """Create a registry for schema migrations."""
    return {}


# Export commonly used items
__all__ = [
    # DateTime utilities
    'utc_now', 'datetime_to_iso', 'iso_to_datetime', 'DateTimeField',
    
    # Base models
    'ImmutableBaseModel', 'MutableBaseModel', 'VersionedSchema',
    'GloballyIdentifiable', 'MetadataSupport', 'TemporalSupport', 'QualityMetrics',
    
    # Validators
    'validate_confidence_score', 'validate_signature_format',
    'validate_non_empty_string', 'validate_uuid_format',
    
    # Utilities
    'generate_entity_id', 'generate_correlation_id', 'generate_workflow_id',
    'generate_task_id', 'calculate_age_seconds', 'is_recent', 'truncate_string',
    
    # Exceptions
    'SchemaValidationError', 'SchemaVersionError',
    
    # Migration
    'MigrationProtocol', 'create_migration_registry'
]
