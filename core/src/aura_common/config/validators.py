"""
⚙️ Configuration Validators
Validate configuration values and constraints.
"""

from typing import List, Optional, Any, Protocol
from pathlib import Path

from .base import AuraConfig
from ..errors import ConfigurationError, ValidationError


class ConfigValidator(Protocol):
    """Protocol for configuration validators."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate configuration and return errors."""
        ...


class EnvironmentValidator:
    """Validate environment-specific settings."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate environment configuration."""
        errors = []
        
        # Production checks
        if config.is_production():
            # Security must be enabled
            if not config.security.encryption_enabled:
                errors.append(
                    ValidationError(
                        "Encryption must be enabled in production",
                        field="security.encryption_enabled",
                        value=False,
                        constraint="required in production"
                    )
                )
            
            # Audit must be enabled
            if not config.security.audit_enabled:
                errors.append(
                    ValidationError(
                        "Audit logging must be enabled in production",
                        field="security.audit_enabled",
                        value=False,
                        constraint="required in production"
                    )
                )
            
            # Observability must be enabled
            if not config.observability.metrics_enabled:
                errors.append(
                    ValidationError(
                        "Metrics must be enabled in production",
                        field="observability.metrics_enabled",
                        value=False,
                        constraint="required in production"
                    )
                )
            
            # Shadow mode should be enabled for safety
            if not config.logging.shadow_mode_enabled:
                errors.append(
                    ValidationError(
                        "Shadow mode should be enabled in production",
                        field="logging.shadow_mode_enabled",
                        value=False,
                        constraint="recommended in production"
                    ).with_suggestion("Enable shadow mode for safe rollouts")
                )
        
        # Development checks
        if config.is_development():
            # Warn about performance features
            if config.observability.profiling_enabled:
                errors.append(
                    ValidationError(
                        "Profiling enabled in development",
                        field="observability.profiling_enabled",
                        value=True,
                        constraint="may impact performance"
                    ).with_suggestion("Disable profiling for better performance")
                )
        
        return errors


class PathValidator:
    """Validate file and directory paths."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate path configuration."""
        errors = []
        
        # Check shadow mode directory
        shadow_dir = config.logging.shadow_mode_dir
        if config.logging.shadow_mode_enabled:
            if not shadow_dir.exists():
                try:
                    shadow_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(
                        ValidationError(
                            f"Cannot create shadow mode directory: {shadow_dir}",
                            field="logging.shadow_mode_dir",
                            value=str(shadow_dir),
                            constraint="must be writable"
                        ).with_suggestion(f"Create directory: mkdir -p {shadow_dir}")
                    )
            elif not os.access(shadow_dir, os.W_OK):
                errors.append(
                    ValidationError(
                        f"Shadow mode directory not writable: {shadow_dir}",
                        field="logging.shadow_mode_dir",
                        value=str(shadow_dir),
                        constraint="must be writable"
                    )
                )
        
        return errors


class ResourceValidator:
    """Validate resource limits and constraints."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate resource configuration."""
        errors = []
        
        # Memory constraints
        total_memory_mb = config.memory.hot_cache_size
        if total_memory_mb > 10000:  # 10GB
            errors.append(
                ValidationError(
                    "Hot cache size exceeds recommended limit",
                    field="memory.hot_cache_size",
                    value=total_memory_mb,
                    constraint="recommended < 10000 MB"
                ).with_suggestion("Consider using tiered caching")
            )
        
        # Agent constraints
        if config.agents.max_agents > 50:
            errors.append(
                ValidationError(
                    "Maximum agents exceeds recommended limit",
                    field="agents.max_agents",
                    value=config.agents.max_agents,
                    constraint="recommended <= 50"
                ).with_suggestion("High agent count may impact consensus performance")
            )
        
        # TDA constraints
        if config.tda.max_points > 100000:
            if config.tda.preferred_engine == "python":
                errors.append(
                    ValidationError(
                        "Python TDA engine may be slow with large datasets",
                        field="tda.max_points",
                        value=config.tda.max_points,
                        constraint="recommended < 10000 for Python engine"
                    ).with_suggestion("Use GPU acceleration for large datasets")
                )
        
        return errors


class SecurityValidator:
    """Validate security configuration."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate security settings."""
        errors = []
        
        # Key rotation
        if config.security.key_rotation_days > 180:
            errors.append(
                ValidationError(
                    "Key rotation period exceeds security best practice",
                    field="security.key_rotation_days",
                    value=config.security.key_rotation_days,
                    constraint="recommended <= 180 days"
                ).with_suggestion("Consider 90-day rotation for better security")
            )
        
        # Algorithm validation
        weak_algorithms = ["HMAC", "RSA"]
        if config.security.signature_algorithm in weak_algorithms:
            errors.append(
                ValidationError(
                    f"Signature algorithm '{config.security.signature_algorithm}' may be weak",
                    field="security.signature_algorithm",
                    value=config.security.signature_algorithm,
                    constraint="consider stronger algorithms"
                ).with_suggestion("Use Ed25519 or post-quantum algorithms")
            )
        
        return errors


class FeatureFlagValidator:
    """Validate feature flag configuration."""
    
    def validate(self, config: AuraConfig) -> List[ValidationError]:
        """Validate feature flags."""
        errors = []
        
        # Check for conflicting flags
        if config.feature_flags.get("disable_tda") and config.feature_flags.get("gpu_tda"):
            errors.append(
                ValidationError(
                    "Conflicting feature flags: disable_tda and gpu_tda",
                    field="feature_flags",
                    value=config.feature_flags,
                    constraint="cannot enable GPU TDA when TDA is disabled"
                )
            )
        
        # Warn about experimental features in production
        experimental_flags = [
            "quantum_tda",
            "federated_learning",
            "neuromorphic_acceleration"
        ]
        
        if config.is_production():
            for flag in experimental_flags:
                if config.feature_flags.get(flag):
                    errors.append(
                        ValidationError(
                            f"Experimental feature '{flag}' enabled in production",
                            field=f"feature_flags.{flag}",
                            value=True,
                            constraint="experimental features not recommended in production"
                        ).with_suggestion("Test thoroughly in staging first")
                    )
        
        return errors


def validate_config(config: AuraConfig) -> None:
    """
    Validate configuration with all validators.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigurationError: If validation fails
    """
    validators = [
        EnvironmentValidator(),
        PathValidator(),
        ResourceValidator(),
        SecurityValidator(),
        FeatureFlagValidator(),
    ]
    
    all_errors = []
    for validator in validators:
        errors = validator.validate(config)
        all_errors.extend(errors)
    
    if all_errors:
        # Group errors by severity
        critical_errors = [e for e in all_errors if "required" in e.details.get("constraint", "")]
        warnings = [e for e in all_errors if e not in critical_errors]
        
        # Log warnings
        if warnings:
            from ..logging import get_logger
            logger = get_logger(__name__)
            for warning in warnings:
                logger.warning(
                    "Configuration warning",
                    field=warning.details.get("field"),
                    message=warning.message,
                    suggestions=warning.suggestions
                )
        
        # Raise on critical errors
        if critical_errors:
            error_messages = [f"- {e.message}" for e in critical_errors]
            raise ConfigurationError(
                f"Configuration validation failed:\n" + "\n".join(error_messages),
                details={"errors": [e.to_dict() for e in critical_errors]}
            )


# Import os for path checks
import os