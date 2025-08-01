"""
Validation utilities for AURA Intelligence.

Provides configuration and environment validation functions.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from ..config import AURASettings
from .logging import get_logger

logger = get_logger(__name__)


def validate_config(config_path: Optional[Path] = None) -> tuple[bool, list[str]]:
    """
    Validate AURA configuration.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Try to load configuration
        if config_path and config_path.exists():
            os.environ["AURA_ENV_FILE"] = str(config_path)
        
        settings = AURASettings.from_env()
        
        # Run built-in validation
        warnings = settings.validate_configuration()
        if warnings:
            errors.extend(warnings)
        
        # Additional validation checks
        if settings.is_production:
            # Production-specific checks
            if not settings.observability.prometheus_enabled:
                errors.append("Prometheus should be enabled in production")
            
            if settings.deployment.production_replicas < 2:
                errors.append("Production should have at least 2 replicas")
            
            if not settings.security.enable_encryption:
                errors.append("Encryption should be enabled in production")
        
        # Check for required directories
        if not settings.data_dir.exists():
            try:
                settings.data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create data directory: {e}")
        
        if not settings.logs_dir.exists():
            try:
                settings.logs_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create logs directory: {e}")
        
        logger.info(
            "Configuration validation completed",
            errors_count=len(errors),
            environment=settings.environment.value
        )
        
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")
    except Exception as e:
        errors.append(f"Configuration validation failed: {str(e)}")
    
    return len(errors) == 0, errors


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate the runtime environment.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    # Python version check
    python_version = sys.version_info
    if python_version < (3, 11):
        errors.append(
            f"Python 3.11+ required, found {python_version.major}.{python_version.minor}"
        )
    elif python_version < (3, 13):
        warnings.append(
            f"Python 3.13+ recommended for latest features, found {python_version.major}.{python_version.minor}"
        )
    
    # Check for required environment variables in production
    if os.getenv("AURA_ENVIRONMENT") in ("production", "enterprise"):
        required_vars = [
            "AURA_API__OPENAI_API_KEY",
            "AURA_SECURITY__JWT_SECRET_KEY",
            "AURA_SECURITY__ENCRYPTION_KEY",
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Required environment variable not set: {var}")
    
    # Check for container environment
    if Path("/.dockerenv").exists() or os.getenv("KUBERNETES_SERVICE_HOST"):
        logger.info("Running in container environment")
        
        # Container-specific checks
        if not os.getenv("AURA_DEPLOYMENT__CONTAINER_TAG"):
            warnings.append("Container tag not specified, using 'latest'")
    
    # Check for required system commands
    required_commands = ["git", "docker"]
    for cmd in required_commands:
        if os.system(f"which {cmd} > /dev/null 2>&1") != 0:
            warnings.append(f"Optional command not found: {cmd}")
    
    # Check available memory
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / 1024 / 1024
                    if available_gb < 2:
                        warnings.append(
                            f"Low available memory: {available_gb:.1f}GB (2GB+ recommended)"
                        )
                    break
    except Exception:
        # Not on Linux or can't read meminfo
        pass
    
    # Check disk space
    try:
        import shutil
        home_dir = Path.home()
        total, used, free = shutil.disk_usage(home_dir)
        free_gb = free / (1024 ** 3)
        if free_gb < 10:
            warnings.append(
                f"Low disk space: {free_gb:.1f}GB free (10GB+ recommended)"
            )
    except Exception:
        pass
    
    # Log results
    logger.info(
        "Environment validation completed",
        errors_count=len(errors),
        warnings_count=len(warnings),
        python_version=f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    )
    
    # Combine errors and warnings for output
    all_issues = errors + [f"Warning: {w}" for w in warnings]
    
    return len(errors) == 0, all_issues


def validate_deployment_readiness() -> tuple[bool, dict[str, Any]]:
    """
    Validate deployment readiness.
    
    Returns:
        Tuple of (is_ready, readiness_report)
    """
    report = {
        "config_valid": False,
        "environment_valid": False,
        "services_ready": {},
        "errors": [],
        "warnings": []
    }
    
    # Validate configuration
    config_valid, config_errors = validate_config()
    report["config_valid"] = config_valid
    if not config_valid:
        report["errors"].extend(config_errors)
    
    # Validate environment
    env_valid, env_issues = validate_environment()
    report["environment_valid"] = env_valid
    report["errors"].extend([i for i in env_issues if not i.startswith("Warning:")])
    report["warnings"].extend([i for i in env_issues if i.startswith("Warning:")])
    
    # Check service dependencies
    try:
        settings = AURASettings.from_env()
        
        # Check database connectivity
        if settings.integration.database_url:
            # TODO: Implement actual database check
            report["services_ready"]["database"] = True
        
        # Check Redis connectivity
        if settings.integration.redis_enabled:
            # TODO: Implement actual Redis check
            report["services_ready"]["redis"] = True
        
        # Check message queue
        if settings.integration.message_queue_url:
            # TODO: Implement actual message queue check
            report["services_ready"]["message_queue"] = True
        
    except Exception as e:
        report["errors"].append(f"Service check failed: {str(e)}")
    
    # Overall readiness
    is_ready = (
        report["config_valid"] and
        report["environment_valid"] and
        len(report["errors"]) == 0
    )
    
    logger.info(
        "Deployment readiness check completed",
        is_ready=is_ready,
        errors_count=len(report["errors"]),
        warnings_count=len(report["warnings"])
    )
    
    return is_ready, report