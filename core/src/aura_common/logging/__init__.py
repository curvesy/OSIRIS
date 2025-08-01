"""
📝 AURA Logging Module
Structured logging with correlation IDs and OpenTelemetry integration.
"""

from .structured import AuraLogger, get_logger
from .correlation import get_correlation_id, set_correlation_id, with_correlation_id
from .shadow_mode import ShadowModeLogger, shadow_log

__all__ = [
    "AuraLogger",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id", 
    "with_correlation_id",
    "ShadowModeLogger",
    "shadow_log",
]