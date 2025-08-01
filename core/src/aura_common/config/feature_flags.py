"""
⚙️ Feature Flag Management
Dynamic feature toggles for safe rollouts.
"""

from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import json
from pathlib import Path

from ..logging import get_logger

logger = get_logger(__name__)


class FeatureState(Enum):
    """Feature flag states."""
    DISABLED = auto()
    ENABLED = auto()
    PERCENTAGE = auto()  # Percentage rollout
    BUCKET = auto()      # User bucket testing


@dataclass
class FeatureFlag:
    """
    Feature flag definition.
    
    Attributes:
        name: Unique flag name
        description: What this feature does
        default_state: Default state
        percentage: Rollout percentage (0-100)
        buckets: User buckets for A/B testing
        metadata: Additional metadata
    """
    name: str
    description: str
    default_state: FeatureState = FeatureState.DISABLED
    percentage: float = 0.0
    buckets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_enabled_for(self, user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled for a user."""
        if self.default_state == FeatureState.ENABLED:
            return True
        
        if self.default_state == FeatureState.DISABLED:
            return False
        
        if self.default_state == FeatureState.PERCENTAGE:
            if user_id:
                # Consistent hashing for user
                hash_value = hash(f"{self.name}:{user_id}") % 100
                return hash_value < self.percentage
            return False
        
        if self.default_state == FeatureState.BUCKET:
            if user_id and self.buckets:
                # Check if user in test bucket
                bucket = self._get_user_bucket(user_id)
                return bucket in self.buckets
            return False
        
        return False
    
    def _get_user_bucket(self, user_id: str) -> str:
        """Get user's bucket for A/B testing."""
        # Simple bucketing based on user ID hash
        buckets = ["control", "test_a", "test_b", "test_c"]
        bucket_index = hash(f"{self.name}:{user_id}") % len(buckets)
        return buckets[bucket_index]


class FeatureManager:
    """
    Centralized feature flag management.
    
    Features:
    - Dynamic flag updates
    - Percentage rollouts
    - A/B testing support
    - Audit logging
    """
    
    def __init__(self, flags_file: Optional[Path] = None):
        """
        Initialize feature manager.
        
        Args:
            flags_file: Optional file to load flags from
        """
        self.flags: Dict[str, FeatureFlag] = {}
        self.flags_file = flags_file
        self._change_callbacks: List[Callable[[str, bool], None]] = []
        
        # Load default flags
        self._load_default_flags()
        
        # Load from file if provided
        if flags_file and flags_file.exists():
            self.load_from_file(flags_file)
    
    def _load_default_flags(self) -> None:
        """Load default feature flags."""
        default_flags = [
            # Core features
            FeatureFlag(
                name="shadow_mode",
                description="Enable shadow mode logging for A/B testing",
                default_state=FeatureState.ENABLED
            ),
            FeatureFlag(
                name="gpu_tda",
                description="Enable GPU acceleration for TDA",
                default_state=FeatureState.ENABLED
            ),
            FeatureFlag(
                name="collective_intelligence",
                description="Enable collective intelligence system",
                default_state=FeatureState.ENABLED
            ),
            
            # Advanced features
            FeatureFlag(
                name="federated_learning",
                description="Enable federated learning capabilities",
                default_state=FeatureState.DISABLED
            ),
            FeatureFlag(
                name="quantum_tda",
                description="Enable quantum TDA algorithms",
                default_state=FeatureState.DISABLED
            ),
            FeatureFlag(
                name="carbon_aware_scheduling",
                description="Enable carbon-aware workload scheduling",
                default_state=FeatureState.DISABLED
            ),
            
            # Experimental features
            FeatureFlag(
                name="neuromorphic_acceleration",
                description="Enable neuromorphic hardware acceleration",
                default_state=FeatureState.DISABLED
            ),
            FeatureFlag(
                name="homomorphic_encryption",
                description="Enable homomorphic encryption for privacy",
                default_state=FeatureState.DISABLED
            ),
            
            # Rollout examples
            FeatureFlag(
                name="new_ui",
                description="New UI experience",
                default_state=FeatureState.PERCENTAGE,
                percentage=10.0  # 10% rollout
            ),
            FeatureFlag(
                name="advanced_analytics",
                description="Advanced analytics dashboard",
                default_state=FeatureState.BUCKET,
                buckets=["test_a"]  # Only test_a bucket
            ),
        ]
        
        for flag in default_flags:
            self.register(flag)
    
    def register(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self.flags[flag.name] = flag
        logger.info(
            "Feature flag registered",
            flag=flag.name,
            state=flag.default_state.name,
            description=flag.description
        )
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        default: bool = False
    ) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            flag_name: Feature flag name
            user_id: Optional user ID for percentage/bucket rollouts
            default: Default value if flag not found
            
        Returns:
            Whether feature is enabled
        """
        flag = self.flags.get(flag_name)
        
        if not flag:
            logger.warning(
                "Unknown feature flag",
                flag=flag_name,
                default=default
            )
            return default
        
        enabled = flag.is_enabled_for(user_id)
        
        # Log flag evaluation
        logger.debug(
            "Feature flag evaluated",
            flag=flag_name,
            user_id=user_id,
            enabled=enabled,
            state=flag.default_state.name
        )
        
        return enabled
    
    def set_enabled(self, flag_name: str, enabled: bool) -> None:
        """Enable or disable a feature flag."""
        flag = self.flags.get(flag_name)
        
        if not flag:
            # Create new flag
            flag = FeatureFlag(
                name=flag_name,
                description=f"Dynamically created flag: {flag_name}",
                default_state=FeatureState.ENABLED if enabled else FeatureState.DISABLED
            )
            self.register(flag)
        else:
            # Update existing flag
            old_state = flag.default_state
            flag.default_state = FeatureState.ENABLED if enabled else FeatureState.DISABLED
            
            logger.info(
                "Feature flag updated",
                flag=flag_name,
                old_state=old_state.name,
                new_state=flag.default_state.name
            )
        
        # Notify callbacks
        for callback in self._change_callbacks:
            try:
                callback(flag_name, enabled)
            except Exception as e:
                logger.error(
                    "Feature flag callback failed",
                    flag=flag_name,
                    error=str(e)
                )
    
    def set_percentage(self, flag_name: str, percentage: float) -> None:
        """Set percentage rollout for a feature."""
        flag = self.flags.get(flag_name)
        
        if not flag:
            flag = FeatureFlag(
                name=flag_name,
                description=f"Percentage rollout: {flag_name}",
                default_state=FeatureState.PERCENTAGE,
                percentage=percentage
            )
            self.register(flag)
        else:
            flag.default_state = FeatureState.PERCENTAGE
            flag.percentage = max(0.0, min(100.0, percentage))
            
            logger.info(
                "Feature flag percentage updated",
                flag=flag_name,
                percentage=flag.percentage
            )
    
    def set_buckets(self, flag_name: str, buckets: List[str]) -> None:
        """Set bucket testing for a feature."""
        flag = self.flags.get(flag_name)
        
        if not flag:
            flag = FeatureFlag(
                name=flag_name,
                description=f"Bucket testing: {flag_name}",
                default_state=FeatureState.BUCKET,
                buckets=buckets
            )
            self.register(flag)
        else:
            flag.default_state = FeatureState.BUCKET
            flag.buckets = buckets
            
            logger.info(
                "Feature flag buckets updated",
                flag=flag_name,
                buckets=buckets
            )
    
    def on_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for flag changes."""
        self._change_callbacks.append(callback)
    
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags and their states."""
        return {
            name: {
                "description": flag.description,
                "state": flag.default_state.name,
                "percentage": flag.percentage,
                "buckets": flag.buckets,
                "metadata": flag.metadata
            }
            for name, flag in self.flags.items()
        }
    
    def save_to_file(self, path: Optional[Path] = None) -> None:
        """Save flags to file."""
        path = path or self.flags_file
        if not path:
            return
        
        data = self.get_all_flags()
        path.write_text(json.dumps(data, indent=2))
        logger.info("Feature flags saved", path=str(path))
    
    def load_from_file(self, path: Path) -> None:
        """Load flags from file."""
        try:
            data = json.loads(path.read_text())
            
            for name, flag_data in data.items():
                state = FeatureState[flag_data["state"]]
                flag = FeatureFlag(
                    name=name,
                    description=flag_data["description"],
                    default_state=state,
                    percentage=flag_data.get("percentage", 0.0),
                    buckets=flag_data.get("buckets", []),
                    metadata=flag_data.get("metadata", {})
                )
                self.register(flag)
            
            logger.info(
                "Feature flags loaded",
                path=str(path),
                count=len(data)
            )
            
        except Exception as e:
            logger.error(
                "Failed to load feature flags",
                path=str(path),
                error=str(e)
            )


# Global feature manager
_feature_manager = FeatureManager()


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
    default: bool = False
) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        flag_name: Feature flag name
        user_id: Optional user ID for rollouts
        default: Default if flag not found
        
    Returns:
        Whether feature is enabled
        
    Example:
        ```python
        if is_feature_enabled("gpu_tda"):
            engine = GPUTDAEngine()
        else:
            engine = CPUTDAEngine()
        ```
    """
    return _feature_manager.is_enabled(flag_name, user_id, default)