"""
🚦 Feature Flag Manager for Progressive Rollout
Atomic module for dynamic feature management with targeting and A/B testing
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger(__name__)

# Metrics
FLAG_EVALUATIONS = Counter('feature_flag_evaluations_total', 'Total flag evaluations', ['flag', 'result'])
FLAG_REFRESH_ERRORS = Counter('feature_flag_refresh_errors_total', 'Flag refresh errors')
ACTIVE_FLAGS = Gauge('feature_flags_active', 'Number of active feature flags')


class RolloutStrategy(Enum):
    """Rollout strategies"""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ATTRIBUTE_MATCH = "attribute_match"
    GRADUAL = "gradual"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool = False
    strategy: RolloutStrategy = RolloutStrategy.ALL_USERS
    percentage: float = 0.0
    user_list: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, float] = field(default_factory=dict)  # For A/B testing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if flag is within active date range"""
        now = datetime.now(timezone.utc)
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True


class FeatureFlagManager:
    """Manages feature flags with dynamic updates and targeting"""
    
    def __init__(self, redis_client=None, refresh_interval: int = 60):
        self.flags: Dict[str, FeatureFlag] = {}
        self.overrides: Dict[str, bool] = {}
        self.redis_client = redis_client
        self.refresh_interval = refresh_interval
        self._refresh_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
        
    async def initialize(self) -> None:
        """Initialize and start refresh loop"""
        await self._load_flags()
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Feature flag manager initialized", flags=len(self.flags))
        
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate if feature flag is enabled for given context"""
        # Check overrides first
        if flag_name in self.overrides:
            result = self.overrides[flag_name]
            FLAG_EVALUATIONS.labels(flag=flag_name, result=str(result)).inc()
            return result
            
        flag = self.flags.get(flag_name)
        if not flag or not flag.enabled or not flag.is_active():
            FLAG_EVALUATIONS.labels(flag=flag_name, result="false").inc()
            return False
            
        result = self._evaluate_flag(flag, context or {})
        FLAG_EVALUATIONS.labels(flag=flag_name, result=str(result)).inc()
        return result
        
    def get_variant(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get A/B test variant for user"""
        flag = self.flags.get(flag_name)
        if not flag or not flag.enabled or not flag.variants:
            return "control"
            
        # Use consistent hashing for variant assignment
        user_id = context.get("user_id", "anonymous") if context else "anonymous"
        hash_input = f"{flag_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Assign variant based on hash
        cumulative = 0.0
        for variant, percentage in flag.variants.items():
            cumulative += percentage
            if (hash_value % 100) < (cumulative * 100):
                return variant
                
        return "control"
        
    def add_override(self, flag_name: str, value: bool) -> None:
        """Add temporary override for testing"""
        self.overrides[flag_name] = value
        logger.info("Flag override added", flag=flag_name, value=value)
        
    def remove_override(self, flag_name: str) -> None:
        """Remove override"""
        self.overrides.pop(flag_name, None)
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for flag changes"""
        self._callbacks.append(callback)
        
    async def refresh_flags(self) -> None:
        """Manually refresh flags from storage"""
        try:
            await self._load_flags()
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(self.flags)
                except Exception as e:
                    logger.error("Callback error", error=str(e))
                    
        except Exception as e:
            FLAG_REFRESH_ERRORS.inc()
            logger.error("Failed to refresh flags", error=str(e))
            
    def _evaluate_flag(self, flag: FeatureFlag, context: Dict[str, Any]) -> bool:
        """Evaluate flag based on strategy"""
        if flag.strategy == RolloutStrategy.ALL_USERS:
            return True
            
        elif flag.strategy == RolloutStrategy.PERCENTAGE:
            user_id = context.get("user_id", "anonymous")
            hash_value = int(hashlib.md5(f"{flag.name}:{user_id}".encode()).hexdigest(), 16)
            return (hash_value % 100) < (flag.percentage * 100)
            
        elif flag.strategy == RolloutStrategy.USER_LIST:
            user_id = context.get("user_id")
            return user_id in flag.user_list if user_id else False
            
        elif flag.strategy == RolloutStrategy.ATTRIBUTE_MATCH:
            for attr, expected in flag.attributes.items():
                if context.get(attr) != expected:
                    return False
            return True
            
        elif flag.strategy == RolloutStrategy.GRADUAL:
            # Gradual rollout based on time
            if not flag.start_date:
                return True
            days_active = (datetime.now(timezone.utc) - flag.start_date).days
            target_percentage = min(days_active * 10, 100)  # 10% per day
            return self._evaluate_flag(
                FeatureFlag(
                    name=flag.name,
                    enabled=True,
                    strategy=RolloutStrategy.PERCENTAGE,
                    percentage=target_percentage / 100
                ),
                context
            )
            
        return False
        
    async def _load_flags(self) -> None:
        """Load flags from Redis or config"""
        if self.redis_client:
            try:
                data = await self.redis_client.get("feature_flags")
                if data:
                    flags_data = json.loads(data)
                    self.flags = {
                        name: FeatureFlag(**config)
                        for name, config in flags_data.items()
                    }
            except Exception as e:
                logger.error("Failed to load flags from Redis", error=str(e))
                
        # Fallback to default flags
        if not self.flags:
            self._load_default_flags()
            
        ACTIVE_FLAGS.set(len([f for f in self.flags.values() if f.enabled]))
        
    def _load_default_flags(self) -> None:
        """Load default feature flags"""
        self.flags = {
            "STREAMING_TDA": FeatureFlag(
                name="STREAMING_TDA",
                enabled=False,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0
            ),
            "KAFKA_EVENT_MESH": FeatureFlag(
                name="KAFKA_EVENT_MESH",
                enabled=False,
                strategy=RolloutStrategy.GRADUAL
            ),
            "DISTRIBUTED_TRACING": FeatureFlag(
                name="DISTRIBUTED_TRACING",
                enabled=True,
                strategy=RolloutStrategy.ALL_USERS
            )
        }
        
    async def _refresh_loop(self) -> None:
        """Periodic refresh of flags"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            await self.refresh_flags()
            
    async def close(self) -> None:
        """Cleanup resources"""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass