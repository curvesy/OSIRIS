"""
Feature Flag System - 2025 Production Standard
Runtime control of features with monitoring and A/B testing support
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)


class FeatureFlagType(str, Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    VARIANT = "variant"
    GRADUAL_ROLLOUT = "gradual_rollout"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    flag_type: FeatureFlagType
    enabled: bool = False
    percentage: float = 0.0
    variants: Dict[str, Any] = field(default_factory=dict)
    rollout_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    

@dataclass
class FeatureFlagEvaluation:
    """Result of feature flag evaluation"""
    flag_name: str
    enabled: bool
    variant: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlags:
    """
    Modern feature flag system with:
    - Runtime configuration
    - A/B testing support
    - Gradual rollouts
    - Monitoring integration
    - Remote configuration
    """
    
    def __init__(self, config_source: Optional[str] = None):
        self.flags: Dict[str, FeatureFlag] = {}
        self.config_source = config_source
        self._update_callbacks: List[Callable] = []
        self._evaluation_cache: Dict[str, FeatureFlagEvaluation] = {}
        self._initialized = False
        
        # Default flags
        self._register_default_flags()
        
    def _register_default_flags(self):
        """Register default feature flags"""
        # TDA Algorithm flags
        self.register_flag(FeatureFlag(
            name="tda.specseq_plus",
            flag_type=FeatureFlagType.BOOLEAN,
            enabled=True,
            metadata={"description": "Enable SpecSeq++ TDA algorithm"}
        ))
        
        self.register_flag(FeatureFlag(
            name="tda.simba_gpu",
            flag_type=FeatureFlagType.PERCENTAGE,
            enabled=True,
            percentage=100.0,
            metadata={"description": "GPU-accelerated SimBa algorithm"}
        ))
        
        self.register_flag(FeatureFlag(
            name="tda.neural_surveillance",
            flag_type=FeatureFlagType.GRADUAL_ROLLOUT,
            enabled=True,
            rollout_config={
                "start_percentage": 10.0,
                "target_percentage": 100.0,
                "increment_per_hour": 10.0,
                "started_at": datetime.now(timezone.utc)
            },
            metadata={"description": "Neural-enhanced TDA surveillance"}
        ))
        
        self.register_flag(FeatureFlag(
            name="tda.auto_fallback",
            flag_type=FeatureFlagType.BOOLEAN,
            enabled=True,
            metadata={"description": "Automatic fallback on TDA failure"}
        ))
        
        # Workflow flags
        self.register_flag(FeatureFlag(
            name="workflow.auto_recovery",
            flag_type=FeatureFlagType.BOOLEAN,
            enabled=True,
            metadata={"description": "Automatic workflow recovery from checkpoints"}
        ))
        
        self.register_flag(FeatureFlag(
            name="workflow.parallel_agents",
            flag_type=FeatureFlagType.VARIANT,
            enabled=True,
            variants={
                "sequential": 20.0,
                "parallel_2": 50.0,
                "parallel_all": 30.0
            },
            metadata={"description": "Agent execution strategy"}
        ))
        
        # Monitoring flags
        self.register_flag(FeatureFlag(
            name="monitoring.enhanced_tracing",
            flag_type=FeatureFlagType.PERCENTAGE,
            enabled=True,
            percentage=10.0,  # Sample 10% of requests
            metadata={"description": "Enhanced distributed tracing"}
        ))
        
    async def initialize(self):
        """Initialize feature flag system"""
        if self.config_source:
            await self._load_remote_config()
            
        # Start background updater
        asyncio.create_task(self._background_updater())
        
        self._initialized = True
        logger.info("Feature flag system initialized", 
                   flags_count=len(self.flags))
        
    def register_flag(self, flag: FeatureFlag):
        """Register a new feature flag"""
        self.flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")
        
    async def is_enabled(
        self, 
        flag_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            context: Optional context for evaluation (user_id, etc.)
            
        Returns:
            True if enabled, False otherwise
        """
        evaluation = await self.evaluate(flag_name, context)
        return evaluation.enabled
        
    async def evaluate(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureFlagEvaluation:
        """
        Evaluate a feature flag with full context.
        
        Args:
            flag_name: Name of the feature flag
            context: Optional context for evaluation
            
        Returns:
            FeatureFlagEvaluation with detailed results
        """
        if flag_name not in self.flags:
            return FeatureFlagEvaluation(
                flag_name=flag_name,
                enabled=False,
                reason="Flag not found"
            )
            
        flag = self.flags[flag_name]
        
        # Check cache
        cache_key = f"{flag_name}:{json.dumps(context or {}, sort_keys=True)}"
        if cache_key in self._evaluation_cache:
            cached = self._evaluation_cache[cache_key]
            if (datetime.now(timezone.utc) - cached.metadata["cached_at"]).seconds < 60:
                return cached
                
        # Evaluate based on flag type
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            evaluation = self._evaluate_boolean(flag)
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            evaluation = self._evaluate_percentage(flag, context)
        elif flag.flag_type == FeatureFlagType.VARIANT:
            evaluation = self._evaluate_variant(flag, context)
        elif flag.flag_type == FeatureFlagType.GRADUAL_ROLLOUT:
            evaluation = self._evaluate_gradual_rollout(flag, context)
        else:
            evaluation = FeatureFlagEvaluation(
                flag_name=flag_name,
                enabled=False,
                reason="Unknown flag type"
            )
            
        # Cache result
        evaluation.metadata["cached_at"] = datetime.now(timezone.utc)
        self._evaluation_cache[cache_key] = evaluation
        
        # Log evaluation
        logger.debug("Feature flag evaluated",
                    flag=flag_name,
                    enabled=evaluation.enabled,
                    variant=evaluation.variant)
        
        return evaluation
        
    def _evaluate_boolean(self, flag: FeatureFlag) -> FeatureFlagEvaluation:
        """Evaluate boolean flag"""
        return FeatureFlagEvaluation(
            flag_name=flag.name,
            enabled=flag.enabled,
            reason="Boolean flag"
        )
        
    def _evaluate_percentage(
        self, 
        flag: FeatureFlag,
        context: Optional[Dict[str, Any]]
    ) -> FeatureFlagEvaluation:
        """Evaluate percentage-based flag"""
        if not flag.enabled:
            return FeatureFlagEvaluation(
                flag_name=flag.name,
                enabled=False,
                reason="Flag disabled"
            )
            
        # Use context to determine if in percentage
        if context and "user_id" in context:
            # Deterministic hash based on user_id
            hash_value = hash(f"{flag.name}:{context['user_id']}")
            in_percentage = (hash_value % 100) < flag.percentage
        else:
            # Random sampling
            import random
            in_percentage = random.random() * 100 < flag.percentage
            
        return FeatureFlagEvaluation(
            flag_name=flag.name,
            enabled=in_percentage,
            reason=f"Percentage rollout: {flag.percentage}%"
        )
        
    def _evaluate_variant(
        self,
        flag: FeatureFlag,
        context: Optional[Dict[str, Any]]
    ) -> FeatureFlagEvaluation:
        """Evaluate variant-based flag"""
        if not flag.enabled or not flag.variants:
            return FeatureFlagEvaluation(
                flag_name=flag.name,
                enabled=False,
                reason="Flag disabled or no variants"
            )
            
        # Determine variant based on context
        if context and "user_id" in context:
            # Deterministic variant selection
            hash_value = hash(f"{flag.name}:{context['user_id']}")
            
            # Convert percentages to ranges
            total = 0.0
            for variant, percentage in flag.variants.items():
                total += percentage
                if (hash_value % 100) < total:
                    return FeatureFlagEvaluation(
                        flag_name=flag.name,
                        enabled=True,
                        variant=variant,
                        reason=f"Variant selected: {variant}"
                    )
        else:
            # Random variant selection
            import random
            rand = random.random() * 100
            total = 0.0
            
            for variant, percentage in flag.variants.items():
                total += percentage
                if rand < total:
                    return FeatureFlagEvaluation(
                        flag_name=flag.name,
                        enabled=True,
                        variant=variant,
                        reason=f"Variant selected: {variant}"
                    )
                    
        return FeatureFlagEvaluation(
            flag_name=flag.name,
            enabled=False,
            reason="No variant matched"
        )
        
    def _evaluate_gradual_rollout(
        self,
        flag: FeatureFlag,
        context: Optional[Dict[str, Any]]
    ) -> FeatureFlagEvaluation:
        """Evaluate gradual rollout flag"""
        if not flag.enabled or not flag.rollout_config:
            return FeatureFlagEvaluation(
                flag_name=flag.name,
                enabled=False,
                reason="Flag disabled or no rollout config"
            )
            
        config = flag.rollout_config
        started_at = config["started_at"]
        hours_elapsed = (datetime.now(timezone.utc) - started_at).total_seconds() / 3600
        
        # Calculate current percentage
        current_percentage = min(
            config["target_percentage"],
            config["start_percentage"] + (hours_elapsed * config["increment_per_hour"])
        )
        
        # Update flag percentage
        flag.percentage = current_percentage
        
        # Evaluate as percentage
        return self._evaluate_percentage(flag, context)
        
    async def enable(self, flag_name: str):
        """Enable a feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = True
            self.flags[flag_name].updated_at = datetime.now(timezone.utc)
            await self._notify_update(flag_name, "enabled")
            
    async def disable(self, flag_name: str):
        """Disable a feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = False
            self.flags[flag_name].updated_at = datetime.now(timezone.utc)
            await self._notify_update(flag_name, "disabled")
            
    async def set_percentage(self, flag_name: str, percentage: float):
        """Set percentage for a flag"""
        if flag_name in self.flags:
            self.flags[flag_name].percentage = max(0.0, min(100.0, percentage))
            self.flags[flag_name].updated_at = datetime.now(timezone.utc)
            await self._notify_update(flag_name, f"percentage={percentage}")
            
    async def set_variant_distribution(
        self,
        flag_name: str,
        variants: Dict[str, float]
    ):
        """Set variant distribution for a flag"""
        if flag_name in self.flags:
            # Normalize to 100%
            total = sum(variants.values())
            if total > 0:
                normalized = {k: (v / total) * 100 for k, v in variants.items()}
                self.flags[flag_name].variants = normalized
                self.flags[flag_name].updated_at = datetime.now(timezone.utc)
                await self._notify_update(flag_name, f"variants updated")
                
    def on_update(self, callback: Callable):
        """Register callback for flag updates"""
        self._update_callbacks.append(callback)
        
    async def _notify_update(self, flag_name: str, change: str):
        """Notify listeners of flag update"""
        logger.info(f"Feature flag updated: {flag_name} - {change}")
        
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(flag_name, change)
                else:
                    callback(flag_name, change)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
                
    async def _load_remote_config(self):
        """Load configuration from remote source"""
        # Implementation depends on your config source
        # Could be: API, S3, ConfigMap, etc.
        pass
        
    async def _background_updater(self):
        """Background task to update flags from remote source"""
        while True:
            try:
                if self.config_source:
                    await self._load_remote_config()
                    
                # Clean up old cache entries
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
                self._evaluation_cache = {
                    k: v for k, v in self._evaluation_cache.items()
                    if v.metadata.get("cached_at", datetime.min) > cutoff
                }
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in background updater: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes
                
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all flag configurations"""
        return {
            name: {
                "type": flag.flag_type.value,
                "enabled": flag.enabled,
                "percentage": flag.percentage,
                "variants": flag.variants,
                "metadata": flag.metadata,
                "updated_at": flag.updated_at.isoformat()
            }
            for name, flag in self.flags.items()
        }
        
    @asynccontextmanager
    async def override(self, overrides: Dict[str, bool]):
        """
        Temporarily override feature flags.
        
        Usage:
            async with feature_flags.override({"tda.simba_gpu": False}):
                # Code runs with override
        """
        original_states = {}
        
        try:
            # Apply overrides
            for flag_name, enabled in overrides.items():
                if flag_name in self.flags:
                    original_states[flag_name] = self.flags[flag_name].enabled
                    self.flags[flag_name].enabled = enabled
                    
            yield
            
        finally:
            # Restore original states
            for flag_name, original in original_states.items():
                self.flags[flag_name].enabled = original