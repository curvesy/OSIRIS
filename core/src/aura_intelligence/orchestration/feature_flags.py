"""
🚩 Orchestration Feature Flags

Manages feature flags for progressive rollout of orchestration capabilities.
Allows enabling/disabling features based on deployment environment and scale.

Key Features:
- Environment-based feature control
- Progressive rollout capabilities
- Safe fallback mechanisms
- Runtime feature toggling
"""

import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

class FeatureFlag(Enum):
    """Available feature flags"""
    # Phase 1: Semantic Foundation
    SEMANTIC_ORCHESTRATION = "semantic_orchestration"
    LANGGRAPH_INTEGRATION = "langgraph_integration"
    TDA_INTEGRATION = "tda_integration"
    
    # Phase 2: Durable Execution
    TEMPORAL_WORKFLOWS = "temporal_workflows"
    SAGA_PATTERNS = "saga_patterns"
    HYBRID_CHECKPOINTING = "hybrid_checkpointing"
    POSTGRESQL_CHECKPOINTING = "postgresql_checkpointing"
    CROSS_THREAD_MEMORY = "cross_thread_memory"
    
    # Phase 3: Distributed Scaling (Behind flags for startup)
    RAY_SERVE_ORCHESTRATION = "ray_serve_orchestration"
    CREWAI_FLOWS = "crewai_flows"
    DISTRIBUTED_COORDINATION = "distributed_coordination"
    AUTO_SCALING = "auto_scaling"
    
    # Phase 4: Production Excellence (Future)
    EVENT_DRIVEN_ORCHESTRATION = "event_driven_orchestration"
    ADVANCED_PATTERN_MATCHING = "advanced_pattern_matching"
    CONSENSUS_ORCHESTRATION = "consensus_orchestration"

@dataclass
class FeatureConfig:
    """Configuration for a feature flag"""
    enabled: bool
    rollout_percentage: float = 100.0
    environment_restrictions: Optional[list] = None
    dependencies: Optional[list] = None
    description: str = ""

class OrchestrationFeatureFlags:
    """
    Manages orchestration feature flags for progressive rollout
    """
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.features = self._initialize_default_features()
        self._load_environment_overrides()
    
    def _initialize_default_features(self) -> Dict[FeatureFlag, FeatureConfig]:
        """Initialize default feature configurations"""
        return {
            # Phase 1: Always enabled (core functionality)
            FeatureFlag.SEMANTIC_ORCHESTRATION: FeatureConfig(
                enabled=True,
                description="Core semantic orchestration capabilities"
            ),
            FeatureFlag.LANGGRAPH_INTEGRATION: FeatureConfig(
                enabled=True,
                description="LangGraph StateGraph integration"
            ),
            FeatureFlag.TDA_INTEGRATION: FeatureConfig(
                enabled=True,
                description="TDA context integration"
            ),
            
            # Phase 2: Enabled for startup (essential durability)
            FeatureFlag.TEMPORAL_WORKFLOWS: FeatureConfig(
                enabled=True,
                description="Temporal.io durable workflows"
            ),
            FeatureFlag.SAGA_PATTERNS: FeatureConfig(
                enabled=True,
                description="Saga pattern compensation"
            ),
            FeatureFlag.HYBRID_CHECKPOINTING: FeatureConfig(
                enabled=True,
                description="Hybrid checkpoint management"
            ),
            FeatureFlag.POSTGRESQL_CHECKPOINTING: FeatureConfig(
                enabled=True,
                description="PostgreSQL-based checkpointing"
            ),
            FeatureFlag.CROSS_THREAD_MEMORY: FeatureConfig(
                enabled=True,
                description="Cross-thread memory and learning"
            ),
            
            # Phase 3: Disabled by default (distributed scaling)
            FeatureFlag.RAY_SERVE_ORCHESTRATION: FeatureConfig(
                enabled=False,
                environment_restrictions=["production", "staging"],
                dependencies=[FeatureFlag.SEMANTIC_ORCHESTRATION],
                description="Ray Serve distributed agent deployments"
            ),
            FeatureFlag.CREWAI_FLOWS: FeatureConfig(
                enabled=False,
                environment_restrictions=["production", "staging"],
                dependencies=[FeatureFlag.SEMANTIC_ORCHESTRATION],
                description="CrewAI Flows hierarchical coordination"
            ),
            FeatureFlag.DISTRIBUTED_COORDINATION: FeatureConfig(
                enabled=False,
                environment_restrictions=["production"],
                dependencies=[
                    FeatureFlag.RAY_SERVE_ORCHESTRATION,
                    FeatureFlag.CREWAI_FLOWS,
                    FeatureFlag.HYBRID_CHECKPOINTING
                ],
                description="Cross-system distributed coordination"
            ),
            FeatureFlag.AUTO_SCALING: FeatureConfig(
                enabled=False,
                environment_restrictions=["production"],
                dependencies=[FeatureFlag.RAY_SERVE_ORCHESTRATION],
                description="Automatic scaling based on load"
            ),
            
            # Phase 4: Future features (disabled)
            FeatureFlag.EVENT_DRIVEN_ORCHESTRATION: FeatureConfig(
                enabled=False,
                description="Event-driven semantic coordination"
            ),
            FeatureFlag.ADVANCED_PATTERN_MATCHING: FeatureConfig(
                enabled=False,
                description="Advanced semantic pattern matching"
            ),
            FeatureFlag.CONSENSUS_ORCHESTRATION: FeatureConfig(
                enabled=False,
                description="Distributed consensus orchestration"
            )
        }
    
    def _load_environment_overrides(self):
        """Load feature flag overrides from environment variables"""
        for feature in FeatureFlag:
            env_var = f"AURA_FEATURE_{feature.value.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                if env_value.lower() in ('true', '1', 'yes', 'on'):
                    self.features[feature].enabled = True
                elif env_value.lower() in ('false', '0', 'no', 'off'):
                    self.features[feature].enabled = False
    
    def is_enabled(self, feature: FeatureFlag) -> bool:
        """Check if a feature is enabled"""
        config = self.features.get(feature)
        if not config:
            return False
        
        # Check if feature is enabled
        if not config.enabled:
            return False
        
        # Check environment restrictions
        if (config.environment_restrictions and 
            self.environment not in config.environment_restrictions):
            return False
        
        # Check dependencies
        if config.dependencies:
            for dependency in config.dependencies:
                if not self.is_enabled(dependency):
                    return False
        
        return True
    
    def enable_feature(self, feature: FeatureFlag, check_dependencies: bool = True):
        """Enable a feature flag"""
        if check_dependencies:
            config = self.features.get(feature)
            if config and config.dependencies:
                for dependency in config.dependencies:
                    if not self.is_enabled(dependency):
                        raise ValueError(
                            f"Cannot enable {feature.value}: dependency {dependency.value} not enabled"
                        )
        
        if feature in self.features:
            self.features[feature].enabled = True
    
    def disable_feature(self, feature: FeatureFlag):
        """Disable a feature flag"""
        if feature in self.features:
            self.features[feature].enabled = False
    
    def get_enabled_features(self) -> list:
        """Get list of currently enabled features"""
        return [
            feature.value for feature, config in self.features.items()
            if self.is_enabled(feature)
        ]
    
    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive feature status"""
        status = {}
        
        for feature, config in self.features.items():
            status[feature.value] = {
                "enabled": self.is_enabled(feature),
                "configured_enabled": config.enabled,
                "environment_restricted": (
                    config.environment_restrictions and 
                    self.environment not in config.environment_restrictions
                ),
                "dependencies_met": (
                    not config.dependencies or 
                    all(self.is_enabled(dep) for dep in config.dependencies)
                ),
                "description": config.description
            }
        
        return status

# Global feature flags instance
_feature_flags = None

def get_feature_flags(environment: Optional[str] = None) -> OrchestrationFeatureFlags:
    """Get the global feature flags instance"""
    global _feature_flags
    
    if _feature_flags is None or (environment and environment != _feature_flags.environment):
        env = environment or os.getenv("AURA_ENVIRONMENT", "development")
        _feature_flags = OrchestrationFeatureFlags(environment=env)
    
    return _feature_flags

def is_feature_enabled(feature: FeatureFlag) -> bool:
    """Quick check if a feature is enabled"""
    return get_feature_flags().is_enabled(feature)

# Convenience functions for common checks
def is_distributed_scaling_enabled() -> bool:
    """Check if distributed scaling features are enabled"""
    flags = get_feature_flags()
    return (flags.is_enabled(FeatureFlag.RAY_SERVE_ORCHESTRATION) or
            flags.is_enabled(FeatureFlag.CREWAI_FLOWS) or
            flags.is_enabled(FeatureFlag.DISTRIBUTED_COORDINATION))

def is_production_ready() -> bool:
    """Check if system is configured for production"""
    flags = get_feature_flags()
    return (flags.is_enabled(FeatureFlag.TEMPORAL_WORKFLOWS) and
            flags.is_enabled(FeatureFlag.HYBRID_CHECKPOINTING) and
            flags.is_enabled(FeatureFlag.POSTGRESQL_CHECKPOINTING))

def get_startup_features() -> list:
    """Get features appropriate for startup deployment"""
    return [
        FeatureFlag.SEMANTIC_ORCHESTRATION,
        FeatureFlag.LANGGRAPH_INTEGRATION,
        FeatureFlag.TDA_INTEGRATION,
        FeatureFlag.TEMPORAL_WORKFLOWS,
        FeatureFlag.SAGA_PATTERNS,
        FeatureFlag.HYBRID_CHECKPOINTING,
        FeatureFlag.POSTGRESQL_CHECKPOINTING,
        FeatureFlag.CROSS_THREAD_MEMORY
    ]