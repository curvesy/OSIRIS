"""
Agent configuration for AURA Intelligence.

Defines settings for agent behavior, enhancement levels, and collaboration.
"""

from typing import Optional

from pydantic import Field, validator

from .base import BaseSettings, EnhancementLevel


class AgentSettings(BaseSettings):
    """
    Advanced agent system configuration.
    
    Controls agent behavior, enhancement levels, and collaboration settings.
    Environment variables: AURA_AGENT__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_AGENT__")
    
    # Core agent settings
    agent_count: int = Field(
        default=7,
        ge=1,
        le=100,
        description="Number of agents in the system"
    )
    enhancement_level: EnhancementLevel = Field(
        default=EnhancementLevel.ULTIMATE,
        description="Agent enhancement level"
    )
    cycle_interval: float = Field(
        default=1.0,
        gt=0.0,
        le=60.0,
        description="Interval between agent cycles in seconds"
    )
    max_cycles: int = Field(
        default=10000,
        ge=1,
        description="Maximum number of cycles to run"
    )
    
    # Feature toggles
    enable_consciousness: bool = Field(
        default=True,
        description="Enable consciousness features"
    )
    enable_learning: bool = Field(
        default=True,
        description="Enable learning capabilities"
    )
    enable_adaptation: bool = Field(
        default=True,
        description="Enable adaptive behavior"
    )
    enable_collaboration: bool = Field(
        default=True,
        description="Enable agent collaboration"
    )
    
    # Performance settings
    performance_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Performance threshold for agent optimization"
    )
    consciousness_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Depth of consciousness processing"
    )
    
    # Advanced features
    causal_reasoning: bool = Field(
        default=True,
        description="Enable causal reasoning capabilities"
    )
    quantum_coherence: bool = Field(
        default=True,
        description="Enable quantum coherence features"
    )
    
    # Resource limits
    max_memory_mb: int = Field(
        default=1024,
        ge=128,
        description="Maximum memory per agent in MB"
    )
    max_cpu_percent: int = Field(
        default=80,
        ge=10,
        le=100,
        description="Maximum CPU usage percentage"
    )
    
    @validator("enhancement_level")
    def validate_enhancement_level(cls, v: EnhancementLevel) -> EnhancementLevel:
        """Validate enhancement level based on features."""
        return v
    
    def get_agent_config_dict(self) -> dict:
        """Get agent configuration as dictionary for legacy compatibility."""
        return self.model_dump(exclude={"max_memory_mb", "max_cpu_percent"})
    
    @property
    def is_advanced_mode(self) -> bool:
        """Check if advanced features are enabled."""
        return self.enhancement_level in (
            EnhancementLevel.ULTIMATE,
            EnhancementLevel.CONSCIOUSNESS
        )