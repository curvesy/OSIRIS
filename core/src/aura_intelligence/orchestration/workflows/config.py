"""
Configuration management for collective intelligence workflows.

Uses Pydantic v2 for robust configuration validation and management.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from langchain_core.runnables import RunnableConfig


class RiskThresholds(BaseModel):
    """Risk threshold configuration."""
    critical: float = Field(0.9, ge=0.0, le=1.0)
    high: float = Field(0.7, ge=0.0, le=1.0)
    medium: float = Field(0.4, ge=0.0, le=1.0)
    low: float = Field(0.1, ge=0.0, le=1.0)
    
    @field_validator('high')
    def validate_high(cls, v: float, info) -> float:
        """Ensure high threshold is less than critical."""
        if 'critical' in info.data and v >= info.data['critical']:
            raise ValueError('high threshold must be less than critical')
        return v
    
    @field_validator('medium')
    def validate_medium(cls, v: float, info) -> float:
        """Ensure medium threshold is less than high."""
        if 'high' in info.data and v >= info.data['high']:
            raise ValueError('medium threshold must be less than high')
        return v
    
    @field_validator('low')
    def validate_low(cls, v: float, info) -> float:
        """Ensure low threshold is less than medium."""
        if 'medium' in info.data and v >= info.data['medium']:
            raise ValueError('low threshold must be less than medium')
        return v


class ModelConfig(BaseModel):
    """Model configuration for different agents."""
    supervisor: str = Field(
        "anthropic/claude-3-5-sonnet-latest",
        description="Model for supervisor agent"
    )
    observer: str = Field(
        "anthropic/claude-3-haiku-latest",
        description="Model for observer agent"
    )
    analyst: str = Field(
        "anthropic/claude-3-5-sonnet-latest",
        description="Model for analyst agent"
    )
    executor: str = Field(
        "anthropic/claude-3-haiku-latest",
        description="Model for executor agent"
    )


class WorkflowConfig(BaseModel):
    """
    Complete workflow configuration using Pydantic v2.
    
    This configuration controls all aspects of the collective
    intelligence workflow behavior.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        use_enum_values=True
    )
    
    # Model configuration
    models: ModelConfig = Field(default_factory=ModelConfig)
    
    # Feature flags
    enable_streaming: bool = Field(True, description="Enable streaming responses")
    enable_human_loop: bool = Field(False, description="Enable human-in-the-loop")
    enable_shadow_mode: bool = Field(True, description="Enable shadow mode logging")
    
    # Infrastructure settings
    checkpoint_mode: Literal["sqlite", "postgres", "memory"] = Field(
        "sqlite",
        description="Checkpoint storage backend"
    )
    memory_provider: Literal["local", "redis", "postgres"] = Field(
        "local",
        description="Memory storage provider"
    )
    
    # Operational parameters
    context_window: int = Field(5, ge=1, le=50, description="Context window size")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    timeout_seconds: int = Field(300, ge=10, le=3600, description="Operation timeout")
    
    # Risk configuration
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    
    # Prompts
    supervisor_prompt: str = Field(
        default=(
            "You are an expert collective intelligence supervisor. "
            "Analyze the current state and decide which tool to call next: "
            "observe_system_event, analyze_risk_patterns, execute_remediation, or FINISH."
        ),
        description="Supervisor agent prompt"
    )
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "WorkflowConfig":
        """
        Create WorkflowConfig from LangGraph RunnableConfig.
        
        Args:
            config: LangGraph runnable configuration
            
        Returns:
            WorkflowConfig instance
        """
        configurable = config.get("configurable", {})
        
        # Extract model configuration
        models = ModelConfig(
            supervisor=configurable.get("supervisor_model", "anthropic/claude-3-5-sonnet-latest"),
            observer=configurable.get("observer_model", "anthropic/claude-3-haiku-latest"),
            analyst=configurable.get("analyst_model", "anthropic/claude-3-5-sonnet-latest"),
            executor=configurable.get("executor_model", "anthropic/claude-3-haiku-latest")
        )
        
        # Extract risk thresholds
        risk_dict = configurable.get("risk_thresholds", {})
        risk_thresholds = RiskThresholds(**risk_dict) if isinstance(risk_dict, dict) else RiskThresholds()
        
        return cls(
            models=models,
            enable_streaming=configurable.get("enable_streaming", True),
            enable_human_loop=configurable.get("enable_human_loop", False),
            enable_shadow_mode=configurable.get("enable_shadow_mode", True),
            checkpoint_mode=configurable.get("checkpoint_mode", "sqlite"),
            memory_provider=configurable.get("memory_provider", "local"),
            context_window=configurable.get("context_window", 5),
            max_retries=configurable.get("max_retries", 3),
            timeout_seconds=configurable.get("timeout_seconds", 300),
            risk_thresholds=risk_thresholds,
            supervisor_prompt=configurable.get("supervisor_prompt", cls.model_fields["supervisor_prompt"].default)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


def extract_config(config: RunnableConfig) -> Dict[str, Any]:
    """
    Extract configuration using latest patterns from assistants-demo.
    
    This function maintains backward compatibility while leveraging
    the new WorkflowConfig model.
    
    Args:
        config: LangGraph runnable configuration
        
    Returns:
        Configuration dictionary
    """
    workflow_config = WorkflowConfig.from_runnable_config(config)
    return workflow_config.to_dict()