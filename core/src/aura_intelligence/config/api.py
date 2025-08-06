"""
API configuration for AURA Intelligence.

Manages API keys and external service configurations securely.
"""

from typing import Optional

from pydantic import Field, validator, SecretStr

from .base import BaseSettings


class APISettings(BaseSettings):
    """
    API configuration for external services.
    
    All API keys are stored as SecretStr for security.
    Environment variables: AURA_API__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_API__")

    
    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model to use"
    )
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="OpenAI temperature setting"
    )
    openai_max_tokens: int = Field(
        default=4000,
        ge=1,
        description="Maximum tokens for OpenAI responses"
    )
    
    # Anthropic Configuration
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-opus-20240229",
        description="Anthropic model to use"
    )
    
    # Google Configuration
    google_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Google API key"
    )
    google_model: str = Field(
        default="gemini-pro",
        description="Google model to use"
    )
    
    # Pinecone Configuration
    pinecone_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Pinecone API key"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment"
    )
    pinecone_index_name: str = Field(
        default="aura-intelligence",
        description="Pinecone index name"
    )
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="API rate limit per minute"
    )
    rate_limit_tokens_per_minute: int = Field(
        default=90000,
        ge=1000,
        description="Token rate limit per minute"
    )
    
    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum API retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Initial retry delay in seconds"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Retry backoff factor"
    )
    
    # Timeout configuration
    request_timeout_seconds: int = Field(
        default=30,
        ge=5,
        description="API request timeout in seconds"
    )
    
    @validator("openai_model")
    def validate_openai_model(cls, v: str) -> str:
        """Validate OpenAI model selection."""
        allowed_prefixes = {"gpt-3.5", "gpt-4", "text-embedding"}
        if not any(v.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"OpenAI model must start with one of {allowed_prefixes}")
        return v
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service (returns actual string value)."""
        key_field = f"{service}_api_key"
        secret = getattr(self, key_field, None)
        return secret.get_secret_value() if secret else None
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None
    
    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None