"""
Security configuration for AURA Intelligence.

Defines security settings, authentication, and authorization.
"""

from typing import List, Optional

from pydantic import Field, SecretStr

from .base import BaseSettings


class SecuritySettings(BaseSettings):
    """
    Security configuration for the AURA Intelligence platform.
    
    Manages authentication, authorization, and security policies.
    Environment variables: AURA_SECURITY__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_SECURITY__")
    """
    Security configuration for the AURA Intelligence platform.
    
    Manages authentication, authorization, and security policies.
    """
    
    # Authentication
    enable_auth: bool = Field(
        default=True,
        description="Enable authentication"
    )
    auth_provider: str = Field(
        default="jwt",
        description="Authentication provider (jwt, oauth2, saml)"
    )
    jwt_secret_key: Optional[SecretStr] = Field(
        default=None,
        description="JWT secret key for token signing"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    jwt_expiration_hours: int = Field(
        default=24,
        ge=1,
        description="JWT token expiration in hours"
    )
    
    # OAuth2 configuration
    oauth2_client_id: Optional[str] = Field(
        default=None,
        description="OAuth2 client ID"
    )
    oauth2_client_secret: Optional[SecretStr] = Field(
        default=None,
        description="OAuth2 client secret"
    )
    oauth2_authorize_url: Optional[str] = Field(
        default=None,
        description="OAuth2 authorization URL"
    )
    oauth2_token_url: Optional[str] = Field(
        default=None,
        description="OAuth2 token URL"
    )
    
    # API key authentication
    enable_api_keys: bool = Field(
        default=True,
        description="Enable API key authentication"
    )
    api_key_header_name: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    
    # CORS configuration
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Rate limit per minute per user"
    )
    
    # Security headers
    enable_security_headers: bool = Field(
        default=True,
        description="Enable security headers"
    )
    hsts_max_age: int = Field(
        default=31536000,
        description="HSTS max age in seconds"
    )
    
    # Encryption
    enable_encryption: bool = Field(
        default=True,
        description="Enable data encryption"
    )
    encryption_key: Optional[SecretStr] = Field(
        default=None,
        description="Data encryption key"
    )
    
    # Audit logging
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    audit_log_retention_days: int = Field(
        default=90,
        ge=30,
        description="Audit log retention in days"
    )
    
    @property
    def requires_auth_setup(self) -> bool:
        """Check if authentication requires setup."""
        if not self.enable_auth:
            return False
        if self.auth_provider == "jwt" and not self.jwt_secret_key:
            return True
        if self.auth_provider == "oauth2" and not self.oauth2_client_id:
            return True
        return False