"""
Tests for AURA Intelligence configuration module.

Tests all configuration classes with parametrized fixtures.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import SecretStr, ValidationError

from aura_intelligence.config import (
    AURASettings,
    EnvironmentType,
    EnhancementLevel,
    AgentSettings,
    MemorySettings,
    APISettings,
    ObservabilitySettings,
    IntegrationSettings,
    SecuritySettings,
    DeploymentSettings,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for testing."""
    # Remove all AURA_ prefixed env vars
    for key in list(os.environ.keys()):
        if key.startswith("AURA_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def test_env(monkeypatch):
    """Set up test environment variables."""
    test_vars = {
        "AURA_ENVIRONMENT": "production",
        "AURA_LOG_LEVEL": "DEBUG",
        "AURA_API__OPENAI_API_KEY": "test-openai-key",
        "AURA_SECURITY__JWT_SECRET_KEY": "test-jwt-secret",
        "AURA_SECURITY__ENCRYPTION_KEY": "test-encryption-key-32-bytes-long",
    }
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    yield test_vars


class TestEnvironmentType:
    """Test EnvironmentType enum."""
    
    @pytest.mark.parametrize("env_type,expected", [
        ("development", EnvironmentType.DEVELOPMENT),
        ("staging", EnvironmentType.STAGING),
        ("production", EnvironmentType.PRODUCTION),
        ("enterprise", EnvironmentType.ENTERPRISE),
    ])
    def test_environment_types(self, env_type, expected):
        assert EnvironmentType(env_type) == expected
    
    def test_invalid_environment_type(self):
        with pytest.raises(ValueError):
            EnvironmentType("invalid")


class TestEnhancementLevel:
    """Test EnhancementLevel enum."""
    
    @pytest.mark.parametrize("level,expected", [
        ("basic", EnhancementLevel.BASIC),
        ("advanced", EnhancementLevel.ADVANCED),
        ("ultimate", EnhancementLevel.ULTIMATE),
        ("consciousness", EnhancementLevel.CONSCIOUSNESS),
    ])
    def test_enhancement_levels(self, level, expected):
        assert EnhancementLevel(level) == expected


class TestAgentSettings:
    """Test AgentSettings configuration."""
    
    def test_default_values(self):
        settings = AgentSettings()
        assert settings.agent_count == 7
        assert settings.enhancement_level == EnhancementLevel.ULTIMATE
        assert settings.cycle_interval == 1.0
        assert settings.enable_consciousness is True
    
    @pytest.mark.parametrize("agent_count,valid", [
        (1, True),
        (50, True),
        (100, True),
        (0, False),
        (101, False),
        (-1, False),
    ])
    def test_agent_count_validation(self, agent_count, valid):
        if valid:
            settings = AgentSettings(agent_count=agent_count)
            assert settings.agent_count == agent_count
        else:
            with pytest.raises(ValidationError):
                AgentSettings(agent_count=agent_count)
    
    def test_is_advanced_mode(self):
        basic = AgentSettings(enhancement_level=EnhancementLevel.BASIC)
        assert not basic.is_advanced_mode
        
        ultimate = AgentSettings(enhancement_level=EnhancementLevel.ULTIMATE)
        assert ultimate.is_advanced_mode


class TestMemorySettings:
    """Test MemorySettings configuration."""
    
    def test_default_values(self):
        settings = MemorySettings()
        assert settings.vector_store_type == "chroma"
        assert settings.max_memories == 10000
        assert settings.enable_cache is True
    
    @pytest.mark.parametrize("store_type,valid", [
        ("chroma", True),
        ("pinecone", True),
        ("weaviate", True),
        ("faiss", True),
        ("qdrant", True),
        ("invalid", False),
    ])
    def test_vector_store_validation(self, store_type, valid):
        if valid:
            settings = MemorySettings(vector_store_type=store_type)
            assert settings.vector_store_type == store_type
        else:
            with pytest.raises(ValidationError):
                MemorySettings(vector_store_type=store_type)
    
    @pytest.mark.parametrize("store_type,requires_key", [
        ("chroma", False),
        ("faiss", False),
        ("pinecone", True),
        ("weaviate", True),
        ("qdrant", True),
    ])
    def test_requires_api_key(self, store_type, requires_key):
        settings = MemorySettings(vector_store_type=store_type)
        assert settings.requires_api_key == requires_key


class TestAPISettings:
    """Test APISettings configuration."""
    
    def test_secret_key_handling(self):
        key = "sk-test-key"
        settings = APISettings(openai_api_key=key)
        
        # Key should be stored as SecretStr
        assert isinstance(settings.openai_api_key, SecretStr)
        
        # get_api_key should return actual string
        assert settings.get_api_key("openai") == key
    
    def test_has_key_properties(self):
        settings = APISettings()
        assert not settings.has_openai_key
        assert not settings.has_anthropic_key
        
        settings = APISettings(
            openai_api_key="test-key",
            anthropic_api_key="test-key"
        )
        assert settings.has_openai_key
        assert settings.has_anthropic_key
    
    @pytest.mark.parametrize("model,valid", [
        ("gpt-3.5-turbo", True),
        ("gpt-4", True),
        ("gpt-4-turbo", True),
        ("text-embedding-ada-002", True),
        ("claude-3", False),
        ("invalid-model", False),
    ])
    def test_openai_model_validation(self, model, valid):
        if valid:
            settings = APISettings(openai_model=model)
            assert settings.openai_model == model
        else:
            with pytest.raises(ValidationError):
                APISettings(openai_model=model)


class TestObservabilitySettings:
    """Test ObservabilitySettings configuration."""
    
    def test_default_values(self):
        settings = ObservabilitySettings()
        assert settings.enable_metrics is True
        assert settings.metrics_port == 9090
        assert settings.prometheus_enabled is True
    
    @pytest.mark.parametrize("log_format,valid", [
        ("json", True),
        ("text", True),
        ("xml", False),
        ("yaml", False),
    ])
    def test_log_format_validation(self, log_format, valid):
        if valid:
            settings = ObservabilitySettings(log_format=log_format)
            assert settings.log_format == log_format
        else:
            with pytest.raises(ValidationError):
                ObservabilitySettings(log_format=log_format)
    
    def test_url_properties(self):
        settings = ObservabilitySettings(
            metrics_port=9999,
            health_check_port=8888
        )
        assert settings.metrics_url == "http://0.0.0.0:9999/metrics"
        assert settings.health_check_url == "http://0.0.0.0:8888/health"


class TestSecuritySettings:
    """Test SecuritySettings configuration."""
    
    def test_secret_handling(self):
        jwt_secret = "super-secret-key"
        settings = SecuritySettings(jwt_secret_key=jwt_secret)
        
        assert isinstance(settings.jwt_secret_key, SecretStr)
        assert settings.jwt_secret_key.get_secret_value() == jwt_secret
    
    def test_requires_auth_setup(self):
        # Auth disabled
        settings = SecuritySettings(enable_auth=False)
        assert not settings.requires_auth_setup
        
        # JWT without secret
        settings = SecuritySettings(
            enable_auth=True,
            auth_provider="jwt",
            jwt_secret_key=None
        )
        assert settings.requires_auth_setup
        
        # OAuth2 without client ID
        settings = SecuritySettings(
            enable_auth=True,
            auth_provider="oauth2",
            oauth2_client_id=None
        )
        assert settings.requires_auth_setup


class TestDeploymentSettings:
    """Test DeploymentSettings configuration."""
    
    @pytest.mark.parametrize("mode,valid", [
        ("shadow", True),
        ("canary", True),
        ("production", True),
        ("testing", False),
        ("development", False),
    ])
    def test_deployment_mode_validation(self, mode, valid):
        if valid:
            settings = DeploymentSettings(deployment_mode=mode)
            assert settings.deployment_mode == mode
        else:
            with pytest.raises(ValidationError):
                DeploymentSettings(deployment_mode=mode)
    
    def test_mode_properties(self):
        shadow = DeploymentSettings(deployment_mode="shadow", shadow_enabled=True)
        assert shadow.is_shadow_mode
        assert not shadow.is_canary_mode
        
        canary = DeploymentSettings(deployment_mode="canary", canary_enabled=True)
        assert not canary.is_shadow_mode
        assert canary.is_canary_mode
    
    def test_full_image_name(self):
        settings = DeploymentSettings(
            container_registry="myregistry.io",
            container_image="aura/core",
            container_tag="v1.2.3"
        )
        assert settings.full_image_name == "myregistry.io/aura/core:v1.2.3"


class TestAURASettings:
    """Test main AURASettings class."""
    
    def test_default_initialization(self, clean_env):
        settings = AURASettings()
        assert settings.environment == EnvironmentType.DEVELOPMENT
        assert isinstance(settings.agent, AgentSettings)
        assert isinstance(settings.memory, MemorySettings)
    
    def test_env_loading(self, test_env):
        settings = AURASettings.from_env()
        assert settings.environment == EnvironmentType.PRODUCTION
        assert settings.log_level == "DEBUG"
        assert settings.api.get_api_key("openai") == "test-openai-key"
    
    def test_validation_warnings(self):
        # Production without API keys
        settings = AURASettings(environment=EnvironmentType.PRODUCTION)
        warnings = settings.validate_configuration()
        assert any("API keys" in w for w in warnings)
        
        # Shadow mode in production
        settings = AURASettings(
            environment=EnvironmentType.PRODUCTION,
            deployment=DeploymentSettings(deployment_mode="shadow")
        )
        warnings = settings.validate_configuration()
        assert any("Shadow mode" in w for w in warnings)
    
    def test_legacy_config_format(self):
        settings = AURASettings(
            environment=EnvironmentType.PRODUCTION,
            api=APISettings(openai_api_key="test-key")
        )
        legacy = settings.get_legacy_config()
        
        assert legacy["environment"] == "production"
        assert "agent_config" in legacy
        assert legacy["api_keys"]["openai"] == "test-key"
    
    def test_print_configuration_summary(self, capsys):
        settings = AURASettings()
        settings.print_configuration_summary()
        
        captured = capsys.readouterr()
        assert "AURA Intelligence Configuration Summary" in captured.out
        assert "Environment: development" in captured.out


@pytest.mark.parametrize("env_vars,expected_env", [
    ({"AURA_ENVIRONMENT": "development"}, EnvironmentType.DEVELOPMENT),
    ({"AURA_ENVIRONMENT": "staging"}, EnvironmentType.STAGING),
    ({"AURA_ENVIRONMENT": "production"}, EnvironmentType.PRODUCTION),
    ({"AURA_ENVIRONMENT": "enterprise"}, EnvironmentType.ENTERPRISE),
])
def test_environment_loading(monkeypatch, clean_env, env_vars, expected_env):
    """Test environment variable loading."""
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    settings = AURASettings.from_env()
    assert settings.environment == expected_env


@pytest.mark.parametrize("nested_var,expected_value", [
    ("AURA_AGENT__AGENT_COUNT", ("agent", "agent_count", 10)),
    ("AURA_MEMORY__VECTOR_STORE_TYPE", ("memory", "vector_store_type", "pinecone")),
    ("AURA_OBSERVABILITY__METRICS_PORT", ("observability", "metrics_port", 8888)),
])
def test_nested_env_loading(monkeypatch, clean_env, nested_var, expected_value):
    """Test nested environment variable loading."""
    attr_path, attr_name, value = expected_value
    monkeypatch.setenv(nested_var, str(value))
    
    settings = AURASettings.from_env()
    nested_obj = getattr(settings, attr_path)
    assert getattr(nested_obj, attr_name) == value