"""
Test Configuration Helper

Provides test-specific configurations without hardcoded values.
"""

import os
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass, field
from datetime import datetime

from ..config.base import AURAConfig, DatabaseConfig, MessagingConfig, ObservabilityConfig, ServicesConfig, SecurityConfig


@dataclass
class TestDatabaseConfig(DatabaseConfig):
    """Test database configuration with in-memory defaults."""
    neo4j_uri: str = field(default="bolt://test-neo4j:7687")
    neo4j_user: str = field(default="test_user")
    neo4j_password: str = field(default="test_password_secure")
    neo4j_database: str = field(default="test_aura")
    
    redis_url: str = field(default="redis://test-redis:6379/1")
    redis_password: Optional[str] = field(default=None)
    redis_db: int = field(default=1)
    
    postgres_uri: str = field(default="postgresql://test:test@test-db:5432/test_aura")
    
    def __post_init__(self):
        """Skip validation for tests."""
        pass


@dataclass
class TestMessagingConfig(MessagingConfig):
    """Test messaging configuration."""
    kafka_bootstrap_servers: list = field(default_factory=lambda: ["test-kafka:9092"])
    kafka_security_protocol: str = field(default="PLAINTEXT")
    
    nats_url: str = field(default="nats://test-nats:4222")
    rabbitmq_url: str = field(default="amqp://test:test@test-rabbit:5672/")


@dataclass
class TestServicesConfig(ServicesConfig):
    """Test services configuration."""
    tda_service_url: str = field(default="http://test-tda:8080")
    temporal_host: str = field(default="test-temporal:7233")
    temporal_namespace: str = field(default="test")
    mem0_base_url: str = field(default="http://test-mem0:8080")


class TestConfig:
    """Helper class for creating test configurations."""
    
    @staticmethod
    def create_test_config(
        environment: str = "test",
        debug: bool = True,
        overrides: Optional[Dict[str, Any]] = None
    ) -> AURAConfig:
        """
        Create a test configuration with sensible defaults.
        
        Args:
            environment: Test environment name
            debug: Enable debug mode
            overrides: Optional configuration overrides
            
        Returns:
            Test configuration
        """
        config = AURAConfig(
            environment=environment,
            debug=debug,
            database=TestDatabaseConfig(),
            messaging=TestMessagingConfig(),
            observability=ObservabilityConfig(
                otel_endpoint="http://test-otel:4317",
                prometheus_port=9091,
                jaeger_endpoint="http://test-jaeger:14268",
                log_level="DEBUG"
            ),
            services=TestServicesConfig(),
            security=SecurityConfig(
                jwt_secret="test_jwt_secret_for_testing_only",
                encryption_key="test_encryption_key_32_chars_ok!",
                min_password_length=8
            )
        )
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        return config
    
    @staticmethod
    def create_mock_neo4j_driver():
        """Create a mock Neo4j driver for testing."""
        driver = AsyncMock()
        session = AsyncMock()
        
        # Mock session methods
        session.run = AsyncMock(return_value=AsyncMock())
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        # Mock driver methods
        driver.session = Mock(return_value=session)
        driver.verify_connectivity = AsyncMock()
        driver.close = AsyncMock()
        
        return driver
    
    @staticmethod
    def create_mock_redis_client():
        """Create a mock Redis client for testing."""
        client = AsyncMock()
        
        # Mock common Redis operations
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.exists = AsyncMock(return_value=0)
        client.expire = AsyncMock(return_value=True)
        client.ttl = AsyncMock(return_value=-1)
        
        # Mock pub/sub
        client.publish = AsyncMock(return_value=1)
        client.subscribe = AsyncMock()
        
        return client
    
    @staticmethod
    def create_mock_kafka_producer():
        """Create a mock Kafka producer for testing."""
        producer = AsyncMock()
        
        producer.send = AsyncMock(return_value=AsyncMock())
        producer.flush = AsyncMock()
        producer.close = AsyncMock()
        
        return producer
    
    @staticmethod
    def create_mock_event_bus():
        """Create a mock event bus for testing."""
        bus = AsyncMock()
        
        bus.publish = AsyncMock()
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        
        return bus
    
    @staticmethod
    def setup_test_environment():
        """Set up test environment variables."""
        test_env = {
            "ENVIRONMENT": "test",
            "DEBUG": "true",
            "NEO4J_PASSWORD": "test_password_secure",
            "JWT_SECRET": "test_jwt_secret",
            "ENCRYPTION_KEY": "test_encryption_key_32_chars_ok!",
            "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
            "REDIS_URL": "redis://test-redis:6379/1"
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
            
    @staticmethod
    def cleanup_test_environment():
        """Clean up test environment variables."""
        test_keys = [
            "ENVIRONMENT", "DEBUG", "NEO4J_PASSWORD", "JWT_SECRET",
            "ENCRYPTION_KEY", "KAFKA_BOOTSTRAP_SERVERS", "REDIS_URL"
        ]
        
        for key in test_keys:
            os.environ.pop(key, None)


# Fixtures for common test scenarios
class TestFixtures:
    """Common test fixtures."""
    
    @staticmethod
    def create_test_tda_result() -> Dict[str, Any]:
        """Create a test TDA result."""
        return {
            "algorithm": "test_algorithm",
            "anomaly_score": 0.75,
            "betti_numbers": [1, 2, 0],
            "persistence_diagrams": [
                {"dimension": 0, "intervals": [[0.1, 0.3], [0.2, 0.5]]},
                {"dimension": 1, "intervals": [[0.3, 0.7]]},
                {"dimension": 2, "intervals": []}
            ],
            "metadata": {
                "computation_time": 0.123,
                "num_points": 1000
            }
        }
    
    @staticmethod
    def create_test_agent_context() -> Dict[str, Any]:
        """Create a test agent context."""
        return {
            "agent_id": "test_agent",
            "task_id": "test_task_123",
            "trace_id": "test_trace_456",
            "tda_insights": TestFixtures.create_test_tda_result(),
            "memory_context": {
                "recent_decisions": [],
                "patterns": []
            }
        }
    
    @staticmethod
    def create_test_workflow_state() -> Dict[str, Any]:
        """Create a test workflow state."""
        return {
            "workflow_id": "test_workflow_789",
            "messages": ["Test message"],
            "evidence_log": [{"type": "test", "value": 123}],
            "tda_insights": {},
            "current_agent": "test_agent",
            "workflow_context": {
                "start_time": datetime.utcnow(),
                "trace_id": "test_trace_456"
            },
            "decision_history": [],
            "risk_assessment": {},
            "collective_decision": {},
            "agents_involved": []
        }