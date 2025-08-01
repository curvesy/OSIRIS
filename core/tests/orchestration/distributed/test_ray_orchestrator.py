"""
🚀 Ray Serve Orchestrator Tests

Tests for the Ray Serve distributed orchestrator including agent ensemble
deployments, TDA-aware load balancing, and distributed coordination.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from aura_intelligence.orchestration.distributed.ray_orchestrator import (
    RayServeOrchestrator,
    AgentEnsembleDeployment,
    TDALoadBalancer,
    DistributedAgentConfig,
    AgentRequest,
    AgentResponse,
    AgentType,
    LoadBalancingStrategy
)

class TestTDALoadBalancer:
    """Test suite for TDALoadBalancer"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            anomaly_severity=0.7,
            complexity_score=0.6,
            pattern_confidence=0.8
        ))
        return mock
    
    @pytest.fixture
    def load_balancer(self, mock_tda_integration):
        """Create TDA load balancer instance"""
        return TDALoadBalancer(tda_integration=mock_tda_integration)
    
    @pytest.fixture
    def sample_request(self):
        """Sample agent request"""
        return AgentRequest(
            request_id="test-request-001",
            agent_type=AgentType.ANALYST,
            payload={"analysis_type": "pattern_detection"},
            tda_correlation_id="test-correlation",
            priority=1
        )
    
    @pytest.mark.asyncio
    async def test_tda_aware_agent_selection(self, load_balancer, sample_request, mock_tda_integration):
        """Test TDA-aware agent selection"""
        available_agents = ["agent-1", "agent-2", "agent-3"]
        
        # Set up agent metrics
        load_balancer.agent_metrics = {
            "agent-1": {"current_load": 0.3, "anomaly_performance": 0.9, "complexity_performance": 0.7},
            "agent-2": {"current_load": 0.5, "anomaly_performance": 0.6, "complexity_performance": 0.8},
            "agent-3": {"current_load": 0.2, "anomaly_performance": 0.8, "complexity_performance": 0.9}
        }
        
        # Test with high anomaly severity
        tda_context = mock_tda_integration.get_context.return_value
        tda_context.anomaly_severity = 0.9
        
        selected_agent = await load_balancer.select_agent(
            sample_request, available_agents, tda_context
        )
        
        # Should select agent-1 due to high anomaly performance
        assert selected_agent == "agent-1"
        assert len(load_balancer.routing_history) == 1
        assert load_balancer.routing_history[0]["selected_agent"] == "agent-1"
    
    @pytest.mark.asyncio
    async def test_performance_based_selection_fallback(self, load_balancer, sample_request):
        """Test performance-based selection when TDA context unavailable"""
        available_agents = ["agent-1", "agent-2", "agent-3"]
        
        # Set up agent metrics with different loads
        load_balancer.agent_metrics = {
            "agent-1": {"current_load": 0.8, "avg_response_time": 500},
            "agent-2": {"current_load": 0.3, "avg_response_time": 200},
            "agent-3": {"current_load": 0.6, "avg_response_time": 300}
        }
        
        selected_agent = await load_balancer.select_agent(
            sample_request, available_agents, None
        )
        
        # Should select agent-2 due to lowest combined load + response time
        assert selected_agent == "agent-2"
    
    @pytest.mark.asyncio
    async def test_agent_selection_with_empty_agents_list(self, load_balancer, sample_request):
        """Test agent selection with empty agents list"""
        with pytest.raises(ValueError, match="No available agents"):
            await load_balancer.select_agent(sample_request, [], None)
    
    def test_update_agent_metrics(self, load_balancer):
        """Test agent metrics update"""
        agent_id = "test-agent"
        metrics = {
            "current_load": 0.5,
            "avg_response_time": 250,
            "anomaly_performance": 0.8
        }
        
        load_balancer.update_agent_metrics(agent_id, metrics)
        
        assert agent_id in load_balancer.agent_metrics
        assert load_balancer.agent_metrics[agent_id]["current_load"] == 0.5
        assert load_balancer.agent_metrics[agent_id]["avg_response_time"] == 250
        assert "last_updated" in load_balancer.agent_metrics[agent_id]
    
    def test_routing_history_management(self, load_balancer, sample_request):
        """Test routing history management and cleanup"""
        # Add many routing decisions to test cleanup
        for i in range(1100):  # More than the 1000 limit
            load_balancer._record_routing_decision(
                sample_request, f"agent-{i}", None
            )
        
        # Should keep only the last 1000 decisions
        assert len(load_balancer.routing_history) == 1000
        assert load_balancer.routing_history[0]["selected_agent"] == "agent-100"  # First kept record

class TestAgentEnsembleDeployment:
    """Test suite for AgentEnsembleDeployment"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            anomaly_severity=0.5,
            complexity_score=0.6
        ))
        mock.send_orchestration_result = AsyncMock()
        return mock
    
    @pytest.fixture
    def agent_config(self):
        """Sample agent configuration"""
        return DistributedAgentConfig(
            agent_type=AgentType.ANALYST,
            num_replicas=2,
            min_replicas=1,
            max_replicas=5,
            cpu_per_replica=2.0,
            gpu_per_replica=0.5,
            enable_tda_integration=True
        )
    
    @pytest.fixture
    def agent_deployment(self, agent_config, mock_tda_integration):
        """Create agent ensemble deployment"""
        return AgentEnsembleDeployment(
            agent_config=agent_config,
            tda_integration=mock_tda_integration
        )
    
    @pytest.fixture
    def sample_request_data(self):
        """Sample request data"""
        return {
            "request_id": "test-request-001",
            "agent_type": AgentType.ANALYST,
            "payload": {"analysis_type": "pattern_detection", "data": "sample_data"},
            "tda_correlation_id": "test-correlation",
            "priority": 1,
            "timeout": 300
        }
    
    @pytest.mark.asyncio
    async def test_agent_deployment_initialization(self, agent_deployment, agent_config):
        """Test agent deployment initialization"""
        assert agent_deployment.agent_config == agent_config
        assert agent_deployment.instance_id.startswith("analyst_")
        assert agent_deployment.request_count == 0
        assert agent_deployment.agent is not None
    
    @pytest.mark.asyncio
    async def test_request_processing_with_tda_context(
        self, 
        agent_deployment, 
        sample_request_data, 
        mock_tda_integration
    ):
        """Test request processing with TDA context"""
        response_data = await agent_deployment(sample_request_data)
        
        # Verify response structure
        assert "request_id" in response_data
        assert "agent_type" in response_data
        assert "result" in response_data
        assert "execution_time" in response_data
        assert "resource_usage" in response_data
        
        # Verify TDA integration was called
        mock_tda_integration.get_context.assert_called_with("test-correlation")
        mock_tda_integration.send_orchestration_result.assert_called()
        
        # Verify metrics were updated
        assert agent_deployment.request_count == 1
        assert agent_deployment.total_execution_time > 0
    
    @pytest.mark.asyncio
    async def test_request_processing_without_tda_context(self, agent_deployment):
        """Test request processing without TDA context"""
        request_data = {
            "request_id": "test-request-002",
            "agent_type": AgentType.ANALYST,
            "payload": {"analysis_type": "simple_analysis"},
            "tda_correlation_id": None
        }
        
        response_data = await agent_deployment(request_data)
        
        assert response_data["request_id"] == "test-request-002"
        assert response_data["tda_correlation_id"] is None
        assert response_data["metadata"]["tda_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, agent_deployment):
        """Test error handling during request processing"""
        # Mock agent to raise exception
        async def failing_process(input_data):
            raise Exception("Processing failed")
        
        agent_deployment.agent.process = failing_process
        
        request_data = {
            "request_id": "test-request-error",
            "agent_type": AgentType.ANALYST,
            "payload": {"test": "data"}
        }
        
        response_data = await agent_deployment(request_data)
        
        # Should return error response instead of raising exception
        assert "error" in response_data["result"]
        assert response_data["result"]["error"] == "Processing failed"
        assert response_data["metadata"]["error"] is True
        assert agent_deployment.error_count == 1
    
    def test_health_check(self, agent_deployment):
        """Test agent deployment health check"""
        # Process some requests to generate metrics
        agent_deployment.request_count = 10
        agent_deployment.total_execution_time = 5.0
        agent_deployment.error_count = 1
        
        health_status = agent_deployment.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["agent_type"] == "analyst"
        assert health_status["request_count"] == 10
        assert health_status["error_rate"] == 0.1
        assert health_status["avg_response_time"] == 0.5
        assert "last_check" in health_status
    
    def test_resource_usage_tracking(self, agent_deployment):
        """Test resource usage tracking"""
        # Simulate some activity
        agent_deployment.request_count = 5
        agent_deployment.total_execution_time = 2.5
        agent_deployment.error_count = 0
        
        resource_usage = agent_deployment._get_resource_usage()
        
        assert "cpu_usage" in resource_usage
        assert "memory_usage" in resource_usage
        assert "gpu_usage" in resource_usage
        assert resource_usage["request_count"] == 5
        assert resource_usage["avg_execution_time"] == 0.5
        assert resource_usage["error_rate"] == 0.0
    
    def test_different_agent_types_initialization(self):
        """Test initialization of different agent types"""
        agent_types = [
            AgentType.OBSERVER,
            AgentType.ANALYST,
            AgentType.SUPERVISOR,
            AgentType.EXECUTOR,
            AgentType.COORDINATOR
        ]
        
        for agent_type in agent_types:
            config = DistributedAgentConfig(agent_type=agent_type)
            deployment = AgentEnsembleDeployment(config)
            
            assert deployment.agent_config.agent_type == agent_type
            assert deployment.agent is not None
            assert deployment.agent.agent_type == agent_type.value

class TestRayServeOrchestrator:
    """Test suite for RayServeOrchestrator"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration"""
        mock = Mock()
        mock.get_context = AsyncMock(return_value=Mock(
            correlation_id="test-correlation",
            anomaly_severity=0.6,
            complexity_score=0.7
        ))
        return mock
    
    @pytest.fixture
    def orchestrator(self, mock_tda_integration):
        """Create Ray Serve orchestrator"""
        return RayServeOrchestrator(tda_integration=mock_tda_integration)
    
    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration"""
        return DistributedAgentConfig(
            agent_type=AgentType.ANALYST,
            num_replicas=2,
            min_replicas=1,
            max_replicas=5,
            target_requests_per_replica=2,
            cpu_per_replica=2.0,
            enable_tda_integration=True
        )
    
    @pytest.fixture
    def sample_request(self):
        """Sample agent request"""
        return AgentRequest(
            request_id="test-request-001",
            agent_type=AgentType.ANALYST,
            payload={"analysis_type": "pattern_detection"},
            tda_correlation_id="test-correlation"
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator, mock_tda_integration):
        """Test orchestrator initialization"""
        assert orchestrator.tda_integration == mock_tda_integration
        assert isinstance(orchestrator.load_balancer, TDALoadBalancer)
        assert orchestrator.deployments == {}
        assert orchestrator.deployment_handles == {}
        assert orchestrator.cluster_metrics["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_deploy_agent_ensemble_mock_mode(self, orchestrator, sample_agent_config):
        """Test agent ensemble deployment in mock mode"""
        deployment_name = "test-analyst-deployment"
        
        result = await orchestrator.deploy_agent_ensemble(
            deployment_name, sample_agent_config
        )
        
        assert result == deployment_name
        assert deployment_name in orchestrator.deployments
        assert orchestrator.deployments[deployment_name]["status"] == "deployed"
        assert orchestrator.deployments[deployment_name]["config"] == sample_agent_config
        assert orchestrator.cluster_metrics["active_deployments"] == 1
    
    @pytest.mark.asyncio
    async def test_route_request_with_auto_selection(
        self, 
        orchestrator, 
        sample_agent_config, 
        sample_request
    ):
        """Test request routing with automatic deployment selection"""
        # Deploy an agent ensemble first
        deployment_name = "test-analyst-deployment"
        await orchestrator.deploy_agent_ensemble(deployment_name, sample_agent_config)
        
        # Route request
        response = await orchestrator.route_request(sample_request)
        
        assert isinstance(response, AgentResponse)
        assert response.request_id == sample_request.request_id
        assert response.agent_type == sample_request.agent_type
        assert orchestrator.cluster_metrics["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_route_request_with_specific_deployment(
        self, 
        orchestrator, 
        sample_agent_config, 
        sample_request
    ):
        """Test request routing to specific deployment"""
        deployment_name = "specific-analyst-deployment"
        await orchestrator.deploy_agent_ensemble(deployment_name, sample_agent_config)
        
        response = await orchestrator.route_request(
            sample_request, deployment_name=deployment_name
        )
        
        assert response.request_id == sample_request.request_id
        assert orchestrator.cluster_metrics["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_route_request_no_matching_deployment(self, orchestrator, sample_request):
        """Test request routing when no matching deployment exists"""
        with pytest.raises(ValueError, match="No deployments available"):
            await orchestrator.route_request(sample_request)
    
    @pytest.mark.asyncio
    async def test_route_request_deployment_not_found(
        self, 
        orchestrator, 
        sample_request
    ):
        """Test request routing to non-existent deployment"""
        with pytest.raises(ValueError, match="Deployment nonexistent not found"):
            await orchestrator.route_request(
                sample_request, deployment_name="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_scale_deployment_mock_mode(self, orchestrator, sample_agent_config):
        """Test deployment scaling in mock mode"""
        deployment_name = "scalable-deployment"
        await orchestrator.deploy_agent_ensemble(deployment_name, sample_agent_config)
        
        result = await orchestrator.scale_deployment(deployment_name, 5)
        
        assert result is True
        assert orchestrator.deployments[deployment_name]["instances"] == 5
    
    @pytest.mark.asyncio
    async def test_scale_nonexistent_deployment(self, orchestrator):
        """Test scaling non-existent deployment"""
        with pytest.raises(ValueError, match="Deployment nonexistent not found"):
            await orchestrator.scale_deployment("nonexistent", 3)
    
    def test_get_cluster_metrics(self, orchestrator, sample_agent_config):
        """Test cluster metrics retrieval"""
        # Add some mock data
        orchestrator.cluster_metrics["total_requests"] = 100
        orchestrator.cluster_metrics["total_errors"] = 5
        orchestrator.deployments["test-deployment"] = {
            "config": sample_agent_config,
            "status": "deployed",
            "instances": 3
        }
        
        metrics = orchestrator.get_cluster_metrics()
        
        assert metrics["total_requests"] == 100
        assert metrics["total_errors"] == 5
        assert "deployments" in metrics
        assert "test-deployment" in metrics["deployments"]
        assert metrics["deployments"]["test-deployment"]["agent_type"] == "analyst"
        assert metrics["deployments"]["test-deployment"]["instances"] == 3
        assert "load_balancer_metrics" in metrics
    
    @pytest.mark.asyncio
    async def test_multiple_agent_types_deployment(self, orchestrator):
        """Test deploying multiple different agent types"""
        agent_types = [AgentType.OBSERVER, AgentType.ANALYST, AgentType.SUPERVISOR]
        
        for agent_type in agent_types:
            config = DistributedAgentConfig(agent_type=agent_type)
            deployment_name = f"{agent_type.value}-deployment"
            
            result = await orchestrator.deploy_agent_ensemble(deployment_name, config)
            assert result == deployment_name
        
        assert len(orchestrator.deployments) == 3
        assert orchestrator.cluster_metrics["active_deployments"] == 3
    
    @pytest.mark.asyncio
    async def test_load_balancer_metrics_update(
        self, 
        orchestrator, 
        sample_agent_config, 
        sample_request
    ):
        """Test load balancer metrics update after request routing"""
        deployment_name = "metrics-test-deployment"
        await orchestrator.deploy_agent_ensemble(deployment_name, sample_agent_config)
        
        # Route a request
        await orchestrator.route_request(sample_request)
        
        # Check that load balancer metrics were updated
        assert deployment_name in orchestrator.load_balancer.agent_metrics
        metrics = orchestrator.load_balancer.agent_metrics[deployment_name]
        assert "current_load" in metrics
        assert "avg_response_time" in metrics
        assert "success_rate" in metrics

class TestDistributedAgentConfig:
    """Test suite for DistributedAgentConfig"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = DistributedAgentConfig(agent_type=AgentType.ANALYST)
        
        assert config.agent_type == AgentType.ANALYST
        assert config.num_replicas == "auto"
        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.target_requests_per_replica == 2
        assert config.cpu_per_replica == 2.0
        assert config.gpu_per_replica == 0.0
        assert config.enable_tda_integration is True
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = DistributedAgentConfig(
            agent_type=AgentType.OBSERVER,
            num_replicas=5,
            min_replicas=2,
            max_replicas=20,
            target_requests_per_replica=4,
            cpu_per_replica=4.0,
            gpu_per_replica=1.0,
            memory_per_replica="4Gi",
            enable_tda_integration=False
        )
        
        assert config.agent_type == AgentType.OBSERVER
        assert config.num_replicas == 5
        assert config.min_replicas == 2
        assert config.max_replicas == 20
        assert config.target_requests_per_replica == 4
        assert config.cpu_per_replica == 4.0
        assert config.gpu_per_replica == 1.0
        assert config.memory_per_replica == "4Gi"
        assert config.enable_tda_integration is False

class TestAgentRequestResponse:
    """Test suite for AgentRequest and AgentResponse"""
    
    def test_agent_request_creation(self):
        """Test agent request creation"""
        request = AgentRequest(
            request_id="test-001",
            agent_type=AgentType.ANALYST,
            payload={"data": "test"},
            tda_correlation_id="correlation-001",
            priority=1,
            timeout=300
        )
        
        assert request.request_id == "test-001"
        assert request.agent_type == AgentType.ANALYST
        assert request.payload == {"data": "test"}
        assert request.tda_correlation_id == "correlation-001"
        assert request.priority == 1
        assert request.timeout == 300
    
    def test_agent_response_creation(self):
        """Test agent response creation"""
        response = AgentResponse(
            request_id="test-001",
            agent_type=AgentType.ANALYST,
            agent_instance_id="analyst-instance-001",
            result={"analysis": "completed"},
            execution_time=1.5,
            resource_usage={"cpu": 0.5, "memory": 0.3},
            tda_correlation_id="correlation-001"
        )
        
        assert response.request_id == "test-001"
        assert response.agent_type == AgentType.ANALYST
        assert response.agent_instance_id == "analyst-instance-001"
        assert response.result == {"analysis": "completed"}
        assert response.execution_time == 1.5
        assert response.resource_usage == {"cpu": 0.5, "memory": 0.3}
        assert response.tda_correlation_id == "correlation-001"