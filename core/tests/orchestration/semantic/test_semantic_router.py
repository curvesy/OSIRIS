"""
🧪 Tests for Semantic Routing Engine

Comprehensive test suite for semantic routing including agent selection,
capability matching, TDA-aware routing, and performance optimization.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from aura_intelligence.orchestration.semantic.semantic_router import (
    SemanticRouter, AgentProfile, AgentCapability, RoutingDecision
)
from aura_intelligence.orchestration.semantic.base_interfaces import (
    TDAContext, SemanticAnalysis, OrchestrationStrategy, UrgencyLevel
)

@pytest.fixture
def semantic_router():
    """Create semantic router for testing"""
    return SemanticRouter()

@pytest.fixture
def sample_semantic_analysis():
    """Sample semantic analysis for testing"""
    return SemanticAnalysis(
        complexity_score=0.6,
        urgency_level=UrgencyLevel.MEDIUM,
        coordination_pattern=OrchestrationStrategy.PARALLEL,
        suggested_agents=["analyst_agent", "observer_agent"],
        confidence=0.8,
        tda_correlation=None
    )

@pytest.fixture
def sample_tda_context():
    """Sample TDA context for testing"""
    return TDAContext(
        correlation_id="test-correlation-123",
        pattern_confidence=0.8,
        anomaly_severity=0.6,
        current_patterns={"anomaly_type": "spike", "confidence": 0.8},
        temporal_window="1h",
        metadata={"source": "test"}
    )

class TestAgentProfileInitialization:
    """Test agent profile initialization and management"""
    
    def test_default_agent_profiles_loaded(self, semantic_router):
        """Test that default agent profiles are loaded correctly"""
        expected_agents = ["supervisor_agent", "analyst_agent", "observer_agent", "executor_agent"]
        
        for agent_id in expected_agents:
            assert agent_id in semantic_router.agent_profiles
            profile = semantic_router.agent_profiles[agent_id]
            assert isinstance(profile, AgentProfile)
            assert profile.agent_id == agent_id
            assert len(profile.capabilities) > 0
            assert 0.0 <= profile.performance_score <= 1.0
    
    def test_agent_capabilities_assignment(self, semantic_router):
        """Test that agents have appropriate capabilities"""
        supervisor = semantic_router.agent_profiles["supervisor_agent"]
        assert AgentCapability.DECISION_MAKING in supervisor.capabilities
        assert AgentCapability.COORDINATION in supervisor.capabilities
        
        analyst = semantic_router.agent_profiles["analyst_agent"]
        assert AgentCapability.ANALYSIS in analyst.capabilities
        assert AgentCapability.PATTERN_RECOGNITION in analyst.capabilities
        
        observer = semantic_router.agent_profiles["observer_agent"]
        assert AgentCapability.MONITORING in observer.capabilities
        assert AgentCapability.ANOMALY_DETECTION in observer.capabilities

class TestAgentFiltering:
    """Test agent filtering and availability"""
    
    def test_filter_available_agents_all_available(self, semantic_router):
        """Test filtering when all agents are available"""
        available = semantic_router._filter_available_agents(None)
        
        # Should return all agents since they're all available by default
        assert len(available) == 4
        assert "supervisor_agent" in available
        assert "analyst_agent" in available
        assert "observer_agent" in available
        assert "executor_agent" in available
    
    def test_filter_available_agents_with_unavailable(self, semantic_router):
        """Test filtering when some agents are unavailable"""
        # Make one agent unavailable
        semantic_router.agent_profiles["supervisor_agent"].availability = False
        
        available = semantic_router._filter_available_agents(None)
        
        assert len(available) == 3
        assert "supervisor_agent" not in available
        assert "analyst_agent" in available
    
    def test_filter_with_specific_agent_list(self, semantic_router):
        """Test filtering with specific agent list"""
        specific_agents = ["analyst_agent", "observer_agent"]
        available = semantic_router._filter_available_agents(specific_agents)
        
        assert len(available) == 2
        assert "analyst_agent" in available
        assert "observer_agent" in available
        assert "supervisor_agent" not in available

class TestCapabilityMatching:
    """Test capability matching logic"""
    
    def test_capability_match_perfect(self, semantic_router):
        """Test perfect capability match"""
        profile = semantic_router.agent_profiles["analyst_agent"]
        
        # Create analysis that matches analyst capabilities
        analysis = SemanticAnalysis(
            complexity_score=0.8,  # High complexity -> needs analysis
            urgency_level=UrgencyLevel.MEDIUM,
            coordination_pattern=OrchestrationStrategy.PARALLEL,
            suggested_agents=[],
            confidence=0.8,
            tda_correlation=None
        )
        
        match_score = semantic_router._calculate_capability_match(profile, analysis)
        assert match_score > 0.5  # Should be a good match
    
    def test_capability_match_poor(self, semantic_router):
        """Test poor capability match"""
        profile = semantic_router.agent_profiles["observer_agent"]  # Monitoring agent
        
        # Create analysis that doesn't match observer capabilities well
        analysis = SemanticAnalysis(
            complexity_score=0.3,  # Low complexity
            urgency_level=UrgencyLevel.LOW,
            coordination_pattern=OrchestrationStrategy.SEQUENTIAL,
            suggested_agents=[],
            confidence=0.8,
            tda_correlation=None
        )
        
        match_score = semantic_router._calculate_capability_match(profile, analysis)
        # Should still have some match due to base capabilities
        assert 0.0 <= match_score <= 1.0
    
    def test_required_capabilities_inference(self, semantic_router):
        """Test inference of required capabilities from analysis"""
        # High complexity analysis
        analysis = SemanticAnalysis(
            complexity_score=0.9,
            urgency_level=UrgencyLevel.HIGH,
            coordination_pattern=OrchestrationStrategy.HIERARCHICAL,
            suggested_agents=[],
            confidence=0.8,
            tda_correlation=None
        )
        
        capabilities = semantic_router._infer_required_capabilities(analysis)
        
        # Should infer multiple capabilities
        assert AgentCapability.COORDINATION in capabilities  # From hierarchical
        assert AgentCapability.DECISION_MAKING in capabilities  # From hierarchical
        assert AgentCapability.ANALYSIS in capabilities  # From high complexity
        assert AgentCapability.MONITORING in capabilities  # From high urgency

class TestAgentScoring:
    """Test agent scoring logic"""
    
    @pytest.mark.asyncio
    async def test_agent_scoring_basic(self, semantic_router, sample_semantic_analysis):
        """Test basic agent scoring"""
        candidate_agents = ["analyst_agent", "observer_agent"]
        
        scores = await semantic_router._score_agents_for_task(
            sample_semantic_analysis, candidate_agents
        )
        
        assert len(scores) == 2
        assert "analyst_agent" in scores
        assert "observer_agent" in scores
        
        # Scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_agent_scoring_with_load_balancing(self, semantic_router, sample_semantic_analysis):
        """Test that load balancing affects scoring"""
        # Set different loads for agents
        semantic_router.agent_profiles["analyst_agent"].current_load = 0.9  # High load
        semantic_router.agent_profiles["observer_agent"].current_load = 0.1  # Low load
        
        candidate_agents = ["analyst_agent", "observer_agent"]
        scores = await semantic_router._score_agents_for_task(
            sample_semantic_analysis, candidate_agents
        )
        
        # Observer should score higher due to lower load
        # (assuming similar capabilities)
        assert scores["observer_agent"] >= scores["analyst_agent"] * 0.8

class TestTDAIntegration:
    """Test TDA integration in routing"""
    
    @pytest.mark.asyncio
    async def test_tda_adjustments_anomaly_boost(self, semantic_router):
        """Test TDA adjustments boost anomaly detection agents during anomalies"""
        base_scores = {
            "analyst_agent": 0.7,
            "observer_agent": 0.6  # Has anomaly detection capability
       
       }
        
        # High anomaly TDA context
        tda_context = TDAContext(
            correlation_id="anomaly-test",
            pattern_confidence=0.8,
            anomaly_severity=0.8,  # High anomaly
            current_patterns={},
            temporal_window="1h",
            metadata={}
        )
        
        adjusted_scores = await semantic_router._apply_tda_adjustments(
            base_scores, tda_context
        )
        
        # Observer should get boost due to anomaly detection capability
        assert adjusted_scores["observer_agent"] > base_scores["observer_agent"]
    
    @pytest.mark.asyncio
    async def test_tda_adjustments_pattern_boost(self, semantic_router):
        """Test TDA adjustments boost pattern recognition agents with strong patterns"""
        base_scores = {
            "analyst_agent": 0.6,  # Has pattern recognition capability
            "executor_agent": 0.7
        }
        
        # Strong pattern TDA context
        tda_context = TDAContext(
            correlation_id="pattern-test",
            pattern_confidence=0.9,  # Strong patterns
            anomaly_severity=0.2,
            current_patterns={"strong_pattern": True},
            temporal_window="1h",
            metadata={}
        )
        
        adjusted_scores = await semantic_router._apply_tda_adjustments(
            base_scores, tda_context
        )
        
        # Analyst should get boost due to pattern recognition capability
        assert adjusted_scores["analyst_agent"] > base_scores["analyst_agent"]

class TestAgentSelectionStrategies:
    """Test agent selection based on orchestration strategies"""
    
    def test_sequential_strategy_selection(self, semantic_router):
        """Test agent selection for sequential strategy"""
        agent_scores = {
            "supervisor_agent": 0.9,
            "analyst_agent": 0.8,
            "observer_agent": 0.7,
            "executor_agent": 0.6
        }
        
        selected = semantic_router._select_agents_by_strategy(
            agent_scores, OrchestrationStrategy.SEQUENTIAL, 5
        )
        
        # Sequential should prefer fewer high-performing agents
        assert len(selected) <= 2
        assert "supervisor_agent" in selected  # Highest score
    
    def test_parallel_strategy_selection(self, semantic_router):
        """Test agent selection for parallel strategy"""
        agent_scores = {
            "supervisor_agent": 0.9,
            "analyst_agent": 0.8,
            "observer_agent": 0.7,
            "executor_agent": 0.6
        }
        
        selected = semantic_router._select_agents_by_strategy(
            agent_scores, OrchestrationStrategy.PARALLEL, 4
        )
        
        # Parallel should use more agents for load distribution
        assert len(selected) == 4
        assert all(agent in selected for agent in agent_scores.keys())
    
    def test_hierarchical_strategy_selection(self, semantic_router):
        """Test agent selection for hierarchical strategy"""
        agent_scores = {
            "supervisor_agent": 0.8,  # Has coordination capability
            "analyst_agent": 0.9,
            "observer_agent": 0.7,
            "executor_agent": 0.6
        }
        
        selected = semantic_router._select_agents_by_strategy(
            agent_scores, OrchestrationStrategy.HIERARCHICAL, 3
        )
        
        # Should prioritize coordinator
        assert "supervisor_agent" in selected
        # Coordinator should be first for hierarchical coordination
        assert selected[0] == "supervisor_agent"
    
    def test_consensus_strategy_selection(self, semantic_router):
        """Test agent selection for consensus strategy"""
        agent_scores = {
            "supervisor_agent": 0.9,  # Has decision making
            "analyst_agent": 0.8,     # Has decision making (indirectly)
            "observer_agent": 0.7,
            "executor_agent": 0.6
        }
        
        selected = semantic_router._select_agents_by_strategy(
            agent_scores, OrchestrationStrategy.CONSENSUS, 5
        )
        
        # Should include decision-making agents
        assert "supervisor_agent" in selected
        assert len(selected) <= 5

class TestRoutingDecisionGeneration:
    """Test complete routing decision generation"""
    
    @pytest.mark.asyncio
    async def test_complete_routing_decision(self, semantic_router, sample_semantic_analysis):
        """Test complete routing decision generation"""
        decision = await semantic_router.route_to_optimal_agents(
            sample_semantic_analysis,
            max_agents=3
        )
        
        assert isinstance(decision, RoutingDecision)
        assert len(decision.selected_agents) <= 3
        assert len(decision.selected_agents) > 0
        assert decision.routing_strategy == sample_semantic_analysis.coordination_pattern
        assert 0.0 <= decision.confidence <= 1.0
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0
        assert 0.0 <= decision.estimated_performance <= 1.0
    
    @pytest.mark.asyncio
    async def test_routing_with_tda_context(self, semantic_router, sample_tda_context):
        """Test routing decision with TDA context"""
        analysis_with_tda = SemanticAnalysis(
            complexity_score=0.7,
            urgency_level=UrgencyLevel.HIGH,
            coordination_pattern=OrchestrationStrategy.PARALLEL,
            suggested_agents=["analyst_agent"],
            confidence=0.8,
            tda_correlation=sample_tda_context
        )
        
        decision = await semantic_router.route_to_optimal_agents(
            analysis_with_tda,
            max_agents=3
        )
        
        assert decision.tda_influence > 0.0
        assert "TDA" in decision.reasoning or "anomaly" in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_fallback_agents_generation(self, semantic_router, sample_semantic_analysis):
        """Test that fallback agents are generated"""
        decision = await semantic_router.route_to_optimal_agents(
            sample_semantic_analysis,
            max_agents=2
        )
        
        # Should have fallback agents
        assert len(decision.fallback_agents) > 0
        
        # Fallback agents should not overlap with selected agents
        for fallback in decision.fallback_agents:
            assert fallback not in decision.selected_agents

class TestPerformanceUpdates:
    """Test agent performance updates"""
    
    @pytest.mark.asyncio
    async def test_agent_performance_update(self, semantic_router):
        """Test updating agent performance metrics"""
        agent_id = "analyst_agent"
        original_performance = semantic_router.agent_profiles[agent_id].performance_score
        
        new_metrics = {
            "performance_score": 0.95,
            "success_rate": 0.9,
            "current_load": 0.3,
            "availability": True
        }
        
        await semantic_router.update_agent_performance(agent_id, new_metrics)
        
        updated_profile = semantic_router.agent_profiles[agent_id]
        
        # Performance should be updated (weighted average)
        assert updated_profile.performance_score != original_performance
        assert updated_profile.last_success_rate == 0.9
        assert updated_profile.current_load == 0.3
        assert updated_profile.availability is True
    
    @pytest.mark.asyncio
    async def test_performance_update_nonexistent_agent(self, semantic_router):
        """Test performance update for non-existent agent"""
        # Should not raise exception
        await semantic_router.update_agent_performance(
            "nonexistent_agent", 
            {"performance_score": 0.8}
        )

class TestRoutingAnalytics:
    """Test routing analytics and insights"""
    
    @pytest.mark.asyncio
    async def test_routing_analytics_empty_history(self, semantic_router):
        """Test analytics with empty routing history"""
        analytics = semantic_router.get_routing_analytics()
        
        assert "message" in analytics
        assert "No routing history" in analytics["message"]
    
    @pytest.mark.asyncio
    async def test_routing_analytics_with_history(self, semantic_router, sample_semantic_analysis):
        """Test analytics with routing history"""
        # Generate some routing decisions
        for _ in range(5):
            await semantic_router.route_to_optimal_agents(sample_semantic_analysis)
        
        analytics = semantic_router.get_routing_analytics()
        
        assert "total_decisions" in analytics
        assert analytics["total_decisions"] == 5
        assert "average_confidence" in analytics
        assert "strategy_distribution" in analytics
        assert "agent_usage_frequency" in analytics
        assert 0.0 <= analytics["average_confidence"] <= 1.0

class TestConfidenceCalculation:
    """Test routing confidence calculation"""
    
    def test_confidence_calculation_high_scores(self, semantic_router, sample_semantic_analysis):
        """Test confidence calculation with high agent scores"""
        selected_agents = ["supervisor_agent", "analyst_agent"]
        agent_scores = {
            "supervisor_agent": 0.9,
            "analyst_agent": 0.85
        }
        
        confidence = semantic_router._calculate_routing_confidence(
            selected_agents, agent_scores, sample_semantic_analysis
        )
        
        assert 0.7 <= confidence <= 0.95  # Should be high confidence
    
    def test_confidence_calculation_low_scores(self, semantic_router, sample_semantic_analysis):
        """Test confidence calculation with low agent scores"""
        selected_agents = ["executor_agent"]
        agent_scores = {
            "executor_agent": 0.3
        }
        
        confidence = semantic_router._calculate_routing_confidence(
            selected_agents, agent_scores, sample_semantic_analysis
        )
        
        assert confidence < 0.7  # Should be lower confidence
    
    def test_confidence_calculation_empty_selection(self, semantic_router, sample_semantic_analysis):
        """Test confidence calculation with no selected agents"""
        confidence = semantic_router._calculate_routing_confidence(
            [], {}, sample_semantic_analysis
        )
        
        assert confidence == 0.0

class TestReasoningGeneration:
    """Test routing reasoning generation"""
    
    def test_reasoning_includes_strategy(self, semantic_router, sample_semantic_analysis):
        """Test that reasoning includes strategy information"""
        selected_agents = ["analyst_agent"]
        agent_scores = {"analyst_agent": 0.8}
        
        reasoning = semantic_router._generate_routing_reasoning(
            selected_agents, sample_semantic_analysis, agent_scores
        )
        
        assert "parallel" in reasoning.lower()  # Strategy from sample analysis
        assert "complexity" in reasoning.lower()
        assert "urgency" in reasoning.lower()
    
    def test_reasoning_includes_tda_influence(self, semantic_router, sample_tda_context):
        """Test that reasoning includes TDA influence when present"""
        analysis_with_tda = SemanticAnalysis(
            complexity_score=0.6,
            urgency_level=UrgencyLevel.MEDIUM,
            coordination_pattern=OrchestrationStrategy.PARALLEL,
            suggested_agents=["analyst_agent"],
            confidence=0.8,
            tda_correlation=sample_tda_context
        )
        
        selected_agents = ["observer_agent"]
        agent_scores = {"observer_agent": 0.8}
        
        reasoning = semantic_router._generate_routing_reasoning(
            selected_agents, analysis_with_tda, agent_scores
        )
        
        assert "tda" in reasoning.lower() or "anomaly" in reasoning.lower()

class TestPerformanceEstimation:
    """Test routing performance estimation"""
    
    def test_performance_estimation_high_performing_agents(self, semantic_router, sample_semantic_analysis):
        """Test performance estimation with high-performing agents"""
        # Set high performance for selected agents
        semantic_router.agent_profiles["supervisor_agent"].performance_score = 0.9
        semantic_router.agent_profiles["supervisor_agent"].last_success_rate = 0.95
        
        selected_agents = ["supervisor_agent"]
        
        estimated_perf = semantic_router._estimate_routing_performance(
            selected_agents, sample_semantic_analysis
        )
        
        assert estimated_perf > 0.8  # Should be high
    
    def test_performance_estimation_empty_selection(self, semantic_router, sample_semantic_analysis):
        """Test performance estimation with no selected agents"""
        estimated_perf = semantic_router._estimate_routing_performance(
            [], sample_semantic_analysis
        )
        
        assert estimated_perf == 0.0

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_emergency_scenario_routing(self, semantic_router):
        """Test routing for emergency scenario"""
        emergency_analysis = SemanticAnalysis(
            complexity_score=0.8,
            urgency_level=UrgencyLevel.CRITICAL,
            coordination_pattern=OrchestrationStrategy.PARALLEL,
            suggested_agents=["observer_agent", "supervisor_agent"],
            confidence=0.9,
            tda_correlation=TDAContext(
                correlation_id="emergency-123",
                pattern_confidence=0.95,
                anomaly_severity=0.9,
                current_patterns={"emergency": True},
                temporal_window="5m",
                metadata={}
            )
        )
        
        decision = await semantic_router.route_to_optimal_agents(
            emergency_analysis,
            max_agents=4
        )
        
        # Should select multiple agents for parallel execution
        assert len(decision.selected_agents) >= 2
        assert decision.routing_strategy == OrchestrationStrategy.PARALLEL
        assert decision.confidence > 0.8
        assert "observer_agent" in decision.selected_agents  # For monitoring
    
    @pytest.mark.asyncio
    async def test_routine_analysis_routing(self, semantic_router):
        """Test routing for routine analysis scenario"""
        routine_analysis = SemanticAnalysis(
            complexity_score=0.4,
            urgency_level=UrgencyLevel.LOW,
            coordination_pattern=OrchestrationStrategy.SEQUENTIAL,
            suggested_agents=["analyst_agent"],
            confidence=0.7,
            tda_correlation=None
        )
        
        decision = await semantic_router.route_to_optimal_agents(
            routine_analysis,
            max_agents=3
        )
        
        # Should select fewer agents for sequential execution
        assert len(decision.selected_agents) <= 2
        assert decision.routing_strategy == OrchestrationStrategy.SEQUENTIAL
        assert "analyst_agent" in decision.selected_agents

if __name__ == "__main__":
    pytest.main([__file__, "-v"])