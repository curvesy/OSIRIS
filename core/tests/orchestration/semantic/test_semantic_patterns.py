"""
🧪 Tests for Semantic Pattern Matcher

Comprehensive test suite for semantic pattern matching with TDA integration.
Tests complexity calculation, urgency determination, pattern selection,
and TDA correlation functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from aura_intelligence.orchestration.semantic.semantic_patterns import SemanticPatternMatcher
from aura_intelligence.orchestration.semantic.base_interfaces import (
    TDAContext, OrchestrationStrategy, UrgencyLevel, TDAIntegration
)

class MockTDAIntegration(TDAIntegration):
    """Mock TDA integration for testing"""
    
    def __init__(self):
        self.contexts = {}
        self.results = []
        self.patterns = {}
    
    async def get_context(self, correlation_id: str) -> TDAContext:
        return self.contexts.get(correlation_id)
    
    async def send_orchestration_result(self, result, correlation_id: str) -> bool:
        self.results.append((result, correlation_id))
        return True
    
    async def get_current_patterns(self, window: str = "1h"):
        return self.patterns.get(window, {})

@pytest.fixture
def pattern_matcher():
    """Create pattern matcher with mock TDA integration"""
    tda_integration = MockTDAIntegration()
    return SemanticPatternMatcher(tda_integration), tda_integration

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

class TestComplexityCalculation:
    """Test complexity score calculation"""
    
    @pytest.mark.asyncio
    async def test_low_complexity_task(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "simple analysis",
            "requirements": ["basic check"]
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.complexity_score < 0.3
    
    @pytest.mark.asyncio
    async def test_high_complexity_task(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "complex multi-agent coordination with consensus",
            "requirements": [f"requirement_{i}" for i in range(15)],
            "agent_dependencies": [f"agent_{i}" for i in range(8)],
            "requires_consensus": True,
            "deadline": (datetime.utcnow() + timedelta(hours=2)).isoformat()
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.complexity_score > 0.7
    
    @pytest.mark.asyncio
    async def test_complexity_with_tda_amplification(self, pattern_matcher, sample_tda_context):
        matcher, tda_mock = pattern_matcher
        tda_mock.contexts["test-correlation-123"] = sample_tda_context
        
        input_data = {
            "task": "medium complexity task",
            "requirements": ["req1", "req2", "req3"]
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data, sample_tda_context)
        
        # Should be amplified by TDA pattern confidence
        assert analysis.complexity_score > 0.3
        assert analysis.tda_correlation == sample_tda_context

class TestUrgencyDetermination:
    """Test urgency level determination"""
    
    @pytest.mark.asyncio
    async def test_explicit_critical_urgency(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "emergency response",
            "critical": True
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.urgency_level == UrgencyLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_deadline_based_urgency(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        # Deadline in 30 minutes - should be critical
        near_deadline = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        input_data = {
            "task": "time-sensitive task",
            "deadline": near_deadline
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.urgency_level == UrgencyLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_tda_urgency_escalation(self, pattern_matcher, sample_tda_context):
        matcher, _ = pattern_matcher
        
        # High anomaly severity should escalate urgency
        high_anomaly_context = TDAContext(
            correlation_id="high-anomaly",
            pattern_confidence=0.9,
            anomaly_severity=0.95,  # Very high anomaly
            current_patterns={},
            temporal_window="1h",
            metadata={}
        )
        
        input_data = {
            "task": "normal task",
            "urgency": "medium"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data, high_anomaly_context)
        assert analysis.urgency_level == UrgencyLevel.CRITICAL

class TestCoordinationPatternSelection:
    """Test coordination pattern selection logic"""
    
    @pytest.mark.asyncio
    async def test_parallel_pattern_for_high_complexity(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "highly complex multi-agent task",
            "requirements": [f"req_{i}" for i in range(20)],
            "agent_dependencies": [f"agent_{i}" for i in range(10)],
            "requires_consensus": True,
            "urgency": "high"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.coordination_pattern == OrchestrationStrategy.PARALLEL
    
    @pytest.mark.asyncio
    async def test_sequential_pattern_for_critical_low_complexity(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "simple but urgent task",
            "requirements": ["quick check"],
            "critical": True
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.coordination_pattern == OrchestrationStrategy.SEQUENTIAL
    
    @pytest.mark.asyncio
    async def test_consensus_pattern_for_medium_complexity(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "medium complexity decision task",
            "requirements": [f"req_{i}" for i in range(8)],
            "urgency": "medium"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert analysis.coordination_pattern == OrchestrationStrategy.CONSENSUS

class TestAgentSuggestion:
    """Test agent suggestion logic"""
    
    @pytest.mark.asyncio
    async def test_analysis_task_agent_suggestion(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task_type": "analysis",
            "domain": "data_analysis"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert "analyst_agent" in analysis.suggested_agents
        assert "observer_agent" in analysis.suggested_agents
    
    @pytest.mark.asyncio
    async def test_decision_task_agent_suggestion(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task_type": "decision",
            "domain": "strategic"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        assert "supervisor_agent" in analysis.suggested_agents
        assert "analyst_agent" in analysis.suggested_agents
    
    @pytest.mark.asyncio
    async def test_tda_enhanced_agent_suggestion(self, pattern_matcher, sample_tda_context):
        matcher, _ = pattern_matcher
        
        # High anomaly should suggest observer agent
        high_anomaly_context = TDAContext(
            correlation_id="anomaly-test",
            pattern_confidence=0.9,
            anomaly_severity=0.8,
            current_patterns={"anomaly_detected": True},
            temporal_window="1h",
            metadata={}
        )
        
        input_data = {
            "task_type": "execution",
            "domain": "system"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data, high_anomaly_context)
        assert "observer_agent" in analysis.suggested_agents

class TestPatternInsights:
    """Test pattern insights generation"""
    
    @pytest.mark.asyncio
    async def test_pattern_insights_generation(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "complex analysis task",
            "requirements": [f"req_{i}" for i in range(10)],
            "urgency": "high"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        insights = await matcher.get_pattern_insights(analysis)
        
        assert "complexity_breakdown" in insights
        assert "urgency_analysis" in insights
        assert "coordination_rationale" in insights
        assert "agent_selection" in insights
        assert "confidence_factors" in insights
        
        # Check structure
        assert "score" in insights["complexity_breakdown"]
        assert "level" in insights["complexity_breakdown"]
        assert "level" in insights["urgency_analysis"]
        assert "pattern" in insights["coordination_rationale"]
        assert "reason" in insights["coordination_rationale"]

class TestConfidenceCalculation:
    """Test confidence calculation"""
    
    @pytest.mark.asyncio
    async def test_confidence_with_tda_boost(self, pattern_matcher, sample_tda_context):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "well-defined task",
            "requirements": ["clear requirement"]
        }
        
        analysis_without_tda = await matcher.analyze_semantic_patterns(input_data)
        analysis_with_tda = await matcher.analyze_semantic_patterns(input_data, sample_tda_context)
        
        # TDA context should boost confidence
        assert analysis_with_tda.confidence > analysis_without_tda.confidence
    
    @pytest.mark.asyncio
    async def test_confidence_bounds(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "any task"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        
        # Confidence should be between 0 and 1
        assert 0.0 <= analysis.confidence <= 1.0

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_emergency_response_scenario(self, pattern_matcher):
        matcher, tda_mock = pattern_matcher
        
        # Emergency scenario with TDA anomaly
        emergency_context = TDAContext(
            correlation_id="emergency-123",
            pattern_confidence=0.95,
            anomaly_severity=0.9,
            current_patterns={"emergency_detected": True, "system_failure": True},
            temporal_window="5m",
            metadata={"alert_level": "critical"}
        )
        
        input_data = {
            "task": "emergency system recovery",
            "task_type": "execution",
            "emergency": True,
            "requires_consensus": False,
            "deadline": (datetime.utcnow() + timedelta(minutes=15)).isoformat()
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data, emergency_context)
        
        # Should be critical urgency with parallel execution
        assert analysis.urgency_level == UrgencyLevel.CRITICAL
        assert analysis.coordination_pattern == OrchestrationStrategy.PARALLEL
        assert "observer_agent" in analysis.suggested_agents
        assert analysis.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_routine_analysis_scenario(self, pattern_matcher):
        matcher, _ = pattern_matcher
        
        input_data = {
            "task": "daily data analysis",
            "task_type": "analysis",
            "requirements": ["data validation", "trend analysis"],
            "urgency": "low"
        }
        
        analysis = await matcher.analyze_semantic_patterns(input_data)
        
        # Should be low urgency with event-driven pattern
        assert analysis.urgency_level == UrgencyLevel.LOW
        assert analysis.coordination_pattern == OrchestrationStrategy.EVENT_DRIVEN
        assert "analyst_agent" in analysis.suggested_agents

if __name__ == "__main__":
    pytest.main([__file__, "-v"])