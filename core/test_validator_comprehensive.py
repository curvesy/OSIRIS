#!/usr/bin/env python3
"""
🧠 Comprehensive Test Suite for ProfessionalPredictiveValidator

Tests the complete "prefrontal cortex" functionality including:
- Rule-based risk detection
- LLM-based validation for novel situations  
- Caching and performance optimization
- Accuracy tracking and meta-learning
- Human-in-the-loop governance
- Error handling and fallback decisions

This validates that our validator is production-ready for Phase 3A.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from aura_intelligence.agents.validator import (
        ProfessionalPredictiveValidator, 
        ValidationResult,
        AccuracyTracker,
        create_professional_validator
    )
    print("✅ Successfully imported validator components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class MockLLM:
    """Mock LLM for testing that returns structured JSON responses."""
    
    def __init__(self, response_template: Dict[str, Any] = None):
        self.response_template = response_template or {
            "predicted_success_probability": 0.7,
            "prediction_confidence_score": 0.8,
            "risk_score": 0.3,
            "predicted_risks": [
                {
                    "risk": "Standard operational risk",
                    "mitigation": "Follow standard procedures and monitoring"
                }
            ],
            "reasoning_trace": "Mock LLM analysis based on provided context"
        }
        self.call_count = 0
        self.last_messages = None
    
    async def ainvoke(self, messages):
        """Mock async LLM invocation that returns structured JSON."""
        self.call_count += 1
        self.last_messages = messages
        
        # Return a mock response object with content
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(json.dumps(self.response_template))


async def test_rule_based_validation():
    """Test that high-risk actions are caught by the rulebook."""
    print("\n🔍 Testing Rule-Based Validation...")
    
    mock_llm = MockLLM()
    validator = ProfessionalPredictiveValidator(mock_llm, risk_threshold=0.75)
    
    # Test critical service restart (should trigger rulebook)
    result = await validator.validate_action(
        proposed_action="restart critical service database",
        memory_context=[],
        current_evidence=[{"type": "system_status", "status": "operational"}]
    )
    
    assert result.risk_score >= 0.9, f"Expected high risk score, got {result.risk_score}"
    assert result.requires_human_approval, "Critical action should require human approval"
    assert "critical service restart" in result.reasoning_trace.lower()
    assert mock_llm.call_count == 0, "Should not call LLM for rule-based decisions"
    
    print("✅ Rule-based validation working correctly")
    return True


async def test_llm_validation():
    """Test LLM-based validation for novel situations."""
    print("\n🧠 Testing LLM-Based Validation...")
    
    # Configure mock LLM with specific response
    mock_response = {
        "predicted_success_probability": 0.85,
        "prediction_confidence_score": 0.9,
        "risk_score": 0.2,
        "predicted_risks": [
            {
                "risk": "Minor configuration drift",
                "mitigation": "Validate configuration before applying"
            }
        ],
        "reasoning_trace": "Low-risk routine operation with high success probability"
    }
    
    mock_llm = MockLLM(mock_response)
    validator = ProfessionalPredictiveValidator(mock_llm, risk_threshold=0.75)
    
    # Test routine operation (should use LLM)
    result = await validator.validate_action(
        proposed_action="update application configuration",
        memory_context=[{"workflow_id": "prev_123", "action": "config_update", "success": True}],
        current_evidence=[{"type": "config_status", "status": "stable"}]
    )
    
    assert result.predicted_success_probability == 0.85
    assert result.prediction_confidence_score == 0.9
    assert result.risk_score == 0.2
    assert not result.requires_human_approval, "Low-risk action should not require approval"
    assert mock_llm.call_count == 1, "Should call LLM for novel situations"
    
    # Verify LLM received proper context
    assert mock_llm.last_messages is not None
    assert len(mock_llm.last_messages) == 2  # System + Human message
    
    print("✅ LLM-based validation working correctly")
    return True


async def test_caching_mechanism():
    """Test that validation results are properly cached."""
    print("\n💾 Testing Caching Mechanism...")
    
    mock_llm = MockLLM()
    validator = ProfessionalPredictiveValidator(mock_llm, cache_ttl_seconds=3600)
    
    action = "routine maintenance task"
    evidence = [{"type": "maintenance", "status": "scheduled"}]
    
    # First call should hit LLM
    result1 = await validator.validate_action(action, [], evidence)
    assert mock_llm.call_count == 1
    assert not result1.cache_hit
    
    # Second identical call should hit cache
    result2 = await validator.validate_action(action, [], evidence)
    assert mock_llm.call_count == 1  # No additional LLM call
    assert result2.cache_hit
    
    # Results should be identical
    assert result1.predicted_success_probability == result2.predicted_success_probability
    assert result1.risk_score == result2.risk_score
    
    print("✅ Caching mechanism working correctly")
    return True


async def test_accuracy_tracking():
    """Test accuracy tracking and meta-learning capabilities."""
    print("\n📊 Testing Accuracy Tracking...")
    
    tracker = AccuracyTracker(max_size=100)
    
    # Create mock validation results
    high_confidence_result = ValidationResult(
        predicted_success_probability=0.9,
        prediction_confidence_score=0.95,
        risk_score=0.1,
        predicted_risks=[],
        reasoning_trace="High confidence prediction"
    )
    
    low_confidence_result = ValidationResult(
        predicted_success_probability=0.6,
        prediction_confidence_score=0.4,
        risk_score=0.5,
        predicted_risks=[],
        reasoning_trace="Low confidence prediction"
    )
    
    # Log some predictions with outcomes
    tracker.log_prediction(high_confidence_result, True)  # Correct prediction
    tracker.log_prediction(high_confidence_result, True)  # Correct prediction
    tracker.log_prediction(low_confidence_result, False)  # Correct prediction
    tracker.log_prediction(high_confidence_result, False) # Incorrect prediction
    
    accuracy = tracker.get_accuracy()
    assert 0.5 <= accuracy <= 1.0, f"Accuracy should be reasonable, got {accuracy}"
    
    calibration = tracker.get_confidence_calibration()
    assert 'calibration_error' in calibration
    assert 'sample_size' in calibration
    
    print(f"✅ Accuracy tracking working correctly (accuracy: {accuracy:.2f})")
    return True


async def test_error_handling():
    """Test robust error handling and fallback decisions."""
    print("\n🛡️ Testing Error Handling...")
    
    # Mock LLM that returns invalid JSON
    class BrokenLLM:
        async def ainvoke(self, messages):
            class MockResponse:
                content = "This is not valid JSON at all!"
            return MockResponse()
    
    broken_llm = BrokenLLM()
    validator = ProfessionalPredictiveValidator(broken_llm)
    
    # Should fallback gracefully on LLM failure
    result = await validator.validate_action(
        proposed_action="some novel action",
        memory_context=[],
        current_evidence=[]
    )
    
    assert result.risk_score >= 0.8, "Should default to high risk on error"
    assert result.requires_human_approval, "Should require approval on error"
    assert "parsing failed" in result.reasoning_trace.lower()
    
    print("✅ Error handling working correctly")
    return True


async def test_performance_metrics():
    """Test performance metrics collection."""
    print("\n📈 Testing Performance Metrics...")
    
    mock_llm = MockLLM()
    validator = ProfessionalPredictiveValidator(mock_llm)
    
    # Perform some validations
    await validator.validate_action("restart service", [], [])  # Rule hit
    await validator.validate_action("novel action", [], [])     # LLM call
    await validator.validate_action("novel action", [], [])     # Cache hit
    
    metrics = validator.get_performance_metrics()
    
    assert metrics['total_validations'] == 3
    assert metrics['rule_hits'] == 1
    assert metrics['llm_calls'] == 1
    assert metrics['cache_hits'] == 1
    assert 0 <= metrics['cache_hit_rate'] <= 1
    
    print("✅ Performance metrics working correctly")
    return True


async def test_factory_function():
    """Test the factory function for easy instantiation."""
    print("\n🏭 Testing Factory Function...")
    
    mock_llm = MockLLM()
    validator = create_professional_validator(
        llm=mock_llm,
        risk_threshold=0.8,
        cache_ttl_seconds=1800
    )
    
    assert isinstance(validator, ProfessionalPredictiveValidator)
    assert validator.risk_threshold == 0.8
    assert validator.cache_ttl.total_seconds() == 1800
    
    print("✅ Factory function working correctly")
    return True


async def main():
    """Run comprehensive test suite."""
    print("🧠 Professional Predictive Validator - Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        test_rule_based_validation,
        test_llm_validation,
        test_caching_mechanism,
        test_accuracy_tracking,
        test_error_handling,
        test_performance_metrics,
        test_factory_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 70)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! ProfessionalPredictiveValidator is ready for Phase 3A!")
        return True
    else:
        print("❌ Some tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
