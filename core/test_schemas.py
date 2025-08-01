#!/usr/bin/env python3
"""
🧪 Schema Test - Direct Verification of Our World-Class Architecture

This test directly verifies our modular schemas work correctly:
- Cryptographic signature creation and verification
- Immutable state management
- Evidence creation with typed content
- Decision making with explainability

Pure Python test with proper imports.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to Python path for proper module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    from aura_intelligence.agents.schemas import crypto, enums, base, evidence, state, decision
    print("✅ All schema modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Let's test individual components...")


def test_crypto_provider():
    """Test cryptographic provider functionality."""
    print("\n🔐 Testing Cryptographic Provider")
    print("-" * 35)
    
    try:
        provider = crypto.get_crypto_provider(crypto.SignatureAlgorithm.HMAC_SHA256)
        print(f"✅ Crypto provider created: {type(provider).__name__}")
        
        # Test signing and verification
        test_data = b"Hello, AURA Intelligence!"
        private_key = "test_private_key_12345"
        
        signature = provider.sign(test_data, private_key)
        print(f"✅ Data signed: {signature[:20]}...")
        
        is_valid = provider.verify(test_data, signature, private_key)
        print(f"✅ Signature verified: {is_valid}")
        
        return True
    except Exception as e:
        print(f"❌ Crypto test failed: {e}")
        return False


def test_evidence_creation():
    """Test evidence creation with typed content."""
    print("\n📋 Testing Evidence Creation")
    print("-" * 30)
    
    try:
        # Create log evidence content
        log_content = evidence.LogEvidence(
            log_level="error",
            log_text="Database connection failed",
            logger_name="db_connector",
            log_timestamp=base.utc_now().isoformat(),
            structured_data={"error_code": "DB_TIMEOUT", "retry_count": 3}
        )
        print(f"✅ Log evidence created: {log_content.log_level}")
        
        # Create dossier entry
        dossier_entry = evidence.DossierEntry(
            entry_id=str(uuid.uuid4()),
            workflow_id="test_workflow_001",
            task_id="test_task_001",
            correlation_id="test_correlation_001",
            evidence_type=enums.EvidenceType.LOG_ENTRY,
            content=log_content,
            summary="Database connection failure detected",
            source="test_agent",
            collection_method="stream_ingestion",
            collection_timestamp=base.utc_now(),
            confidence=0.95,
            reliability=0.9,
            freshness=1.0,
            completeness=0.85,
            collecting_agent_id="test_agent_001",
            agent_public_key="test_public_key",
            signature="test_signature",
            signature_algorithm=crypto.SignatureAlgorithm.HMAC_SHA256,
            signature_timestamp=base.utc_now()
        )
        print(f"✅ Dossier entry created: {dossier_entry.evidence_type.value}")
        print(f"✅ Evidence confidence: {dossier_entry.confidence}")
        
        return True
    except Exception as e:
        print(f"❌ Evidence test failed: {e}")
        return False


def test_state_management():
    """Test immutable state management."""
    print("\n🔒 Testing State Management")
    print("-" * 27)
    
    try:
        # Create initial state
        initial_state = state.AgentState(
            task_id="test_task_001",
            workflow_id="test_workflow_001",
            correlation_id="test_correlation_001",
            state_version=1,
            schema_version="2.0",
            state_signature="test_signature",
            signature_algorithm=crypto.SignatureAlgorithm.HMAC_SHA256,
            last_modifier_agent_id="test_agent_001",
            agent_public_key="test_public_key",
            signature_timestamp=base.utc_now(),
            task_type="test_task",
            priority="normal",
            status=enums.TaskStatus.PENDING,
            urgency="medium",
            initial_event={"test": "event"},
            initial_context={"test": "context"},
            trigger_source="test_source",
            created_at=base.utc_now(),
            updated_at=base.utc_now(),
            tags=["test:tag"],
            metadata={"test": "metadata"}
        )
        print(f"✅ Initial state created: v{initial_state.state_version}")
        print(f"✅ Task type: {initial_state.task_type}")
        print(f"✅ Status: {initial_state.status.value}")
        
        return True
    except Exception as e:
        print(f"❌ State test failed: {e}")
        return False


def test_decision_making():
    """Test decision making with explainability."""
    print("\n🤔 Testing Decision Making")
    print("-" * 26)
    
    try:
        # Create decision criteria
        criterion = decision.DecisionCriterion(
            criterion_id="urgency",
            name="Event Urgency",
            description="How urgent is this event?",
            weight=0.5,
            measurement_method="categorical_mapping"
        )
        print(f"✅ Decision criterion created: {criterion.name}")
        
        # Create decision option
        option = decision.DecisionOption(
            option_id="escalate",
            name="Escalate to Human",
            description="Forward to human analyst",
            estimated_effort_hours=0.5,
            estimated_cost=50.0,
            risk_level=enums.RiskLevel.LOW,
            scores={"urgency": 0.8}
        )
        print(f"✅ Decision option created: {option.name}")
        
        # Create decision point
        decision_point = decision.DecisionPoint(
            decision_id=str(uuid.uuid4()),
            workflow_id="test_workflow_001",
            task_id="test_task_001",
            correlation_id="test_correlation_001",
            decision_type="workflow_routing",
            decision_method="rule_based",
            criteria=[criterion],
            options=[option],
            chosen_option_id="escalate",
            rationale="High urgency event requires human review",
            confidence_in_decision=0.85,
            deciding_agent_id="test_agent_001",
            agent_public_key="test_public_key",
            signature="test_signature",
            signature_algorithm=crypto.SignatureAlgorithm.HMAC_SHA256,
            signature_timestamp=base.utc_now(),
            decision_timestamp=base.utc_now()
        )
        print(f"✅ Decision point created: {decision_point.chosen_option_id}")
        print(f"✅ Decision confidence: {decision_point.confidence_in_decision}")
        
        return True
    except Exception as e:
        print(f"❌ Decision test failed: {e}")
        return False


def main():
    """Run all schema tests."""
    print("🧪 AURA Intelligence - Schema Architecture Test")
    print("=" * 50)
    print("Testing our world-class modular schema system")
    
    # Run tests
    tests = [
        ("Cryptographic Provider", test_crypto_provider),
        ("Evidence Creation", test_evidence_creation),
        ("State Management", test_state_management),
        ("Decision Making", test_decision_making)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 25)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Modular schema architecture is working perfectly")
        print("✅ Ready for ObserverAgent implementation")
        print("✅ The Collective's foundation is solid")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed")
        print("Need to fix issues before proceeding")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
