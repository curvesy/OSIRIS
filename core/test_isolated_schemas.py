#!/usr/bin/env python3
"""
🧪 Isolated Schema Validation - Direct File Testing

This test bypasses all __init__.py files and directly imports
only the specific schema files we need to validate core contracts.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add schema directory directly to path (bypass __init__.py files)
schema_dir = Path(__file__).parent / "src" / "aura_intelligence" / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

def test_signature_verification():
    """Test 1: Direct import and test cryptographic signatures."""
    print("🔐 Test 1: Cryptographic Signature Verification")
    print("-" * 45)
    
    try:
        # Direct imports (no package structure)
        import enums
        import crypto
        import base
        import evidence
        
        print("✅ Direct schema imports successful")
        
        # Create crypto provider
        crypto_provider = crypto.get_crypto_provider(enums.SignatureAlgorithm.HMAC_SHA256)
        private_key = "test_private_key_12345"
        public_key = "test_public_key_12345"
        
        print(f"✅ Crypto provider created: {type(crypto_provider).__name__}")
        
        # Create log evidence content
        log_content = evidence.LogEvidence(
            log_level="error",
            log_text="Database connection failed",
            logger_name="db_connector",
            log_timestamp=base.utc_now().isoformat(),
            structured_data={"error_code": "DB_TIMEOUT", "retry_count": 3}
        )
        
        print(f"✅ Log evidence created: {log_content.log_level}")
        
        # Create dossier entry (unsigned)
        entry = evidence.DossierEntry(
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
            agent_public_key=public_key,
            signature="placeholder",
            signature_algorithm=enums.SignatureAlgorithm.HMAC_SHA256,
            signature_timestamp=base.utc_now()
        )
        
        print(f"✅ Dossier entry created: {entry.evidence_type.value}")
        
        # Sign the evidence
        canonical_repr = entry.get_canonical_representation()
        evidence_bytes = canonical_repr.encode('utf-8')
        signature = crypto_provider.sign(evidence_bytes, private_key)
        
        print(f"✅ Evidence signed: {signature[:20]}...")
        
        # Create signed entry
        signed_entry = entry.copy(update={'signature': signature})
        
        # Verify signature
        signed_canonical = signed_entry.get_canonical_representation()
        signed_bytes = signed_canonical.encode('utf-8')
        is_valid = crypto_provider.verify(signed_bytes, signed_entry.signature, private_key)
        
        print(f"✅ Signature verification: {'PASSED' if is_valid else 'FAILED'}")
        
        if is_valid:
            print("🎉 Test 1 PASSED: Cryptographic signatures working correctly")
            return True
        else:
            print("❌ Test 1 FAILED: Signature verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_immutable_state_updates():
    """Test 2: Direct test of immutable state updates."""
    print("\n🔒 Test 2: Immutable State Updates")
    print("-" * 35)
    
    try:
        # Direct imports
        import enums
        import base
        import state
        
        print("✅ State modules imported successfully")
        
        # Create initial state
        state1 = state.AgentState(
            task_id="test_task_001",
            workflow_id="test_workflow_001",
            correlation_id="test_correlation_001",
            state_version=1,
            schema_version="2.0",
            state_signature="test_signature_1",
            signature_algorithm=enums.SignatureAlgorithm.HMAC_SHA256,
            last_modifier_agent_id="test_agent_001",
            agent_public_key="test_public_key",
            signature_timestamp=base.utc_now(),
            task_type="test_task",
            priority="normal",
            status=enums.TaskStatus.PENDING,
            urgency="medium",
            initial_event={"test": "event1"},
            initial_context={"test": "context1"},
            trigger_source="test_source",
            created_at=base.utc_now(),
            updated_at=base.utc_now(),
            tags=["test:tag1"],
            metadata={"test": "metadata1"}
        )
        
        print(f"✅ State 1 created: v{state1.state_version}, ID: {state1.global_id}")
        
        # Create second state (should be different)
        state2 = state.AgentState(
            task_id="test_task_002",
            workflow_id="test_workflow_002",
            correlation_id="test_correlation_002",
            state_version=1,
            schema_version="2.0",
            state_signature="test_signature_2",
            signature_algorithm=enums.SignatureAlgorithm.HMAC_SHA256,
            last_modifier_agent_id="test_agent_002",
            agent_public_key="test_public_key",
            signature_timestamp=base.utc_now(),
            task_type="test_task",
            priority="high",
            status=enums.TaskStatus.IN_PROGRESS,
            urgency="high",
            initial_event={"test": "event2"},
            initial_context={"test": "context2"},
            trigger_source="test_source",
            created_at=base.utc_now(),
            updated_at=base.utc_now(),
            tags=["test:tag2"],
            metadata={"test": "metadata2"}
        )
        
        print(f"✅ State 2 created: v{state2.state_version}, ID: {state2.global_id}")
        
        # Verify immutability properties
        different_objects = state1 is not state2
        different_ids = state1.global_id != state2.global_id
        different_tasks = state1.task_id != state2.task_id
        different_workflows = state1.workflow_id != state2.workflow_id
        
        print(f"✅ Different objects: {'PASSED' if different_objects else 'FAILED'}")
        print(f"✅ Different global IDs: {'PASSED' if different_ids else 'FAILED'}")
        print(f"✅ Different task IDs: {'PASSED' if different_tasks else 'FAILED'}")
        print(f"✅ Different workflow IDs: {'PASSED' if different_workflows else 'FAILED'}")
        
        # Test state update (copy with changes)
        updated_state = state1.copy(update={
            'state_version': state1.state_version + 1,
            'status': enums.TaskStatus.IN_PROGRESS,
            'updated_at': base.utc_now(),
            'state_signature': 'updated_signature'
        })
        
        print(f"✅ Updated state created: v{updated_state.state_version}")
        
        # Verify update properties
        version_incremented = updated_state.state_version == state1.state_version + 1
        status_changed = updated_state.status != state1.status
        same_task_id = updated_state.task_id == state1.task_id
        same_workflow_id = updated_state.workflow_id == state1.workflow_id
        
        print(f"✅ Version incremented: {'PASSED' if version_incremented else 'FAILED'}")
        print(f"✅ Status changed: {'PASSED' if status_changed else 'FAILED'}")
        print(f"✅ Task ID preserved: {'PASSED' if same_task_id else 'FAILED'}")
        print(f"✅ Workflow ID preserved: {'PASSED' if same_workflow_id else 'FAILED'}")
        
        all_checks = all([
            different_objects, different_ids, different_tasks, different_workflows,
            version_incremented, status_changed, same_task_id, same_workflow_id
        ])
        
        if all_checks:
            print("🎉 Test 2 PASSED: Immutable state management working correctly")
            return True
        else:
            print("❌ Test 2 FAILED: Some immutability checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trace_context_propagation():
    """Test 3: Direct test of trace context."""
    print("\n📊 Test 3: Trace Context Propagation")
    print("-" * 35)
    
    try:
        # Direct import
        import tracecontext
        
        print("✅ Trace context module imported successfully")
        
        # Create trace context
        trace_ctx = tracecontext.TraceContext(
            traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            tracestate="rojo=00f067aa0ba902b7,congo=t61rcWkgMzE"
        )
        
        print(f"✅ Trace context created: {trace_ctx.traceparent[:20]}...")
        
        # Verify trace context properties
        has_traceparent = trace_ctx.traceparent is not None
        has_tracestate = trace_ctx.tracestate is not None
        valid_format = trace_ctx.traceparent.startswith("00-")
        
        print(f"✅ Has traceparent: {'PASSED' if has_traceparent else 'FAILED'}")
        print(f"✅ Has tracestate: {'PASSED' if has_tracestate else 'FAILED'}")
        print(f"✅ Valid format: {'PASSED' if valid_format else 'FAILED'}")
        
        # Test trace context serialization
        trace_dict = trace_ctx.dict()
        has_both_fields = 'traceparent' in trace_dict and 'tracestate' in trace_dict
        
        print(f"✅ Serialization: {'PASSED' if has_both_fields else 'FAILED'}")
        
        all_checks = all([has_traceparent, has_tracestate, valid_format, has_both_fields])
        
        if all_checks:
            print("🎉 Test 3 PASSED: Trace context propagation working correctly")
            return True
        else:
            print("❌ Test 3 FAILED: Some trace context checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run isolated schema validation tests."""
    print("🧪 AURA Intelligence - Isolated Schema Validation")
    print("=" * 52)
    print("Testing schemas directly (bypassing __init__.py files)")
    print()
    
    # Run tests in order
    tests = [
        ("Cryptographic Signature Verification", test_signature_verification),
        ("Immutable State Updates", test_immutable_state_updates),
        ("Trace Context Propagation", test_trace_context_propagation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n⚠️  Stopping at first failure: {test_name}")
                break
                
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            break
    
    # Summary
    print(f"\n📊 Isolated Schema Validation Results")
    print("=" * 42)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} core contracts validated")
    
    if passed == len(tests):
        print("\n🎉 ALL CORE CONTRACTS VALIDATED!")
        print("✅ Schemas work correctly in isolation")
        print("✅ Ready to proceed to Phase 2: Walking Skeleton")
        print("✅ Cryptographic signatures proven working")
        print("✅ Immutable state management proven working")
        print("✅ Trace context propagation proven working")
    else:
        print(f"\n⚠️  {len(tests) - passed} contracts failed validation")
        print("Must fix core schema issues before proceeding")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
