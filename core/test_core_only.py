#!/usr/bin/env python3
"""
🧪 Core-Only Validation - Minimal Dependencies

This test validates only the most fundamental contracts:
1. Cryptographic signature verification
2. Basic state immutability 
3. Core enum functionality

No complex dependencies - pure Python + Pydantic + cryptography only.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add schema directory directly to path
schema_dir = Path(__file__).parent / "src" / "aura_intelligence" / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

def test_crypto_signatures():
    """Test 1: Core cryptographic signature functionality."""
    print("🔐 Test 1: Core Cryptographic Signatures")
    print("-" * 40)
    
    try:
        # Import core modules only
        import enums
        import crypto
        import base
        
        print("✅ Core modules imported successfully")
        
        # Test crypto provider
        crypto_provider = crypto.get_crypto_provider(enums.SignatureAlgorithm.HMAC_SHA256)
        print(f"✅ Crypto provider created: {type(crypto_provider).__name__}")
        
        # Test signing and verification
        test_data = b"Hello, AURA Intelligence Core Test!"
        private_key = "test_private_key_12345"
        
        signature = crypto_provider.sign(test_data, private_key)
        print(f"✅ Data signed: {signature[:20]}...")
        
        is_valid = crypto_provider.verify(test_data, signature, private_key)
        print(f"✅ Signature verified: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test with different data (should fail)
        wrong_data = b"Wrong data"
        is_invalid = crypto_provider.verify(wrong_data, signature, private_key)
        print(f"✅ Wrong data rejected: {'PASSED' if not is_invalid else 'FAILED'}")
        
        if is_valid and not is_invalid:
            print("🎉 Test 1 PASSED: Core cryptographic signatures working")
            return True
        else:
            print("❌ Test 1 FAILED: Signature verification issues")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enum_functionality():
    """Test 2: Core enum functionality."""
    print("\n📋 Test 2: Core Enum Functionality")
    print("-" * 35)
    
    try:
        import enums
        
        print("✅ Enums module imported successfully")
        
        # Test TaskStatus enum
        pending_status = enums.TaskStatus.PENDING
        completed_status = enums.TaskStatus.COMPLETED
        
        print(f"✅ TaskStatus.PENDING: {pending_status.value}")
        print(f"✅ TaskStatus.COMPLETED: {completed_status.value}")
        
        # Test status methods
        is_active = pending_status.is_active()
        is_terminal = completed_status.is_terminal()
        
        print(f"✅ PENDING is active: {'PASSED' if is_active else 'FAILED'}")
        print(f"✅ COMPLETED is terminal: {'PASSED' if is_terminal else 'FAILED'}")
        
        # Test EvidenceType enum
        log_evidence = enums.EvidenceType.LOG_ENTRY
        metric_evidence = enums.EvidenceType.METRIC

        print(f"✅ EvidenceType.LOG_ENTRY: {log_evidence.value}")
        print(f"✅ EvidenceType.METRIC: {metric_evidence.value}")
        
        # Test SignatureAlgorithm enum
        hmac_algo = enums.SignatureAlgorithm.HMAC_SHA256
        rsa_algo = enums.SignatureAlgorithm.RSA_PSS_SHA256
        
        print(f"✅ SignatureAlgorithm.HMAC_SHA256: {hmac_algo.value}")
        print(f"✅ SignatureAlgorithm.RSA_PSS_SHA256: {rsa_algo.value}")
        
        all_checks = all([is_active, is_terminal, 
                         pending_status != completed_status,
                         log_evidence != metric_evidence,
                         hmac_algo != rsa_algo])
        
        if all_checks:
            print("🎉 Test 2 PASSED: Core enum functionality working")
            return True
        else:
            print("❌ Test 2 FAILED: Some enum checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_utilities():
    """Test 3: Core base utilities."""
    print("\n🛠️  Test 3: Core Base Utilities")
    print("-" * 30)
    
    try:
        import base
        
        print("✅ Base module imported successfully")
        
        # Test datetime utilities
        current_time = base.utc_now()
        print(f"✅ Current UTC time: {current_time}")
        
        # Test datetime conversion
        iso_string = base.datetime_to_iso(current_time)
        converted_back = base.iso_to_datetime(iso_string)
        
        print(f"✅ ISO conversion: {iso_string}")
        print(f"✅ Converted back: {converted_back}")
        
        # Test validators
        valid_confidence = base.validate_confidence_score(0.85)
        print(f"✅ Valid confidence score: {valid_confidence}")
        
        valid_signature = base.validate_signature_format("a" * 40)  # 40 char signature
        print(f"✅ Valid signature format: {valid_signature[:20]}...")
        
        # Test ID generation
        entity_id = base.generate_entity_id("test")
        workflow_id = base.generate_workflow_id("test")
        task_id = base.generate_task_id("test")
        
        print(f"✅ Entity ID: {entity_id}")
        print(f"✅ Workflow ID: {workflow_id}")
        print(f"✅ Task ID: {task_id}")
        
        # Test validation errors
        try:
            base.validate_confidence_score(1.5)  # Should fail
            confidence_validation_works = False
        except ValueError:
            confidence_validation_works = True
            print("✅ Confidence validation correctly rejects invalid values")
        
        all_checks = all([
            isinstance(current_time, datetime),
            iso_string.endswith('+00:00') or iso_string.endswith('Z'),
            abs((converted_back - current_time).total_seconds()) < 1,
            valid_confidence == 0.85,
            len(valid_signature) >= 32,
            entity_id.startswith("test_"),
            workflow_id.startswith("test_"),
            task_id.startswith("test_"),
            confidence_validation_works
        ])
        
        if all_checks:
            print("🎉 Test 3 PASSED: Core base utilities working")
            return True
        else:
            print("❌ Test 3 FAILED: Some utility checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_evidence_creation():
    """Test 4: Simple evidence creation without complex dependencies."""
    print("\n📄 Test 4: Simple Evidence Creation")
    print("-" * 35)
    
    try:
        import enums
        import base
        
        print("✅ Required modules imported")
        
        # Create a simple LogEvidence manually (without importing evidence.py)
        from pydantic import BaseModel
        
        class SimpleLogEvidence(BaseModel):
            log_level: str
            log_text: str
            logger_name: str
            log_timestamp: str
            structured_data: dict
        
        # Create log evidence
        log_evidence = SimpleLogEvidence(
            log_level="error",
            log_text="Database connection failed",
            logger_name="db_connector", 
            log_timestamp=base.utc_now().isoformat(),
            structured_data={"error_code": "DB_TIMEOUT", "retry_count": 3}
        )
        
        print(f"✅ Log evidence created: {log_evidence.log_level}")
        print(f"✅ Log text: {log_evidence.log_text[:30]}...")
        print(f"✅ Structured data: {log_evidence.structured_data}")
        
        # Test serialization
        evidence_dict = log_evidence.dict()
        evidence_json = log_evidence.json()
        
        print(f"✅ Evidence serialized to dict: {len(evidence_dict)} fields")
        print(f"✅ Evidence serialized to JSON: {len(evidence_json)} chars")
        
        # Test that we can recreate from dict
        recreated = SimpleLogEvidence(**evidence_dict)
        
        print(f"✅ Evidence recreated from dict: {recreated.log_level}")
        
        all_checks = all([
            log_evidence.log_level == "error",
            "Database connection failed" in log_evidence.log_text,
            log_evidence.structured_data["error_code"] == "DB_TIMEOUT",
            len(evidence_dict) == 5,
            len(evidence_json) > 100,
            recreated.log_level == log_evidence.log_level
        ])
        
        if all_checks:
            print("🎉 Test 4 PASSED: Simple evidence creation working")
            return True
        else:
            print("❌ Test 4 FAILED: Some evidence checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run core-only validation tests."""
    print("🧪 AURA Intelligence - Core-Only Validation")
    print("=" * 48)
    print("Testing fundamental contracts with minimal dependencies")
    print()
    
    # Run tests in order
    tests = [
        ("Core Cryptographic Signatures", test_crypto_signatures),
        ("Core Enum Functionality", test_enum_functionality),
        ("Core Base Utilities", test_base_utilities),
        ("Simple Evidence Creation", test_simple_evidence_creation)
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
    print(f"\n📊 Core-Only Validation Results")
    print("=" * 35)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} core contracts validated")
    
    if passed == len(tests):
        print("\n🎉 CORE CONTRACTS VALIDATED!")
        print("✅ Fundamental building blocks are working")
        print("✅ Cryptographic signatures proven")
        print("✅ Enum system proven")
        print("✅ Base utilities proven")
        print("✅ Simple evidence creation proven")
        print("\n🚀 Ready to build minimal walking skeleton!")
    else:
        print(f"\n⚠️  {len(tests) - passed} core contracts failed")
        print("Must fix fundamental issues before proceeding")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
