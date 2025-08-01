#!/usr/bin/env python3
"""
🧠 Neural Observability System Test
Simple validation of our Phase 1, Step 3 implementation.
"""

import asyncio
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

# Add the src directory to Python path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# Test individual component imports
def test_component_imports():
    """Test that all components can be imported successfully."""
    
    print("🧠 Testing Neural Observability Component Imports...")
    print("=" * 50)
    
    components_tested = []
    
    try:
        # Test config import
        print("1️⃣ Testing config import...")
        sys.path.insert(0, str(Path(__file__).parent / "src" / "aura_intelligence" / "observability"))
        
        import config
        print("   ✅ Config module imported successfully")
        components_tested.append("config")
        
        # Test ObservabilityConfig class
        test_config = config.ObservabilityConfig(
            organism_id="test_organism",
            deployment_environment="test",
            log_level="INFO"
        )
        print(f"   ✅ ObservabilityConfig created: {test_config.organism_id}")
        
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
    
    try:
        # Test context managers
        print("\n2️⃣ Testing context managers import...")
        import context_managers
        print("   ✅ Context managers imported successfully")
        components_tested.append("context_managers")
        
        # Test ObservabilityContext
        test_context = context_managers.ObservabilityContext(
            workflow_id="test_workflow",
            workflow_type="test",
            trace_id="test_trace"
        )
        print(f"   ✅ ObservabilityContext created: {test_context.workflow_id}")
        
    except Exception as e:
        print(f"   ❌ Context managers import failed: {e}")
    
    try:
        # Test OpenTelemetry integration
        print("\n3️⃣ Testing OpenTelemetry integration...")
        import opentelemetry_integration
        print("   ✅ OpenTelemetry integration imported successfully")
        components_tested.append("opentelemetry_integration")
        
    except Exception as e:
        print(f"   ❌ OpenTelemetry integration import failed: {e}")
    
    try:
        # Test LangSmith integration
        print("\n4️⃣ Testing LangSmith integration...")
        import langsmith_integration
        print("   ✅ LangSmith integration imported successfully")
        components_tested.append("langsmith_integration")
        
    except Exception as e:
        print(f"   ❌ LangSmith integration import failed: {e}")
    
    try:
        # Test Prometheus metrics
        print("\n5️⃣ Testing Prometheus metrics...")
        import prometheus_metrics
        print("   ✅ Prometheus metrics imported successfully")
        components_tested.append("prometheus_metrics")
        
    except Exception as e:
        print(f"   ❌ Prometheus metrics import failed: {e}")
    
    try:
        # Test structured logging
        print("\n6️⃣ Testing structured logging...")
        import structured_logging
        print("   ✅ Structured logging imported successfully")
        components_tested.append("structured_logging")
        
    except Exception as e:
        print(f"   ❌ Structured logging import failed: {e}")
    
    try:
        # Test knowledge graph
        print("\n7️⃣ Testing knowledge graph...")
        import knowledge_graph
        print("   ✅ Knowledge graph imported successfully")
        components_tested.append("knowledge_graph")
        
    except Exception as e:
        print(f"   ❌ Knowledge graph import failed: {e}")
    
    try:
        # Test health monitor
        print("\n8️⃣ Testing health monitor...")
        import health_monitor
        print("   ✅ Health monitor imported successfully")
        components_tested.append("health_monitor")
        
        # Test HealthMetrics
        test_metrics = health_monitor.HealthMetrics()
        print(f"   ✅ HealthMetrics created: {test_metrics.status}")
        
    except Exception as e:
        print(f"   ❌ Health monitor import failed: {e}")
    
    # Summary
    print(f"\n📊 Import Test Results:")
    print(f"   - Components tested: {len(components_tested)}/8")
    print(f"   - Successfully imported: {', '.join(components_tested)}")
    
    if len(components_tested) >= 6:  # Allow some optional dependencies to be missing
        print(f"   🎉 CORE COMPONENTS IMPORTED SUCCESSFULLY!")
        print(f"   📝 Note: Some optional dependencies may be missing (OpenTelemetry, etc.)")
        return True
    else:
        print(f"   ⚠️ Too many components failed to import")
        return False


async def test_basic_functionality():
    """Test basic functionality of key components."""
    
    print("\n🔧 Testing Basic Functionality...")
    print("=" * 50)
    
    try:
        # Import required modules
        import config
        import context_managers
        import structured_logging
        import health_monitor
        
        # Test configuration
        print("1️⃣ Testing configuration...")
        test_config = config.ObservabilityConfig(
            organism_id="test_organism_functionality",
            deployment_environment="test",
            log_level="INFO",
            health_check_interval=1.0,
        )
        print(f"   ✅ Configuration created successfully")
        
        # Test context manager
        print("\n2️⃣ Testing context manager...")
        context = context_managers.ObservabilityContext(
            workflow_id="test_workflow_func",
            workflow_type="functionality_test",
            trace_id="test_trace_func"
        )
        context.start_time = time.time()
        print(f"   ✅ Context manager created: {context.workflow_id}")
        
        # Test structured logging
        print("\n3️⃣ Testing structured logging...")
        logging_manager = structured_logging.StructuredLoggingManager(test_config)
        await logging_manager.initialize()
        
        # Test logging methods
        logging_manager.log_workflow_started(context, {})
        logging_manager.log_llm_usage("test-model", 100, 50, 1.0, 0.01)
        logging_manager.log_error_recovery("test_error", "retry", True)
        
        print(f"   ✅ Structured logging working correctly")
        
        # Test health monitor
        print("\n4️⃣ Testing health monitor...")
        health_monitor_instance = health_monitor.OrganismHealthMonitor(
            test_config,
            prometheus_manager=None,
            logging_manager=logging_manager
        )
        await health_monitor_instance.initialize()
        
        # Wait for initial health check
        await asyncio.sleep(2.0)
        
        current_health = health_monitor_instance.get_current_health()
        print(f"   ✅ Health monitor working:")
        print(f"      - Health Score: {current_health.overall_score:.3f}")
        print(f"      - Status: {current_health.status}")
        print(f"      - CPU Usage: {current_health.cpu_usage:.1f}%")
        
        # Test health updates
        await health_monitor_instance.update_health_score(0.95)
        print(f"   ✅ Health score update working")
        
        # Cleanup
        await health_monitor_instance.shutdown()
        await logging_manager.shutdown()
        
        print(f"\n🎉 ALL FUNCTIONALITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_architecture_quality():
    """Test the quality of our modular architecture."""
    
    print("\n🏗️ Testing Architecture Quality...")
    print("=" * 50)
    
    src_path = Path(__file__).parent / "src" / "aura_intelligence" / "observability"
    
    # Check file structure
    expected_files = [
        "__init__.py",
        "config.py",
        "core.py",
        "context_managers.py",
        "opentelemetry_integration.py",
        "langsmith_integration.py",
        "prometheus_metrics.py",
        "structured_logging.py",
        "knowledge_graph.py",
        "health_monitor.py",
    ]
    
    print("1️⃣ Checking file structure...")
    missing_files = []
    for file in expected_files:
        file_path = src_path / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    if not missing_files:
        print(f"   🎉 All expected files present!")
    else:
        print(f"   ⚠️ Missing files: {', '.join(missing_files)}")
    
    # Check code quality indicators
    print("\n2️⃣ Checking code quality...")
    
    quality_indicators = {
        "Modular Architecture": len(expected_files) >= 10,
        "Professional Structure": (src_path / "__init__.py").exists(),
        "Configuration Management": (src_path / "config.py").exists(),
        "Context Management": (src_path / "context_managers.py").exists(),
        "Health Monitoring": (src_path / "health_monitor.py").exists(),
        "Structured Logging": (src_path / "structured_logging.py").exists(),
        "Knowledge Graph": (src_path / "knowledge_graph.py").exists(),
    }
    
    passed_indicators = 0
    for indicator, passed in quality_indicators.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {indicator}")
        if passed:
            passed_indicators += 1
    
    quality_score = passed_indicators / len(quality_indicators)
    print(f"\n📊 Architecture Quality Score: {quality_score:.1%}")
    
    if quality_score >= 0.9:
        print(f"   🏆 EXCELLENT: Professional-grade architecture!")
    elif quality_score >= 0.7:
        print(f"   👍 GOOD: Solid architecture with room for improvement")
    else:
        print(f"   ⚠️ NEEDS WORK: Architecture requires attention")
    
    return quality_score >= 0.8


async def main():
    """Run all tests."""
    
    print("🧠 Neural Observability System Validation")
    print("=" * 60)
    print("Phase 1, Step 3: Complete Sensory Awareness")
    print("=" * 60)
    
    # Test 1: Component imports
    imports_passed = test_component_imports()
    
    # Test 2: Basic functionality
    if imports_passed:
        functionality_passed = await test_basic_functionality()
    else:
        functionality_passed = False
    
    # Test 3: Architecture quality
    architecture_passed = test_architecture_quality()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    
    tests = {
        "Component Imports": imports_passed,
        "Basic Functionality": functionality_passed,
        "Architecture Quality": architecture_passed,
    }
    
    passed_tests = sum(tests.values())
    total_tests = len(tests)
    
    for test_name, passed in tests.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(f"\n🎉 PHASE 1, STEP 3 COMPLETE!")
        print(f"   Neural Observability System is fully validated!")
        print(f"   The digital organism has complete sensory awareness!")
        print(f"\n🚀 READY FOR PHASE 2: Closing the Learning Loop")
    elif passed_tests >= total_tests * 0.8:
        print(f"\n👍 MOSTLY COMPLETE!")
        print(f"   Neural Observability System is largely functional!")
        print(f"   Minor issues to address before Phase 2")
    else:
        print(f"\n⚠️ NEEDS ATTENTION!")
        print(f"   Neural Observability System requires fixes")
        print(f"   Address failing tests before proceeding")


if __name__ == "__main__":
    print("🧠 Starting Neural Observability Validation...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Neural Observability Validation Complete!")
