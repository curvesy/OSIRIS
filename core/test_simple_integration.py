#!/usr/bin/env python3
"""
🧪 Simple Phase 2C Integration Validation

Quick validation that Phase 2C components are properly integrated
with the UltimateAURASystem without running full cycles.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all Phase 2C components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test core system import
        from aura_intelligence.core.system import UltimateAURASystem
        print("✅ UltimateAURASystem import: SUCCESS")
        
        # Test Phase 2C component imports
        from aura_intelligence.enterprise.mem0_hot import HotEpisodicIngestor, DEV_SETTINGS
        from aura_intelligence.enterprise.mem0_semantic import SemanticMemorySync, MemoryRankingService
        from aura_intelligence.enterprise.mem0_search import create_search_router
        print("✅ Phase 2C component imports: SUCCESS")
        
        # Test data structures
        from aura_intelligence.enterprise.data_structures import TopologicalSignature
        print("✅ Data structure imports: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_system_creation():
    """Test that UltimateAURASystem can be created with Phase 2C components."""
    print("\n🔧 Testing system creation...")
    
    try:
        from aura_intelligence import get_development_config, create_ultimate_aura_system
        
        # Create system with development configuration
        config = get_development_config()
        ultimate_system = create_ultimate_aura_system(config)
        
        # Validate Phase 2C components are present
        assert hasattr(ultimate_system, 'hot_memory_settings'), "hot_memory_settings missing"
        assert hasattr(ultimate_system, 'hot_memory'), "hot_memory missing"
        assert hasattr(ultimate_system, 'semantic_memory'), "semantic_memory missing"
        assert hasattr(ultimate_system, 'ranking_service'), "ranking_service missing"
        assert hasattr(ultimate_system, 'search_router'), "search_router missing"
        
        print("✅ System creation: SUCCESS")
        print(f"   - Hot memory settings: ✅ Present")
        print(f"   - Hot memory: ✅ Present")
        print(f"   - Semantic memory: ✅ Present")
        print(f"   - Ranking service: ✅ Present")
        print(f"   - Search router: ✅ Present")
        
        return True
        
    except Exception as e:
        print(f"❌ System creation test failed: {e}")
        return False

def test_intelligence_flywheel_method():
    """Test that the Intelligence Flywheel method exists."""
    print("\n🧠 Testing Intelligence Flywheel method...")
    
    try:
        from aura_intelligence import get_development_config, create_ultimate_aura_system
        
        config = get_development_config()
        ultimate_system = create_ultimate_aura_system(config)
        
        # Check if Intelligence Flywheel method exists
        assert hasattr(ultimate_system, '_execute_intelligence_flywheel'), "Intelligence Flywheel method missing"
        
        print("✅ Intelligence Flywheel method: SUCCESS")
        print(f"   - _execute_intelligence_flywheel: ✅ Present")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligence Flywheel method test failed: {e}")
        return False

def test_search_api_integration():
    """Test that SearchAPI has Phase 2C routes."""
    print("\n🌐 Testing SearchAPI Phase 2C integration...")
    
    try:
        from aura_intelligence.enterprise.search_api import SearchAPIService
        
        # Create SearchAPI service
        search_api = SearchAPIService()
        
        # Check if Phase 2C route method exists
        assert hasattr(search_api, '_add_phase2c_routes'), "Phase 2C routes method missing"
        
        print("✅ SearchAPI Phase 2C integration: SUCCESS")
        print(f"   - _add_phase2c_routes method: ✅ Present")
        
        return True
        
    except Exception as e:
        print(f"❌ SearchAPI Phase 2C integration test failed: {e}")
        return False

def test_data_structure_compatibility():
    """Test TopologicalSignature data structure compatibility."""
    print("\n📊 Testing data structure compatibility...")
    
    try:
        from aura_intelligence.enterprise.data_structures import TopologicalSignature
        from datetime import datetime
        
        # Test new format (betti_numbers list)
        signature_new = TopologicalSignature(
            betti_numbers=[5, 3, 1],
            persistence_diagram={"birth_death_pairs": [[0.0, 1.0]]},
            agent_context={"test": True},
            timestamp=datetime.now(),
            signature_hash="test_signature_new"
        )
        
        print("✅ Data structure compatibility: SUCCESS")
        print(f"   - New format (betti_numbers): ✅ Working")
        print(f"   - Signature hash: {signature_new.signature_hash}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure compatibility test failed: {e}")
        return False

def main():
    """Run all simple integration tests."""
    print("🚀 AURA Intelligence Phase 2C Simple Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_system_creation,
        test_intelligence_flywheel_method,
        test_search_api_integration,
        test_data_structure_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 SIMPLE INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    success_rate = passed / total
    
    if success_rate == 1.0:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Phase 2C integration is properly configured")
        print("✅ Ready for full integration testing")
    elif success_rate >= 0.8:
        print("⚠️ MOSTLY SUCCESSFUL")
        print(f"✅ {passed}/{total} tests passed ({success_rate:.1%})")
        print("🔧 Minor issues need attention")
    else:
        print("❌ INTEGRATION ISSUES DETECTED")
        print(f"❌ {passed}/{total} tests passed ({success_rate:.1%})")
        print("🔧 Significant fixes needed")
    
    print(f"\n📈 Summary:")
    print(f"   - Tests passed: {passed}/{total}")
    print(f"   - Success rate: {success_rate:.1%}")
    print(f"   - Status: {'READY' if success_rate == 1.0 else 'NEEDS WORK'}")
    
    return 0 if success_rate == 1.0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
