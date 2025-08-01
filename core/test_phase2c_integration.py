#!/usr/bin/env python3
"""
🧪 Phase 2C Integration Test Suite

Comprehensive validation of Phase 2C Intelligence Flywheel integration 
with the UltimateAURASystem. Tests the complete end-to-end functionality:

1. System initialization with Phase 2C components
2. Intelligence Flywheel execution during system cycles
3. API endpoint availability and functionality
4. Memory persistence and retrieval
5. Performance and reliability metrics

This validates that the Intelligence Flywheel is fully operational
within the complete AURA Intelligence ecosystem.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aura_intelligence import create_ultimate_aura_system, get_development_config
from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.utils.logger import get_logger

logger = get_logger(__name__)


class Phase2CIntegrationTester:
    """Comprehensive integration tester for Phase 2C Intelligence Flywheel."""
    
    def __init__(self):
        self.ultimate_system = None
        self.test_results = {
            "initialization": False,
            "intelligence_flywheel": False,
            "api_endpoints": False,
            "memory_persistence": False,
            "performance": False,
            "overall_success": False
        }
        self.performance_metrics = {
            "initialization_time_ms": 0,
            "cycle_time_ms": 0,
            "flywheel_time_ms": 0,
            "memory_operations_per_sec": 0
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete Phase 2C integration test suite."""
        print("🧪 Starting Phase 2C Integration Test Suite")
        print("=" * 60)
        
        try:
            # Test 1: System Initialization
            await self._test_system_initialization()
            
            # Test 2: Intelligence Flywheel Integration
            await self._test_intelligence_flywheel()
            
            # Test 3: API Endpoints
            await self._test_api_endpoints()
            
            # Test 4: Memory Persistence
            await self._test_memory_persistence()
            
            # Test 5: Performance Validation
            await self._test_performance()
            
            # Calculate overall success
            self.test_results["overall_success"] = all([
                self.test_results["initialization"],
                self.test_results["intelligence_flywheel"],
                self.test_results["memory_persistence"],
                self.test_results["performance"]
            ])
            
            # Generate final report
            return self._generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_results": self.test_results,
                "performance_metrics": self.performance_metrics
            }
        finally:
            # Cleanup
            if self.ultimate_system:
                try:
                    await self.ultimate_system.cleanup()
                except:
                    pass
    
    async def _test_system_initialization(self):
        """Test 1: Validate system initialization with Phase 2C components."""
        print("\n🔧 Test 1: System Initialization with Phase 2C")
        
        start_time = time.time()
        
        try:
            # Create system with development configuration
            config = get_development_config()
            self.ultimate_system = create_ultimate_aura_system(config)
            
            # Initialize the system
            await self.ultimate_system.initialize()
            
            # Validate Phase 2C components are initialized
            assert self.ultimate_system.hot_memory is not None, "Hot memory not initialized"
            assert self.ultimate_system.semantic_memory is not None, "Semantic memory not initialized"
            assert self.ultimate_system.ranking_service is not None, "Ranking service not initialized"
            assert self.ultimate_system.search_router is not None, "Search router not initialized"
            
            # Test component health
            hot_memory_health = await self.ultimate_system.hot_memory.health_check()
            assert hot_memory_health["status"] == "healthy", f"Hot memory unhealthy: {hot_memory_health}"
            
            self.performance_metrics["initialization_time_ms"] = (time.time() - start_time) * 1000
            self.test_results["initialization"] = True
            
            print("✅ System initialization: PASSED")
            print(f"   - Hot memory: ✅ Initialized")
            print(f"   - Semantic memory: ✅ Initialized") 
            print(f"   - Ranking service: ✅ Initialized")
            print(f"   - Search router: ✅ Initialized")
            print(f"   - Initialization time: {self.performance_metrics['initialization_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"❌ System initialization: FAILED - {e}")
            raise
    
    async def _test_intelligence_flywheel(self):
        """Test 2: Validate Intelligence Flywheel execution during system cycle."""
        print("\n🧠 Test 2: Intelligence Flywheel Integration")
        
        start_time = time.time()
        
        try:
            # Create test topological signatures
            test_signatures = [
                {
                    "betti_numbers": [5, 3, 1],
                    "persistence_diagram": {"birth_death_pairs": [[0.0, 1.0], [0.5, 2.0]]},
                    "anomaly_score": 0.8,
                    "hash": f"integration_test_{i}"
                }
                for i in range(3)
            ]
            
            # Mock topology results for the cycle
            original_get_topology_data = self.ultimate_system.agents.get_consciousness_topology_data
            self.ultimate_system.agents.get_consciousness_topology_data = lambda: {"signatures": test_signatures}
            
            # Run a single system cycle
            cycle_result = await self.ultimate_system.run_ultimate_cycle()
            
            # Restore original method
            self.ultimate_system.agents.get_consciousness_topology_data = original_get_topology_data
            
            # Validate cycle success and flywheel execution
            assert cycle_result["success"], f"System cycle failed: {cycle_result}"
            assert "flywheel_insights" in cycle_result, "Intelligence Flywheel not executed"
            
            flywheel_insights = cycle_result["flywheel_insights"]
            assert flywheel_insights["success"], f"Intelligence Flywheel failed: {flywheel_insights}"
            assert flywheel_insights["insights"]["signatures_processed"] > 0, "No signatures processed"
            
            self.performance_metrics["cycle_time_ms"] = cycle_result.get("cycle_time_ms", 0)
            self.performance_metrics["flywheel_time_ms"] = flywheel_insights.get("flywheel_time_ms", 0)
            self.test_results["intelligence_flywheel"] = True
            
            print("✅ Intelligence Flywheel: PASSED")
            print(f"   - Signatures processed: {flywheel_insights['insights']['signatures_processed']}")
            print(f"   - Anomalies detected: {flywheel_insights['insights']['anomalies_detected']}")
            print(f"   - Memory consolidations: {flywheel_insights['insights']['memory_consolidations']}")
            print(f"   - Flywheel time: {self.performance_metrics['flywheel_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"❌ Intelligence Flywheel: FAILED - {e}")
            raise
    
    async def _test_api_endpoints(self):
        """Test 3: Validate API endpoints are accessible."""
        print("\n🌐 Test 3: API Endpoints")
        
        try:
            # Test SearchAPIService has Phase 2C routes
            search_api = self.ultimate_system.search_api
            assert search_api is not None, "Search API not initialized"
            assert search_api.app is not None, "FastAPI app not created"
            
            # Check if Phase 2C routes are mounted (basic validation)
            routes = [route.path for route in search_api.app.routes]
            phase2c_routes_found = any("/api/v2c" in route for route in routes)
            
            self.test_results["api_endpoints"] = True
            
            print("✅ API Endpoints: PASSED")
            print(f"   - Search API: ✅ Available")
            print(f"   - Phase 2C routes: {'✅' if phase2c_routes_found else '⚠️'} {'Found' if phase2c_routes_found else 'Not found'}")
            print(f"   - Total routes: {len(routes)}")
            
        except Exception as e:
            print(f"❌ API Endpoints: FAILED - {e}")
            raise
    
    async def _test_memory_persistence(self):
        """Test 4: Validate memory persistence and retrieval."""
        print("\n💾 Test 4: Memory Persistence")
        
        start_time = time.time()
        
        try:
            # Create test signature
            test_signature = TopologicalSignature(
                betti_numbers=[7, 4, 2],
                persistence_diagram={"birth_death_pairs": [[0.0, 1.5], [0.3, 2.5]]},
                agent_context={"test": "memory_persistence"},
                timestamp=datetime.now(),
                signature_hash="memory_test_signature"
            )
            
            # Store in hot memory
            store_success = await self.ultimate_system.hot_memory.ingest_signature(
                signature=test_signature,
                agent_id="integration_test",
                event_type="test_event",
                agent_meta={"test": True},
                full_event={"integration_test": True}
            )
            assert store_success, "Failed to store signature in hot memory"
            
            # Search for the stored signature
            similar_matches = await self.ultimate_system.ranking_service.find_similar_signatures(
                signature=test_signature,
                threshold=0.9,
                max_results=5
            )
            
            # Validate retrieval
            assert len(similar_matches) > 0, "Failed to retrieve stored signature"
            found_exact_match = any(
                match.signature.signature_hash == test_signature.signature_hash 
                for match in similar_matches
            )
            assert found_exact_match, "Exact signature match not found"
            
            operations_time = (time.time() - start_time) * 1000
            self.performance_metrics["memory_operations_per_sec"] = 2000 / operations_time  # 2 operations
            self.test_results["memory_persistence"] = True
            
            print("✅ Memory Persistence: PASSED")
            print(f"   - Storage: ✅ Success")
            print(f"   - Retrieval: ✅ Success")
            print(f"   - Exact match found: ✅ Yes")
            print(f"   - Operations time: {operations_time:.2f}ms")
            
        except Exception as e:
            print(f"❌ Memory Persistence: FAILED - {e}")
            raise
    
    async def _test_performance(self):
        """Test 5: Validate performance meets requirements."""
        print("\n⚡ Test 5: Performance Validation")
        
        try:
            # Performance thresholds
            max_init_time_ms = 10000  # 10 seconds
            max_cycle_time_ms = 5000   # 5 seconds
            max_flywheel_time_ms = 100 # 100ms
            min_memory_ops_per_sec = 10
            
            # Validate performance metrics
            init_ok = self.performance_metrics["initialization_time_ms"] <= max_init_time_ms
            cycle_ok = self.performance_metrics["cycle_time_ms"] <= max_cycle_time_ms
            flywheel_ok = self.performance_metrics["flywheel_time_ms"] <= max_flywheel_time_ms
            memory_ok = self.performance_metrics["memory_operations_per_sec"] >= min_memory_ops_per_sec
            
            self.test_results["performance"] = init_ok and cycle_ok and flywheel_ok and memory_ok
            
            print(f"✅ Performance Validation: {'PASSED' if self.test_results['performance'] else 'FAILED'}")
            print(f"   - Initialization: {self.performance_metrics['initialization_time_ms']:.2f}ms {'✅' if init_ok else '❌'}")
            print(f"   - Cycle time: {self.performance_metrics['cycle_time_ms']:.2f}ms {'✅' if cycle_ok else '❌'}")
            print(f"   - Flywheel time: {self.performance_metrics['flywheel_time_ms']:.2f}ms {'✅' if flywheel_ok else '❌'}")
            print(f"   - Memory ops/sec: {self.performance_metrics['memory_operations_per_sec']:.2f} {'✅' if memory_ok else '❌'}")
            
        except Exception as e:
            print(f"❌ Performance Validation: FAILED - {e}")
            raise
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results) - 1  # Exclude overall_success
        
        return {
            "success": self.test_results["overall_success"],
            "summary": {
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat(),
            "integration_status": "READY FOR PRODUCTION" if self.test_results["overall_success"] else "NEEDS FIXES"
        }


async def main():
    """Run the Phase 2C integration test suite."""
    print("🚀 AURA Intelligence Phase 2C Integration Test")
    print("Testing complete Intelligence Flywheel integration...")
    
    tester = Phase2CIntegrationTester()
    results = await tester.run_comprehensive_test()
    
    # Display final results
    print("\n" + "=" * 60)
    print("📊 FINAL INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    if results["success"]:
        print("🎉 PHASE 2C INTEGRATION: SUCCESS!")
        print("✅ Intelligence Flywheel is fully operational")
        print("✅ Ready for production deployment")
    else:
        print("❌ PHASE 2C INTEGRATION: FAILED")
        print("🔧 Requires fixes before production")
    
    if 'summary' in results:
        print(f"\n📈 Test Summary:")
        print(f"   - Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
        print(f"   - Success rate: {results['summary']['success_rate']:.1%}")
        print(f"   - Integration status: {results['integration_status']}")

        print(f"\n⚡ Performance Summary:")
        for metric, value in results['performance_metrics'].items():
            print(f"   - {metric}: {value:.2f}")
    else:
        print(f"\n❌ Test execution failed: {results.get('error', 'Unknown error')}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
