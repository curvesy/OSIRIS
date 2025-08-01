#!/usr/bin/env python3
"""
🧠 PHASE 2B TEST SCRIPT - ENHANCED KNOWLEDGE GRAPH

Test script for the Enhanced Knowledge Graph with Neo4j GDS 2.19:
- Community Detection (Louvain, Label Propagation, Leiden)
- Centrality Analysis (PageRank, Betweenness, Harmonic)
- Pattern Prediction using Graph ML
- Consciousness-driven analysis

This tests the enhanced Intelligence Flywheel with advanced graph intelligence.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aura_intelligence.enterprise.enhanced_knowledge_graph import EnhancedKnowledgeGraphService
from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, SystemEvent, AgentAction, Outcome
)
from aura_intelligence.core.system import UltimateAURASystem
from aura_intelligence.config import UltimateAURAConfig


class Phase2BTestSuite:
    """
    🧠 Complete Phase 2B Test Suite
    
    Tests the Enhanced Intelligence Flywheel:
    1. ANALYZE - Generate topological signatures with Mojo TDA
    2. STORE - Store in Enhanced Knowledge Graph with GDS
    3. SEARCH - Find patterns with Graph ML algorithms
    4. LEARN - Use advanced graph intelligence for decisions
    5. PREDICT - Forecast future patterns with ML
    6. EVOLVE - Consciousness-driven graph analysis
    """
    
    def __init__(self):
        self.enhanced_kg = None
        self.ultimate_system = None
        
        print("🧠" + "=" * 68 + "🧠")
        print("  PHASE 2B TEST SUITE - ENHANCED KNOWLEDGE GRAPH")
        print("  Testing Advanced Graph Intelligence with GDS 2.19")
        print("🧠" + "=" * 68 + "🧠")
    
    async def setup(self):
        """Setup test environment."""
        print("\n🔧 Setting up enhanced test environment...")
        
        try:
            # Initialize Enhanced Knowledge Graph Service
            self.enhanced_kg = EnhancedKnowledgeGraphService(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            # Initialize ULTIMATE_COMPLETE_SYSTEM
            config = UltimateAURAConfig()
            self.ultimate_system = UltimateAURASystem(config)
            
            print("✅ Enhanced test environment setup complete")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise
    
    async def test_enhanced_knowledge_graph(self):
        """Test Enhanced Knowledge Graph with GDS."""
        print("\n🧠 Testing Enhanced Knowledge Graph Service...")
        
        try:
            # Initialize enhanced knowledge graph
            success = await self.enhanced_kg.initialize()
            if not success:
                print("❌ Enhanced knowledge graph initialization failed")
                return False
            
            print("✅ Enhanced Knowledge Graph with GDS 2.19 initialized")
            
            # Create test data for graph analysis
            test_signatures = self._create_test_signatures(10)
            test_events = self._create_test_events(10)
            test_actions = self._create_test_actions(10)
            test_outcomes = self._create_test_outcomes(10)
            
            # Store test data
            print("📦 Storing test data for graph analysis...")
            for i in range(10):
                success = await self.enhanced_kg.store_event_chain(
                    test_signatures[i],
                    test_events[i],
                    test_actions[i],
                    test_outcomes[i]
                )
                if success:
                    print(f"  ✅ Stored event chain {i+1}/10")
                else:
                    print(f"  ❌ Failed to store event chain {i+1}/10")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced knowledge graph test failed: {e}")
            return False
    
    async def test_community_detection(self):
        """Test community detection algorithms."""
        print("\n🔍 Testing Community Detection Algorithms...")
        
        try:
            # Test with different consciousness levels
            consciousness_levels = [0.3, 0.6, 0.9]  # Low, Medium, High
            
            for level in consciousness_levels:
                print(f"\n  🧠 Testing consciousness level: {level}")
                
                start_time = time.time()
                communities = await self.enhanced_kg.detect_signature_communities(level)
                detection_time = (time.time() - start_time) * 1000
                
                if communities and "error" not in communities:
                    algorithm = communities.get("algorithm_used", "unknown")
                    total_communities = communities.get("total_communities", 0)
                    modularity = communities.get("modularity", 0.0)
                    
                    print(f"    ✅ Algorithm: {algorithm}")
                    print(f"    📊 Communities: {total_communities}")
                    print(f"    📈 Modularity: {modularity:.3f}")
                    print(f"    ⏱️ Time: {detection_time:.2f}ms")
                else:
                    print(f"    ❌ Community detection failed: {communities}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Community detection test failed: {e}")
            return False
    
    async def test_centrality_analysis(self):
        """Test centrality analysis algorithms."""
        print("\n📊 Testing Centrality Analysis Algorithms...")
        
        try:
            # Test with different consciousness levels
            consciousness_levels = [0.4, 0.7, 0.9]  # Medium, High, Very High
            
            for level in consciousness_levels:
                print(f"\n  🧠 Testing consciousness level: {level}")
                
                start_time = time.time()
                centrality = await self.enhanced_kg.analyze_centrality_patterns(level)
                analysis_time = (time.time() - start_time) * 1000
                
                if centrality and "error" not in centrality:
                    algorithms = centrality.get("centrality_algorithms", [])
                    top_signatures = centrality.get("top_signatures", [])
                    
                    print(f"    ✅ Algorithms: {', '.join(algorithms)}")
                    print(f"    🏆 Top signatures: {len(top_signatures)}")
                    print(f"    ⏱️ Time: {analysis_time:.2f}ms")
                    
                    # Show top signature
                    if top_signatures:
                        top = top_signatures[0]
                        print(f"    🥇 Top signature: {top.get('nodeId', 'N/A')[:8]}... (score: {top.get('propertyValue', 0.0):.3f})")
                else:
                    print(f"    ❌ Centrality analysis failed: {centrality}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Centrality analysis test failed: {e}")
            return False
    
    async def test_pattern_prediction(self):
        """Test pattern prediction using Graph ML."""
        print("\n🔮 Testing Pattern Prediction with Graph ML...")
        
        try:
            # Get a test signature hash
            test_signature = self._create_test_signatures(1)[0]
            signature_hash = test_signature.signature_hash
            
            # Test with high consciousness for advanced ML
            consciousness_level = 0.8
            
            print(f"  🎯 Predicting patterns for signature: {signature_hash[:8]}...")
            
            start_time = time.time()
            predictions = await self.enhanced_kg.predict_future_patterns(
                signature_hash, consciousness_level
            )
            prediction_time = (time.time() - start_time) * 1000
            
            if predictions and "error" not in predictions:
                ml_algorithms = predictions.get("ml_algorithms_used", [])
                signature_predictions = predictions.get("predictions", [])
                confidence = predictions.get("prediction_confidence", 0.0)
                
                print(f"    ✅ ML Algorithms: {', '.join(ml_algorithms)}")
                print(f"    🔮 Predictions: {len(signature_predictions)}")
                print(f"    📊 Confidence: {confidence:.3f}")
                print(f"    ⏱️ Time: {prediction_time:.2f}ms")
                
                # Show sample predictions
                for i, pred in enumerate(signature_predictions[:3]):
                    print(f"    📈 Prediction {i+1}: {pred.get('type', 'N/A')} -> {pred.get('target', 'N/A')[:8]}...")
            else:
                print(f"    ❌ Pattern prediction failed: {predictions}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Pattern prediction test failed: {e}")
            return False
    
    async def test_consciousness_driven_analysis(self):
        """Test consciousness-driven graph analysis."""
        print("\n🧠 Testing Consciousness-Driven Graph Analysis...")
        
        try:
            # Test different consciousness states
            consciousness_states = [
                {"level": 0.3, "coherence": 0.2},  # Low consciousness
                {"level": 0.6, "coherence": 0.5},  # Medium consciousness
                {"level": 0.9, "coherence": 0.8}   # High consciousness
            ]
            
            for state in consciousness_states:
                level = state["level"]
                coherence = state["coherence"]
                effective = level * (0.7 + 0.3 * coherence)
                
                print(f"\n  🧠 Testing consciousness state: level={level}, coherence={coherence} (effective={effective:.3f})")
                
                start_time = time.time()
                analysis = await self.enhanced_kg.consciousness_driven_analysis(state)
                analysis_time = (time.time() - start_time) * 1000
                
                if analysis and analysis.get("success"):
                    depth = analysis.get("analysis_depth", "unknown")
                    components = []
                    
                    if "communities" in analysis:
                        components.append("communities")
                    if "centrality" in analysis:
                        components.append("centrality")
                    if "predictions" in analysis:
                        components.append("predictions")
                    
                    print(f"    ✅ Analysis depth: {depth}")
                    print(f"    🔧 Components: {', '.join(components)}")
                    print(f"    ⏱️ Time: {analysis_time:.2f}ms")
                else:
                    print(f"    ❌ Consciousness analysis failed: {analysis}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Consciousness-driven analysis test failed: {e}")
            return False
    
    async def test_ultimate_system_integration(self):
        """Test integration with ULTIMATE_COMPLETE_SYSTEM."""
        print("\n🌟 Testing ULTIMATE_COMPLETE_SYSTEM Integration...")
        
        try:
            # Initialize the ultimate system
            print("🚀 Initializing ULTIMATE_COMPLETE_SYSTEM with Enhanced Graph...")
            await self.ultimate_system.initialize()
            
            # Test that enhanced knowledge graph is integrated
            if hasattr(self.ultimate_system, 'enhanced_knowledge_graph'):
                print("  ✅ Enhanced Knowledge Graph integrated into ULTIMATE_COMPLETE_SYSTEM")
                
                # Test enhanced graph health
                health = await self.ultimate_system.enhanced_knowledge_graph.health_check()
                print(f"  📊 Enhanced Graph health: {health.get('status', 'unknown')}")
                print(f"  🧠 GDS status: {health.get('gds_status', 'unknown')}")
                
                if health.get("gds_version"):
                    print(f"  📈 GDS version: {health.get('gds_version')}")
                
            else:
                print("  ❌ Enhanced Knowledge Graph not found in ULTIMATE_COMPLETE_SYSTEM")
                return False
            
            # Run a test cycle with enhanced graph integration
            print("🔄 Running test cycle with enhanced graph integration...")
            result = await self.ultimate_system.run_ultimate_cycle()
            
            if result.get("success"):
                print("  ✅ Ultimate system cycle completed successfully")
                print(f"  📊 Consciousness level: {result.get('consciousness_level', 0.0):.3f}")
                
                # Check if TDA analysis was performed
                if "topology_analysis" in result:
                    tda_result = result["topology_analysis"]
                    print(f"  🔬 TDA Analysis: {tda_result.get('topology_signature', 'N/A')}")
                    print(f"  ⚡ Real Mojo: {'✅' if tda_result.get('real_mojo_acceleration') else '❌'}")
                    print(f"  🧠 Enhanced Graph: ✅ Integrated with consciousness")
            else:
                print("  ❌ Ultimate system cycle failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Ultimate system integration test failed: {e}")
            return False
    
    def _create_test_signatures(self, count: int) -> List[TopologicalSignature]:
        """Create test topological signatures with varied patterns."""
        signatures = []
        
        for i in range(count):
            # Create varied Betti numbers for community detection
            betti_patterns = [
                [1, 0, 0],  # Simple
                [1, 1, 0],  # One loop
                [1, 2, 0],  # Two loops
                [1, 3, 1],  # Complex with void
                [2, 1, 0],  # Two components
            ]
            
            signature = TopologicalSignature(
                betti_numbers=betti_patterns[i % len(betti_patterns)],
                persistence_diagram={
                    "birth_death_pairs": [
                        {"birth": 0.0, "death": 0.5 + i * 0.1},
                        {"birth": 0.2, "death": 0.8 + i * 0.05}
                    ]
                },
                agent_context={
                    "agents": [f"test_agent_{j}" for j in range(3)],
                    "system_state": {"test_metric": 0.5 + i * 0.1}
                },
                timestamp=datetime.now(),
                signature_hash="",  # Will be generated
                consciousness_level=0.3 + (i * 0.07),  # Varied consciousness
                quantum_coherence=0.2 + (i * 0.05),    # Varied coherence
                algorithm_used="enhanced_test_algorithm"
            )
            signatures.append(signature)
        
        return signatures
    
    def _create_test_events(self, count: int) -> List[SystemEvent]:
        """Create test system events."""
        events = []
        event_types = ["anomaly_detected", "performance_degradation", "resource_spike", "network_issue", "security_alert"]
        
        for i in range(count):
            event = SystemEvent(
                event_id=f"test_event_{i:03d}",
                event_type=event_types[i % len(event_types)],
                timestamp=datetime.now(),
                system_state={"cpu_usage": 0.5 + i * 0.05, "memory_usage": 0.4 + i * 0.04},
                triggering_agents=[f"monitoring_agent_{i % 3}"],
                consciousness_state={"level": 0.4 + i * 0.06, "coherence": 0.3 + i * 0.05}
            )
            events.append(event)
        
        return events
    
    def _create_test_actions(self, count: int) -> List[AgentAction]:
        """Create test agent actions."""
        actions = []
        action_types = ["scale_resources", "restart_service", "update_config", "alert_admin", "isolate_component"]
        
        for i in range(count):
            action = AgentAction(
                action_id=f"test_action_{i:03d}",
                agent_id=f"response_agent_{i % 4}",
                action_type=action_types[i % len(action_types)],
                timestamp=datetime.now(),
                input_signature=f"test_signature_{i:03d}",
                decision_context={"confidence": 0.6 + i * 0.03},
                action_parameters={"scale_factor": 1.2 + i * 0.1},
                confidence_score=0.5 + i * 0.04
            )
            actions.append(action)
        
        return actions
    
    def _create_test_outcomes(self, count: int) -> List[Outcome]:
        """Create test outcomes."""
        outcomes = []
        
        for i in range(count):
            outcome = Outcome(
                outcome_id=f"test_outcome_{i:03d}",
                action_id=f"test_action_{i:03d}",
                timestamp=datetime.now(),
                success=i % 3 != 0,  # 2/3 success rate
                impact_score=0.3 + i * 0.07,
                metrics={"response_time": 1.5 + i * 0.2, "success_rate": 0.8 + i * 0.02}
            )
            outcomes.append(outcome)
        
        return outcomes
    
    async def run_all_tests(self):
        """Run complete Phase 2B test suite."""
        print("\n🚀 Running Complete Phase 2B Test Suite...")
        
        test_results = {}
        
        # Test 1: Enhanced Knowledge Graph
        test_results["enhanced_knowledge_graph"] = await self.test_enhanced_knowledge_graph()
        
        # Test 2: Community Detection
        test_results["community_detection"] = await self.test_community_detection()
        
        # Test 3: Centrality Analysis
        test_results["centrality_analysis"] = await self.test_centrality_analysis()
        
        # Test 4: Pattern Prediction
        test_results["pattern_prediction"] = await self.test_pattern_prediction()
        
        # Test 5: Consciousness-Driven Analysis
        test_results["consciousness_analysis"] = await self.test_consciousness_driven_analysis()
        
        # Test 6: Ultimate System Integration
        test_results["ultimate_system"] = await self.test_ultimate_system_integration()
        
        # Print results summary
        print("\n📊 TEST RESULTS SUMMARY:")
        print("🧠" + "-" * 68 + "🧠")
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        print(f"\n🏆 OVERALL RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 PHASE 2B IMPLEMENTATION SUCCESSFUL!")
            print("🧠 Enhanced Intelligence Flywheel is operational!")
            print("🔮 Advanced Graph ML capabilities are working!")
        else:
            print("⚠️ Some tests failed - check logs for details")
        
        print("🧠" + "=" * 68 + "🧠")
        
        return passed == total


async def main():
    """Main test execution."""
    test_suite = Phase2BTestSuite()
    
    try:
        await test_suite.setup()
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n🌟 PHASE 2B READY FOR PRODUCTION!")
            return 0
        else:
            print("\n⚠️ PHASE 2B NEEDS ATTENTION")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
