#!/usr/bin/env python3
"""
🧠 PHASE 2B SIMPLE TEST - ENHANCED KNOWLEDGE GRAPH

Simple test to verify Enhanced Knowledge Graph with GDS 2.19 works.
This creates basic test data and tests the enhanced graph capabilities.
"""

import asyncio
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aura_intelligence.enterprise.enhanced_knowledge_graph import EnhancedKnowledgeGraphService


async def test_enhanced_graph():
    """Simple test of Enhanced Knowledge Graph."""
    print("🧠 Testing Enhanced Knowledge Graph with GDS 2.19...")
    
    # Initialize enhanced knowledge graph
    enhanced_kg = EnhancedKnowledgeGraphService(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    try:
        # Initialize
        print("🔧 Initializing Enhanced Knowledge Graph...")
        success = await enhanced_kg.initialize()
        if not success:
            print("❌ Initialization failed")
            return False
        
        print("✅ Enhanced Knowledge Graph initialized")
        
        # Clear existing data first
        print("🧹 Clearing existing test data...")
        async with enhanced_kg.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        # Create some simple test data
        print("📦 Creating simple test data...")

        # Create basic nodes and relationships
        async with enhanced_kg.driver.session() as session:
            # Create test signatures
            await session.run("""
                CREATE (s1:Signature {hash: 'test_sig_1', consciousness_level: 0.5, timestamp: datetime()})
                CREATE (s2:Signature {hash: 'test_sig_2', consciousness_level: 0.7, timestamp: datetime()})
                CREATE (s3:Signature {hash: 'test_sig_3', consciousness_level: 0.3, timestamp: datetime()})
                
                CREATE (e1:Event {event_id: 'test_event_1', timestamp: datetime()})
                CREATE (e2:Event {event_id: 'test_event_2', timestamp: datetime()})
                
                CREATE (a1:Action {action_id: 'test_action_1', timestamp: datetime()})
                CREATE (a2:Action {action_id: 'test_action_2', timestamp: datetime()})
                
                CREATE (o1:Outcome {outcome_id: 'test_outcome_1', timestamp: datetime()})
                CREATE (o2:Outcome {outcome_id: 'test_outcome_2', timestamp: datetime()})
                
                CREATE (s1)-[:GENERATED_BY {weight: 1.0}]->(e1)
                CREATE (s2)-[:GENERATED_BY {weight: 0.8}]->(e2)
                CREATE (e1)-[:TRIGGERED_BY {weight: 0.9}]->(a1)
                CREATE (e2)-[:TRIGGERED_BY {weight: 0.7}]->(a2)
                CREATE (a1)-[:LED_TO {weight: 0.6}]->(o1)
                CREATE (a2)-[:LED_TO {weight: 0.8}]->(o2)
                CREATE (s1)-[:INFLUENCES {weight: 0.5}]->(s2)
                CREATE (s2)-[:INFLUENCES {weight: 0.4}]->(s3)
            """)
        
        print("✅ Test data created")
        
        # Test community detection
        print("\n🔍 Testing Community Detection...")
        communities = await enhanced_kg.detect_signature_communities(0.6)
        
        if communities and "error" not in communities:
            print(f"  ✅ Algorithm: {communities.get('algorithm_used', 'unknown')}")
            print(f"  📊 Communities: {communities.get('total_communities', 0)}")
            print(f"  ⏱️ Time: {communities.get('computation_time_ms', 0):.2f}ms")
        else:
            print(f"  ❌ Community detection failed: {communities}")
            return False
        
        # Test centrality analysis
        print("\n📊 Testing Centrality Analysis...")
        centrality = await enhanced_kg.analyze_centrality_patterns(0.7)
        
        if centrality and "error" not in centrality:
            algorithms = centrality.get('centrality_algorithms', [])
            print(f"  ✅ Algorithms: {', '.join(algorithms)}")
            print(f"  ⏱️ Time: {centrality.get('computation_time_ms', 0):.2f}ms")
        else:
            print(f"  ❌ Centrality analysis failed: {centrality}")
            return False
        
        # Test pattern prediction
        print("\n🔮 Testing Pattern Prediction...")
        predictions = await enhanced_kg.predict_future_patterns('test_sig_1', 0.8)
        
        if predictions and "error" not in predictions:
            print(f"  ✅ Predictions: {len(predictions.get('predictions', []))}")
            print(f"  📊 Confidence: {predictions.get('prediction_confidence', 0.0):.3f}")
            print(f"  ⏱️ Time: {predictions.get('computation_time_ms', 0):.2f}ms")
        else:
            print(f"  ❌ Pattern prediction failed: {predictions}")
            return False
        
        # Test consciousness-driven analysis
        print("\n🧠 Testing Consciousness-Driven Analysis...")
        consciousness_state = {"level": 0.8, "coherence": 0.6}
        analysis = await enhanced_kg.consciousness_driven_analysis(consciousness_state)
        
        if analysis and analysis.get("success"):
            depth = analysis.get("analysis_depth", "unknown")
            components = []
            if "communities" in analysis:
                components.append("communities")
            if "centrality" in analysis:
                components.append("centrality")
            if "predictions" in analysis:
                components.append("predictions")
            
            print(f"  ✅ Analysis depth: {depth}")
            print(f"  🔧 Components: {', '.join(components)}")
        else:
            print(f"  ❌ Consciousness analysis failed: {analysis}")
            return False
        
        # Test enhanced stats
        print("\n📊 Testing Enhanced Stats...")
        stats = await enhanced_kg.get_enhanced_stats()
        
        if stats.get("gds_available"):
            print(f"  ✅ GDS Version: {stats.get('gds_version', 'unknown')}")
            print(f"  📊 Graph projections: {stats.get('graph_projections', 0)}")
            print(f"  🧠 ML queries: {stats.get('ml_queries_executed', 0)}")
            print(f"  ⏱️ Avg ML time: {stats.get('avg_ml_time_ms', 0):.2f}ms")
        else:
            print(f"  ❌ GDS not available: {stats}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🧠 Enhanced Intelligence Flywheel is operational!")
        print("🔮 Advanced Graph ML capabilities are working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        await enhanced_kg.close()


async def main():
    """Main test execution."""
    print("🧠" + "=" * 60 + "🧠")
    print("  PHASE 2B SIMPLE TEST - ENHANCED KNOWLEDGE GRAPH")
    print("  Testing Advanced Graph Intelligence with GDS 2.19")
    print("🧠" + "=" * 60 + "🧠")
    
    success = await test_enhanced_graph()
    
    if success:
        print("\n🌟 PHASE 2B ENHANCED KNOWLEDGE GRAPH IS READY!")
        return 0
    else:
        print("\n⚠️ PHASE 2B NEEDS ATTENTION")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
