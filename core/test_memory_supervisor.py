#!/usr/bin/env python3
"""
🧠 Memory-Aware Supervisor Test Script
Tests the Phase 2 memory-enhanced decision making functionality.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test imports with strict error handling
def test_imports():
    """Test all required imports with strict error handling."""
    print("🔍 Testing imports...")
    
    try:
        from aura_intelligence.agents.supervisor import Supervisor, CollectiveState, create_memory_aware_supervisor
        print("✅ Supervisor imports successful")
    except ImportError as e:
        print(f"❌ CRITICAL: Supervisor import failed: {e}")
        return False
    
    try:
        from aura_intelligence.observability.knowledge_graph import KnowledgeGraphManager
        from aura_intelligence.observability.config import ObservabilityConfig
        print("✅ Knowledge graph imports successful")
    except ImportError as e:
        print(f"❌ CRITICAL: Knowledge graph import failed: {e}")
        return False
    
    try:
        import langchain_core
        print("✅ LangChain core available")
    except ImportError as e:
        print(f"❌ CRITICAL: LangChain not available: {e}")
        return False
    
    return True


class MockLLM:
    """Mock LLM for testing without external dependencies."""
    
    def __init__(self):
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Mock async invocation."""
        self.call_count += 1
        
        # Mock response that follows the expected format
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
        
        # Simulate a realistic supervisor decision
        mock_content = """
        REASONING: Based on the evidence log showing user query about system status, and historical context indicating that ObserverAgent was successful in similar situations, I will call the ObserverAgent to gather current system information.
        ACTION: ObserverAgent
        """
        
        return MockResponse(mock_content)


async def test_memory_retrieval():
    """Test the knowledge graph memory retrieval functionality."""
    print("\n🧠 Testing memory retrieval...")
    
    try:
        from aura_intelligence.observability.config import ObservabilityConfig
        from aura_intelligence.observability.knowledge_graph import KnowledgeGraphManager
        
        # Create test config (will work without actual Neo4j connection)
        config = ObservabilityConfig()
        kg_manager = KnowledgeGraphManager(config)
        
        # Test with mock evidence
        mock_evidence = [
            {"evidence_type": "user_query", "content": "system status"},
            {"evidence_type": "system_health", "confidence": 0.8}
        ]
        
        # This should return empty list without Neo4j, but shouldn't crash
        historical_context = await kg_manager.get_historical_context(mock_evidence)
        
        print(f"✅ Memory retrieval test passed - returned {len(historical_context)} contexts")
        return True
        
    except Exception as e:
        print(f"❌ Memory retrieval test failed: {e}")
        return False


async def test_supervisor_decision_making():
    """Test the supervisor's memory-enhanced decision making."""
    print("\n🎯 Testing supervisor decision making...")
    
    try:
        from aura_intelligence.agents.supervisor import Supervisor, CollectiveState
        from aura_intelligence.observability.config import ObservabilityConfig
        from aura_intelligence.observability.knowledge_graph import KnowledgeGraphManager
        
        # Create mock components
        mock_llm = MockLLM()
        tools = ["ObserverAgent", "AnalystAgent", "ExecutorAgent"]
        supervisor = Supervisor(mock_llm, tools)
        
        # Create test state with evidence
        test_state = CollectiveState({
            'evidence_log': [
                {"evidence_type": "user_query", "content": "What is the system status?"},
                {"evidence_type": "workflow_start", "timestamp": datetime.now().isoformat()}
            ]
        })
        
        # Create knowledge graph manager
        config = ObservabilityConfig()
        kg_manager = KnowledgeGraphManager(config)
        
        # Test the supervisor's invoke method
        decision = await supervisor.invoke(test_state, kg_manager)
        
        # Validate the decision structure
        required_keys = ['reasoning', 'next_action', 'memory_enhanced', 'timestamp', 'memory_context']
        for key in required_keys:
            if key not in decision:
                print(f"❌ Missing key in decision: {key}")
                return False
        
        print(f"✅ Supervisor decision test passed")
        print(f"   - Reasoning: {decision['reasoning'][:50]}...")
        print(f"   - Next Action: {decision['next_action']}")
        print(f"   - Memory Enhanced: {decision['memory_enhanced']}")
        print(f"   - Historical Contexts Found: {decision['memory_context']['historical_contexts_found']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Supervisor decision test failed: {e}")
        return False


async def test_historical_context_formatting():
    """Test the historical context formatting functionality."""
    print("\n📝 Testing historical context formatting...")
    
    try:
        from aura_intelligence.agents.supervisor import Supervisor
        
        mock_llm = MockLLM()
        supervisor = Supervisor(mock_llm, ["test_tool"])
        
        # Test with empty context
        empty_result = supervisor._format_historical_context([])
        if empty_result != "No similar historical context found.":
            print(f"❌ Empty context formatting failed")
            return False
        
        # Test with mock historical context
        mock_context = [
            {
                'workflowId': 'workflow_123',
                'successfulActions': ['ObserverAgent', 'AnalystAgent'],
                'similarityScore': 2
            },
            {
                'workflowId': 'workflow_456', 
                'successfulActions': ['ExecutorAgent'],
                'similarityScore': 1
            }
        ]
        
        formatted = supervisor._format_historical_context(mock_context)
        
        # Validate formatting
        if 'workflow_123' not in formatted or 'ObserverAgent → AnalystAgent' not in formatted:
            print(f"❌ Context formatting failed")
            return False
        
        print("✅ Historical context formatting test passed")
        return True
        
    except Exception as e:
        print(f"❌ Historical context formatting test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("🚀 Starting Memory-Aware Supervisor Tests")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ CRITICAL: Import tests failed - cannot continue")
        return False
    
    # Test 2: Memory retrieval
    memory_test = await test_memory_retrieval()
    
    # Test 3: Supervisor decision making
    decision_test = await test_supervisor_decision_making()
    
    # Test 4: Context formatting
    formatting_test = await test_historical_context_formatting()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Memory Retrieval: {'✅ PASSED' if memory_test else '❌ FAILED'}")
    print(f"Decision Making: {'✅ PASSED' if decision_test else '❌ FAILED'}")
    print(f"Context Formatting: {'✅ PASSED' if formatting_test else '❌ FAILED'}")
    
    all_passed = memory_test and decision_test and formatting_test
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Memory-aware supervisor is working!")
        print("✅ Phase 2 'read' part of the learning loop is functional")
    else:
        print("\n⚠️ SOME TESTS FAILED - Review errors above")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
