#!/usr/bin/env python3
"""
🧠 Isolated Memory-Aware Supervisor Test
Tests just the supervisor functionality without any complex imports.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


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


class MockKnowledgeGraphManager:
    """Mock knowledge graph manager for testing."""
    
    async def get_historical_context(self, current_evidence: List[dict], top_k: int = 3) -> List[dict]:
        """Mock historical context retrieval."""
        # Return mock historical context
        return [
            {
                'workflowId': 'workflow_123',
                'successfulActions': ['ObserverAgent', 'AnalystAgent'],
                'similarityScore': 2,
                'duration': 45.2,
                'workflowType': 'system_status_check'
            },
            {
                'workflowId': 'workflow_456', 
                'successfulActions': ['ExecutorAgent'],
                'similarityScore': 1,
                'duration': 23.1,
                'workflowType': 'user_query'
            }
        ]


def test_direct_supervisor_import():
    """Test importing supervisor module directly."""
    print("🔍 Testing direct supervisor import...")
    
    try:
        # Import the supervisor file directly
        import importlib.util
        
        supervisor_path = os.path.join(
            os.path.dirname(__file__), 
            'src', 
            'aura_intelligence', 
            'agents', 
            'supervisor.py'
        )
        
        spec = importlib.util.spec_from_file_location("supervisor", supervisor_path)
        supervisor_module = importlib.util.module_from_spec(spec)
        
        # Mock the dependencies that might not be available
        sys.modules['aura_intelligence.observability.knowledge_graph'] = type('MockModule', (), {
            'KnowledgeGraphManager': MockKnowledgeGraphManager
        })()
        sys.modules['aura_intelligence.observability.config'] = type('MockModule', (), {
            'ObservabilityConfig': type('MockConfig', (), {})
        })()
        
        spec.loader.exec_module(supervisor_module)
        
        # Test that we can access the classes
        Supervisor = supervisor_module.Supervisor
        CollectiveState = supervisor_module.CollectiveState
        create_memory_aware_supervisor = supervisor_module.create_memory_aware_supervisor
        
        print("✅ Direct supervisor import successful")
        return True, supervisor_module
        
    except Exception as e:
        print(f"❌ Direct supervisor import failed: {e}")
        return False, None


async def test_collective_state(supervisor_module):
    """Test the CollectiveState class."""
    print("\n📊 Testing CollectiveState...")
    
    try:
        CollectiveState = supervisor_module.CollectiveState
        
        # Test basic functionality
        state = CollectiveState()
        state['test_key'] = 'test_value'
        
        if state.get('test_key') != 'test_value':
            print("❌ CollectiveState get/set failed")
            return False
        
        if state.get('missing_key', 'default') != 'default':
            print("❌ CollectiveState default value failed")
            return False
        
        print("✅ CollectiveState test passed")
        return True
        
    except Exception as e:
        print(f"❌ CollectiveState test failed: {e}")
        return False


async def test_supervisor_creation(supervisor_module):
    """Test supervisor creation and basic functionality."""
    print("\n🤖 Testing supervisor creation...")
    
    try:
        Supervisor = supervisor_module.Supervisor
        create_memory_aware_supervisor = supervisor_module.create_memory_aware_supervisor
        
        mock_llm = MockLLM()
        tools = ["ObserverAgent", "AnalystAgent", "ExecutorAgent"]
        
        # Test direct creation
        supervisor1 = Supervisor(mock_llm, tools)
        if supervisor1.tool_names != tools:
            print("❌ Direct supervisor creation failed")
            return False
        
        # Test factory function
        supervisor2 = create_memory_aware_supervisor(mock_llm, tools)
        if supervisor2.tool_names != tools:
            print("❌ Factory supervisor creation failed")
            return False
        
        print("✅ Supervisor creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Supervisor creation test failed: {e}")
        return False


async def test_context_formatting(supervisor_module):
    """Test historical context formatting."""
    print("\n📝 Testing context formatting...")
    
    try:
        Supervisor = supervisor_module.Supervisor
        
        mock_llm = MockLLM()
        supervisor = Supervisor(mock_llm, ["test_tool"])
        
        # Test empty context
        empty_result = supervisor._format_historical_context([])
        if empty_result != "No similar historical context found.":
            print(f"❌ Empty context formatting failed: {empty_result}")
            return False
        
        # Test with mock context
        mock_context = [
            {
                'workflowId': 'workflow_123',
                'successfulActions': ['ObserverAgent', 'AnalystAgent'],
                'similarityScore': 2
            }
        ]
        
        formatted = supervisor._format_historical_context(mock_context)
        if 'workflow_123' not in formatted or 'ObserverAgent → AnalystAgent' not in formatted:
            print(f"❌ Context formatting failed: {formatted}")
            return False
        
        print("✅ Context formatting test passed")
        return True
        
    except Exception as e:
        print(f"❌ Context formatting test failed: {e}")
        return False


async def test_response_parsing(supervisor_module):
    """Test supervisor response parsing."""
    print("\n🔍 Testing response parsing...")
    
    try:
        Supervisor = supervisor_module.Supervisor
        
        mock_llm = MockLLM()
        supervisor = Supervisor(mock_llm, ["test_tool"])
        
        # Test response parsing
        test_response = """
        REASONING: This is my reasoning for the decision
        ACTION: ObserverAgent
        """
        
        parsed = supervisor._parse_supervisor_response(test_response)
        
        if parsed['reasoning'] != "This is my reasoning for the decision":
            print(f"❌ Reasoning parsing failed: {parsed['reasoning']}")
            return False
        
        if parsed['next_action'] != "ObserverAgent":
            print(f"❌ Action parsing failed: {parsed['next_action']}")
            return False
        
        if not parsed['memory_enhanced']:
            print("❌ Memory enhanced flag not set")
            return False
        
        print("✅ Response parsing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Response parsing test failed: {e}")
        return False


async def test_full_supervisor_invoke(supervisor_module):
    """Test the full supervisor invoke method."""
    print("\n🎯 Testing full supervisor invoke...")
    
    try:
        Supervisor = supervisor_module.Supervisor
        CollectiveState = supervisor_module.CollectiveState
        
        mock_llm = MockLLM()
        tools = ["ObserverAgent", "AnalystAgent", "ExecutorAgent"]
        supervisor = Supervisor(mock_llm, tools)
        
        # Create test state
        test_state = CollectiveState({
            'evidence_log': [
                {"evidence_type": "user_query", "content": "What is the system status?"},
                {"evidence_type": "workflow_start", "timestamp": datetime.now().isoformat()}
            ]
        })
        
        # Create mock knowledge graph manager
        mock_kg = MockKnowledgeGraphManager()
        
        # Test invoke
        decision = await supervisor.invoke(test_state, mock_kg)
        
        # Validate decision structure
        required_keys = ['reasoning', 'next_action', 'memory_enhanced', 'timestamp', 'memory_context']
        for key in required_keys:
            if key not in decision:
                print(f"❌ Missing key in decision: {key}")
                return False
        
        print(f"✅ Full supervisor invoke test passed")
        print(f"   - Next Action: {decision['next_action']}")
        print(f"   - Memory Enhanced: {decision['memory_enhanced']}")
        print(f"   - Historical Contexts: {decision['memory_context']['historical_contexts_found']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full supervisor invoke test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("🚀 Starting Isolated Memory-Aware Supervisor Tests")
    print("=" * 60)
    
    # Test 1: Direct import
    import_success, supervisor_module = test_direct_supervisor_import()
    if not import_success:
        print("\n❌ CRITICAL: Cannot import supervisor module - stopping tests")
        return False
    
    # Run all tests
    tests = [
        ("CollectiveState", test_collective_state(supervisor_module)),
        ("Supervisor Creation", test_supervisor_creation(supervisor_module)),
        ("Context Formatting", test_context_formatting(supervisor_module)),
        ("Response Parsing", test_response_parsing(supervisor_module)),
        ("Full Invoke", test_full_supervisor_invoke(supervisor_module))
    ]
    
    results = {}
    for test_name, test_coro in tests:
        results[test_name] = await test_coro
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    passed = 1  # Import test passed
    total = len(results) + 1  # +1 for import test
    
    print(f"Direct Import: ✅ PASSED")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Memory-aware supervisor is working!")
        print("✅ Phase 2 'read' part of the learning loop is functional")
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED - Review errors above")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
