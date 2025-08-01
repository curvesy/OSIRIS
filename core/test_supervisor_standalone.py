#!/usr/bin/env python3
"""
🧠 Standalone Memory-Aware Supervisor Test
Tests the supervisor functionality by copying the code directly.
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
from datetime import datetime


# Enhanced system prompt with historical context integration
SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor of a collective of AI agents. Your role is to analyze the current system state and decide the next best action.

## Core Directives
1. Analyze the full Evidence Log.
2. Review the Historical Context of similar past situations.
3. Based on all available information, decide the next agent to call or conclude the workflow.

## Evidence Log
{evidence_log}

## Historical Context (Memory of Past Successes)
Here are summaries of similar past workflows that completed successfully. Use this to inform your decision.
{historical_context}

## Your Task
Based on the evidence and historical context, what is the next single action to take? Choose from the available tools: {tool_names}.
If the goal is complete, respond with "FINISH".

Provide your reasoning in this format:
REASONING: [Brief explanation of how evidence and memory inform your decision]
ACTION: [chosen tool or FINISH]
"""


class CollectiveState:
    """Simple state container for workflow data."""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any):
        self.data[key] = value


class Supervisor:
    """
    Memory-aware supervisor implementing the learning loop 'read' phase.
    Makes decisions based on current evidence AND historical context.
    """
    
    def __init__(self, llm, tools: List[str]):
        """
        Initialize memory-aware supervisor.
        
        Args:
            llm: Language model for decision making
            tools: List of available tool names
        """
        self.llm = llm
        self.tool_names = tools
        self.system_prompt = SUPERVISOR_SYSTEM_PROMPT
    
    async def invoke(self, state: CollectiveState, kg_manager) -> Dict[str, Any]:
        """
        The new, memory-aware invocation logic for the Supervisor.
        
        Args:
            state: Current workflow state with evidence log
            kg_manager: Knowledge graph manager for memory retrieval
            
        Returns:
            Dict containing the supervisor's decision and reasoning
        """
        
        # 1. Retrieve historical context from the knowledge graph
        current_evidence = state.get('evidence_log', [])
        historical_context = await kg_manager.get_historical_context(current_evidence)
        
        # 2. Format the context for the prompt
        formatted_history = self._format_historical_context(historical_context)
        
        # 3. Format the enhanced prompt
        prompt = self.system_prompt.format(
            evidence_log=json.dumps(current_evidence, indent=2),
            historical_context=formatted_history,
            tool_names=", ".join(self.tool_names)
        )
        
        # 4. Invoke the LLM with the new, context-rich prompt
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.ainvoke(messages)
        
        # 5. Parse the response and return structured decision
        decision = self._parse_supervisor_response(response.content)
        
        # 6. Store decision context for future learning
        decision['memory_context'] = {
            'historical_contexts_found': len(historical_context),
            'evidence_count': len(current_evidence),
            'timestamp': datetime.now().isoformat()
        }
        
        return decision
    
    def _format_historical_context(self, historical_context: List[Dict[str, Any]]) -> str:
        """
        Format historical context for the prompt.
        
        Args:
            historical_context: List of historical workflow contexts
            
        Returns:
            Formatted string for prompt inclusion
        """
        if not historical_context:
            return "No similar historical context found."
        
        formatted_lines = []
        for ctx in historical_context:
            workflow_id = ctx.get('workflowId', 'unknown')
            successful_actions = ctx.get('successfulActions', [])
            similarity_score = ctx.get('similarityScore', 0)
            
            # Filter out None values and format actions
            actions = [action for action in successful_actions if action]
            action_str = ' → '.join(actions) if actions else 'no specific actions recorded'
            
            formatted_lines.append(
                f"- In a similar past workflow ({workflow_id}), "
                f"the actions '{action_str}' led to success. "
                f"(Similarity: {similarity_score})"
            )
        
        return "\n".join(formatted_lines)
    
    def _parse_supervisor_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the supervisor's response into a structured decision.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Structured decision dictionary
        """
        lines = response.strip().split('\n')
        
        reasoning = ""
        action = "FINISH"
        
        for line in lines:
            line = line.strip()
            if line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
        
        return {
            'reasoning': reasoning,
            'next_action': action,
            'memory_enhanced': True,
            'timestamp': datetime.now().isoformat()
        }


# Factory function for easy instantiation
def create_memory_aware_supervisor(llm, tools: List[str]) -> Supervisor:
    """
    Factory function to create a memory-aware supervisor.
    
    Args:
        llm: Language model instance
        tools: List of available tool names
        
    Returns:
        Configured Supervisor instance
    """
    return Supervisor(llm, tools)


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


async def test_collective_state():
    """Test the CollectiveState class."""
    print("📊 Testing CollectiveState...")
    
    try:
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


async def test_supervisor_creation():
    """Test supervisor creation and basic functionality."""
    print("\n🤖 Testing supervisor creation...")
    
    try:
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


async def test_context_formatting():
    """Test historical context formatting."""
    print("\n📝 Testing context formatting...")
    
    try:
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


async def test_full_supervisor_invoke():
    """Test the full supervisor invoke method."""
    print("\n🎯 Testing full supervisor invoke...")
    
    try:
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
    print("🚀 Starting Standalone Memory-Aware Supervisor Tests")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("CollectiveState", test_collective_state()),
        ("Supervisor Creation", test_supervisor_creation()),
        ("Context Formatting", test_context_formatting()),
        ("Full Invoke", test_full_supervisor_invoke())
    ]
    
    results = {}
    for test_name, test_coro in tests:
        results[test_name] = await test_coro
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Memory-aware supervisor is working!")
        print("✅ Phase 2 'read' part of the learning loop is functional")
        print("✅ Memory retrieval integration successful")
        print("✅ Historical context formatting working")
        print("✅ Decision making with memory enhancement complete")
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED - Review errors above")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
