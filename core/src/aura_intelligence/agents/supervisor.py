"""
🧠 Memory-Aware Supervisor Agent - Phase 2 Implementation
Transforms reactive decision-making into reflective, learning-based choices.
"""

import json
from typing import Dict, Any, List
from datetime import datetime

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available - install with: pip install langchain-core")

try:
    from ..observability.knowledge_graph import KnowledgeGraphManager
    from ..observability.config import ObservabilityConfig
except ImportError:
    # Mock classes for testing
    class KnowledgeGraphManager:
        async def get_historical_context(self, evidence, top_k=3):
            return []

    class ObservabilityConfig:
        pass


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
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for Supervisor agent")
            
        self.llm = llm
        self.tool_names = tools
        self.system_prompt = SUPERVISOR_SYSTEM_PROMPT
    
    async def invoke(self, state: CollectiveState, kg_manager: KnowledgeGraphManager) -> Dict[str, Any]:
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
        messages = [HumanMessage(content=prompt)]
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
