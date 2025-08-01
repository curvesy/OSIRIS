"""
Simple Chain Workflow - Research and Analysis Agents

Demonstrates a basic two-agent workflow where a research agent
gathers information and an analysis agent processes it.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from langgraph.graph import StateGraph, END
from pydantic import Field

from ..base import AgentBase, AgentConfig, AgentState
from ..observability import AgentInstrumentor, GenAIAttributes


class ResearchAnalysisState(AgentState):
    """State for research and analysis workflow."""
    
    # Research phase
    research_query: str = ""
    research_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Analysis phase
    analysis_complete: bool = False
    analysis_summary: str = ""
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Decision tracking
    last_decision: str = ""
    decision_reason: str = ""


class ResearchAgent(AgentBase[str, Dict[str, Any], ResearchAnalysisState]):
    """
    Research agent that gathers information on a topic.
    
    This is a mock implementation - in production, this would
    integrate with search APIs, databases, or knowledge graphs.
    """
    
    def __init__(self, config: AgentConfig = None):
        """Initialize research agent."""
        if config is None:
            config = AgentConfig(name="research_agent", model="gpt-4")
        super().__init__(config)
        self.instrumentor = AgentInstrumentor()
    
    def build_graph(self) -> StateGraph:
        """Build the research workflow graph."""
        workflow = StateGraph(ResearchAnalysisState)
        
        # Define nodes
        workflow.add_node("research", self.research_step)
        workflow.add_node("validate", self.validate_step)
        
        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "validate")
        workflow.add_edge("validate", END)
        
        return workflow
    
    async def _execute_step(self, state: ResearchAnalysisState, step_name: str) -> ResearchAnalysisState:
        """Execute a specific step in the research workflow."""
        if step_name == "research":
            return await self.research_step(state)
        elif step_name == "validate":
            return await self.validate_step(state)
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    async def research_step(self, state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Perform research on the given query."""
        # In production, this would call search APIs or databases
        async with self.instrumentor.trace_llm_call(
            agent_name=self.name,
            model=self.config.model,
            prompt=f"Research the following topic: {state.research_query}",
            temperature=self.config.temperature
        ) as span:
            # Mock research results
            state.research_results = [
                {
                    "source": "Academic Paper A",
                    "content": f"Research findings about {state.research_query}",
                    "relevance": 0.95,
                    "date": "2025-07-15"
                },
                {
                    "source": "Industry Report B",
                    "content": f"Market analysis related to {state.research_query}",
                    "relevance": 0.87,
                    "date": "2025-07-20"
                },
                {
                    "source": "Technical Blog C",
                    "content": f"Implementation details for {state.research_query}",
                    "relevance": 0.82,
                    "date": "2025-07-25"
                }
            ]
            
            # Record decision
            state.last_decision = "research_complete"
            state.decision_reason = f"Found {len(state.research_results)} relevant sources"
            
            # Add message
            state.add_message(
                "assistant",
                f"Research completed. Found {len(state.research_results)} relevant sources.",
                sources=len(state.research_results)
            )
            
            # Update span with results
            span.set_attribute(GenAIAttributes.AGENT_DECISION, state.last_decision)
            span.set_attribute("research.sources_found", len(state.research_results))
            
            # Record tokens (mock)
            if hasattr(self, '_metrics'):
                self._metrics.record_tokens(50, 200, self.config.model)
        
        state.next_step = "validate"
        return state
    
    async def validate_step(self, state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Validate research results."""
        # Check if we have enough quality results
        high_quality_results = [
            r for r in state.research_results 
            if r.get('relevance', 0) > 0.8
        ]
        
        if len(high_quality_results) >= 2:
            state.last_decision = "validation_passed"
            state.decision_reason = f"{len(high_quality_results)} high-quality sources found"
            state.add_message(
                "assistant",
                "Research validation passed. Ready for analysis.",
                validation_status="passed"
            )
        else:
            state.last_decision = "validation_failed"
            state.decision_reason = "Insufficient high-quality sources"
            state.add_message(
                "assistant",
                "Research validation failed. More sources needed.",
                validation_status="failed"
            )
        
        state.next_step = None  # End of research phase
        return state
    
    def _create_initial_state(self, input_data: str) -> ResearchAnalysisState:
        """Create initial state from research query."""
        state = ResearchAnalysisState(research_query=input_data)
        state.add_message("user", f"Research request: {input_data}")
        return state
    
    def _extract_output(self, final_state: ResearchAnalysisState) -> Dict[str, Any]:
        """Extract research results from final state."""
        return {
            "query": final_state.research_query,
            "results": final_state.research_results,
            "validation": final_state.last_decision,
            "message_count": len(final_state.messages)
        }


class AnalysisAgent(AgentBase[ResearchAnalysisState, Dict[str, Any], ResearchAnalysisState]):
    """
    Analysis agent that processes research results.
    
    Takes the output from research agent and generates insights.
    """
    
    def __init__(self, config: AgentConfig = None):
        """Initialize analysis agent."""
        if config is None:
            config = AgentConfig(name="analysis_agent", model="gpt-4", temperature=0.3)
        super().__init__(config)
        self.instrumentor = AgentInstrumentor()
    
    def build_graph(self) -> StateGraph:
        """Build the analysis workflow graph."""
        workflow = StateGraph(ResearchAnalysisState)
        
        # Define nodes
        workflow.add_node("analyze", self.analyze_step)
        workflow.add_node("synthesize", self.synthesize_step)
        workflow.add_node("recommend", self.recommend_step)
        
        # Define edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", "recommend")
        workflow.add_edge("recommend", END)
        
        return workflow
    
    async def _execute_step(self, state: ResearchAnalysisState, step_name: str) -> ResearchAnalysisState:
        """Execute a specific step in the analysis workflow."""
        if step_name == "analyze":
            return await self.analyze_step(state)
        elif step_name == "synthesize":
            return await self.synthesize_step(state)
        elif step_name == "recommend":
            return await self.recommend_step(state)
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    async def analyze_step(self, state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Analyze research results to extract insights."""
        async with self.instrumentor.trace_llm_call(
            agent_name=self.name,
            model=self.config.model,
            prompt=f"Analyze research results for: {state.research_query}",
            temperature=self.config.temperature
        ) as span:
            # Extract key insights (mock)
            state.key_insights = [
                f"Primary finding: Strong correlation in {state.research_query}",
                f"Secondary finding: Emerging trends indicate growth",
                f"Risk factor: Market volatility affects outcomes"
            ]
            
            state.last_decision = "analysis_complete"
            state.decision_reason = f"Extracted {len(state.key_insights)} key insights"
            
            state.add_message(
                "assistant",
                f"Analysis complete. Identified {len(state.key_insights)} key insights.",
                insights_count=len(state.key_insights)
            )
            
            # Record metrics
            if hasattr(self, '_metrics'):
                self._metrics.record_quality(0.92, 0.88)  # Quality and relevance scores
        
        state.next_step = "synthesize"
        return state
    
    async def synthesize_step(self, state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Synthesize findings into a summary."""
        # Create summary from insights
        state.analysis_summary = (
            f"Analysis of '{state.research_query}' reveals several key findings. "
            f"Based on {len(state.research_results)} sources, the analysis shows "
            f"significant patterns and trends. {' '.join(state.key_insights[:2])}"
        )
        
        state.last_decision = "synthesis_complete"
        state.decision_reason = "Summary generated from insights"
        
        state.add_message(
            "assistant",
            "Synthesis complete. Summary generated.",
            summary_length=len(state.analysis_summary)
        )
        
        state.next_step = "recommend"
        return state
    
    async def recommend_step(self, state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Generate recommendations based on analysis."""
        # Generate recommendations (mock)
        state.recommendations = [
            f"Consider implementing solution A for {state.research_query}",
            "Monitor emerging trends quarterly",
            "Mitigate identified risks through diversification"
        ]
        
        state.analysis_complete = True
        state.last_decision = "recommendations_generated"
        state.decision_reason = f"Generated {len(state.recommendations)} actionable recommendations"
        
        state.add_message(
            "assistant",
            f"Analysis workflow complete. Generated {len(state.recommendations)} recommendations.",
            recommendations_count=len(state.recommendations)
        )
        
        state.next_step = None  # End of workflow
        return state
    
    def _create_initial_state(self, input_data: ResearchAnalysisState) -> ResearchAnalysisState:
        """Use existing state from research phase."""
        input_data.current_step = "analyze"  # Reset to analysis start
        input_data.add_message("system", "Starting analysis phase")
        return input_data
    
    def _extract_output(self, final_state: ResearchAnalysisState) -> Dict[str, Any]:
        """Extract analysis results from final state."""
        return {
            "query": final_state.research_query,
            "summary": final_state.analysis_summary,
            "insights": final_state.key_insights,
            "recommendations": final_state.recommendations,
            "sources_analyzed": len(final_state.research_results),
            "analysis_complete": final_state.analysis_complete
        }


class ResearchAnalysisWorkflow:
    """
    Orchestrates the research and analysis workflow.
    
    This demonstrates how multiple agents work together in a chain.
    """
    
    def __init__(self):
        """Initialize the workflow with both agents."""
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        
        # Add instrumentation
        instrumentor = AgentInstrumentor()
        self.research_agent = instrumentor.instrument_agent(self.research_agent)
        self.analysis_agent = instrumentor.instrument_agent(self.analysis_agent)
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run the complete research and analysis workflow.
        
        Args:
            query: Research query to process
            
        Returns:
            Complete analysis results
        """
        # Phase 1: Research
        research_result, research_metrics = await self.research_agent.process(query)
        
        # Extract state for next phase
        research_state = self.research_agent._create_initial_state(query)
        research_state.research_results = research_result['results']
        
        # Phase 2: Analysis
        analysis_result, analysis_metrics = await self.analysis_agent.process(research_state)
        
        # Combine results
        return {
            "query": query,
            "research": research_result,
            "analysis": analysis_result,
            "metrics": {
                "research": research_metrics.to_dict(),
                "analysis": analysis_metrics.to_dict(),
                "total_execution_time_ms": (
                    research_metrics.execution_time_ms + 
                    analysis_metrics.execution_time_ms
                )
            }
        }


# Simple demonstration workflow
class SimpleChainWorkflow(ResearchAnalysisWorkflow):
    """Alias for the research-analysis workflow."""
    pass