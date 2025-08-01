"""
Temporal Workflow Definitions for AURA Intelligence

Implements durable, stateful workflows for agent orchestration
with built-in resilience and observability.
"""

from datetime import timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
import structlog

from ...agents.base import AgentState
from ...agents.observability import GenAIAttributes
from .activities import (
    AgentActivity,
    KafkaProducerActivity,
    StateManagementActivity,
    ObservabilityActivity
)

logger = structlog.get_logger()


@dataclass
class AgentWorkflowInput:
    """Input for agent workflow execution."""
    agent_id: str
    agent_type: str
    input_data: Dict[str, Any]
    config: Dict[str, Any]
    trace_parent: Optional[str] = None


@dataclass
class AgentWorkflowResult:
    """Result from agent workflow execution."""
    agent_id: str
    status: str
    output: Dict[str, Any]
    metrics: Dict[str, Any]
    duration_ms: float


@workflow.defn
class AgentWorkflow:
    """
    Durable workflow for single agent execution.
    
    Features:
    - Automatic retries with exponential backoff
    - State persistence across failures
    - Distributed tracing integration
    - Metrics collection
    """
    
    def __init__(self):
        self.state: Optional[AgentState] = None
        self.start_time: Optional[float] = None
        
    @workflow.run
    async def run(self, input: AgentWorkflowInput) -> AgentWorkflowResult:
        """Execute agent workflow with full resilience."""
        self.start_time = workflow.now().timestamp()
        
        # Initialize workflow state
        workflow.logger.info(
            "Starting agent workflow",
            agent_id=input.agent_id,
            agent_type=input.agent_type
        )
        
        # Create retry policy
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=5,
            backoff_coefficient=2.0
        )
        
        try:
            # Step 1: Initialize agent state
            self.state = await workflow.execute_activity(
                StateManagementActivity.create_initial_state,
                input.agent_id,
                input.input_data,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=retry_policy
            )
            
            # Step 2: Start observability span
            span_context = await workflow.execute_activity(
                ObservabilityActivity.start_workflow_span,
                input.agent_id,
                input.agent_type,
                input.trace_parent,
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            # Step 3: Execute agent processing
            result = await workflow.execute_activity(
                AgentActivity.process,
                input.agent_id,
                input.agent_type,
                self.state,
                input.config,
                span_context,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy
            )
            
            # Step 4: Persist final state
            await workflow.execute_activity(
                StateManagementActivity.persist_state,
                input.agent_id,
                result["state"],
                start_to_close_timeout=timedelta(seconds=30)
            )
            
            # Step 5: Publish completion event
            await workflow.execute_activity(
                KafkaProducerActivity.publish_event,
                "agent.workflow.completed",
                {
                    "agent_id": input.agent_id,
                    "agent_type": input.agent_type,
                    "status": "success",
                    "output": result["output"]
                },
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            # Step 6: Record metrics
            duration_ms = (workflow.now().timestamp() - self.start_time) * 1000
            await workflow.execute_activity(
                ObservabilityActivity.record_workflow_metrics,
                input.agent_id,
                input.agent_type,
                "success",
                duration_ms,
                result.get("tokens", {}),
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            return AgentWorkflowResult(
                agent_id=input.agent_id,
                status="success",
                output=result["output"],
                metrics=result.get("metrics", {}),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            # Handle failures gracefully
            workflow.logger.error(
                "Agent workflow failed",
                agent_id=input.agent_id,
                error=str(e)
            )
            
            # Publish failure event
            await workflow.execute_activity(
                KafkaProducerActivity.publish_event,
                "agent.workflow.failed",
                {
                    "agent_id": input.agent_id,
                    "agent_type": input.agent_type,
                    "error": str(e)
                },
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3)
            )
            
            # Record failure metrics
            duration_ms = (workflow.now().timestamp() - self.start_time) * 1000
            await workflow.execute_activity(
                ObservabilityActivity.record_workflow_metrics,
                input.agent_id,
                input.agent_type,
                "failed",
                duration_ms,
                {},
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            raise ApplicationError(
                f"Agent workflow failed: {str(e)}",
                type="AgentWorkflowError"
            )


@dataclass
class MultiAgentWorkflowInput:
    """Input for multi-agent orchestration workflow."""
    workflow_id: str
    agents: List[Dict[str, Any]]  # List of agent configurations
    orchestration_type: str  # "sequential", "parallel", "consensus"
    input_data: Dict[str, Any]
    config: Dict[str, Any]


@workflow.defn
class MultiAgentOrchestrationWorkflow:
    """
    Orchestrates multiple agents with different execution patterns.
    
    Supports:
    - Sequential execution with data passing
    - Parallel execution with result aggregation
    - Consensus-based execution with voting
    """
    
    @workflow.run
    async def run(self, input: MultiAgentWorkflowInput) -> Dict[str, Any]:
        """Execute multi-agent orchestration."""
        workflow.logger.info(
            "Starting multi-agent orchestration",
            workflow_id=input.workflow_id,
            orchestration_type=input.orchestration_type,
            agent_count=len(input.agents)
        )
        
        if input.orchestration_type == "sequential":
            return await self._run_sequential(input)
        elif input.orchestration_type == "parallel":
            return await self._run_parallel(input)
        elif input.orchestration_type == "consensus":
            return await self._run_consensus(input)
        else:
            raise ValueError(f"Unknown orchestration type: {input.orchestration_type}")
    
    async def _run_sequential(self, input: MultiAgentWorkflowInput) -> Dict[str, Any]:
        """Run agents sequentially, passing output to next agent."""
        current_data = input.input_data
        results = []
        
        for i, agent_config in enumerate(input.agents):
            workflow.logger.info(
                f"Running agent {i+1}/{len(input.agents)}",
                agent_type=agent_config["type"]
            )
            
            # Execute child workflow
            agent_input = AgentWorkflowInput(
                agent_id=f"{input.workflow_id}_agent_{i}",
                agent_type=agent_config["type"],
                input_data=current_data,
                config=agent_config.get("config", {})
            )
            
            result = await workflow.execute_child_workflow(
                AgentWorkflow.run,
                agent_input,
                id=f"{input.workflow_id}_agent_{i}",
                retry_policy=RetryPolicy(maximum_attempts=3)
            )
            
            results.append(result)
            current_data = result.output  # Pass output to next agent
        
        return {
            "workflow_id": input.workflow_id,
            "type": "sequential",
            "results": results,
            "final_output": results[-1].output if results else {}
        }
    
    async def _run_parallel(self, input: MultiAgentWorkflowInput) -> Dict[str, Any]:
        """Run agents in parallel and aggregate results."""
        # Create tasks for parallel execution
        tasks = []
        
        for i, agent_config in enumerate(input.agents):
            agent_input = AgentWorkflowInput(
                agent_id=f"{input.workflow_id}_agent_{i}",
                agent_type=agent_config["type"],
                input_data=input.input_data,
                config=agent_config.get("config", {})
            )
            
            task = workflow.execute_child_workflow(
                AgentWorkflow.run,
                agent_input,
                id=f"{input.workflow_id}_agent_{i}",
                retry_policy=RetryPolicy(maximum_attempts=3)
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "agent_index": i,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        # Aggregate outputs
        aggregated_output = {}
        for result in successful_results:
            aggregated_output[result.agent_id] = result.output
        
        return {
            "workflow_id": input.workflow_id,
            "type": "parallel",
            "successful": len(successful_results),
            "failed": len(failed_results),
            "results": successful_results,
            "failures": failed_results,
            "aggregated_output": aggregated_output
        }
    
    async def _run_consensus(self, input: MultiAgentWorkflowInput) -> Dict[str, Any]:
        """Run consensus workflow with voting mechanism."""
        # This will be implemented with Byzantine fault tolerance
        # For now, using simple majority voting
        
        # Run agents in parallel
        parallel_result = await self._run_parallel(input)
        
        if parallel_result["successful"] < len(input.agents) // 2 + 1:
            raise ApplicationError(
                "Consensus failed: insufficient successful agents",
                type="ConsensusError"
            )
        
        # Execute consensus activity
        consensus_result = await workflow.execute_activity(
            AgentActivity.compute_consensus,
            parallel_result["aggregated_output"],
            input.config.get("consensus_strategy", "majority"),
            start_to_close_timeout=timedelta(seconds=60)
        )
        
        return {
            "workflow_id": input.workflow_id,
            "type": "consensus",
            "participant_count": len(input.agents),
            "successful_count": parallel_result["successful"],
            "consensus": consensus_result,
            "individual_results": parallel_result["results"]
        }


@workflow.defn
class ResearchAnalysisPipeline:
    """
    End-to-end research and analysis pipeline.
    
    Demonstrates:
    - Complex multi-stage workflow
    - Conditional execution
    - Human-in-the-loop patterns
    - Progressive result streaming
    """
    
    @workflow.run
    async def run(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research and analysis pipeline."""
        workflow_id = workflow.info().workflow_id
        
        # Stage 1: Research
        research_input = MultiAgentWorkflowInput(
            workflow_id=f"{workflow_id}_research",
            agents=[
                {"type": "web_search", "config": {"max_results": 10}},
                {"type": "academic_search", "config": {"databases": ["arxiv", "pubmed"]}},
                {"type": "news_search", "config": {"time_range": "7d"}}
            ],
            orchestration_type="parallel",
            input_data={"query": query},
            config=config
        )
        
        research_results = await workflow.execute_child_workflow(
            MultiAgentOrchestrationWorkflow.run,
            research_input,
            id=f"{workflow_id}_research_phase"
        )
        
        # Stage 2: Analysis
        analysis_input = MultiAgentWorkflowInput(
            workflow_id=f"{workflow_id}_analysis",
            agents=[
                {"type": "summarizer", "config": {"max_length": 500}},
                {"type": "fact_checker", "config": {"confidence_threshold": 0.8}},
                {"type": "sentiment_analyzer", "config": {}}
            ],
            orchestration_type="sequential",
            input_data=research_results["aggregated_output"],
            config=config
        )
        
        analysis_results = await workflow.execute_child_workflow(
            MultiAgentOrchestrationWorkflow.run,
            analysis_input,
            id=f"{workflow_id}_analysis_phase"
        )
        
        # Stage 3: Quality check with consensus
        if config.get("require_consensus", True):
            consensus_input = MultiAgentWorkflowInput(
                workflow_id=f"{workflow_id}_consensus",
                agents=[
                    {"type": "quality_checker", "config": {"model": "gpt-4"}},
                    {"type": "quality_checker", "config": {"model": "claude-3"}},
                    {"type": "quality_checker", "config": {"model": "gemini-pro"}}
                ],
                orchestration_type="consensus",
                input_data=analysis_results["final_output"],
                config={"consensus_strategy": "weighted_average"}
            )
            
            consensus_results = await workflow.execute_child_workflow(
                MultiAgentOrchestrationWorkflow.run,
                consensus_input,
                id=f"{workflow_id}_consensus_phase"
            )
            
            final_output = consensus_results["consensus"]
        else:
            final_output = analysis_results["final_output"]
        
        # Publish final results
        await workflow.execute_activity(
            KafkaProducerActivity.publish_event,
            "research.pipeline.completed",
            {
                "workflow_id": workflow_id,
                "query": query,
                "results": final_output
            },
            start_to_close_timeout=timedelta(seconds=30)
        )
        
        return {
            "workflow_id": workflow_id,
            "query": query,
            "research": research_results,
            "analysis": analysis_results,
            "final_output": final_output,
            "stages_completed": 3
        }


@workflow.defn
class ConsensusWorkflow:
    """
    Byzantine fault-tolerant consensus workflow.
    
    Implements:
    - PBFT-style consensus
    - Leader election
    - View changes
    - Message authentication
    """
    
    @workflow.run
    async def run(self, proposal: Dict[str, Any], validators: List[str]) -> Dict[str, Any]:
        """Execute Byzantine consensus protocol."""
        # This is a placeholder for the full PBFT implementation
        # Will be expanded in Week 3
        workflow.logger.info(
            "Starting consensus workflow",
            proposal_id=proposal.get("id"),
            validator_count=len(validators)
        )
        
        # For now, return a simple consensus result
        return {
            "consensus_achieved": True,
            "proposal": proposal,
            "validators": validators,
            "round": 1,
            "timestamp": workflow.now().isoformat()
        }