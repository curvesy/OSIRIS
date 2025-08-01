"""
Temporal Client for AURA Intelligence

Provides high-level interface for executing workflows with
observability, error handling, and result management.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass
from datetime import timedelta
import uuid
import asyncio

from temporalio.client import Client, WorkflowHandle as TemporalWorkflowHandle
from temporalio.common import RetryPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
import structlog
from opentelemetry import trace, metrics

from .workflows import (
    AgentWorkflowInput,
    AgentWorkflowResult,
    MultiAgentWorkflowInput
)

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
workflow_started = meter.create_counter(
    name="temporal.client.workflow.started",
    description="Number of workflows started"
)

workflow_completed = meter.create_counter(
    name="temporal.client.workflow.completed",
    description="Number of workflows completed"
)

workflow_failed = meter.create_counter(
    name="temporal.client.workflow.failed",
    description="Number of workflows failed"
)

T = TypeVar('T')


@dataclass
class WorkflowExecutionConfig:
    """Configuration for workflow execution."""
    workflow_id: Optional[str] = None
    task_queue: str = "agent-workflows"
    execution_timeout: timedelta = timedelta(minutes=30)
    run_timeout: timedelta = timedelta(minutes=10)
    task_timeout: timedelta = timedelta(seconds=30)
    retry_policy: Optional[RetryPolicy] = None
    id_reuse_policy: WorkflowIDReusePolicy = WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY
    search_attributes: Optional[Dict[str, Any]] = None
    memo: Optional[Dict[str, Any]] = None
    cron_schedule: Optional[str] = None


class WorkflowHandle(Generic[T]):
    """
    Enhanced workflow handle with observability and convenience methods.
    """
    
    def __init__(
        self,
        handle: TemporalWorkflowHandle,
        workflow_type: str,
        input_data: Any
    ):
        self.handle = handle
        self.workflow_type = workflow_type
        self.input_data = input_data
        self.workflow_id = handle.id
        self.run_id = handle.result_run_id
        
    async def result(self, timeout: Optional[timedelta] = None) -> T:
        """Get workflow result with timeout."""
        with tracer.start_as_current_span(
            "temporal.client.workflow.result",
            attributes={
                "workflow.id": self.workflow_id,
                "workflow.type": self.workflow_type
            }
        ) as span:
            try:
                result = await self.handle.result(timeout=timeout)
                
                workflow_completed.add(
                    1,
                    {"workflow_type": self.workflow_type, "status": "success"}
                )
                
                span.set_attribute("workflow.status", "completed")
                return result
                
            except Exception as e:
                workflow_failed.add(
                    1,
                    {"workflow_type": self.workflow_type, "error": type(e).__name__}
                )
                
                span.record_exception(e)
                span.set_attribute("workflow.status", "failed")
                raise
    
    async def query(self, query_type: str, *args) -> Any:
        """Query workflow state."""
        return await self.handle.query(query_type, *args)
    
    async def signal(self, signal_name: str, *args) -> None:
        """Send signal to workflow."""
        await self.handle.signal(signal_name, *args)
    
    async def cancel(self) -> None:
        """Cancel workflow execution."""
        await self.handle.cancel()
    
    async def terminate(self, reason: str) -> None:
        """Terminate workflow execution."""
        await self.handle.terminate(reason)
    
    async def describe(self) -> Dict[str, Any]:
        """Get workflow description."""
        description = await self.handle.describe()
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "type": self.workflow_type,
            "status": description.status,
            "start_time": description.start_time,
            "execution_time": description.execution_time,
            "memo": description.memo,
            "search_attributes": description.search_attributes
        }


class TemporalClient:
    """
    High-level Temporal client for AURA Intelligence.
    
    Features:
    - Simplified workflow execution
    - Automatic observability
    - Error handling and retries
    - Result caching
    """
    
    def __init__(
        self,
        host: str = "localhost:7233",
        namespace: str = "default"
    ):
        self.host = host
        self.namespace = namespace
        self.client: Optional[Client] = None
        self._connected = False
        
    async def connect(self) -> None:
        """Connect to Temporal server."""
        if self._connected:
            return
            
        logger.info(f"Connecting to Temporal at {self.host}")
        
        try:
            self.client = await Client.connect(
                self.host,
                namespace=self.namespace
            )
            self._connected = True
            logger.info("Connected to Temporal successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Temporal: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Temporal server."""
        if self.client and self._connected:
            await self.client.close()
            self._connected = False
            logger.info("Disconnected from Temporal")
    
    async def execute_agent_workflow(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        config: Optional[WorkflowExecutionConfig] = None
    ) -> WorkflowHandle[AgentWorkflowResult]:
        """Execute a single agent workflow."""
        await self.connect()
        
        if config is None:
            config = WorkflowExecutionConfig()
        
        # Generate workflow ID if not provided
        if not config.workflow_id:
            config.workflow_id = f"agent-{agent_type}-{uuid.uuid4()}"
        
        # Create workflow input
        workflow_input = AgentWorkflowInput(
            agent_id=config.workflow_id,
            agent_type=agent_type,
            input_data=input_data,
            config=config.search_attributes or {}
        )
        
        # Start workflow with span
        with tracer.start_as_current_span(
            "temporal.client.start_workflow",
            attributes={
                "workflow.type": "AgentWorkflow",
                "agent.type": agent_type,
                "workflow.id": config.workflow_id
            }
        ) as span:
            try:
                handle = await self.client.start_workflow(
                    "AgentWorkflow.run",
                    workflow_input,
                    id=config.workflow_id,
                    task_queue=config.task_queue,
                    execution_timeout=config.execution_timeout,
                    run_timeout=config.run_timeout,
                    task_timeout=config.task_timeout,
                    retry_policy=config.retry_policy,
                    id_reuse_policy=config.id_reuse_policy,
                    search_attributes=config.search_attributes,
                    memo=config.memo
                )
                
                workflow_started.add(
                    1,
                    {"workflow_type": "AgentWorkflow", "agent_type": agent_type}
                )
                
                logger.info(
                    f"Started agent workflow",
                    workflow_id=config.workflow_id,
                    agent_type=agent_type
                )
                
                return WorkflowHandle(handle, "AgentWorkflow", workflow_input)
                
            except WorkflowAlreadyStartedError:
                # Get existing workflow handle
                handle = self.client.get_workflow_handle(config.workflow_id)
                logger.info(f"Workflow already exists: {config.workflow_id}")
                return WorkflowHandle(handle, "AgentWorkflow", workflow_input)
                
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Failed to start workflow: {e}")
                raise
    
    async def execute_multi_agent_workflow(
        self,
        agents: List[Dict[str, Any]],
        orchestration_type: str,
        input_data: Dict[str, Any],
        config: Optional[WorkflowExecutionConfig] = None
    ) -> WorkflowHandle[Dict[str, Any]]:
        """Execute a multi-agent orchestration workflow."""
        await self.connect()
        
        if config is None:
            config = WorkflowExecutionConfig()
        
        # Generate workflow ID if not provided
        if not config.workflow_id:
            config.workflow_id = f"multi-agent-{orchestration_type}-{uuid.uuid4()}"
        
        # Create workflow input
        workflow_input = MultiAgentWorkflowInput(
            workflow_id=config.workflow_id,
            agents=agents,
            orchestration_type=orchestration_type,
            input_data=input_data,
            config=config.search_attributes or {}
        )
        
        # Start workflow
        handle = await self.client.start_workflow(
            "MultiAgentOrchestrationWorkflow.run",
            workflow_input,
            id=config.workflow_id,
            task_queue=config.task_queue,
            execution_timeout=config.execution_timeout,
            run_timeout=config.run_timeout,
            task_timeout=config.task_timeout,
            retry_policy=config.retry_policy,
            id_reuse_policy=config.id_reuse_policy,
            search_attributes=config.search_attributes,
            memo=config.memo
        )
        
        workflow_started.add(
            1,
            {
                "workflow_type": "MultiAgentOrchestrationWorkflow",
                "orchestration_type": orchestration_type
            }
        )
        
        logger.info(
            f"Started multi-agent workflow",
            workflow_id=config.workflow_id,
            orchestration_type=orchestration_type,
            agent_count=len(agents)
        )
        
        return WorkflowHandle(handle, "MultiAgentOrchestrationWorkflow", workflow_input)
    
    async def execute_research_pipeline(
        self,
        query: str,
        config: Optional[WorkflowExecutionConfig] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowHandle[Dict[str, Any]]:
        """Execute a research and analysis pipeline."""
        await self.connect()
        
        if config is None:
            config = WorkflowExecutionConfig(
                task_queue="research-pipelines",
                execution_timeout=timedelta(hours=1)
            )
        
        # Generate workflow ID if not provided
        if not config.workflow_id:
            config.workflow_id = f"research-{uuid.uuid4()}"
        
        # Start workflow
        handle = await self.client.start_workflow(
            "ResearchAnalysisPipeline.run",
            query,
            pipeline_config or {},
            id=config.workflow_id,
            task_queue=config.task_queue,
            execution_timeout=config.execution_timeout,
            run_timeout=config.run_timeout,
            task_timeout=config.task_timeout,
            retry_policy=config.retry_policy,
            id_reuse_policy=config.id_reuse_policy,
            search_attributes=config.search_attributes,
            memo=config.memo
        )
        
        workflow_started.add(
            1,
            {"workflow_type": "ResearchAnalysisPipeline"}
        )
        
        logger.info(
            f"Started research pipeline",
            workflow_id=config.workflow_id,
            query=query
        )
        
        return WorkflowHandle(handle, "ResearchAnalysisPipeline", query)
    
    async def list_workflows(
        self,
        query: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """List workflows with optional query."""
        await self.connect()
        
        workflows = []
        async for workflow in self.client.list_workflows(query=query, page_size=page_size):
            workflows.append({
                "workflow_id": workflow.id,
                "type": workflow.type_name,
                "status": workflow.status,
                "start_time": workflow.start_time,
                "execution_time": workflow.execution_time,
                "search_attributes": workflow.search_attributes
            })
        
        return workflows


# Convenience function for quick workflow execution
async def execute_workflow(
    workflow_type: str,
    *args,
    task_queue: str = "agent-workflows",
    timeout: timedelta = timedelta(minutes=30),
    **kwargs
) -> Any:
    """
    Execute a workflow and wait for result.
    
    This is a convenience function for simple use cases.
    For more control, use TemporalClient directly.
    """
    client = TemporalClient()
    
    try:
        await client.connect()
        
        if workflow_type == "agent":
            handle = await client.execute_agent_workflow(
                args[0],  # agent_type
                args[1],  # input_data
                WorkflowExecutionConfig(
                    task_queue=task_queue,
                    execution_timeout=timeout
                )
            )
        elif workflow_type == "multi_agent":
            handle = await client.execute_multi_agent_workflow(
                args[0],  # agents
                args[1],  # orchestration_type
                args[2],  # input_data
                WorkflowExecutionConfig(
                    task_queue=task_queue,
                    execution_timeout=timeout
                )
            )
        elif workflow_type == "research":
            handle = await client.execute_research_pipeline(
                args[0],  # query
                WorkflowExecutionConfig(
                    task_queue="research-pipelines",
                    execution_timeout=timeout
                ),
                kwargs.get("pipeline_config", {})
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Wait for result
        return await handle.result(timeout=timeout)
        
    finally:
        await client.disconnect()


# Example usage
if __name__ == "__main__":
    async def main():
        # Example 1: Execute single agent
        result = await execute_workflow(
            "agent",
            "observer",
            {"query": "What is the system status?"},
            timeout=timedelta(minutes=5)
        )
        print(f"Agent result: {result}")
        
        # Example 2: Execute multi-agent workflow
        result = await execute_workflow(
            "multi_agent",
            [
                {"type": "observer", "config": {}},
                {"type": "analyst", "config": {}}
            ],
            "sequential",
            {"query": "Analyze system performance"},
            timeout=timedelta(minutes=10)
        )
        print(f"Multi-agent result: {result}")
        
        # Example 3: Execute research pipeline
        result = await execute_workflow(
            "research",
            "Latest developments in quantum computing",
            timeout=timedelta(hours=1),
            pipeline_config={"require_consensus": True}
        )
        print(f"Research result: {result}")
    
    asyncio.run(main())