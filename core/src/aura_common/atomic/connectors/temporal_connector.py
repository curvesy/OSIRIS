"""
Temporal connector atomic components.

Provides workflow and activity execution components for durable
execution with Temporal workflow engine.
"""

from typing import TypeVar, Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
import uuid
import asyncio

from ..base import AtomicComponent
from ..base.exceptions import ConnectionError, ProcessingError

T = TypeVar('T')
R = TypeVar('R')


class WorkflowStatus(Enum):
    """Workflow execution status."""
    
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINATED = "terminated"
    TIMED_OUT = "timed_out"


@dataclass
class TemporalConfig:
    """Configuration for Temporal connection."""
    
    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "aura-tasks"
    identity: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.host:
            raise ValueError("host required")
        if not self.namespace:
            raise ValueError("namespace required")
        if not self.task_queue:
            raise ValueError("task_queue required")


@dataclass
class WorkflowOptions:
    """Options for workflow execution."""
    
    workflow_id: Optional[str] = None
    task_queue: Optional[str] = None
    execution_timeout: Optional[timedelta] = None
    run_timeout: Optional[timedelta] = None
    task_timeout: Optional[timedelta] = None
    retry_policy: Optional[Dict[str, Any]] = None
    memo: Optional[Dict[str, Any]] = None
    search_attributes: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowExecution:
    """Workflow execution details."""
    
    workflow_id: str
    run_id: str
    status: WorkflowStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    close_time: Optional[str] = None


class TemporalWorkflowStarter(AtomicComponent[Dict[str, Any], WorkflowExecution, TemporalConfig]):
    """
    Atomic component for starting Temporal workflows.
    
    Features:
    - Workflow execution with options
    - Automatic ID generation
    - Execution tracking
    - Error handling
    """
    
    def __init__(self, name: str, config: TemporalConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._client = None
        self._stub_mode = True  # Using stub for now
    
    def _validate_config(self) -> None:
        """Validate Temporal configuration."""
        self.config.validate()
    
    async def _process(self, workflow_request: Dict[str, Any]) -> WorkflowExecution:
        """
        Start a Temporal workflow.
        
        Args:
            workflow_request: Dict containing:
                - workflow_type: Name of workflow to start
                - args: Workflow arguments
                - options: Optional WorkflowOptions
                
        Returns:
            WorkflowExecution details
        """
        # Extract request components
        workflow_type = workflow_request.get("workflow_type")
        args = workflow_request.get("args", [])
        options = workflow_request.get("options", {})
        
        if not workflow_type:
            raise ValueError("workflow_type required")
        
        # Generate workflow ID if not provided
        workflow_id = options.get("workflow_id") or f"{workflow_type}-{uuid.uuid4()}"
        
        # Initialize client if needed
        if self._client is None:
            await self._connect()
        
        try:
            if self._stub_mode:
                # Stub implementation
                self.logger.info(
                    f"Starting workflow (stub mode)",
                    workflow_type=workflow_type,
                    workflow_id=workflow_id,
                    args=args
                )
                
                # Simulate workflow start
                run_id = str(uuid.uuid4())
                
                return WorkflowExecution(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    status=WorkflowStatus.RUNNING,
                    start_time="2025-07-31T12:00:00Z"
                )
            
            else:
                # Real implementation would use temporal-sdk
                # handle = await self._client.start_workflow(...)
                pass
                
        except Exception as e:
            self.logger.error(f"Failed to start workflow: {e}")
            raise ProcessingError(
                f"Workflow start failed: {str(e)}",
                component_name=self.name
            )
    
    async def _connect(self) -> None:
        """Connect to Temporal server."""
        self.logger.info(
            "Connecting to Temporal",
            host=self.config.host,
            namespace=self.config.namespace
        )
        
        if self._stub_mode:
            self._client = MockTemporalClient()
        else:
            # Real implementation would create Temporal client
            pass


class TemporalWorkflowExecutor(AtomicComponent[str, WorkflowExecution, TemporalConfig]):
    """
    Atomic component for executing and monitoring workflows.
    
    Features:
    - Workflow status checking
    - Result retrieval
    - Cancellation support
    - Query capabilities
    """
    
    def __init__(self, name: str, config: TemporalConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._client = None
        self._stub_mode = True
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.config.validate()
    
    async def _process(self, workflow_id: str) -> WorkflowExecution:
        """
        Get workflow execution status.
        
        Args:
            workflow_id: Workflow ID to check
            
        Returns:
            Current WorkflowExecution status
        """
        if not workflow_id:
            raise ValueError("workflow_id required")
        
        # Initialize client if needed
        if self._client is None:
            await self._connect()
        
        try:
            if self._stub_mode:
                # Stub implementation
                return WorkflowExecution(
                    workflow_id=workflow_id,
                    run_id=str(uuid.uuid4()),
                    status=WorkflowStatus.COMPLETED,
                    result={"status": "success", "data": "stub_result"},
                    start_time="2025-07-31T12:00:00Z",
                    close_time="2025-07-31T12:01:00Z"
                )
            else:
                # Real implementation would query workflow
                pass
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            raise ProcessingError(
                f"Workflow query failed: {str(e)}",
                component_name=self.name
            )
    
    async def _connect(self) -> None:
        """Connect to Temporal server."""
        self.logger.info("Connecting to Temporal")
        
        if self._stub_mode:
            self._client = MockTemporalClient()
        else:
            # Real implementation
            pass
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            if self._stub_mode:
                self.logger.info(f"Cancelling workflow: {workflow_id}")
                return True
            else:
                # Real implementation
                pass
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow: {e}")
            return False


class TemporalActivityExecutor(AtomicComponent[Dict[str, Any], Any, TemporalConfig]):
    """
    Atomic component for executing Temporal activities.
    
    Features:
    - Activity registration
    - Heartbeat support
    - Cancellation handling
    - Result serialization
    """
    
    def __init__(self, name: str, config: TemporalConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._activities: Dict[str, Callable] = {}
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.config.validate()
    
    def register_activity(self, name: str, handler: Callable) -> None:
        """Register an activity handler."""
        self._activities[name] = handler
        self.logger.info(f"Registered activity: {name}")
    
    async def _process(self, activity_request: Dict[str, Any]) -> Any:
        """
        Execute a Temporal activity.
        
        Args:
            activity_request: Dict containing:
                - activity_type: Name of activity
                - args: Activity arguments
                
        Returns:
            Activity result
        """
        activity_type = activity_request.get("activity_type")
        args = activity_request.get("args", [])
        
        if not activity_type:
            raise ValueError("activity_type required")
        
        handler = self._activities.get(activity_type)
        if not handler:
            raise ValueError(f"Unknown activity: {activity_type}")
        
        try:
            # Execute activity
            self.logger.info(
                f"Executing activity",
                activity_type=activity_type,
                args_count=len(args)
            )
            
            # Call handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args)
            else:
                result = handler(*args)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Activity execution failed: {e}")
            raise ProcessingError(
                f"Activity failed: {str(e)}",
                component_name=self.name
            )


# Mock classes for testing
class MockTemporalClient:
    """Mock Temporal client for testing."""
    
    async def start_workflow(self, workflow_type: str, args: List[Any], **kwargs):
        """Mock workflow start."""
        return MockWorkflowHandle(
            workflow_id=kwargs.get("id", str(uuid.uuid4())),
            run_id=str(uuid.uuid4())
        )
    
    async def get_workflow_handle(self, workflow_id: str):
        """Mock get workflow handle."""
        return MockWorkflowHandle(workflow_id, str(uuid.uuid4()))


class MockWorkflowHandle:
    """Mock workflow handle."""
    
    def __init__(self, workflow_id: str, run_id: str):
        self.workflow_id = workflow_id
        self.run_id = run_id
    
    async def result(self):
        """Mock get result."""
        return {"status": "success", "data": "mock_result"}
    
    async def cancel(self):
        """Mock cancel."""
        pass