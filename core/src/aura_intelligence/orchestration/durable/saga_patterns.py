"""
🔄 Saga Pattern Implementation

Implements distributed transaction patterns for multi-agent workflows.
Provides compensation logic, rollback mechanisms, and TDA-aware error handling
for maintaining consistency across agent operations.

Key Features:
- Saga orchestration with compensation actions
- Forward and backward recovery strategies
- TDA context integration for intelligent compensation
- Automatic rollback on failure scenarios

TDA Integration:
- Uses TDA context for compensation decision making
- Correlates saga failures with TDA anomaly patterns
- Implements TDA-aware rollback strategies
- Tracks saga success rates for TDA analysis
"""

from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# TDA integration
try:
    from aura_intelligence.observability.tracing import get_tracer
    from ..semantic.tda_integration import TDAContextIntegration
    from ..semantic.base_interfaces import TDAContext
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None
    TDAContextIntegration = None
    TDAContext = None

class SagaStepStatus(Enum):
    """Status of individual saga steps"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

class CompensationType(Enum):
    """Types of compensation actions"""
    ROLLBACK = "rollback"
    FORWARD_RECOVERY = "forward_recovery"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP = "skip"

@dataclass
class SagaStep:
    """Individual step in a saga transaction"""
    step_id: str
    name: str
    action: Callable
    compensation_action: Optional[Callable]
    parameters: Dict[str, Any]
    compensation_parameters: Optional[Dict[str, Any]] = None
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[datetime] = None
    compensation_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class CompensationHandler:
    """Handler for compensation logic"""
    handler_id: str
    step_name: str
    compensation_type: CompensationType
    handler_function: Callable
    parameters: Dict[str, Any]
    tda_context_required: bool = False
    priority: int = 0  # Higher priority compensations execute first

class SagaOrchestrator:
    """
    Orchestrates saga patterns for distributed transactions
    """
    
    def __init__(self, tda_integration: Optional[TDAContextIntegration] = None):
        self.tda_integration = tda_integration or TDAContextIntegration() if TDAContextIntegration else None
        self.active_sagas: Dict[str, List[SagaStep]] = {}
        self.compensation_handlers: Dict[str, List[CompensationHandler]] = {}
        self.saga_history: List[Dict[str, Any]] = []
    
    async def execute_saga(
        self,
        saga_id: str,
        steps: List[SagaStep],
        tda_correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a saga transaction with automatic compensation on failure
        """
        if tracer:
            with tracer.start_as_current_span("saga_execution") as span:
                span.set_attributes({
                    "saga.id": saga_id,
                    "saga.steps_count": len(steps),
                    "tda.correlation_id": tda_correlation_id or "none"
                })
        
        start_time = datetime.now(timezone.utc)
        self.active_sagas[saga_id] = steps
        executed_steps = []
        
        try:
            # Get TDA context for saga planning
            tda_context = None
            if self.tda_integration and tda_correlation_id:
                tda_context = await self.tda_integration.get_context(tda_correlation_id)
            
            # Execute steps sequentially
            for step in steps:
                step.status = SagaStepStatus.EXECUTING
                step.execution_time = datetime.now(timezone.utc)
                
                try:
                    # Execute step with TDA context
                    step_input = {
                        "parameters": step.parameters,
                        "tda_context": asdict(tda_context) if tda_context else None,
                        "saga_id": saga_id,
                        "step_id": step.step_id
                    }
                    
                    step.result = await self._execute_step_with_retry(step, step_input)
                    step.status = SagaStepStatus.COMPLETED
                    executed_steps.append(step)
                    
                except Exception as e:
                    step.status = SagaStepStatus.FAILED
                    step.error = str(e)
                    
                    # Trigger compensation for all executed steps
                    await self._compensate_saga_steps(
                        saga_id, executed_steps, tda_context, tda_correlation_id
                    )
                    
                    raise Exception(f"Saga {saga_id} failed at step {step.name}: {str(e)}")
            
            # Saga completed successfully
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                "saga_id": saga_id,
                "status": "completed",
                "steps_executed": len(executed_steps),
                "execution_time": execution_time,
                "results": {step.name: step.result for step in executed_steps},
                "tda_correlation_id": tda_correlation_id
            }
            
            self.saga_history.append(result)
            
            # Send success to TDA
            if self.tda_integration and tda_correlation_id:
                await self.tda_integration.send_orchestration_result(
                    result, tda_correlation_id
                )
            
            return result
            
        except Exception as e:
            # Saga failed, record failure
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                "saga_id": saga_id,
                "status": "failed",
                "steps_executed": len(executed_steps),
                "execution_time": execution_time,
                "error": str(e),
                "compensated_steps": [step.name for step in executed_steps if step.status == SagaStepStatus.COMPENSATED],
                "tda_correlation_id": tda_correlation_id
            }
            
            self.saga_history.append(result)
            
            # Send failure to TDA
            if self.tda_integration and tda_correlation_id:
                await self.tda_integration.send_orchestration_result(
                    result, tda_correlation_id
                )
            
            return result
            
        finally:
            # Clean up active saga
            if saga_id in self.active_sagas:
                del self.active_sagas[saga_id]
    
    async def _execute_step_with_retry(
        self,
        step: SagaStep,
        step_input: Dict[str, Any]
    ) -> Any:
        """
        Execute a saga step with retry logic
        """
        for attempt in range(step.max_retries + 1):
            try:
                step.retry_count = attempt
                
                # Execute the step action
                if asyncio.iscoroutinefunction(step.action):
                    result = await step.action(step_input)
                else:
                    result = step.action(step_input)
                
                return result
                
            except Exception as e:
                if attempt == step.max_retries:
                    raise  # Last attempt, re-raise the exception
                
                # Wait before retry with exponential backoff
                delay = min(2 ** attempt, 30)  # Max 30 seconds
                await asyncio.sleep(delay)
    
    async def _compensate_saga_steps(
        self,
        saga_id: str,
        executed_steps: List[SagaStep],
        tda_context: Optional[TDAContext],
        tda_correlation_id: Optional[str]
    ):
        """
        Compensate executed saga steps in reverse order
        """
        if tracer:
            with tracer.start_as_current_span("saga_compensation") as span:
                span.set_attributes({
                    "saga.id": saga_id,
                    "compensation.steps_count": len(executed_steps),
                    "tda.correlation_id": tda_correlation_id or "none"
                })
        
        # Execute compensation in reverse order
        for step in reversed(executed_steps):
            if step.compensation_action:
                try:
                    step.status = SagaStepStatus.COMPENSATING
                    step.compensation_time = datetime.now(timezone.utc)
                    
                    # Prepare compensation input
                    compensation_input = {
                        "original_parameters": step.parameters,
                        "original_result": step.result,
                        "compensation_parameters": step.compensation_parameters or {},
                        "tda_context": asdict(tda_context) if tda_context else None,
                        "saga_id": saga_id,
                        "step_id": step.step_id
                    }
                    
                    # Execute compensation
                    if asyncio.iscoroutinefunction(step.compensation_action):
                        await step.compensation_action(compensation_input)
                    else:
                        step.compensation_action(compensation_input)
                    
                    step.status = SagaStepStatus.COMPENSATED
                    
                except Exception as comp_error:
                    # Log compensation failure but continue with other compensations
                    step.error = f"Compensation failed: {str(comp_error)}"
                    
                    # Notify TDA about compensation failure
                    if self.tda_integration and tda_correlation_id:
                        await self.tda_integration.send_orchestration_result(
                            {
                                "saga_id": saga_id,
                                "step_id": step.step_id,
                                "compensation_error": str(comp_error),
                                "requires_manual_intervention": True
                            },
                            tda_correlation_id
                        )
    
    def register_compensation_handler(
        self,
        step_name: str,
        handler: CompensationHandler
    ):
        """
        Register a compensation handler for a specific step type
        """
        if step_name not in self.compensation_handlers:
            self.compensation_handlers[step_name] = []
        
        self.compensation_handlers[step_name].append(handler)
        
        # Sort by priority (higher priority first)
        self.compensation_handlers[step_name].sort(key=lambda h: h.priority, reverse=True)
    
    async def execute_custom_compensation(
        self,
        saga_id: str,
        step_name: str,
        compensation_type: CompensationType,
        parameters: Dict[str, Any],
        tda_correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute custom compensation logic for specific scenarios
        """
        handlers = self.compensation_handlers.get(step_name, [])
        
        # Find appropriate handler
        handler = None
        for h in handlers:
            if h.compensation_type == compensation_type:
                handler = h
                break
        
        if not handler:
            return {
                "status": "failed",
                "error": f"No compensation handler found for {step_name} with type {compensation_type.value}"
            }
        
        try:
            # Get TDA context if required
            tda_context = None
            if handler.tda_context_required and self.tda_integration and tda_correlation_id:
                tda_context = await self.tda_integration.get_context(tda_correlation_id)
            
            # Prepare handler input
            handler_input = {
                "saga_id": saga_id,
                "step_name": step_name,
                "parameters": parameters,
                "handler_parameters": handler.parameters,
                "tda_context": asdict(tda_context) if tda_context else None
            }
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler.handler_function):
                result = await handler.handler_function(handler_input)
            else:
                result = handler.handler_function(handler_input)
            
            return {
                "status": "completed",
                "handler_id": handler.handler_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "handler_id": handler.handler_id,
                "error": str(e)
            }
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a saga
        """
        if saga_id in self.active_sagas:
            steps = self.active_sagas[saga_id]
            return {
                "saga_id": saga_id,
                "status": "running",
                "total_steps": len(steps),
                "completed_steps": sum(1 for s in steps if s.status == SagaStepStatus.COMPLETED),
                "failed_steps": sum(1 for s in steps if s.status == SagaStepStatus.FAILED),
                "compensated_steps": sum(1 for s in steps if s.status == SagaStepStatus.COMPENSATED),
                "current_step": next((s.name for s in steps if s.status == SagaStepStatus.EXECUTING), None)
            }
        
        # Check history
        for saga_record in reversed(self.saga_history):
            if saga_record["saga_id"] == saga_id:
                return saga_record
        
        return None
    
    def get_saga_metrics(self) -> Dict[str, Any]:
        """
        Get saga execution metrics for monitoring
        """
        if not self.saga_history:
            return {
                "total_sagas": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "compensation_rate": 0.0
            }
        
        total_sagas = len(self.saga_history)
        successful_sagas = sum(1 for s in self.saga_history if s["status"] == "completed")
        compensated_sagas = sum(1 for s in self.saga_history if "compensated_steps" in s and s["compensated_steps"])
        
        total_execution_time = sum(s.get("execution_time", 0) for s in self.saga_history)
        
        return {
            "total_sagas": total_sagas,
            "success_rate": successful_sagas / total_sagas,
            "average_execution_time": total_execution_time / total_sagas,
            "compensation_rate": compensated_sagas / total_sagas,
            "active_sagas": len(self.active_sagas)
        }