#!/usr/bin/env python3
"""
⚡ Executor Agent - Professional Action Execution

Advanced executor agent for taking actions based on analysis.
Built on your proven patterns with enterprise-grade execution.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Import schemas
schema_dir = Path(__file__).parent.parent / "schemas"
sys.path.insert(0, str(schema_dir))

try:
    import enums
    import base
    from production_observer_agent import ProductionAgentState, ProductionEvidence, AgentConfig
except ImportError:
    # Fallback for testing
    class ProductionAgentState:
        def __init__(self): pass
    class ProductionEvidence:
        def __init__(self, **kwargs): pass
    class AgentConfig:
        def __init__(self): pass

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    Professional executor agent using your proven patterns.
    
    The executor specializes in:
    1. Action planning based on analysis results
    2. Safe execution with rollback capabilities
    3. Result validation and reporting
    4. Integration with external systems
    5. Governance and compliance tracking
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = f"executor_{config.agent_id}"
        
        # Execution configuration
        self.action_registry = self._initialize_action_registry()
        self.safety_checks = True
        self.dry_run_mode = config.get("dry_run_mode", False)
        
        # Execution limits
        self.max_concurrent_actions = 3
        self.action_timeout_seconds = 30
        self.retry_attempts = 2
        
        logger.info(f"⚡ ExecutorAgent initialized: {self.agent_id}")
    
    async def execute_action(self, state: ProductionAgentState) -> ProductionAgentState:
        """
        Main execution function - the executor's core capability.
        
        Args:
            state: Current workflow state with analysis results
            
        Returns:
            State enriched with execution evidence
        """
        
        logger.info(f"⚡ ExecutorAgent executing: {state.workflow_id}")
        
        try:
            # Step 1: Extract analysis results
            analysis_evidence = self._get_latest_analysis(state)
            
            if not analysis_evidence:
                logger.warning("No analysis found for execution")
                return self._create_no_analysis_execution(state)
            
            # Step 2: Plan actions based on analysis
            action_plan = self._create_action_plan(analysis_evidence)
            
            # Step 3: Validate action plan
            validation_result = await self._validate_action_plan(action_plan, state)
            
            if not validation_result["valid"]:
                logger.warning(f"Action plan validation failed: {validation_result['reason']}")
                return self._create_validation_failed_execution(state, validation_result)
            
            # Step 4: Execute actions
            execution_results = await self._execute_action_plan(action_plan)
            
            # Step 5: Validate execution results
            result_validation = self._validate_execution_results(execution_results)
            
            # Step 6: Create execution evidence
            execution_evidence = self._create_execution_evidence(
                state,
                action_plan,
                execution_results,
                result_validation
            )
            
            # Step 7: Update state immutably
            new_state = state.add_evidence(execution_evidence, self.config)
            
            # Step 8: Update state status based on results
            new_state = self._update_state_status(new_state, result_validation)
            
            success_count = sum(1 for r in execution_results if r.get("success", False))
            logger.info(f"✅ Execution complete: {success_count}/{len(execution_results)} actions succeeded")
            
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return self._create_error_execution(state, str(e))
    
    def _get_latest_analysis(self, state: ProductionAgentState) -> Optional[Any]:
        """Get the latest analysis evidence from state."""
        
        try:
            evidence_entries = getattr(state, 'evidence_entries', [])
            
            for evidence in reversed(evidence_entries):
                evidence_type = getattr(evidence, 'evidence_type', None)
                if evidence_type and str(evidence_type) == "EvidenceType.PATTERN":
                    return evidence
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest analysis: {e}")
            return None
    
    def _create_action_plan(self, analysis_evidence: Any) -> Dict[str, Any]:
        """Create action plan based on analysis results."""
        
        try:
            content = getattr(analysis_evidence, 'content', {})
            
            risk_score = content.get('risk_score', 0.5)
            risk_level = content.get('risk_level', 'medium')
            recommendations = content.get('recommendations', [])
            patterns = content.get('patterns_detected', [])
            
            # Determine actions based on risk level and recommendations
            actions = []
            
            if risk_level == "critical":
                actions.extend(self._get_critical_actions(patterns, recommendations))
            elif risk_level == "high":
                actions.extend(self._get_high_risk_actions(patterns, recommendations))
            elif risk_level == "medium":
                actions.extend(self._get_medium_risk_actions(patterns, recommendations))
            else:
                actions.extend(self._get_low_risk_actions(patterns, recommendations))
            
            # Add monitoring actions
            actions.extend(self._get_monitoring_actions(risk_level))
            
            action_plan = {
                "plan_id": f"plan_{base.utc_now().strftime('%Y%m%d_%H%M%S')}",
                "risk_level": risk_level,
                "risk_score": risk_score,
                "actions": actions,
                "execution_strategy": "sequential",  # or "parallel" for independent actions
                "rollback_plan": self._create_rollback_plan(actions),
                "estimated_duration_seconds": sum(a.get("estimated_duration", 10) for a in actions),
                "created_timestamp": base.utc_now().isoformat()
            }
            
            logger.info(f"📋 Action plan created: {len(actions)} actions for {risk_level} risk")
            return action_plan
            
        except Exception as e:
            logger.error(f"Failed to create action plan: {e}")
            return {"actions": [], "error": str(e)}
    
    def _get_critical_actions(self, patterns: List[str], recommendations: List[str]) -> List[Dict[str, Any]]:
        """Get actions for critical risk situations."""
        
        actions = []
        
        # Immediate alerting
        actions.append({
            "type": "send_critical_alert",
            "priority": "immediate",
            "target": "incident_response_team",
            "message": "Critical risk detected - immediate attention required",
            "estimated_duration": 5
        })
        
        # Stop operations if recommended
        if "stop_current_operations" in recommendations:
            actions.append({
                "type": "halt_operations",
                "priority": "immediate",
                "scope": "current_workflow",
                "reason": "Critical risk mitigation",
                "estimated_duration": 10
            })
        
        # Escalate to human
        actions.append({
            "type": "escalate_to_human",
            "priority": "immediate",
            "escalation_level": "senior_engineer",
            "context": "Critical risk situation requires human intervention",
            "estimated_duration": 60
        })
        
        return actions
    
    def _get_high_risk_actions(self, patterns: List[str], recommendations: List[str]) -> List[Dict[str, Any]]:
        """Get actions for high risk situations."""
        
        actions = []
        
        # High priority alerting
        actions.append({
            "type": "send_high_priority_alert",
            "priority": "high",
            "target": "operations_team",
            "message": "High risk detected - urgent attention required",
            "estimated_duration": 5
        })
        
        # Increase monitoring
        if "increase_monitoring" in recommendations:
            actions.append({
                "type": "increase_monitoring",
                "priority": "high",
                "monitoring_level": "enhanced",
                "duration_minutes": 60,
                "estimated_duration": 15
            })
        
        # Create incident ticket
        actions.append({
            "type": "create_incident_ticket",
            "priority": "high",
            "severity": "high",
            "description": "High risk situation detected by collective intelligence",
            "estimated_duration": 10
        })
        
        return actions
    
    def _get_medium_risk_actions(self, patterns: List[str], recommendations: List[str]) -> List[Dict[str, Any]]:
        """Get actions for medium risk situations."""
        
        actions = []
        
        # Standard alerting
        actions.append({
            "type": "send_standard_alert",
            "priority": "medium",
            "target": "monitoring_team",
            "message": "Medium risk detected - investigation recommended",
            "estimated_duration": 5
        })
        
        # Schedule investigation
        if "schedule_investigation" in recommendations:
            actions.append({
                "type": "schedule_investigation",
                "priority": "medium",
                "investigation_type": "pattern_analysis",
                "scheduled_within_hours": 24,
                "estimated_duration": 10
            })
        
        return actions
    
    def _get_low_risk_actions(self, patterns: List[str], recommendations: List[str]) -> List[Dict[str, Any]]:
        """Get actions for low risk situations."""
        
        actions = []
        
        # Log observation
        actions.append({
            "type": "log_observation",
            "priority": "low",
            "log_level": "info",
            "message": "Low risk situation - continuing normal operations",
            "estimated_duration": 2
        })
        
        # Update metrics
        actions.append({
            "type": "update_metrics",
            "priority": "low",
            "metric_type": "workflow_health",
            "value": "healthy",
            "estimated_duration": 3
        })
        
        return actions
    
    def _get_monitoring_actions(self, risk_level: str) -> List[Dict[str, Any]]:
        """Get monitoring actions based on risk level."""
        
        actions = []
        
        if risk_level in ["critical", "high"]:
            actions.append({
                "type": "enable_enhanced_monitoring",
                "priority": "high",
                "monitoring_duration_minutes": 120,
                "monitoring_frequency_seconds": 30,
                "estimated_duration": 5
            })
        else:
            actions.append({
                "type": "update_monitoring_status",
                "priority": "low",
                "status": "normal",
                "estimated_duration": 2
            })
        
        return actions
    
    def _create_rollback_plan(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create rollback plan for actions."""
        
        rollback_actions = []
        
        for action in actions:
            action_type = action.get("type", "unknown")
            
            # Define rollback for each action type
            if action_type == "halt_operations":
                rollback_actions.append({
                    "type": "resume_operations",
                    "reason": "rollback_halt_operations"
                })
            elif action_type == "increase_monitoring":
                rollback_actions.append({
                    "type": "reset_monitoring_level",
                    "reason": "rollback_enhanced_monitoring"
                })
            # Add more rollback mappings as needed
        
        return {
            "rollback_actions": rollback_actions,
            "rollback_strategy": "reverse_order",
            "created_timestamp": base.utc_now().isoformat()
        }
    
    async def _validate_action_plan(self, action_plan: Dict[str, Any], state: Any) -> Dict[str, Any]:
        """Validate action plan before execution."""
        
        try:
            actions = action_plan.get("actions", [])
            
            if not actions:
                return {
                    "valid": False,
                    "reason": "no_actions_in_plan",
                    "details": "Action plan contains no actions to execute"
                }
            
            # Check for dangerous actions in dry run mode
            if self.dry_run_mode:
                dangerous_actions = ["halt_operations", "delete_data", "restart_service"]
                for action in actions:
                    if action.get("type") in dangerous_actions:
                        return {
                            "valid": False,
                            "reason": "dangerous_action_in_dry_run",
                            "details": f"Action {action.get('type')} not allowed in dry run mode"
                        }
            
            # Check execution limits
            if len(actions) > self.max_concurrent_actions * 2:  # Allow some buffer
                return {
                    "valid": False,
                    "reason": "too_many_actions",
                    "details": f"Action plan has {len(actions)} actions, limit is {self.max_concurrent_actions * 2}"
                }
            
            # Validate individual actions
            for action in actions:
                if not self._validate_single_action(action):
                    return {
                        "valid": False,
                        "reason": "invalid_action",
                        "details": f"Action {action.get('type', 'unknown')} failed validation"
                    }
            
            return {
                "valid": True,
                "reason": "validation_passed",
                "details": f"All {len(actions)} actions validated successfully"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "reason": "validation_error",
                "details": str(e)
            }
    
    def _validate_single_action(self, action: Dict[str, Any]) -> bool:
        """Validate a single action."""
        
        required_fields = ["type", "priority"]
        
        for field in required_fields:
            if field not in action:
                logger.warning(f"Action missing required field: {field}")
                return False
        
        # Check if action type is supported
        action_type = action.get("type")
        if action_type not in self.action_registry:
            logger.warning(f"Unsupported action type: {action_type}")
            return False
        
        return True
    
    async def _execute_action_plan(self, action_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the action plan."""
        
        actions = action_plan.get("actions", [])
        execution_strategy = action_plan.get("execution_strategy", "sequential")
        
        if execution_strategy == "parallel":
            return await self._execute_actions_parallel(actions)
        else:
            return await self._execute_actions_sequential(actions)
    
    async def _execute_actions_sequential(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute actions sequentially."""
        
        results = []
        
        for i, action in enumerate(actions):
            logger.info(f"⚡ Executing action {i+1}/{len(actions)}: {action.get('type')}")
            
            try:
                result = await self._execute_single_action(action)
                results.append(result)
                
                # Stop on critical failure
                if not result.get("success", False) and action.get("priority") == "immediate":
                    logger.error("Critical action failed - stopping execution")
                    break
                    
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                results.append({
                    "action_type": action.get("type", "unknown"),
                    "success": False,
                    "error": str(e),
                    "execution_time_ms": 0
                })
        
        return results
    
    async def _execute_actions_parallel(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute actions in parallel (limited concurrency)."""
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_actions)
        
        async def execute_with_semaphore(action):
            async with semaphore:
                return await self._execute_single_action(action)
        
        # Execute all actions concurrently
        tasks = [execute_with_semaphore(action) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "action_type": actions[i].get("type", "unknown"),
                    "success": False,
                    "error": str(result),
                    "execution_time_ms": 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        
        start_time = datetime.utcnow()
        action_type = action.get("type", "unknown")
        
        try:
            # Get action executor from registry
            executor = self.action_registry.get(action_type)
            
            if not executor:
                return {
                    "action_type": action_type,
                    "success": False,
                    "error": f"No executor found for action type: {action_type}",
                    "execution_time_ms": 0
                }
            
            # Execute with timeout
            result = await asyncio.wait_for(
                executor(action),
                timeout=self.action_timeout_seconds
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "action_type": action_type,
                "success": True,
                "result": result,
                "execution_time_ms": execution_time,
                "executed_at": start_time.isoformat()
            }
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "action_type": action_type,
                "success": False,
                "error": f"Action timed out after {self.action_timeout_seconds} seconds",
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "action_type": action_type,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    def _initialize_action_registry(self) -> Dict[str, Callable]:
        """Initialize the action executor registry."""
        
        return {
            "send_critical_alert": self._execute_send_alert,
            "send_high_priority_alert": self._execute_send_alert,
            "send_standard_alert": self._execute_send_alert,
            "halt_operations": self._execute_halt_operations,
            "escalate_to_human": self._execute_escalate_to_human,
            "increase_monitoring": self._execute_increase_monitoring,
            "create_incident_ticket": self._execute_create_ticket,
            "schedule_investigation": self._execute_schedule_investigation,
            "log_observation": self._execute_log_observation,
            "update_metrics": self._execute_update_metrics,
            "enable_enhanced_monitoring": self._execute_enable_monitoring,
            "update_monitoring_status": self._execute_update_monitoring_status
        }
    
    # Action executors (simplified implementations)
    
    async def _execute_send_alert(self, action: Dict[str, Any]) -> str:
        """Execute alert sending."""
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Alert sent to {action.get('target', 'unknown')}: {action.get('message', '')}"
    
    async def _execute_halt_operations(self, action: Dict[str, Any]) -> str:
        """Execute operations halt."""
        if self.dry_run_mode:
            return "DRY RUN: Would halt operations"
        await asyncio.sleep(0.2)  # Simulate halt process
        return f"Operations halted: {action.get('reason', 'unknown')}"
    
    async def _execute_escalate_to_human(self, action: Dict[str, Any]) -> str:
        """Execute human escalation."""
        await asyncio.sleep(0.1)  # Simulate escalation
        return f"Escalated to {action.get('escalation_level', 'unknown')}"
    
    async def _execute_increase_monitoring(self, action: Dict[str, Any]) -> str:
        """Execute monitoring increase."""
        await asyncio.sleep(0.1)  # Simulate monitoring setup
        return f"Monitoring increased to {action.get('monitoring_level', 'enhanced')}"
    
    async def _execute_create_ticket(self, action: Dict[str, Any]) -> str:
        """Execute ticket creation."""
        await asyncio.sleep(0.1)  # Simulate ticket API
        return f"Ticket created with {action.get('severity', 'medium')} severity"
    
    async def _execute_schedule_investigation(self, action: Dict[str, Any]) -> str:
        """Execute investigation scheduling."""
        await asyncio.sleep(0.1)  # Simulate scheduling
        return f"Investigation scheduled: {action.get('investigation_type', 'unknown')}"
    
    async def _execute_log_observation(self, action: Dict[str, Any]) -> str:
        """Execute observation logging."""
        return f"Logged: {action.get('message', 'observation')}"
    
    async def _execute_update_metrics(self, action: Dict[str, Any]) -> str:
        """Execute metrics update."""
        return f"Metrics updated: {action.get('metric_type', 'unknown')} = {action.get('value', 'unknown')}"
    
    async def _execute_enable_monitoring(self, action: Dict[str, Any]) -> str:
        """Execute enhanced monitoring."""
        await asyncio.sleep(0.1)  # Simulate monitoring setup
        return f"Enhanced monitoring enabled for {action.get('monitoring_duration_minutes', 60)} minutes"
    
    async def _execute_update_monitoring_status(self, action: Dict[str, Any]) -> str:
        """Execute monitoring status update."""
        return f"Monitoring status updated to {action.get('status', 'normal')}"
    
    def _validate_execution_results(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate execution results."""
        
        total_actions = len(execution_results)
        successful_actions = sum(1 for r in execution_results if r.get("success", False))
        failed_actions = total_actions - successful_actions
        
        success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        # Determine overall execution status
        if success_rate == 1.0:
            status = "complete_success"
        elif success_rate >= 0.8:
            status = "mostly_successful"
        elif success_rate >= 0.5:
            status = "partial_success"
        else:
            status = "mostly_failed"
        
        return {
            "status": status,
            "success_rate": success_rate,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "validation_timestamp": base.utc_now().isoformat()
        }
    
    def _create_execution_evidence(self, state: Any, action_plan: Dict[str, Any], 
                                 execution_results: List[Dict[str, Any]], 
                                 result_validation: Dict[str, Any]) -> Any:
        """Create comprehensive execution evidence."""
        
        try:
            execution_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.OBSERVATION,
                content={
                    "execution_type": "collective_intelligence_execution",
                    "action_plan_id": action_plan.get("plan_id", "unknown"),
                    "actions_planned": len(action_plan.get("actions", [])),
                    "actions_executed": len(execution_results),
                    "execution_results": execution_results,
                    "result_validation": result_validation,
                    "success_rate": result_validation.get("success_rate", 0.0),
                    "execution_status": result_validation.get("status", "unknown"),
                    "dry_run_mode": self.dry_run_mode,
                    "execution_timestamp": base.utc_now().isoformat(),
                    "executor_id": self.agent_id,
                    "execution_version": "v1.0"
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return execution_evidence
            
        except Exception as e:
            logger.error(f"Failed to create execution evidence: {e}")
            return None
    
    def _update_state_status(self, state: Any, result_validation: Dict[str, Any]) -> Any:
        """Update state status based on execution results."""
        
        try:
            execution_status = result_validation.get("status", "unknown")
            
            if execution_status in ["complete_success", "mostly_successful"]:
                if hasattr(state, 'status'):
                    state.status = enums.TaskStatus.COMPLETED
            elif execution_status in ["partial_success", "mostly_failed"]:
                if hasattr(state, 'status'):
                    state.status = enums.TaskStatus.FAILED
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to update state status: {e}")
            return state
    
    def _create_no_analysis_execution(self, state: Any) -> Any:
        """Create execution evidence when no analysis is available."""
        
        try:
            no_analysis_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.OBSERVATION,
                content={
                    "execution_type": "no_analysis_execution",
                    "status": "skipped",
                    "reason": "no_analysis_available",
                    "recommendation": "ensure_analysis_before_execution",
                    "execution_timestamp": base.utc_now().isoformat(),
                    "executor_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return state.add_evidence(no_analysis_evidence, self.config)
            
        except Exception as e:
            logger.error(f"Failed to create no-analysis execution: {e}")
            return state
    
    def _create_validation_failed_execution(self, state: Any, validation_result: Dict[str, Any]) -> Any:
        """Create execution evidence when validation fails."""
        
        try:
            validation_failed_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.OBSERVATION,
                content={
                    "execution_type": "validation_failed_execution",
                    "status": "validation_failed",
                    "validation_result": validation_result,
                    "recommendation": "review_action_plan",
                    "execution_timestamp": base.utc_now().isoformat(),
                    "executor_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return state.add_evidence(validation_failed_evidence, self.config)
            
        except Exception as e:
            logger.error(f"Failed to create validation-failed execution: {e}")
            return state
    
    def _create_error_execution(self, state: Any, error_message: str) -> Any:
        """Create execution evidence for execution errors."""
        
        try:
            error_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.OBSERVATION,
                content={
                    "execution_type": "error_execution",
                    "status": "execution_error",
                    "error_message": error_message,
                    "recommendation": "investigate_execution_error",
                    "execution_timestamp": base.utc_now().isoformat(),
                    "executor_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return state.add_evidence(error_evidence, self.config)
            
        except Exception as e:
            logger.error(f"Failed to create error execution: {e}")
            return state
