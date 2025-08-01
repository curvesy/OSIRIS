"""
⚡ Action Executor - Professional Action Execution
Clean, focused action execution with proper error handling.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .schemas import ActiveModeDecision, ActionStatus

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    ⚡ Professional Action Executor
    
    Handles safe execution of approved actions:
    - Action validation
    - Execution with timeout
    - Result tracking
    - Error handling and recovery
    """
    
    def __init__(self, execution_timeout: float = 30.0):
        self.execution_timeout = execution_timeout
        self.execution_history: Dict[str, Dict[str, Any]] = {}
        
        logger.info("⚡ Action Executor initialized")
    
    async def execute_action(self, decision: ActiveModeDecision) -> Dict[str, Any]:
        """
        Execute an approved action safely.
        
        Args:
            decision: The decision containing the action to execute
            
        Returns:
            Execution result dictionary
        """
        if decision.status != ActionStatus.APPROVED:
            logger.error(f"❌ Cannot execute non-approved action: {decision.decision_id}")
            return {
                'success': False,
                'error': 'Action not approved for execution',
                'execution_time': datetime.now().isoformat()
            }
        
        logger.info(f"⚡ Executing action: {decision.proposed_action}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_action_impl(decision),
                timeout=self.execution_timeout
            )
            
            # Store execution history
            self.execution_history[decision.decision_id] = result
            
            logger.info(f"✅ Action executed successfully: {decision.decision_id}")
            return result
            
        except asyncio.TimeoutError:
            error_result = {
                'success': False,
                'error': f'Action execution timed out after {self.execution_timeout}s',
                'execution_time': datetime.now().isoformat()
            }
            
            self.execution_history[decision.decision_id] = error_result
            logger.error(f"⏰ Action execution timeout: {decision.decision_id}")
            return error_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f'Action execution failed: {str(e)}',
                'execution_time': datetime.now().isoformat()
            }
            
            self.execution_history[decision.decision_id] = error_result
            logger.error(f"❌ Action execution failed: {decision.decision_id} - {e}")
            return error_result
    
    async def _execute_action_impl(self, decision: ActiveModeDecision) -> Dict[str, Any]:
        """
        Internal action execution implementation.
        
        In production, this would integrate with actual systems:
        - Kubernetes API for scaling
        - Service management APIs
        - Infrastructure automation tools
        - Monitoring system APIs
        
        For demo, we simulate realistic execution.
        """
        action = decision.proposed_action.lower()
        
        # Simulate execution time based on action complexity
        if 'scale' in action or 'deploy' in action:
            await asyncio.sleep(0.5)  # Complex actions take longer
        elif 'restart' in action or 'update' in action:
            await asyncio.sleep(0.3)  # Medium complexity
        else:
            await asyncio.sleep(0.1)  # Simple actions
        
        # Simulate different execution outcomes based on action type
        if 'block' in action:
            # Security actions
            return {
                'success': True,
                'message': f"Security action '{decision.proposed_action}' executed successfully",
                'execution_time': datetime.now().isoformat(),
                'cost_impact': 5000.0,  # Security incident prevention value
                'action_type': 'security',
                'affected_resources': ['firewall_rules', 'access_control']
            }
        
        elif 'scale' in action:
            # Scaling actions
            return {
                'success': True,
                'message': f"Scaling action '{decision.proposed_action}' executed successfully",
                'execution_time': datetime.now().isoformat(),
                'cost_impact': 2000.0,  # Performance optimization value
                'action_type': 'scaling',
                'affected_resources': ['compute_instances', 'load_balancer']
            }
        
        elif 'restart' in action:
            # Service management actions
            return {
                'success': True,
                'message': f"Service action '{decision.proposed_action}' executed successfully",
                'execution_time': datetime.now().isoformat(),
                'cost_impact': 1500.0,  # Service recovery value
                'action_type': 'service_management',
                'affected_resources': ['application_services']
            }
        
        else:
            # Generic actions
            return {
                'success': True,
                'message': f"Action '{decision.proposed_action}' executed successfully",
                'execution_time': datetime.now().isoformat(),
                'cost_impact': 1000.0,  # General operational value
                'action_type': 'general',
                'affected_resources': ['system_configuration']
            }
    
    def get_execution_history(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution history for a specific decision.
        
        Args:
            decision_id: ID of the decision to get history for
            
        Returns:
            Execution result if found, None otherwise
        """
        return self.execution_history.get(decision_id)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring."""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'average_cost_impact': 0.0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for result in self.execution_history.values() if result.get('success', False))
        failed = total - successful
        
        total_cost_impact = sum(
            result.get('cost_impact', 0.0) 
            for result in self.execution_history.values() 
            if result.get('success', False)
        )
        
        return {
            'total_executions': total,
            'successful_executions': successful,
            'failed_executions': failed,
            'success_rate': (successful / total) * 100 if total > 0 else 0.0,
            'average_cost_impact': total_cost_impact / successful if successful > 0 else 0.0
        }
    
    def clear_history(self):
        """Clear execution history (for testing or maintenance)."""
        self.execution_history.clear()
        logger.info("🧹 Execution history cleared")
