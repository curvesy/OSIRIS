"""
🚀 Active Mode Deployment - Professional Main Controller
Clean orchestration of all governance components.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..schemas import ActiveModeDecision, ActionStatus, RiskThresholds
from ..risk_engine import RiskAssessmentEngine
from ..database import GovernanceDatabase
from ..metrics import MetricsManager
from ..executor import ActionExecutor
from .human_approval import HumanApprovalManager

logger = logging.getLogger(__name__)


class ActiveModeDeployment:
    """
    🚀 Professional Active Mode Deployment Controller
    
    Orchestrates all governance components:
    - Risk assessment
    - Decision processing
    - Human approval workflows
    - Action execution
    - Metrics tracking
    """
    
    def __init__(self, 
                 db_path: str = "governance.db",
                 risk_thresholds: RiskThresholds = None):
        
        # Initialize all components
        self.risk_engine = RiskAssessmentEngine(risk_thresholds)
        self.database = GovernanceDatabase(db_path)
        self.metrics_manager = MetricsManager()
        self.executor = ActionExecutor()
        self.human_approval = HumanApprovalManager()
        
        # Decision tracking
        self.decisions_log: List[ActiveModeDecision] = []
        
        logger.info("🚀 Active Mode Deployment initialized")
    
    async def process_decision(self, 
                             evidence_log: List[Dict[str, Any]], 
                             proposed_action: str, 
                             reasoning: str) -> ActiveModeDecision:
        """
        Process a decision through the complete governance pipeline.
        
        Args:
            evidence_log: Current evidence from the workflow
            proposed_action: Action proposed by the supervisor
            reasoning: Reasoning behind the proposed action
            
        Returns:
            ActiveModeDecision with complete processing results
        """
        start_time = time.time()
        
        # Generate unique decision ID
        decision_id = f"active_{int(time.time() * 1000)}"
        
        logger.info(f"🔍 Processing decision {decision_id}: {proposed_action}")
        
        # 1. Risk Assessment
        risk_score = await self.risk_engine.calculate_risk_score(evidence_log, proposed_action)
        risk_level = self.risk_engine.determine_risk_level(risk_score)
        
        # 2. Create decision object
        decision = ActiveModeDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            evidence_log=evidence_log,
            proposed_action=proposed_action,
            risk_score=risk_score,
            risk_level=risk_level,
            reasoning=reasoning,
            status=ActionStatus.PENDING
        )
        
        # 3. Process based on risk level
        await self._process_by_risk_level(decision)
        
        # 4. Record metrics
        response_time = time.time() - start_time
        self.metrics_manager.record_decision(decision, response_time)
        
        # 5. Store in database
        self.database.store_decision(decision)
        
        # 6. Add to local log
        self.decisions_log.append(decision)
        
        logger.info(f"✅ Decision processed: {decision_id} (Risk: {risk_score:.3f}, Status: {decision.status.value})")
        
        return decision
    
    async def _process_by_risk_level(self, decision: ActiveModeDecision):
        """Process decision based on its risk level."""
        
        if decision.risk_level.value == 'low':
            # Auto-execute low-risk actions
            decision.status = ActionStatus.APPROVED
            execution_result = await self.executor.execute_action(decision)
            decision.execution_result = execution_result
            decision.execution_time = datetime.now()
            
            if execution_result.get('success'):
                decision.status = ActionStatus.EXECUTED
            else:
                decision.status = ActionStatus.FAILED
            
        elif decision.risk_level.value == 'medium':
            # Queue for human approval
            decision.status = ActionStatus.PENDING
            await self.human_approval.queue_for_approval(decision)
            logger.info(f"🤔 Queued for human approval: {decision.decision_id}")
            
        else:  # HIGH risk
            # Block high-risk actions
            decision.status = ActionStatus.BLOCKED
            logger.warning(f"🚫 High-risk action blocked: {decision.decision_id}")
    
    async def approve_decision(self, decision_id: str, reviewer: str, approved: bool) -> bool:
        """
        Process human approval for a decision.
        
        Args:
            decision_id: ID of the decision to approve/reject
            reviewer: Name/ID of the human reviewer
            approved: Whether the action was approved
            
        Returns:
            True if successfully processed, False otherwise
        """
        # Find the decision
        decision = self._find_decision(decision_id)
        if not decision:
            logger.error(f"❌ Decision {decision_id} not found")
            return False
        
        if decision.status != ActionStatus.PENDING:
            logger.error(f"❌ Decision {decision_id} is not pending approval")
            return False
        
        # Process approval
        decision.human_reviewer = reviewer
        
        if approved:
            decision.status = ActionStatus.APPROVED
            
            # Execute the approved action
            execution_result = await self.executor.execute_action(decision)
            decision.execution_result = execution_result
            decision.execution_time = datetime.now()
            
            if execution_result.get('success'):
                decision.status = ActionStatus.EXECUTED
            else:
                decision.status = ActionStatus.FAILED
            
            logger.info(f"✅ Decision {decision_id} approved and executed by {reviewer}")
        else:
            decision.status = ActionStatus.BLOCKED
            logger.info(f"🚫 Decision {decision_id} rejected by {reviewer}")
        
        # Update database
        self.database.store_decision(decision)
        
        # Remove from approval queue
        await self.human_approval.remove_from_queue(decision_id)
        
        return True
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Get current production metrics for dashboard."""
        base_metrics = self.metrics_manager.get_current_metrics()
        
        # Add recent decisions
        recent_decisions = [
            {
                'decision_id': d.decision_id,
                'timestamp': d.timestamp.isoformat(),
                'risk_score': d.risk_score,
                'risk_level': d.risk_level.value,
                'status': d.status.value,
                'proposed_action': d.proposed_action
            }
            for d in self.decisions_log[-10:]  # Last 10 decisions
        ]
        
        base_metrics['recent_decisions'] = recent_decisions
        base_metrics['risk_thresholds'] = {
            'auto_execute': self.risk_engine.thresholds.auto_execute,
            'human_approval': self.risk_engine.thresholds.human_approval,
            'block_threshold': self.risk_engine.thresholds.block_threshold
        }
        
        return base_metrics
    
    def generate_roi_report(self) -> Dict[str, Any]:
        """Generate comprehensive ROI report for stakeholders."""
        roi_report = self.metrics_manager.generate_roi_report()
        
        # Add execution statistics
        execution_stats = self.executor.get_execution_stats()
        
        # Convert to dictionary for JSON serialization
        return {
            'report_timestamp': roi_report.report_timestamp,
            'reporting_period': roi_report.reporting_period,
            'financial_impact': roi_report.financial_impact,
            'operational_metrics': roi_report.operational_metrics,
            'business_value': roi_report.business_value,
            'execution_statistics': execution_stats
        }
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of decisions pending human approval."""
        return self.human_approval.get_pending_approvals()
    
    async def tune_risk_thresholds(self, accuracy_data: List[Dict[str, Any]]):
        """Automatically tune risk thresholds based on accuracy data."""
        if len(accuracy_data) < 10:
            logger.info("📊 Insufficient data for threshold tuning")
            return
        
        # Simple threshold tuning logic
        false_positives = sum(1 for d in accuracy_data 
                            if d.get('predicted_risk') == 'high' and d.get('actual_outcome') == 'success')
        false_negatives = sum(1 for d in accuracy_data 
                            if d.get('predicted_risk') == 'low' and d.get('actual_outcome') == 'failure')
        
        current_thresholds = self.risk_engine.thresholds
        
        # Adjust thresholds based on error rates
        if false_positives > len(accuracy_data) * 0.1:  # Too many false positives
            current_thresholds.human_approval += 0.05
            current_thresholds.block_threshold += 0.05
            logger.info("📈 Increased risk thresholds due to false positives")
        
        if false_negatives > len(accuracy_data) * 0.05:  # Too many false negatives
            current_thresholds.auto_execute -= 0.05
            current_thresholds.human_approval -= 0.05
            logger.info("📉 Decreased risk thresholds due to false negatives")
        
        # Ensure thresholds stay within reasonable bounds
        current_thresholds.auto_execute = max(0.1, min(0.5, current_thresholds.auto_execute))
        current_thresholds.human_approval = max(0.4, min(0.9, current_thresholds.human_approval))
        current_thresholds.block_threshold = max(0.6, min(1.0, current_thresholds.block_threshold))
        
        self.risk_engine.update_thresholds(current_thresholds)
        logger.info(f"🎯 Updated risk thresholds: {current_thresholds}")
    
    def _find_decision(self, decision_id: str) -> Optional[ActiveModeDecision]:
        """Find decision by ID in local log."""
        for decision in self.decisions_log:
            if decision.decision_id == decision_id:
                return decision
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'risk_engine': 'healthy',
                'database': 'healthy',
                'metrics_manager': 'healthy',
                'executor': 'healthy',
                'human_approval': 'healthy'
            },
            'pending_approvals': len(self.get_pending_approvals()),
            'total_decisions': len(self.decisions_log),
            'last_decision': self.decisions_log[-1].timestamp.isoformat() if self.decisions_log else None
        }
