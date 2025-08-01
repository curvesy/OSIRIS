"""
🤔 Human Approval Manager - Professional Human-in-the-Loop
Clean management of human approval workflows.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque

from ..schemas import ActiveModeDecision

logger = logging.getLogger(__name__)


class HumanApprovalManager:
    """
    🤔 Professional Human Approval Manager
    
    Manages human-in-the-loop workflows:
    - Approval queue management
    - Notification handling
    - Timeout management
    - Escalation procedures
    """
    
    def __init__(self, approval_timeout_minutes: int = 30):
        self.approval_queue: deque = deque()
        self.approval_timeout = timedelta(minutes=approval_timeout_minutes)
        self.notification_handlers: List[callable] = []
        
        logger.info(f"🤔 Human Approval Manager initialized (timeout: {approval_timeout_minutes}min)")
    
    async def queue_for_approval(self, decision: ActiveModeDecision):
        """
        Queue a decision for human approval.
        
        Args:
            decision: The decision requiring human approval
        """
        approval_item = {
            'decision': decision,
            'queued_at': datetime.now(),
            'notified': False,
            'escalated': False
        }
        
        self.approval_queue.append(approval_item)
        
        # Send notification
        await self._notify_reviewers(decision)
        approval_item['notified'] = True
        
        logger.info(f"📋 Decision queued for approval: {decision.decision_id}")
    
    async def remove_from_queue(self, decision_id: str) -> bool:
        """
        Remove a decision from the approval queue.
        
        Args:
            decision_id: ID of the decision to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, item in enumerate(self.approval_queue):
            if item['decision'].decision_id == decision_id:
                del self.approval_queue[i]
                logger.info(f"📋 Decision removed from queue: {decision_id}")
                return True
        
        logger.warning(f"⚠️ Decision not found in queue: {decision_id}")
        return False
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """
        Get list of decisions pending approval.
        
        Returns:
            List of pending approval items with metadata
        """
        pending = []
        
        for item in self.approval_queue:
            decision = item['decision']
            queued_at = item['queued_at']
            time_waiting = datetime.now() - queued_at
            
            pending.append({
                'decision_id': decision.decision_id,
                'proposed_action': decision.proposed_action,
                'risk_score': decision.risk_score,
                'reasoning': decision.reasoning,
                'queued_at': queued_at.isoformat(),
                'time_waiting_minutes': int(time_waiting.total_seconds() / 60),
                'requires_escalation': time_waiting > self.approval_timeout,
                'evidence_summary': self._summarize_evidence(decision.evidence_log)
            })
        
        return pending
    
    async def check_timeouts(self):
        """
        Check for timed-out approvals and handle escalation.
        
        This should be called periodically by a background task.
        """
        current_time = datetime.now()
        escalated_count = 0
        
        for item in self.approval_queue:
            if not item['escalated']:
                time_waiting = current_time - item['queued_at']
                
                if time_waiting > self.approval_timeout:
                    await self._escalate_approval(item)
                    item['escalated'] = True
                    escalated_count += 1
        
        if escalated_count > 0:
            logger.warning(f"⏰ Escalated {escalated_count} timed-out approvals")
    
    def add_notification_handler(self, handler: callable):
        """
        Add a notification handler for approval requests.
        
        Args:
            handler: Async function that takes (decision, urgency_level)
        """
        self.notification_handlers.append(handler)
        logger.info(f"📧 Added notification handler: {handler.__name__}")
    
    async def _notify_reviewers(self, decision: ActiveModeDecision):
        """Send notifications to human reviewers."""
        urgency_level = self._determine_urgency(decision)
        
        for handler in self.notification_handlers:
            try:
                await handler(decision, urgency_level)
            except Exception as e:
                logger.error(f"❌ Notification handler failed: {e}")
        
        logger.info(f"📧 Notifications sent for decision: {decision.decision_id}")
    
    async def _escalate_approval(self, approval_item: Dict[str, Any]):
        """Escalate a timed-out approval to higher authority."""
        decision = approval_item['decision']
        
        # In production, this would:
        # - Notify senior reviewers
        # - Create high-priority tickets
        # - Send alerts to management
        # - Potentially auto-reject based on policy
        
        logger.warning(f"🚨 Escalating timed-out approval: {decision.decision_id}")
        
        # For demo, we'll just log the escalation
        escalation_data = {
            'decision_id': decision.decision_id,
            'proposed_action': decision.proposed_action,
            'risk_score': decision.risk_score,
            'time_waiting': datetime.now() - approval_item['queued_at'],
            'escalation_reason': 'approval_timeout'
        }
        
        logger.warning(f"🚨 ESCALATION: {escalation_data}")
    
    def _determine_urgency(self, decision: ActiveModeDecision) -> str:
        """Determine urgency level for notifications."""
        if decision.risk_score > 0.7:
            return 'high'
        elif decision.risk_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _summarize_evidence(self, evidence_log: List[Dict[str, Any]]) -> str:
        """Create a human-readable summary of evidence."""
        if not evidence_log:
            return "No evidence provided"
        
        evidence_types = [e.get('type', 'unknown') for e in evidence_log]
        unique_types = list(set(evidence_types))
        
        summary_parts = []
        
        # Count evidence by type
        for evidence_type in unique_types:
            count = evidence_types.count(evidence_type)
            if count == 1:
                summary_parts.append(evidence_type)
            else:
                summary_parts.append(f"{evidence_type} ({count})")
        
        return ", ".join(summary_parts)
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval queue statistics."""
        if not self.approval_queue:
            return {
                'pending_count': 0,
                'average_wait_time_minutes': 0,
                'escalated_count': 0,
                'oldest_request_age_minutes': 0
            }
        
        current_time = datetime.now()
        wait_times = []
        escalated_count = 0
        
        for item in self.approval_queue:
            wait_time = current_time - item['queued_at']
            wait_times.append(wait_time.total_seconds() / 60)
            
            if item['escalated']:
                escalated_count += 1
        
        return {
            'pending_count': len(self.approval_queue),
            'average_wait_time_minutes': sum(wait_times) / len(wait_times),
            'escalated_count': escalated_count,
            'oldest_request_age_minutes': max(wait_times)
        }
    
    def clear_queue(self):
        """Clear the approval queue (for testing or maintenance)."""
        cleared_count = len(self.approval_queue)
        self.approval_queue.clear()
        logger.info(f"🧹 Cleared {cleared_count} items from approval queue")
