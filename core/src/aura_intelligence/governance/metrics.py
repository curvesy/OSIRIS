"""
📊 Governance Metrics Manager - Professional Metrics Tracking
Clean, focused metrics calculation and reporting.
"""

import time
import logging
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, List

from .schemas import ProductionMetrics, ROIReport, ActiveModeDecision

logger = logging.getLogger(__name__)


class MetricsManager:
    """
    📊 Professional Metrics Manager
    
    Handles all metrics calculation and reporting:
    - Real-time metrics tracking
    - ROI calculation
    - Performance monitoring
    - Business value reporting
    """
    
    def __init__(self):
        self.metrics = ProductionMetrics(
            total_decisions=0,
            auto_executed=0,
            human_approved=0,
            blocked_actions=0,
            cost_savings=0.0,
            incidents_prevented=0,
            average_response_time=0.0,
            accuracy_rate=0.0
        )
        
        self._response_times: List[float] = []
        logger.info("📊 Metrics Manager initialized")
    
    def record_decision(self, decision: ActiveModeDecision, response_time: float):
        """
        Record a new decision and update metrics.
        
        Args:
            decision: The decision that was processed
            response_time: Time taken to process the decision
        """
        self.metrics.total_decisions += 1
        
        # Update status counters
        if decision.status.value == 'executed':
            self.metrics.auto_executed += 1
        elif decision.status.value == 'approved':
            self.metrics.human_approved += 1
        elif decision.status.value == 'blocked':
            self.metrics.blocked_actions += 1
            self.metrics.incidents_prevented += 1
        
        # Update response time
        self._response_times.append(response_time)
        self.metrics.average_response_time = sum(self._response_times) / len(self._response_times)
        
        # Update cost savings if execution result available
        if decision.execution_result and 'cost_impact' in decision.execution_result:
            self.metrics.cost_savings += decision.execution_result['cost_impact']
        
        # Calculate accuracy rate
        successful_decisions = self.metrics.auto_executed + self.metrics.human_approved
        self.metrics.accuracy_rate = successful_decisions / self.metrics.total_decisions if self.metrics.total_decisions > 0 else 0.0
        
        logger.debug(f"📊 Metrics updated for decision {decision.decision_id}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.metrics),
            'performance': {
                'decisions_per_minute': self._calculate_decisions_per_minute(),
                'automation_rate': self._calculate_automation_rate(),
                'risk_prevention_rate': self._calculate_risk_prevention_rate()
            }
        }
    
    def generate_roi_report(self, reporting_period: str = "30_days") -> ROIReport:
        """
        Generate comprehensive ROI report.
        
        Args:
            reporting_period: Period for the report (e.g., "30_days", "weekly")
            
        Returns:
            ROIReport with financial and operational metrics
        """
        # Calculate financial impact
        direct_savings = self.metrics.cost_savings
        incident_prevention_savings = self.metrics.incidents_prevented * 50000  # $50K per incident
        total_roi = direct_savings + incident_prevention_savings
        
        financial_impact = {
            'direct_cost_savings': direct_savings,
            'incident_prevention_savings': incident_prevention_savings,
            'total_roi': total_roi,
            'roi_per_decision': total_roi / max(self.metrics.total_decisions, 1)
        }
        
        # Calculate operational metrics
        operational_metrics = {
            'total_decisions_processed': float(self.metrics.total_decisions),
            'automation_rate': self._calculate_automation_rate(),
            'human_intervention_rate': self._calculate_human_intervention_rate(),
            'risk_prevention_rate': self._calculate_risk_prevention_rate(),
            'average_response_time_ms': self.metrics.average_response_time * 1000,
            'system_accuracy': self.metrics.accuracy_rate * 100
        }
        
        # Calculate business value
        business_value = {
            'incidents_prevented': str(self.metrics.incidents_prevented),
            'operational_efficiency_gain': f"{self._calculate_automation_rate():.1f}%",
            'risk_reduction': f"{self._calculate_risk_prevention_rate():.1f}%",
            'human_productivity_gain': f"Reduced manual intervention by {100 - self._calculate_human_intervention_rate():.1f}%"
        }
        
        return ROIReport(
            report_timestamp=datetime.now().isoformat(),
            reporting_period=reporting_period,
            financial_impact=financial_impact,
            operational_metrics=operational_metrics,
            business_value=business_value
        )
    
    def _calculate_automation_rate(self) -> float:
        """Calculate percentage of decisions that were automated."""
        if self.metrics.total_decisions == 0:
            return 0.0
        return (self.metrics.auto_executed / self.metrics.total_decisions) * 100
    
    def _calculate_human_intervention_rate(self) -> float:
        """Calculate percentage of decisions requiring human intervention."""
        if self.metrics.total_decisions == 0:
            return 0.0
        return (self.metrics.human_approved / self.metrics.total_decisions) * 100
    
    def _calculate_risk_prevention_rate(self) -> float:
        """Calculate percentage of decisions that were blocked for risk."""
        if self.metrics.total_decisions == 0:
            return 0.0
        return (self.metrics.blocked_actions / self.metrics.total_decisions) * 100
    
    def _calculate_decisions_per_minute(self) -> float:
        """Calculate decisions processed per minute (simplified)."""
        # This would use actual time tracking in production
        return self.metrics.total_decisions / max(1, len(self._response_times) / 60)
    
    def reset_metrics(self):
        """Reset all metrics (for testing or new reporting periods)."""
        self.metrics = ProductionMetrics(
            total_decisions=0,
            auto_executed=0,
            human_approved=0,
            blocked_actions=0,
            cost_savings=0.0,
            incidents_prevented=0,
            average_response_time=0.0,
            accuracy_rate=0.0
        )
        self._response_times.clear()
        logger.info("📊 Metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for external systems."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.metrics),
            'calculated_kpis': {
                'automation_rate': self._calculate_automation_rate(),
                'human_intervention_rate': self._calculate_human_intervention_rate(),
                'risk_prevention_rate': self._calculate_risk_prevention_rate(),
                'decisions_per_minute': self._calculate_decisions_per_minute()
            }
        }
