"""
⚖️ Risk Assessment Engine - Professional Risk Calculation
Modular, testable risk assessment for governance decisions.
"""

import logging
from typing import Dict, Any, List
from .schemas import RiskLevel, RiskThresholds

logger = logging.getLogger(__name__)


class RiskAssessmentEngine:
    """
    🎯 Professional Risk Assessment Engine
    
    Calculates risk scores using multiple factors:
    - Action type analysis
    - Evidence quality assessment  
    - Historical pattern matching
    - Context-aware adjustments
    """
    
    def __init__(self, thresholds: RiskThresholds = None):
        self.thresholds = thresholds or RiskThresholds()
        
        # Risk patterns (configurable)
        self.high_risk_actions = ['delete', 'remove', 'terminate', 'shutdown', 'drop']
        self.medium_risk_actions = ['modify', 'update', 'restart', 'deploy', 'scale']
        self.low_risk_actions = ['read', 'get', 'list', 'monitor', 'check']
        
        logger.info("⚖️ Risk Assessment Engine initialized")
    
    async def calculate_risk_score(self, evidence_log: List[Dict[str, Any]], 
                                 proposed_action: str) -> float:
        """
        Calculate comprehensive risk score.
        
        Args:
            evidence_log: Current evidence from workflow
            proposed_action: Action proposed by supervisor
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        risk_score = 0.0
        
        # 1. Action-based risk (40% weight)
        action_risk = self._assess_action_risk(proposed_action)
        risk_score += action_risk * 0.4
        
        # 2. Evidence quality risk (30% weight)
        evidence_risk = self._assess_evidence_risk(evidence_log)
        risk_score += evidence_risk * 0.3
        
        # 3. Context risk (20% weight)
        context_risk = self._assess_context_risk(evidence_log, proposed_action)
        risk_score += context_risk * 0.2
        
        # 4. Historical pattern risk (10% weight)
        pattern_risk = await self._assess_pattern_risk(evidence_log, proposed_action)
        risk_score += pattern_risk * 0.1
        
        # Normalize to 0-1 range
        final_score = min(max(risk_score, 0.0), 1.0)
        
        logger.debug(f"🎯 Risk calculated: {final_score:.3f} for action '{proposed_action}'")
        return final_score
    
    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score and thresholds."""
        if risk_score < self.thresholds.auto_execute:
            return RiskLevel.LOW
        elif risk_score < self.thresholds.human_approval:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _assess_action_risk(self, proposed_action: str) -> float:
        """Assess risk based on action type."""
        action_lower = proposed_action.lower()
        
        if any(risk_action in action_lower for risk_action in self.high_risk_actions):
            return 0.8
        elif any(risk_action in action_lower for risk_action in self.medium_risk_actions):
            return 0.5
        elif any(risk_action in action_lower for risk_action in self.low_risk_actions):
            return 0.1
        else:
            return 0.3  # Unknown action type
    
    def _assess_evidence_risk(self, evidence_log: List[Dict[str, Any]]) -> float:
        """Assess risk based on evidence quality and content."""
        if not evidence_log:
            return 0.9  # No evidence is high risk
        
        if len(evidence_log) < 2:
            return 0.6  # Insufficient evidence
        
        # Check for error indicators in evidence
        error_indicators = ['error', 'fail', 'critical', 'alert', 'warning']
        error_count = 0
        
        for evidence in evidence_log:
            evidence_str = str(evidence).lower()
            if any(indicator in evidence_str for indicator in error_indicators):
                error_count += 1
        
        # More errors = higher risk
        error_ratio = error_count / len(evidence_log)
        return min(error_ratio * 0.8, 0.7)
    
    def _assess_context_risk(self, evidence_log: List[Dict[str, Any]], 
                           proposed_action: str) -> float:
        """Assess contextual risk factors."""
        risk_factors = 0.0
        
        # Check for time-sensitive contexts
        for evidence in evidence_log:
            if isinstance(evidence, dict):
                # High severity increases risk
                severity = evidence.get('severity', '').lower()
                if severity in ['critical', 'high']:
                    risk_factors += 0.3
                elif severity in ['medium', 'warning']:
                    risk_factors += 0.1
                
                # User impact increases risk
                if 'user_impact' in evidence or 'affected_users' in evidence:
                    risk_factors += 0.2
        
        return min(risk_factors, 0.8)
    
    async def _assess_pattern_risk(self, evidence_log: List[Dict[str, Any]], 
                                 proposed_action: str) -> float:
        """
        Assess risk based on historical patterns.
        
        In production, this would query the knowledge graph
        for similar past scenarios and their outcomes.
        """
        # Simplified pattern assessment for demo
        # In production: query knowledge graph for similar patterns
        
        # For now, return baseline pattern risk
        return 0.2
    
    def update_thresholds(self, new_thresholds: RiskThresholds):
        """Update risk thresholds (for adaptive tuning)."""
        self.thresholds = new_thresholds
        logger.info(f"🎯 Risk thresholds updated: {self.thresholds}")
    
    def get_risk_explanation(self, evidence_log: List[Dict[str, Any]], 
                           proposed_action: str, risk_score: float) -> Dict[str, Any]:
        """
        Generate human-readable explanation of risk assessment.
        
        Returns:
            Dictionary with risk breakdown and reasoning
        """
        action_risk = self._assess_action_risk(proposed_action)
        evidence_risk = self._assess_evidence_risk(evidence_log)
        context_risk = self._assess_context_risk(evidence_log, proposed_action)
        
        return {
            'total_risk_score': risk_score,
            'risk_breakdown': {
                'action_risk': action_risk,
                'evidence_risk': evidence_risk,
                'context_risk': context_risk,
                'pattern_risk': 0.2  # Simplified for demo
            },
            'risk_factors': {
                'action_type': self._get_action_category(proposed_action),
                'evidence_quality': 'sufficient' if len(evidence_log) >= 2 else 'insufficient',
                'error_indicators': self._count_error_indicators(evidence_log),
                'severity_level': self._get_max_severity(evidence_log)
            },
            'recommendation': self._get_risk_recommendation(risk_score)
        }
    
    def _get_action_category(self, proposed_action: str) -> str:
        """Categorize action type for explanation."""
        action_lower = proposed_action.lower()
        
        if any(risk_action in action_lower for risk_action in self.high_risk_actions):
            return 'high_risk_action'
        elif any(risk_action in action_lower for risk_action in self.medium_risk_actions):
            return 'medium_risk_action'
        elif any(risk_action in action_lower for risk_action in self.low_risk_actions):
            return 'low_risk_action'
        else:
            return 'unknown_action'
    
    def _count_error_indicators(self, evidence_log: List[Dict[str, Any]]) -> int:
        """Count error indicators in evidence."""
        error_indicators = ['error', 'fail', 'critical', 'alert', 'warning']
        count = 0
        
        for evidence in evidence_log:
            evidence_str = str(evidence).lower()
            if any(indicator in evidence_str for indicator in error_indicators):
                count += 1
        
        return count
    
    def _get_max_severity(self, evidence_log: List[Dict[str, Any]]) -> str:
        """Get maximum severity level from evidence."""
        severity_levels = ['low', 'medium', 'high', 'critical']
        max_severity = 'low'
        
        for evidence in evidence_log:
            if isinstance(evidence, dict):
                severity = evidence.get('severity', '').lower()
                if severity in severity_levels:
                    if severity_levels.index(severity) > severity_levels.index(max_severity):
                        max_severity = severity
        
        return max_severity
    
    def _get_risk_recommendation(self, risk_score: float) -> str:
        """Get recommendation based on risk score."""
        risk_level = self.determine_risk_level(risk_score)
        
        if risk_level == RiskLevel.LOW:
            return 'Safe to auto-execute'
        elif risk_level == RiskLevel.MEDIUM:
            return 'Requires human approval'
        else:
            return 'Block and escalate - too risky'
