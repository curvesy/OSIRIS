"""
📋 Governance Schemas - Professional Data Models
Clean, focused data structures for governance operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for decision making."""
    LOW = "low"           # < 0.3: Auto-execute
    MEDIUM = "medium"     # 0.3-0.8: Human approval required
    HIGH = "high"         # > 0.8: Block and escalate


class ActionStatus(Enum):
    """Status of actions in active mode."""
    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class ActiveModeDecision:
    """Decision made in active mode with risk assessment."""
    decision_id: str
    timestamp: datetime
    evidence_log: List[Dict[str, Any]]
    proposed_action: str
    risk_score: float
    risk_level: RiskLevel
    reasoning: str
    status: ActionStatus
    human_reviewer: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    execution_time: Optional[datetime] = None


@dataclass
class ProductionMetrics:
    """Real-time production metrics."""
    total_decisions: int
    auto_executed: int
    human_approved: int
    blocked_actions: int
    cost_savings: float
    incidents_prevented: int
    average_response_time: float
    accuracy_rate: float


@dataclass
class RiskThresholds:
    """Configurable risk thresholds."""
    auto_execute: float = 0.3
    human_approval: float = 0.8
    block_threshold: float = 0.8


@dataclass
class ROIReport:
    """ROI report structure for stakeholders."""
    report_timestamp: str
    reporting_period: str
    financial_impact: Dict[str, float]
    operational_metrics: Dict[str, float]
    business_value: Dict[str, str]
