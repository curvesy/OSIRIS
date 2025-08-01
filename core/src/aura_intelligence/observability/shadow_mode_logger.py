#!/usr/bin/env python3
"""
🌙 Shadow Mode Logger - Phase 3C Implementation

Logs predictions vs actual outcomes without blocking execution to populate 
training dataset and prove ROI. This is the data collection foundation for 
our eventual PredictiveDecisionEngine.

Based on the strategic plan from ksksksk.md:
- Log every (situation, action, prediction, outcome) tuple
- Track prediction accuracy vs actual results  
- Build training dataset for meta-learning
- Generate ROI metrics and governance dashboard data
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

# Handle optional dependencies gracefully
try:
    import aiofiles
    import aiosqlite
    HAS_ASYNC_DEPS = True
except ImportError:
    HAS_ASYNC_DEPS = False

logger = logging.getLogger(__name__)


@dataclass
class ShadowModeEntry:
    """Single shadow mode log entry for training data collection."""
    
    # Identifiers
    workflow_id: str
    thread_id: str
    timestamp: datetime
    
    # Situation Context
    evidence_log: List[Dict[str, Any]]
    memory_context: Dict[str, Any]
    supervisor_decision: Dict[str, Any]
    
    # Validator Prediction
    predicted_success_probability: float
    prediction_confidence_score: float
    risk_score: float
    predicted_risks: List[Dict[str, Any]]
    reasoning_trace: str
    requires_human_approval: bool
    
    # Routing Decision
    routing_decision: str  # "tools", "supervisor", "error_handler"
    decision_score: float  # success_probability * confidence
    
    # Actual Outcome (populated later)
    actual_outcome: Optional[str] = None  # "success", "failure", "partial"
    actual_execution_time: Optional[float] = None
    actual_error_details: Optional[Dict[str, Any]] = None
    outcome_timestamp: Optional[datetime] = None
    
    # Accuracy Metrics (calculated)
    prediction_accuracy: Optional[float] = None
    risk_assessment_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        if self.outcome_timestamp:
            data['outcome_timestamp'] = self.outcome_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShadowModeEntry':
        """Create from dictionary (for loading from storage)."""
        # Convert ISO strings back to datetime objects
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('outcome_timestamp'):
            data['outcome_timestamp'] = datetime.fromisoformat(data['outcome_timestamp'])
        return cls(**data)


class ShadowModeLogger:
    """
    🌙 Shadow Mode Logger for Phase 3C
    
    Logs all validator predictions and actual outcomes for:
    1. ROI validation and stakeholder reporting
    2. Training data collection for future neural models
    3. Accuracy tracking and threshold tuning
    4. Governance dashboard metrics
    """
    
    def __init__(self, 
                 db_path: str = "shadow_mode_logs.db",
                 json_backup_dir: str = "shadow_logs_backup"):
        self.db_path = db_path
        self.json_backup_dir = Path(json_backup_dir)
        self.json_backup_dir.mkdir(exist_ok=True)
        
        # Performance metrics
        self.entries_logged = 0
        self.outcomes_recorded = 0
        
    async def initialize(self):
        """Initialize the shadow mode logging system."""
        if not HAS_ASYNC_DEPS:
            logger.warning("⚠️ Async dependencies not available, using fallback mode")
            return
        await self._create_database_schema()
        logger.info("🌙 Shadow Mode Logger initialized")
    
    async def _create_database_schema(self):
        """Create SQLite database schema for shadow mode logs."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS shadow_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    
                    -- Prediction data
                    predicted_success_probability REAL,
                    prediction_confidence_score REAL,
                    risk_score REAL,
                    routing_decision TEXT,
                    decision_score REAL,
                    requires_human_approval BOOLEAN,
                    
                    -- Outcome data (populated later)
                    actual_outcome TEXT,
                    actual_execution_time REAL,
                    outcome_timestamp TEXT,
                    
                    -- Accuracy metrics
                    prediction_accuracy REAL,
                    risk_assessment_accuracy REAL,
                    
                    -- Full context (JSON)
                    full_context TEXT,
                    
                    -- Indexing
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_workflow_id ON shadow_logs(workflow_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON shadow_logs(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_routing_decision ON shadow_logs(routing_decision)")
            
            await db.commit()
    
    async def log_prediction(self, entry: ShadowModeEntry) -> str:
        """
        Log a validator prediction in shadow mode.

        Returns:
            str: Entry ID for later outcome correlation
        """
        if not HAS_ASYNC_DEPS:
            logger.debug("🌙 Shadow mode logging skipped (dependencies not available)")
            return "fallback_id"

        try:
            # Store in SQLite for fast queries
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    INSERT INTO shadow_logs (
                        workflow_id, thread_id, timestamp,
                        predicted_success_probability, prediction_confidence_score,
                        risk_score, routing_decision, decision_score,
                        requires_human_approval, full_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.workflow_id,
                    entry.thread_id, 
                    entry.timestamp.isoformat(),
                    entry.predicted_success_probability,
                    entry.prediction_confidence_score,
                    entry.risk_score,
                    entry.routing_decision,
                    entry.decision_score,
                    entry.requires_human_approval,
                    json.dumps(entry.to_dict())
                ))
                
                entry_id = cursor.lastrowid
                await db.commit()
            
            # Also backup to JSON for data science analysis
            backup_file = self.json_backup_dir / f"shadow_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
            async with aiofiles.open(backup_file, 'a') as f:
                await f.write(json.dumps(entry.to_dict()) + '\n')
            
            self.entries_logged += 1
            
            logger.debug(f"🌙 Logged shadow mode prediction: {entry.workflow_id} -> {entry.routing_decision}")
            return str(entry_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to log shadow mode prediction: {e}")
            raise
    
    async def record_outcome(self,
                           workflow_id: str,
                           actual_outcome: str,
                           execution_time: Optional[float] = None,
                           error_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record the actual outcome for a previously logged prediction.

        Args:
            workflow_id: The workflow ID to match
            actual_outcome: "success", "failure", "partial"
            execution_time: Actual execution time in seconds
            error_details: Error details if outcome was failure

        Returns:
            bool: True if outcome was successfully recorded
        """
        if not HAS_ASYNC_DEPS:
            logger.debug("🌙 Shadow mode outcome recording skipped (dependencies not available)")
            return True

        try:
            outcome_timestamp = datetime.now()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Update the most recent entry for this workflow
                await db.execute("""
                    UPDATE shadow_logs 
                    SET actual_outcome = ?,
                        actual_execution_time = ?,
                        outcome_timestamp = ?
                    WHERE workflow_id = ? 
                    AND actual_outcome IS NULL
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (
                    actual_outcome,
                    execution_time,
                    outcome_timestamp.isoformat(),
                    workflow_id
                ))
                
                await db.commit()
                
                # Calculate accuracy metrics
                await self._calculate_accuracy_metrics(workflow_id)
            
            self.outcomes_recorded += 1
            logger.debug(f"🌙 Recorded outcome for {workflow_id}: {actual_outcome}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record outcome: {e}")
            return False
    
    async def _calculate_accuracy_metrics(self, workflow_id: str):
        """Calculate prediction accuracy metrics for a completed entry."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get the entry we just updated
            async with db.execute("""
                SELECT predicted_success_probability, actual_outcome, risk_score
                FROM shadow_logs 
                WHERE workflow_id = ? AND actual_outcome IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (workflow_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    predicted_prob, actual_outcome, risk_score = row
                    
                    # Calculate prediction accuracy (1.0 if correct, 0.0 if wrong)
                    actual_success = 1.0 if actual_outcome == "success" else 0.0
                    prediction_accuracy = 1.0 - abs(predicted_prob - actual_success)
                    
                    # Calculate risk assessment accuracy
                    actual_risk = 1.0 - actual_success  # High risk if failed
                    risk_assessment_accuracy = 1.0 - abs(risk_score - actual_risk)
                    
                    # Update the entry with calculated metrics
                    await db.execute("""
                        UPDATE shadow_logs 
                        SET prediction_accuracy = ?,
                            risk_assessment_accuracy = ?
                        WHERE workflow_id = ? AND actual_outcome IS NOT NULL
                        ORDER BY timestamp DESC LIMIT 1
                    """, (prediction_accuracy, risk_assessment_accuracy, workflow_id))
                    
                    await db.commit()
    
    async def get_accuracy_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get accuracy metrics for the governance dashboard.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with accuracy metrics and ROI data
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Overall accuracy
            async with db.execute("""
                SELECT 
                    AVG(prediction_accuracy) as avg_prediction_accuracy,
                    AVG(risk_assessment_accuracy) as avg_risk_accuracy,
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN actual_outcome = 'success' THEN 1 ELSE 0 END) as successful_outcomes,
                    SUM(CASE WHEN requires_human_approval = 1 THEN 1 ELSE 0 END) as human_approvals_required
                FROM shadow_logs 
                WHERE timestamp >= ? AND actual_outcome IS NOT NULL
            """, (cutoff_date,)) as cursor:
                overall_stats = await cursor.fetchone()
            
            # Accuracy by routing decision
            routing_stats = []
            async with db.execute("""
                SELECT 
                    routing_decision,
                    AVG(prediction_accuracy) as accuracy,
                    COUNT(*) as count
                FROM shadow_logs 
                WHERE timestamp >= ? AND actual_outcome IS NOT NULL
                GROUP BY routing_decision
            """, (cutoff_date,)) as cursor:
                async for row in cursor:
                    routing_stats.append({
                        'routing_decision': row[0],
                        'accuracy': row[1],
                        'count': row[2]
                    })
        
        return {
            'period_days': days,
            'overall_prediction_accuracy': overall_stats[0] or 0.0,
            'overall_risk_accuracy': overall_stats[1] or 0.0,
            'total_predictions': overall_stats[2] or 0,
            'successful_outcomes': overall_stats[3] or 0,
            'human_approvals_required': overall_stats[4] or 0,
            'routing_accuracy': routing_stats,
            'entries_logged': self.entries_logged,
            'outcomes_recorded': self.outcomes_recorded
        }
