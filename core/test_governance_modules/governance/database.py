"""
🗄️ Governance Database Manager - Professional Data Persistence
Clean, focused database operations for governance data.
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from .schemas import ActiveModeDecision, ProductionMetrics

logger = logging.getLogger(__name__)


class GovernanceDatabase:
    """
    🗄️ Professional Database Manager for Governance
    
    Handles all database operations for active mode governance:
    - Decision logging and retrieval
    - Metrics persistence
    - Query optimization
    - Data integrity
    """
    
    def __init__(self, db_path: str = "governance.db"):
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"🗄️ Governance database initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_decisions (
                decision_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                evidence_log TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                status TEXT NOT NULL,
                human_reviewer TEXT,
                execution_result TEXT,
                execution_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                timestamp TEXT PRIMARY KEY,
                total_decisions INTEGER,
                auto_executed INTEGER,
                human_approved INTEGER,
                blocked_actions INTEGER,
                cost_savings REAL,
                incidents_prevented INTEGER,
                average_response_time REAL,
                accuracy_rate REAL
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON active_decisions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_status ON active_decisions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_risk_level ON active_decisions(risk_level)")
        
        conn.commit()
        conn.close()
    
    def store_decision(self, decision: ActiveModeDecision) -> bool:
        """
        Store decision in database.
        
        Args:
            decision: ActiveModeDecision to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO active_decisions 
                (decision_id, timestamp, evidence_log, proposed_action, risk_score, 
                 risk_level, reasoning, status, human_reviewer, execution_result, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id,
                decision.timestamp.isoformat(),
                json.dumps(decision.evidence_log),
                decision.proposed_action,
                decision.risk_score,
                decision.risk_level.value,
                decision.reasoning,
                decision.status.value,
                decision.human_reviewer,
                json.dumps(decision.execution_result) if decision.execution_result else None,
                decision.execution_time.isoformat() if decision.execution_time else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"💾 Decision stored: {decision.decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store decision {decision.decision_id}: {e}")
            return False
    
    def get_decision(self, decision_id: str) -> Optional[ActiveModeDecision]:
        """
        Retrieve decision by ID.
        
        Args:
            decision_id: ID of decision to retrieve
            
        Returns:
            ActiveModeDecision if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM active_decisions WHERE decision_id = ?
            """, (decision_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_decision(row)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve decision {decision_id}: {e}")
            return None
    
    def get_recent_decisions(self, limit: int = 10) -> List[ActiveModeDecision]:
        """
        Get recent decisions for dashboard.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent ActiveModeDecisions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM active_decisions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_decision(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve recent decisions: {e}")
            return []
    
    def store_metrics_snapshot(self, metrics: ProductionMetrics) -> bool:
        """
        Store metrics snapshot.
        
        Args:
            metrics: ProductionMetrics to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO metrics_snapshots
                (timestamp, total_decisions, auto_executed, human_approved,
                 blocked_actions, cost_savings, incidents_prevented,
                 average_response_time, accuracy_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metrics.total_decisions,
                metrics.auto_executed,
                metrics.human_approved,
                metrics.blocked_actions,
                metrics.cost_savings,
                metrics.incidents_prevented,
                metrics.average_response_time,
                metrics.accuracy_rate
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug("📊 Metrics snapshot stored")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store metrics snapshot: {e}")
            return False
    
    def get_decision_stats(self) -> dict:
        """Get decision statistics for reporting."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'executed' THEN 1 ELSE 0 END) as executed,
                    SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) as blocked,
                    AVG(risk_score) as avg_risk_score
                FROM active_decisions
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            return {
                'total_decisions': row[0] or 0,
                'executed_decisions': row[1] or 0,
                'blocked_decisions': row[2] or 0,
                'average_risk_score': row[3] or 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get decision stats: {e}")
            return {}
    
    def _row_to_decision(self, row) -> ActiveModeDecision:
        """Convert database row to ActiveModeDecision."""
        from .schemas import RiskLevel, ActionStatus
        
        return ActiveModeDecision(
            decision_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            evidence_log=json.loads(row[2]),
            proposed_action=row[3],
            risk_score=row[4],
            risk_level=RiskLevel(row[5]),
            reasoning=row[6],
            status=ActionStatus(row[7]),
            human_reviewer=row[8],
            execution_result=json.loads(row[9]) if row[9] else None,
            execution_time=datetime.fromisoformat(row[10]) if row[10] else None
        )
