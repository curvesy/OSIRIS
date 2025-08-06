#!/usr/bin/env python3
"""
🎼 Workflow Checkpoint Manager - State Persistence

Professional checkpoint management for LangGraph workflows.
Handles state persistence, recovery, and workflow continuity.
"""

import logging
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowCheckpointManager:
    """
    Professional checkpoint manager for workflow state persistence.
    
    Provides:
    1. Workflow state persistence
    2. Recovery from failures
    3. Workflow history tracking
    4. Performance analytics
    5. Audit trail maintenance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("db_path", "collective_checkpoints.db")
        self.retention_days = config.get("retention_days", 30)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"🎼 Checkpoint manager initialized: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize the checkpoint database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create checkpoints table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        node_name TEXT NOT NULL,
                        state_data TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(workflow_id, checkpoint_id)
                    )
                """)
                
                # Create workflow summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_summary (
                        workflow_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        total_nodes INTEGER DEFAULT 0,
                        evidence_count INTEGER DEFAULT 0,
                        final_risk_score REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        node_name TEXT NOT NULL,
                        execution_time_ms INTEGER,
                        memory_usage_mb REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_id ON workflow_checkpoints(workflow_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON workflow_checkpoints(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_status ON workflow_summary(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_workflow ON workflow_metrics(workflow_id)")
                
                conn.commit()
                logger.info("✅ Checkpoint database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def save_checkpoint(self, workflow_id: str, checkpoint_id: str, 
                            node_name: str, state_data: Any, 
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a workflow checkpoint."""
        
        try:
            # Serialize state data
            serialized_state = json.dumps(self._serialize_state(state_data))
            serialized_metadata = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_checkpoints 
                    (workflow_id, checkpoint_id, node_name, state_data, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (workflow_id, checkpoint_id, node_name, serialized_state, serialized_metadata))
                
                conn.commit()
                
            logger.info(f"✅ Checkpoint saved: {workflow_id}:{checkpoint_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(self, workflow_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific workflow checkpoint."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT node_name, state_data, metadata, created_at
                    FROM workflow_checkpoints
                    WHERE workflow_id = ? AND checkpoint_id = ?
                """, (workflow_id, checkpoint_id))
                
                result = cursor.fetchone()
                
                if result:
                    node_name, state_data, metadata, created_at = result
                    
                    return {
                        "workflow_id": workflow_id,
                        "checkpoint_id": checkpoint_id,
                        "node_name": node_name,
                        "state_data": json.loads(state_data),
                        "metadata": json.loads(metadata) if metadata else None,
                        "created_at": created_at
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return None
    
    async def get_workflow_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all checkpoints for a workflow."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT checkpoint_id, node_name, created_at
                    FROM workflow_checkpoints
                    WHERE workflow_id = ?
                    ORDER BY created_at ASC
                """, (workflow_id,))
                
                results = cursor.fetchall()
                
                checkpoints = []
                for checkpoint_id, node_name, created_at in results:
                    checkpoints.append({
                        "checkpoint_id": checkpoint_id,
                        "node_name": node_name,
                        "created_at": created_at
                    })
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"❌ Failed to get workflow checkpoints: {e}")
            return []
    
    async def save_workflow_summary(self, workflow_id: str, summary: Dict[str, Any]) -> None:
        """Save workflow summary information."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_summary
                    (workflow_id, status, start_time, end_time, total_nodes, 
                     evidence_count, final_risk_score, success, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow_id,
                    summary.get("status", "unknown"),
                    summary.get("start_time"),
                    summary.get("end_time"),
                    summary.get("total_nodes", 0),
                    summary.get("evidence_count", 0),
                    summary.get("final_risk_score"),
                    summary.get("success", False),
                    summary.get("error_message"),
                    json.dumps(summary.get("metadata", {}))
                ))
                
                conn.commit()
                
            logger.info(f"✅ Workflow summary saved: {workflow_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save workflow summary: {e}")
    
    async def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow summary."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT status, start_time, end_time, total_nodes, evidence_count,
                           final_risk_score, success, error_message, metadata, 
                           created_at, updated_at
                    FROM workflow_summary
                    WHERE workflow_id = ?
                """, (workflow_id,))
                
                result = cursor.fetchone()
                
                if result:
                    (status, start_time, end_time, total_nodes, evidence_count,
                     final_risk_score, success, error_message, metadata,
                     created_at, updated_at) = result
                    
                    return {
                        "workflow_id": workflow_id,
                        "status": status,
                        "start_time": start_time,
                        "end_time": end_time,
                        "total_nodes": total_nodes,
                        "evidence_count": evidence_count,
                        "final_risk_score": final_risk_score,
                        "success": bool(success),
                        "error_message": error_message,
                        "metadata": json.loads(metadata) if metadata else {},
                        "created_at": created_at,
                        "updated_at": updated_at
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get workflow summary: {e}")
            return None
    
    async def record_node_metrics(self, workflow_id: str, node_name: str,
                                execution_time_ms: int, memory_usage_mb: float,
                                success: bool, error_message: Optional[str] = None) -> None:
        """Record performance metrics for a node execution."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO workflow_metrics
                    (workflow_id, node_name, execution_time_ms, memory_usage_mb, 
                     success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (workflow_id, node_name, execution_time_ms, memory_usage_mb,
                      success, error_message))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to record node metrics: {e}")
    
    async def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance analytics for recent workflows."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get workflow statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_workflows,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_workflows,
                        AVG(evidence_count) as avg_evidence_count,
                        AVG(final_risk_score) as avg_risk_score
                    FROM workflow_summary
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days))
                
                workflow_stats = cursor.fetchone()
                
                # Get node performance statistics
                cursor.execute("""
                    SELECT 
                        node_name,
                        COUNT(*) as executions,
                        AVG(execution_time_ms) as avg_execution_time,
                        AVG(memory_usage_mb) as avg_memory_usage,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions
                    FROM workflow_metrics
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY node_name
                """.format(days))
                
                node_stats = cursor.fetchall()
                
                # Format results
                analytics = {
                    "period_days": days,
                    "workflow_statistics": {
                        "total_workflows": workflow_stats[0] or 0,
                        "successful_workflows": workflow_stats[1] or 0,
                        "success_rate": (workflow_stats[1] or 0) / max(workflow_stats[0] or 1, 1),
                        "avg_evidence_count": workflow_stats[2] or 0,
                        "avg_risk_score": workflow_stats[3] or 0
                    },
                    "node_performance": []
                }
                
                for node_name, executions, avg_time, avg_memory, successful in node_stats:
                    analytics["node_performance"].append({
                        "node_name": node_name,
                        "executions": executions,
                        "avg_execution_time_ms": avg_time or 0,
                        "avg_memory_usage_mb": avg_memory or 0,
                        "success_rate": successful / max(executions, 1)
                    })
                
                return analytics
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance analytics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_checkpoints(self) -> int:
        """Clean up old checkpoints based on retention policy."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old checkpoints
                cursor.execute("""
                    DELETE FROM workflow_checkpoints
                    WHERE created_at < datetime('now', '-{} days')
                """.format(self.retention_days))
                
                deleted_checkpoints = cursor.rowcount
                
                # Delete old workflow summaries
                cursor.execute("""
                    DELETE FROM workflow_summary
                    WHERE created_at < datetime('now', '-{} days')
                """.format(self.retention_days))
                
                deleted_summaries = cursor.rowcount
                
                # Delete old metrics
                cursor.execute("""
                    DELETE FROM workflow_metrics
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(self.retention_days))
                
                deleted_metrics = cursor.rowcount
                
                conn.commit()
                
                total_deleted = deleted_checkpoints + deleted_summaries + deleted_metrics
                
                if total_deleted > 0:
                    logger.info(f"🧹 Cleaned up {total_deleted} old records")
                
                return total_deleted
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 0
    
    def _serialize_state(self, state_data: Any) -> Dict[str, Any]:
        """Serialize state data for storage."""
        
        try:
            # Handle different state types
            if hasattr(state_data, '__dict__'):
                # Convert object to dictionary
                serialized = {}
                for key, value in state_data.__dict__.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        serialized[key] = value
                    elif isinstance(value, datetime):
                        serialized[key] = value.isoformat()
                    elif isinstance(value, list):
                        serialized[key] = [self._serialize_item(item) for item in value]
                    elif isinstance(value, dict):
                        serialized[key] = value
                    else:
                        serialized[key] = str(value)
                
                return serialized
            
            elif isinstance(state_data, dict):
                return state_data
            
            else:
                return {"serialized_state": str(state_data)}
                
        except Exception as e:
            logger.error(f"State serialization failed: {e}")
            return {"error": str(e)}
    
    def _serialize_item(self, item: Any) -> Any:
        """Serialize individual items."""
        
        if isinstance(item, (str, int, float, bool, type(None))):
            return item
        elif isinstance(item, datetime):
            return item.isoformat()
        elif hasattr(item, '__dict__'):
            return self._serialize_state(item)
        else:
            return str(item)


# Alias for backward compatibility
CheckpointManager = WorkflowCheckpointManager
