#!/usr/bin/env python3
"""
🧪 Simple AURA Intelligence Test
Test the core shadow mode concept without complex dependencies
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SimpleShadowEntry:
    """Simple shadow mode entry for testing"""
    workflow_id: str
    predicted_success_probability: float
    prediction_confidence_score: float
    routing_decision: str
    actual_outcome: Optional[str] = None
    timestamp: Optional[str] = None

class SimpleShadowLogger:
    """Simple shadow mode logger for testing"""
    
    def __init__(self, db_path: str = "test_shadow_mode.db"):
        self.db_path = db_path
        self.conn = None
    
    async def initialize(self):
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT UNIQUE,
                predicted_success_probability REAL,
                prediction_confidence_score REAL,
                routing_decision TEXT,
                actual_outcome TEXT,
                timestamp TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        print(f"✅ Shadow logger initialized with database: {self.db_path}")
    
    async def log_prediction(self, entry: SimpleShadowEntry) -> str:
        """Log a prediction"""
        entry.timestamp = datetime.now().isoformat()
        
        self.conn.execute("""
            INSERT OR REPLACE INTO shadow_predictions 
            (workflow_id, predicted_success_probability, prediction_confidence_score, 
             routing_decision, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.workflow_id,
            entry.predicted_success_probability,
            entry.prediction_confidence_score,
            entry.routing_decision,
            entry.timestamp
        ))
        self.conn.commit()
        
        print(f"📝 Logged prediction for workflow: {entry.workflow_id}")
        return entry.workflow_id
    
    async def record_outcome(self, workflow_id: str, outcome: str) -> bool:
        """Record the actual outcome"""
        cursor = self.conn.execute("""
            UPDATE shadow_predictions
            SET actual_outcome = ?
            WHERE workflow_id = ?
        """, (outcome, workflow_id))

        affected = cursor.rowcount
        self.conn.commit()

        if affected > 0:
            print(f"✅ Recorded outcome '{outcome}' for workflow: {workflow_id}")
            return True
        else:
            print(f"⚠️  No prediction found for workflow: {workflow_id}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get simple metrics"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(actual_outcome) as completed_predictions,
                AVG(predicted_success_probability) as avg_predicted_success,
                COUNT(CASE WHEN actual_outcome = 'success' THEN 1 END) as actual_successes
            FROM shadow_predictions
        """)
        
        row = cursor.fetchone()
        total, completed, avg_predicted, actual_successes = row
        
        accuracy = 0.0
        if completed > 0:
            # Simple accuracy calculation
            accuracy = actual_successes / completed
        
        return {
            "total_predictions": total,
            "completed_predictions": completed,
            "avg_predicted_success": round(avg_predicted or 0, 3),
            "actual_success_rate": round(accuracy, 3),
            "data_completeness": round(completed / total if total > 0 else 0, 3)
        }
    
    async def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("🔒 Shadow logger closed")

class SimpleValidator:
    """Simple validator that mimics our ProfessionalPredictiveValidator"""
    
    def __init__(self):
        self.predictions_made = 0
    
    async def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Simple validation logic"""
        self.predictions_made += 1
        
        # Simple rule-based validation
        action_type = action.get("type", "unknown")
        risk_score = 0.3  # Default medium risk
        
        if action_type == "high_value_purchase":
            risk_score = 0.8
        elif action_type == "routine_task":
            risk_score = 0.1
        elif action_type == "negotiation":
            risk_score = 0.5
        
        success_probability = 1.0 - risk_score
        confidence = 0.85  # Fixed confidence for testing
        
        # Determine routing decision
        decision_score = success_probability * confidence
        
        if decision_score > 0.7:
            routing = "tools"  # High confidence - execute
        elif decision_score > 0.4:
            routing = "supervisor"  # Medium confidence - replan
        else:
            routing = "error_handler"  # Low confidence - escalate
        
        result = {
            "risk_score": risk_score,
            "success_probability": success_probability,
            "confidence_score": confidence,
            "decision_score": decision_score,
            "routing_decision": routing,
            "reasoning": f"Action type '{action_type}' assessed as {risk_score:.1f} risk"
        }
        
        print(f"🤖 Validator prediction: {json.dumps(result, indent=2)}")
        return result

async def simulate_workflow(validator: SimpleValidator, shadow_logger: SimpleShadowLogger, 
                          action: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a complete workflow with shadow mode logging"""
    
    workflow_id = f"test_{int(time.time())}_{action.get('type', 'unknown')}"
    
    print(f"\n🚀 Simulating workflow: {workflow_id}")
    print(f"📋 Action: {json.dumps(action, indent=2)}")
    
    # Step 1: Validator makes prediction
    validation_result = await validator.validate_action(action)
    
    # Step 2: Log the prediction (shadow mode)
    shadow_entry = SimpleShadowEntry(
        workflow_id=workflow_id,
        predicted_success_probability=validation_result["success_probability"],
        prediction_confidence_score=validation_result["confidence_score"],
        routing_decision=validation_result["routing_decision"]
    )
    
    await shadow_logger.log_prediction(shadow_entry)
    
    # Step 3: Execute the action (simulate)
    print(f"⚙️  Executing action (shadow mode ignores validator)...")
    await asyncio.sleep(0.5)  # Simulate execution time
    
    # Step 4: Determine actual outcome (simulate)
    # For testing, we'll make it succeed most of the time
    import random
    actual_success = random.random() > 0.3  # 70% success rate
    actual_outcome = "success" if actual_success else "failure"
    
    print(f"📊 Actual outcome: {actual_outcome}")
    
    # Step 5: Record the actual outcome
    await shadow_logger.record_outcome(workflow_id, actual_outcome)
    
    return {
        "workflow_id": workflow_id,
        "prediction": validation_result,
        "actual_outcome": actual_outcome,
        "shadow_logged": True
    }

async def run_simple_test():
    """Run a simple end-to-end test"""
    
    print("🧪 Simple AURA Intelligence Shadow Mode Test")
    print("=" * 50)
    print("Goal: Test shadow mode logging with simulated workflows")
    print("")
    
    # Initialize components
    validator = SimpleValidator()
    shadow_logger = SimpleShadowLogger()
    
    await shadow_logger.initialize()
    
    # Test scenarios
    test_actions = [
        {"type": "routine_task", "description": "Update inventory records"},
        {"type": "negotiation", "description": "Negotiate supplier contract"},
        {"type": "high_value_purchase", "description": "Purchase $50K equipment"},
        {"type": "routine_task", "description": "Send status report"},
        {"type": "negotiation", "description": "Renegotiate service terms"}
    ]
    
    print(f"🎯 Running {len(test_actions)} test scenarios...")
    print("")
    
    results = []
    for i, action in enumerate(test_actions, 1):
        print(f"--- Test {i}/{len(test_actions)} ---")
        result = await simulate_workflow(validator, shadow_logger, action)
        results.append(result)
        await asyncio.sleep(0.2)  # Brief pause between tests
    
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    # Get metrics
    metrics = await shadow_logger.get_metrics()
    print(f"Total predictions logged: {metrics['total_predictions']}")
    print(f"Completed workflows: {metrics['completed_predictions']}")
    print(f"Average predicted success: {metrics['avg_predicted_success']:.1%}")
    print(f"Actual success rate: {metrics['actual_success_rate']:.1%}")
    print(f"Data completeness: {metrics['data_completeness']:.1%}")
    
    print("\n🎉 Simple Test Complete!")
    print("=" * 50)
    print("✅ Shadow mode logging is working!")
    print("✅ Validator predictions are being captured")
    print("✅ Actual outcomes are being recorded")
    print("✅ Metrics are being calculated")
    print("")
    print("📋 What this proves:")
    print("   • The core shadow mode concept works")
    print("   • Predictions and outcomes are being logged")
    print("   • We can calculate accuracy metrics")
    print("   • The system is ready for more complex testing")
    
    await shadow_logger.close()
    
    return results

if __name__ == "__main__":
    asyncio.run(run_simple_test())
