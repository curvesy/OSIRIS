"""
Shadow mode logging infrastructure for Phase 3C.

This module provides shadow mode capabilities for logging predictions
and outcomes to build training data for the PredictiveDecisionEngine.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

from ...observability.shadow_mode_logger import ShadowModeLogger, ShadowModeEntry
from .state import CollectiveState

logger = logging.getLogger(__name__)

# Global shadow logger instance
_shadow_logger: Optional[ShadowModeLogger] = None
_shadow_lock = asyncio.Lock()


async def get_shadow_logger() -> ShadowModeLogger:
    """
    Get or create the global shadow mode logger.
    
    Uses double-checked locking pattern for thread safety.
    
    Returns:
        ShadowModeLogger instance
    """
    global _shadow_logger
    
    if _shadow_logger is None:
        async with _shadow_lock:
            if _shadow_logger is None:
                _shadow_logger = ShadowModeLogger()
                await _shadow_logger.initialize()
                logger.info("Shadow mode logger initialized")
    
    return _shadow_logger


async def log_shadow_mode_prediction(
    state: CollectiveState,
    validation_result: Dict[str, Any],
    proposed_action: str
) -> None:
    """
    Log validator prediction in shadow mode for Phase 3C.
    
    This creates the training data foundation for our eventual PredictiveDecisionEngine.
    
    Args:
        state: Current workflow state
        validation_result: Result from risk validation
        proposed_action: Action that would be taken
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        # Calculate routing decision based on validation result
        risk_level = validation_result.get("risk_level", "unknown")
        confidence = validation_result.get("confidence", 0.0)
        
        # Determine predicted route
        predicted_route = _determine_route(risk_level, confidence)
        
        # Create shadow mode entry
        entry = ShadowModeEntry(
            workflow_id=state["workflow_id"],
            timestamp=datetime.now(timezone.utc),
            state_snapshot={
                "evidence_count": len(state.get("evidence_log", [])),
                "current_step": state.get("current_step", "unknown"),
                "risk_assessment": state.get("risk_assessment", {}),
                "system_health": state.get("system_health", {}).to_dict() if hasattr(state.get("system_health", {}), 'to_dict') else state.get("system_health", {})
            },
            validator_output=validation_result,
            predicted_route=predicted_route,
            confidence_score=confidence,
            metadata={
                "proposed_action": proposed_action,
                "thread_id": state.get("thread_id", ""),
                "shadow_mode_version": "3C"
            }
        )
        
        # Log the entry
        await shadow_logger.log_prediction(entry)
        
        logger.debug(
            f"Shadow mode prediction logged: workflow={state['workflow_id']}, "
            f"route={predicted_route}, confidence={confidence:.2f}"
        )
        
    except Exception as e:
        # Shadow mode failures should not affect main workflow
        logger.error(f"Failed to log shadow mode prediction: {e}", exc_info=True)


async def record_shadow_mode_outcome(
    workflow_id: str,
    outcome: str,
    execution_time: float,
    error_details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record the actual outcome of a workflow execution.
    
    Args:
        workflow_id: Workflow identifier
        outcome: Execution outcome ("success" or "failure")
        execution_time: Time taken to execute
        error_details: Optional error information
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        await shadow_logger.record_outcome(
            workflow_id=workflow_id,
            actual_route=outcome,
            execution_time_ms=execution_time * 1000,  # Convert to milliseconds
            error_details=error_details
        )
        
        logger.debug(
            f"Shadow mode outcome recorded: workflow={workflow_id}, "
            f"outcome={outcome}, time={execution_time:.2f}s"
        )
        
    except Exception as e:
        logger.error(f"Failed to record shadow mode outcome: {e}", exc_info=True)


def _determine_route(risk_level: str, confidence: float) -> str:
    """
    Determine the routing decision based on risk and confidence.
    
    Args:
        risk_level: Risk assessment level
        confidence: Confidence score
        
    Returns:
        Predicted route string
    """
    # High confidence critical risks require immediate action
    if risk_level == "critical" and confidence > 0.8:
        return "execute_immediate"
    
    # High risks with good confidence need remediation
    elif risk_level == "high" and confidence > 0.6:
        return "execute_remediation"
    
    # Medium risks require more analysis
    elif risk_level == "medium":
        return "analyze_further"
    
    # Low risks or low confidence continue observation
    else:
        return "continue_observation"


async def get_shadow_mode_metrics() -> Dict[str, Any]:
    """
    Get current shadow mode metrics and statistics.
    
    Returns:
        Dictionary of shadow mode metrics
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        # Get basic metrics
        metrics = await shadow_logger.get_metrics()
        
        # Add prediction accuracy if available
        if metrics.get("total_predictions", 0) > 0:
            accuracy = metrics.get("correct_predictions", 0) / metrics["total_predictions"]
            metrics["prediction_accuracy"] = accuracy
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get shadow mode metrics: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "unavailable"
        }


async def export_shadow_mode_data(
    output_path: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> bool:
    """
    Export shadow mode data for training or analysis.
    
    Args:
        output_path: Path to export data
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        True if export successful
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        # Export data within date range
        data = await shadow_logger.export_training_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            import json
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Shadow mode data exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export shadow mode data: {e}", exc_info=True)
        return False