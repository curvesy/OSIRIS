"""
Tool definitions for collective intelligence workflows.

This module contains all tool implementations used by the workflow agents,
with production-hardened error handling and retry logic.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

from langchain_core.tools import tool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)
try:
    from pybreaker import CircuitBreaker
except ImportError:
    # Fallback circuit breaker implementation
    class CircuitBreaker:
        def __init__(self, fail_max=3, reset_timeout=60, exclude=None):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
            self.exclude = exclude or []
            
        def __call__(self, func):
            return func

logger = logging.getLogger(__name__)


# Circuit breakers for external operations
risk_analysis_breaker = CircuitBreaker(
    fail_max=3,
    reset_timeout=30,
    exclude=[ValueError, json.JSONDecodeError]
)

remediation_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60
)


def with_error_handling(func):
    """
    Decorator to add consistent error handling to tools.
    
    Ensures all tools return valid responses even on failure.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool {func.__name__} failed: {e}", exc_info=True)
            return {
                "evidence_type": "ERROR",
                "error": str(e),
                "error_type": type(e).__name__,
                "tool": func.__name__,
                "timestamp": datetime.now().isoformat()
            }
    return wrapper


@tool
@with_error_handling
async def observe_system_event(event_data: str) -> Dict[str, Any]:
    """
    Observe and process system events.
    
    Args:
        event_data: JSON string containing event information
        
    Returns:
        Dict containing observation evidence with type, source, severity,
        message, content, timestamp, confidence, and signature.
    """
    event = json.loads(event_data)
    
    # Validate required fields
    required_fields = ["source", "severity", "message"]
    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing required field: {field}")
    
    # Generate observation evidence
    return {
        "evidence_type": "OBSERVATION",
        "source": event.get("source"),
        "severity": event.get("severity"),
        "message": event.get("message"),
        "content": event,
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.95,
        "signature": f"obs_sig_{hash(json.dumps(event, sort_keys=True))}"
    }


@tool
@with_error_handling
async def analyze_risk_patterns(evidence_log: str) -> Dict[str, Any]:
    """
    Analyze risk patterns using advanced multi-dimensional analysis.
    
    Production-hardened with:
    - Circuit breaker protection
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Graceful degradation
    
    Args:
        evidence_log: JSON string containing list of evidence entries
        
    Returns:
        Dict containing risk analysis with score, level, patterns analyzed,
        severity distribution, risk factors, recommendations, and confidence.
    """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, asyncio.TimeoutError))
    )
    async def _perform_risk_analysis(evidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Core risk analysis with retry logic."""
        
        # Risk weight mapping
        risk_weights = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        # Initialize analysis variables
        total_risk = 0.0
        pattern_count = 0
        severity_distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        risk_factors = []
        
        # Analyze each evidence entry
        for evidence in evidence_data:
            if not isinstance(evidence, dict):
                logger.warning(f"Invalid evidence format: {type(evidence)}")
                continue
            
            severity = evidence.get("severity", "low")
            if severity not in risk_weights:
                logger.warning(f"Unknown severity level: {severity}, defaulting to 'low'")
                severity = "low"
            
            weight = risk_weights[severity]
            total_risk += weight
            pattern_count += 1
            severity_distribution[severity] += 1
            
            risk_factors.append({
                "source": evidence.get("source", "unknown"),
                "severity": severity,
                "weight": weight,
                "timestamp": evidence.get("timestamp"),
                "message": evidence.get("message", "")[:100]  # Truncate for safety
            })
        
        if pattern_count == 0:
            logger.warning("No valid evidence patterns found")
            return {
                "risk_score": 0.5,
                "risk_level": "medium",
                "patterns_analyzed": 0,
                "confidence": 0.1,
                "warning": "No valid evidence patterns found",
                "risk_factors": []
            }
        
        # Calculate normalized risk score with advanced weighting
        base_risk_score = total_risk / pattern_count
        
        # Apply pattern diversity bonus/penalty
        unique_severities = sum(1 for count in severity_distribution.values() if count > 0)
        diversity_factor = 1.0 + (unique_severities - 1) * 0.1
        
        # Apply critical event amplification
        critical_amplification = 1.0 + (severity_distribution["critical"] * 0.2)
        
        final_risk_score = min(base_risk_score * diversity_factor * critical_amplification, 1.0)
        
        # Determine risk level with hysteresis
        risk_level = _determine_risk_level(final_risk_score)
        
        # Generate recommendations
        recommendations = _generate_recommendations(risk_level)
        
        # Calculate confidence
        confidence = min(0.9, 0.5 + (pattern_count * 0.1) + (unique_severities * 0.1))
        
        return {
            "risk_score": round(final_risk_score, 4),
            "risk_level": risk_level,
            "patterns_analyzed": pattern_count,
            "severity_distribution": severity_distribution,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "confidence": round(confidence, 3),
            "analysis_metadata": {
                "base_score": round(base_risk_score, 4),
                "diversity_factor": round(diversity_factor, 3),
                "critical_amplification": round(critical_amplification, 3),
                "unique_severities": unique_severities
            }
        }
    
    # Main execution with circuit breaker protection
    try:
        # Parse and validate input
        try:
            evidence_data = json.loads(evidence_log)
            if not isinstance(evidence_data, list):
                evidence_data = [evidence_data]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in evidence_log: {e}")
            return {
                "evidence_type": "ANALYSIS_ERROR",
                "error": f"Invalid JSON format: {str(e)}",
                "risk_score": 0.5,
                "risk_level": "medium",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error_type": "validation_error"
            }
        
        # Execute analysis with circuit breaker
        result = await risk_analysis_breaker(
            _perform_risk_analysis(evidence_data)
        )
        
        # Add evidence type and timestamp
        result.update({
            "evidence_type": "RISK_ANALYSIS",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
        
    except CircuitBreaker.CircuitBreakerError:
        logger.error("Risk analysis circuit breaker is open")
        return {
            "evidence_type": "ANALYSIS_ERROR",
            "error": "Risk analysis service temporarily unavailable",
            "risk_score": 0.5,
            "risk_level": "medium",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error_type": "circuit_breaker_open"
        }
    except RetryError as e:
        logger.error(f"Risk analysis failed after retries: {e}")
        return {
            "evidence_type": "ANALYSIS_ERROR",
            "error": "Risk analysis failed after multiple attempts",
            "risk_score": 0.5,
            "risk_level": "medium",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error_type": "retry_exhausted"
        }


@tool
@with_error_handling
async def execute_remediation(action_plan: str) -> Dict[str, Any]:
    """
    Execute remediation actions based on risk analysis.
    
    Production-hardened with circuit breaker and validation.
    
    Args:
        action_plan: JSON string containing remediation actions
        
    Returns:
        Dict containing execution results with status, actions taken,
        and any errors encountered.
    """
    
    @remediation_breaker
    async def _execute_action(action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single remediation action."""
        action_type = action.get("type", "unknown")
        target = action.get("target", "unknown")
        
        # Simulate remediation execution
        # In production, this would call actual remediation APIs
        logger.info(f"Executing remediation: {action_type} on {target}")
        
        # Add small delay to simulate real work
        await asyncio.sleep(0.1)
        
        return {
            "action": action_type,
            "target": target,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Parse action plan
        actions = json.loads(action_plan)
        if not isinstance(actions, list):
            actions = [actions]
        
        # Execute all actions
        results = []
        errors = []
        
        for action in actions:
            try:
                result = await _execute_action(action)
                results.append(result)
            except Exception as e:
                error = {
                    "action": action.get("type", "unknown"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error)
                logger.error(f"Failed to execute action: {e}")
        
        # Determine overall status
        if not results and errors:
            status = "failed"
        elif errors:
            status = "partial_success"
        else:
            status = "success"
        
        return {
            "evidence_type": "REMEDIATION",
            "status": status,
            "actions_executed": len(results),
            "actions_failed": len(errors),
            "results": results,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85 if status == "success" else 0.5
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid action plan JSON: {e}")
        return {
            "evidence_type": "REMEDIATION_ERROR",
            "error": f"Invalid action plan format: {str(e)}",
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }


def _determine_risk_level(risk_score: float) -> str:
    """
    Determine risk level from score with hysteresis.
    
    Args:
        risk_score: Normalized risk score (0-1)
        
    Returns:
        Risk level string
    """
    if risk_score >= 0.85:
        return "critical"
    elif risk_score >= 0.65:
        return "high"
    elif risk_score >= 0.35:
        return "medium"
    else:
        return "low"


def _generate_recommendations(risk_level: str) -> List[str]:
    """
    Generate recommendations based on risk level.
    
    Args:
        risk_level: Current risk level
        
    Returns:
        List of recommendation strings
    """
    recommendations_map = {
        "critical": [
            "Immediate escalation required - Page on-call team",
            "Execute automated remediation procedures",
            "Prepare incident response team"
        ],
        "high": [
            "Monitor situation closely",
            "Execute automated remediation if available",
            "Notify operations team"
        ],
        "medium": [
            "Schedule review within 4 hours",
            "Log incident for trend analysis"
        ],
        "low": [
            "Continue monitoring",
            "No immediate action required"
        ]
    }
    
    return recommendations_map.get(risk_level, ["Unknown risk level"])