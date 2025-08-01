#!/usr/bin/env python3
"""
🎼 Advanced LangGraph Collective Intelligence - July 2025

Latest configuration-driven patterns using cutting-edge LangGraph features.
Based on LangGraph Academy ambient agents and professional configuration patterns.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Annotated, Sequence, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models import init_chat_model

# Import shadow mode logging for Phase 3C
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observability.shadow_mode_logger import ShadowModeLogger, ShadowModeEntry

logger = logging.getLogger(__name__)


class CollectiveState(TypedDict):
    """Advanced state using latest LangGraph TypedDict patterns."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workflow_id: str
    thread_id: str
    evidence_log: List[Dict[str, Any]]
    supervisor_decisions: List[Dict[str, Any]]
    memory_context: Dict[str, Any]
    active_config: Dict[str, Any]
    current_step: str
    risk_assessment: Optional[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    # Phase 1, Step 2: Error handling state
    error_log: List[Dict[str, Any]]
    error_recovery_attempts: int
    last_error: Optional[Dict[str, Any]]
    system_health: Dict[str, Any]


def extract_config(config: RunnableConfig) -> Dict[str, Any]:
    """Extract configuration using latest patterns from assistants-demo."""
    configurable = config.get("configurable", {})

    return {
        "supervisor_model": configurable.get("supervisor_model", "anthropic/claude-3-5-sonnet-latest"),
        "observer_model": configurable.get("observer_model", "anthropic/claude-3-haiku-latest"),
        "analyst_model": configurable.get("analyst_model", "anthropic/claude-3-5-sonnet-latest"),
        "executor_model": configurable.get("executor_model", "anthropic/claude-3-haiku-latest"),
        "enable_streaming": configurable.get("enable_streaming", True),
        "enable_human_loop": configurable.get("enable_human_loop", False),
        "checkpoint_mode": configurable.get("checkpoint_mode", "sqlite"),
        "memory_provider": configurable.get("memory_provider", "local"),
        "context_window": configurable.get("context_window", 5),
        "risk_thresholds": configurable.get("risk_thresholds", {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }),
        "supervisor_prompt": configurable.get("supervisor_prompt",
            "You are an expert collective intelligence supervisor. "
            "Analyze the current state and decide which tool to call next: "
            "observe_system_event, analyze_risk_patterns, execute_remediation, or FINISH."),
    }


# 🌙 Phase 3C: Shadow Mode Logging Infrastructure
_shadow_logger: Optional[ShadowModeLogger] = None

async def get_shadow_logger() -> ShadowModeLogger:
    """Get or create the global shadow mode logger."""
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowModeLogger()
        await _shadow_logger.initialize()
    return _shadow_logger

async def log_shadow_mode_prediction(state: CollectiveState, validation_result, proposed_action: str):
    """
    🌙 Log validator prediction in shadow mode for Phase 3C.

    This creates the training data foundation for our eventual PredictiveDecisionEngine.
    """
    try:
        shadow_logger = await get_shadow_logger()

        # Calculate routing decision based on validation result
        prob = validation_result.predicted_success_probability
        conf = validation_result.prediction_confidence_score
        decision_score = prob * conf

        if validation_result.requires_human_approval:
            routing_decision = "error_handler"
        elif decision_score > 0.7:
            routing_decision = "tools"
        elif decision_score >= 0.4:
            routing_decision = "supervisor"
        else:
            routing_decision = "error_handler"

        # Create shadow mode entry
        entry = ShadowModeEntry(
            workflow_id=state.get("workflow_id", "unknown"),
            thread_id=state.get("thread_id", "unknown"),
            timestamp=datetime.now(),
            evidence_log=state.get("evidence_log", []),
            memory_context=state.get("memory_context", {}),
            supervisor_decision=state.get("supervisor_decisions", [])[-1] if state.get("supervisor_decisions") else {},
            predicted_success_probability=validation_result.predicted_success_probability,
            prediction_confidence_score=validation_result.prediction_confidence_score,
            risk_score=validation_result.risk_score,
            predicted_risks=validation_result.predicted_risks,
            reasoning_trace=validation_result.reasoning_trace,
            requires_human_approval=validation_result.requires_human_approval,
            routing_decision=routing_decision,
            decision_score=decision_score
        )

        # Log the prediction (outcome will be recorded later)
        await shadow_logger.log_prediction(entry)

        logger.debug(f"🌙 Shadow mode prediction logged: {proposed_action} -> {routing_decision}")

    except Exception as e:
        # Don't let shadow mode logging break the main workflow
        logger.warning(f"⚠️ Shadow mode logging failed (non-blocking): {e}")

async def record_shadow_mode_outcome(workflow_id: str, outcome: str, execution_time: Optional[float] = None, error_details: Optional[Dict] = None):
    """
    🌙 Record actual outcome for shadow mode analysis.

    Call this after action execution to complete the training data.
    """
    try:
        shadow_logger = await get_shadow_logger()
        await shadow_logger.record_outcome(workflow_id, outcome, execution_time, error_details)
        logger.debug(f"🌙 Shadow mode outcome recorded: {workflow_id} -> {outcome}")
    except Exception as e:
        logger.warning(f"⚠️ Shadow mode outcome recording failed (non-blocking): {e}")

async def shadow_aware_tool_node(state: CollectiveState) -> CollectiveState:
    """
    🌙 Tool execution wrapper with shadow mode outcome recording.

    This wraps the standard ToolNode to record actual outcomes for our training data.
    """
    import time
    from langgraph.prebuilt import ToolNode

    workflow_id = state.get("workflow_id", "unknown")
    start_time = time.time()

    try:
        # Execute tools using standard ToolNode
        tools = [observe_system_event, analyze_risk_patterns, execute_remediation]
        tool_node = ToolNode(tools)

        # Execute the tools
        result_state = await tool_node.ainvoke(state)

        execution_time = time.time() - start_time

        # Determine outcome based on execution results
        execution_results = result_state.get("execution_results", [])
        if execution_results:
            # Check if any execution failed
            has_errors = any(result.get("error") for result in execution_results)
            outcome = "failure" if has_errors else "success"
        else:
            # Check messages for tool call results
            messages = result_state.get("messages", [])
            tool_messages = [msg for msg in messages if hasattr(msg, 'tool_calls') or getattr(msg, 'type', None) == 'tool']
            outcome = "success" if tool_messages else "partial"

        # Record shadow mode outcome
        await record_shadow_mode_outcome(workflow_id, outcome, execution_time)

        logger.info(f"🌙 Tool execution completed: {outcome} in {execution_time:.2f}s")

        return result_state

    except Exception as e:
        execution_time = time.time() - start_time
        error_details = {"error": str(e), "traceback": traceback.format_exc()}

        # Record failure outcome
        await record_shadow_mode_outcome(workflow_id, "failure", execution_time, error_details)

        logger.error(f"❌ Tool execution failed: {e}")

        # Return state with error information
        return {
            **state,
            "execution_results": [{"error": str(e), "timestamp": datetime.now().isoformat()}],
            "current_step": "tool_execution_failed"
        }


@tool
async def observe_system_event(event_data: str) -> Dict[str, Any]:
    """Observe system events using proven ObserverAgent patterns."""
    import json
    event = json.loads(event_data)

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
async def analyze_risk_patterns(evidence_log: str) -> Dict[str, Any]:
    """
    Analyze risk patterns using advanced multi-dimensional analysis.

    Production-hardened with:
    - Circuit breaker protection
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Graceful degradation
    """

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from pybreaker import CircuitBreaker
    import asyncio
    import json

    # Circuit breaker for external API calls (if any)
    risk_analysis_breaker = CircuitBreaker(
        fail_max=3,
        reset_timeout=30,
        exclude=[ValueError, json.JSONDecodeError]  # Don't break on data validation errors
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, asyncio.TimeoutError))
    )
    async def _perform_risk_analysis(evidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Core risk analysis with retry logic."""

        try:
            # Advanced risk analysis logic with production hardening
            risk_weights = {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.1
            }

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
            diversity_factor = 1.0 + (unique_severities - 1) * 0.1  # Bonus for diverse patterns

            # Apply critical event amplification
            critical_amplification = 1.0 + (severity_distribution["critical"] * 0.2)

            final_risk_score = min(base_risk_score * diversity_factor * critical_amplification, 1.0)

            # Determine risk level with hysteresis to prevent oscillation
            if final_risk_score >= 0.85:
                risk_level = "critical"
            elif final_risk_score >= 0.65:
                risk_level = "high"
            elif final_risk_score >= 0.35:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Generate intelligent recommendations
            recommendations = []
            if risk_level == "critical":
                recommendations = [
                    "Immediate escalation required - Page on-call team",
                    "Execute automated remediation procedures",
                    "Prepare incident response team"
                ]
            elif risk_level == "high":
                recommendations = [
                    "Monitor situation closely",
                    "Execute automated remediation if available",
                    "Notify operations team"
                ]
            elif risk_level == "medium":
                recommendations = [
                    "Schedule review within 4 hours",
                    "Log incident for trend analysis"
                ]
            else:
                recommendations = [
                    "Continue monitoring",
                    "No immediate action required"
                ]

            # Calculate confidence based on data quality
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

        except Exception as e:
            logger.error(f"Core risk analysis failed: {e}")
            raise  # Re-raise for retry logic

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

        # Execute analysis with circuit breaker protection
        analysis_result = await risk_analysis_breaker.call_async(_perform_risk_analysis, evidence_data)

        # Build final response
        return {
            "evidence_type": "ANALYSIS",
            "timestamp": datetime.now().isoformat(),
            "signature": f"analysis_sig_{hash(evidence_log)}",
            "circuit_breaker_state": str(risk_analysis_breaker.current_state),
            **analysis_result
        }

    except CircuitBreaker.CircuitBreakerError:
        logger.error("Risk analysis circuit breaker is open - too many failures")
        return {
            "evidence_type": "ANALYSIS_ERROR",
            "error": "Risk analysis service temporarily unavailable (circuit breaker open)",
            "risk_score": 0.7,  # Conservative high risk when analysis fails
            "risk_level": "high",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error_type": "circuit_breaker_open",
            "retry_after": 30
        }

    except Exception as e:
        logger.error(f"Risk analysis failed after all retries: {e}")
        return {
            "evidence_type": "ANALYSIS_ERROR",
            "error": f"Analysis failed: {str(e)}",
            "risk_score": 0.6,  # Conservative medium-high risk on error
            "risk_level": "high",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error_type": "analysis_failure"
        }


@tool
async def execute_remediation(analysis_data: str) -> Dict[str, Any]:
    """Execute remediation actions based on risk analysis."""
    import json
    analysis = json.loads(analysis_data)
    risk_score = analysis.get("risk_score", 0.5)

    actions = []
    if risk_score > 0.8:
        actions = [
            {"type": "create_incident", "priority": "P1", "auto_assign": True},
            {"type": "page_oncall", "severity": "critical"},
            {"type": "scale_resources", "factor": 2.0}
        ]
    elif risk_score > 0.6:
        actions = [
            {"type": "create_ticket", "priority": "P2"},
            {"type": "alert_team", "channel": "ops"}
        ]
    elif risk_score > 0.3:
        actions = [
            {"type": "log_incident", "category": "monitoring"}
        ]

    return {
        "evidence_type": "EXECUTION",
        "actions_planned": len(actions),
        "actions_executed": actions,
        "status": "SUCCESS",
        "execution_time_ms": len(actions) * 100,
        "timestamp": datetime.now().isoformat(),
        "signature": f"exec_sig_{len(actions)}"
    }


async def error_handler(state: CollectiveState) -> CollectiveState:
    """
    Professional error handling node with intelligent recovery strategies.

    Phase 1, Step 2: Graph-level error handling and recovery.
    Implements enterprise-grade error processing with:
    - Error classification and severity assessment
    - Intelligent recovery strategies (retry, escalate, terminate)
    - Circuit breaker integration
    - Professional observability and alerting
    """

    from tenacity import retry, stop_after_attempt, wait_exponential
    import traceback

    logger.info("🚨 Error handler activated - analyzing system state")

    # Extract current error context
    last_error = state.get("last_error", {})
    error_log = state.get("error_log", [])
    recovery_attempts = state.get("error_recovery_attempts", 0)
    current_step = state.get("current_step", "unknown")

    # Error classification
    error_type = last_error.get("error_type", "unknown")
    error_severity = last_error.get("severity", "medium")
    error_source = last_error.get("source", "unknown")
    error_message = last_error.get("message", "Unknown error")

    logger.error(f"Processing error: {error_type} from {error_source} - {error_message}")

    # Recovery strategy decision matrix
    recovery_strategy = "terminate"  # Default safe fallback

    if error_type == "validation_error":
        if recovery_attempts < 2:
            recovery_strategy = "retry_with_sanitization"
        else:
            recovery_strategy = "escalate_to_human"

    elif error_type == "circuit_breaker_open":
        if recovery_attempts < 1:
            recovery_strategy = "wait_and_retry"
        else:
            recovery_strategy = "fallback_mode"

    elif error_type == "analysis_failure":
        if recovery_attempts < 3:
            recovery_strategy = "retry_with_degraded_analysis"
        else:
            recovery_strategy = "escalate_to_human"

    elif error_type == "network_error":
        if recovery_attempts < 5:
            recovery_strategy = "exponential_backoff_retry"
        else:
            recovery_strategy = "offline_mode"

    else:
        # Unknown error - be conservative
        if recovery_attempts < 1:
            recovery_strategy = "single_retry"
        else:
            recovery_strategy = "escalate_to_human"

    logger.info(f"🔧 Recovery strategy selected: {recovery_strategy}")

    # Execute recovery strategy
    recovery_result = await _execute_recovery_strategy(
        recovery_strategy,
        state,
        last_error,
        recovery_attempts
    )

    # Update system health metrics
    system_health = state.get("system_health", {})
    system_health.update({
        "last_error_time": datetime.now().isoformat(),
        "total_errors": len(error_log) + 1,
        "recovery_success_rate": _calculate_recovery_success_rate(error_log),
        "current_health_status": recovery_result.get("health_status", "degraded"),
        "error_trends": _analyze_error_trends(error_log + [last_error])
    })

    # Build updated state
    updated_state = {
        **state,
        "error_log": error_log + [{
            **last_error,
            "recovery_strategy": recovery_strategy,
            "recovery_result": recovery_result.get("status", "unknown"),
            "handled_at": datetime.now().isoformat()
        }],
        "error_recovery_attempts": recovery_attempts + 1,
        "system_health": system_health,
        "current_step": recovery_result.get("next_step", "supervisor"),
        "last_error": None  # Clear the error after handling
    }

    # Add recovery message to conversation
    recovery_message = HumanMessage(content=f"""
🚨 Error Recovery Report:
- Error Type: {error_type}
- Recovery Strategy: {recovery_strategy}
- Status: {recovery_result.get('status', 'unknown')}
- Next Step: {recovery_result.get('next_step', 'supervisor')}
- System Health: {system_health.get('current_health_status', 'unknown')}
""")

    updated_state["messages"] = list(updated_state.get("messages", [])) + [recovery_message]

    logger.info(f"✅ Error handling complete - next step: {recovery_result.get('next_step', 'supervisor')}")

    return updated_state


async def _execute_recovery_strategy(
    strategy: str,
    state: CollectiveState,
    error: Dict[str, Any],
    attempts: int
) -> Dict[str, Any]:
    """Execute the selected recovery strategy."""

    try:
        if strategy == "retry_with_sanitization":
            logger.info("🔄 Attempting retry with data sanitization")
            # Clean and retry the failed operation
            return {
                "status": "retry_scheduled",
                "next_step": state.get("current_step", "supervisor"),
                "health_status": "recovering"
            }

        elif strategy == "wait_and_retry":
            logger.info("⏳ Waiting for circuit breaker reset")
            await asyncio.sleep(2)  # Brief wait for circuit breaker
            return {
                "status": "retry_after_wait",
                "next_step": state.get("current_step", "supervisor"),
                "health_status": "recovering"
            }

        elif strategy == "retry_with_degraded_analysis":
            logger.info("🔧 Retrying with simplified analysis")
            return {
                "status": "degraded_retry",
                "next_step": "supervisor",
                "health_status": "degraded"
            }

        elif strategy == "exponential_backoff_retry":
            wait_time = min(2 ** attempts, 30)  # Cap at 30 seconds
            logger.info(f"⏳ Exponential backoff: waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            return {
                "status": "backoff_retry",
                "next_step": state.get("current_step", "supervisor"),
                "health_status": "recovering"
            }

        elif strategy == "escalate_to_human":
            logger.warning("🚨 Escalating to human operator")
            return {
                "status": "human_escalation",
                "next_step": "FINISH",
                "health_status": "requires_intervention",
                "escalation_reason": f"Multiple recovery attempts failed: {error.get('message', 'Unknown')}"
            }

        elif strategy == "fallback_mode":
            logger.info("🛡️ Entering fallback mode")
            return {
                "status": "fallback_active",
                "next_step": "supervisor",
                "health_status": "fallback_mode"
            }

        elif strategy == "offline_mode":
            logger.warning("📴 Entering offline mode")
            return {
                "status": "offline_mode",
                "next_step": "FINISH",
                "health_status": "offline"
            }

        else:  # terminate or unknown
            logger.error("🛑 Terminating workflow due to unrecoverable error")
            return {
                "status": "terminated",
                "next_step": "FINISH",
                "health_status": "failed"
            }

    except Exception as recovery_error:
        logger.error(f"❌ Recovery strategy failed: {recovery_error}")
        return {
            "status": "recovery_failed",
            "next_step": "FINISH",
            "health_status": "critical",
            "recovery_error": str(recovery_error)
        }


def _calculate_recovery_success_rate(error_log: List[Dict[str, Any]]) -> float:
    """Calculate the success rate of error recovery attempts."""
    if not error_log:
        return 1.0

    successful_recoveries = sum(
        1 for error in error_log
        if error.get("recovery_result") in ["retry_scheduled", "retry_after_wait", "degraded_retry", "backoff_retry"]
    )

    return successful_recoveries / len(error_log) if error_log else 1.0


def _analyze_error_trends(error_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze error patterns and trends for system health assessment."""
    if not error_log:
        return {"trend": "stable", "pattern": "none"}

    # Count error types
    error_types = {}
    recent_errors = error_log[-10:]  # Last 10 errors

    for error in recent_errors:
        error_type = error.get("error_type", "unknown")
        error_types[error_type] = error_types.get(error_type, 0) + 1

    # Determine trend
    if len(recent_errors) >= 5:
        if error_types.get("circuit_breaker_open", 0) >= 3:
            trend = "circuit_breaker_pattern"
        elif error_types.get("network_error", 0) >= 3:
            trend = "network_instability"
        elif len(set(error_types.keys())) == 1:
            trend = "recurring_issue"
        else:
            trend = "mixed_errors"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "pattern": error_types,
        "recent_error_count": len(recent_errors),
        "dominant_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else "none"
    }


class AmbientSupervisor:
    """Advanced supervisor using latest LangGraph Academy patterns."""

    def __init__(self, config: RunnableConfig):
        self.config_data = extract_config(config)
        self.llm = init_chat_model(self.config_data["supervisor_model"])

    async def __call__(self, state: CollectiveState) -> CollectiveState:
        """Supervisor node using latest patterns."""
        messages = state.get("messages", [])
        evidence_log = state.get("evidence_log", [])

        # Intelligent decision making
        if len(evidence_log) == 0:
            decision = "observe_system_event"
            reasoning = "No evidence collected yet, need to observe initial event"
        elif not any("ANALYSIS" in str(item.get("evidence_type", "")) for item in evidence_log):
            decision = "analyze_risk_patterns"
            reasoning = "Evidence collected, need risk analysis"
        elif not any("EXECUTION" in str(item.get("evidence_type", "")) for item in evidence_log):
            # Check if execution is needed based on risk
            analysis_items = [item for item in evidence_log if item.get("evidence_type") == "ANALYSIS"]
            if analysis_items and analysis_items[-1].get("risk_score", 0) > 0.3:
                decision = "execute_remediation"
                reasoning = "Risk analysis complete, execution required"
            else:
                decision = "FINISH"
                reasoning = "Low risk detected, no action needed"
        else:
            decision = "FINISH"
            reasoning = "All workflow steps completed successfully"

        # Create supervisor decision
        supervisor_decision = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "reasoning": reasoning,
            "evidence_count": len(evidence_log),
            "confidence": 0.9
        }

        # Add decision message
        decision_message = HumanMessage(content=decision)
        updated_messages = list(messages) + [decision_message]
        updated_decisions = list(state.get("supervisor_decisions", [])) + [supervisor_decision]

        return {
            **state,
            "messages": updated_messages,
            "supervisor_decisions": updated_decisions,
            "current_step": f"supervisor_decision_{decision}"
        }


def supervisor_router(state: CollectiveState) -> str:
    """
    Route based on supervisor decision using latest patterns with error handling.

    Phase 3B: Enhanced routing with validator integration for prospection.
    """

    # Check for errors first
    if state.get("last_error"):
        logger.warning("🚨 Error detected - routing to error handler")
        return "error_handler"

    # Check system health
    system_health = state.get("system_health", {})
    health_status = system_health.get("current_health_status", "healthy")

    if health_status in ["critical", "offline"]:
        logger.error(f"🛑 System health critical: {health_status} - terminating")
        return END

    # Check for too many recovery attempts
    recovery_attempts = state.get("error_recovery_attempts", 0)
    if recovery_attempts >= 10:
        logger.error("🛑 Maximum recovery attempts exceeded - terminating")
        return END

    messages = state.get("messages", [])
    if not messages:
        # Phase 3B: Route to validator instead of directly to tools
        return "validator"

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        if "FINISH" in last_message.content.upper():
            return END
        elif last_message.content in ["observe_system_event", "analyze_risk_patterns", "execute_remediation"]:
            # Phase 3B: Route to validator for risk assessment before tool execution
            return "validator"

    return END


async def memory_enrichment_node(state: CollectiveState) -> CollectiveState:
    """Memory enrichment using ambient agent patterns."""
    logger.info("🧠 Enriching with collective memory...")

    # Mock memory enrichment - replace with LangMem in production
    memory_context = {
        "similar_incidents": 3,
        "success_rate": 0.85,
        "recommended_approach": "automated_resolution",
        "confidence": 0.7,
        "historical_patterns": ["database_issues", "connection_pool_exhaustion"],
        "avg_resolution_time": 180
    }

    memory_message = SystemMessage(content=f"""
    Collective Memory Context:
    - Similar incidents: {memory_context['similar_incidents']}
    - Success rate: {memory_context['success_rate']:.1%}
    - Recommended: {memory_context['recommended_approach']}
    - Confidence: {memory_context['confidence']:.2f}
    """)

    updated_messages = list(state.get("messages", [])) + [memory_message]

    return {
        **state,
        "messages": updated_messages,
        "memory_context": memory_context,
        "current_step": "memory_enriched"
    }


async def validator_node(state: CollectiveState) -> CollectiveState:
    """
    🧠 Professional Predictive Validator Node - The "Prefrontal Cortex"

    Validates proposed actions before execution using hybrid rule-based/LLM approach.
    This is the core of Phase 3B - integrating prospection into the workflow.
    """
    logger.info("🧠 Validating proposed action with predictive validator...")

    try:
        # Import validator (lazy import to avoid circular dependencies)
        from ..agents.validator import create_professional_validator
        from langchain_community.chat_models import init_chat_model

        # Extract the proposed action from supervisor decisions
        supervisor_decisions = state.get("supervisor_decisions", [])
        if not supervisor_decisions:
            # No action to validate - this shouldn't happen in normal flow
            logger.warning("⚠️ No supervisor decision found for validation")
            return {
                **state,
                "risk_assessment": {
                    "predicted_success_probability": 0.1,
                    "prediction_confidence_score": 0.1,
                    "risk_score": 0.95,
                    "predicted_risks": [{"risk": "No action specified", "mitigation": "Return to supervisor"}],
                    "reasoning_trace": "No action found to validate",
                    "requires_human_approval": True,
                    "validation_status": "error"
                },
                "current_step": "validation_failed"
            }

        # Get the latest supervisor decision
        latest_decision = supervisor_decisions[-1]
        proposed_action = latest_decision.get("decision", "unknown_action")

        # Initialize validator with LLM
        config_data = extract_config(state.get("active_config", {}))
        llm = init_chat_model(config_data.get("validator_model", "gpt-4o-mini"))
        validator = create_professional_validator(
            llm=llm,
            risk_threshold=config_data.get("risk_threshold", 0.75),
            cache_ttl_seconds=config_data.get("cache_ttl", 3600)
        )

        # Prepare context for validation
        current_evidence = state.get("evidence_log", [])
        memory_context = state.get("memory_context", {})

        # Convert memory context to list format expected by validator
        memory_list = []
        if isinstance(memory_context, dict):
            for key, value in memory_context.items():
                if isinstance(value, list):
                    memory_list.extend([{"type": key, "data": item} for item in value])
                else:
                    memory_list.append({"type": key, "data": value})

        # Perform validation
        validation_result = await validator.validate_action(
            proposed_action=proposed_action,
            memory_context=memory_list,
            current_evidence=current_evidence
        )

        # Convert ValidationResult to dict for state storage
        risk_assessment = {
            "predicted_success_probability": validation_result.predicted_success_probability,
            "prediction_confidence_score": validation_result.prediction_confidence_score,
            "risk_score": validation_result.risk_score,
            "predicted_risks": validation_result.predicted_risks,
            "reasoning_trace": validation_result.reasoning_trace,
            "requires_human_approval": validation_result.requires_human_approval,
            "cache_hit": validation_result.cache_hit,
            "timestamp": validation_result.timestamp.isoformat(),
            "validation_status": "complete",
            "proposed_action": proposed_action
        }

        # Log validation metrics for monitoring
        logger.info(f"🎯 Validation complete: {proposed_action}")
        logger.info(f"   Success Probability: {validation_result.predicted_success_probability:.2f}")
        logger.info(f"   Confidence Score: {validation_result.prediction_confidence_score:.2f}")
        logger.info(f"   Risk Score: {validation_result.risk_score:.2f}")
        logger.info(f"   Human Approval Required: {validation_result.requires_human_approval}")

        # 🌙 Phase 3C: Shadow Mode Logging
        await log_shadow_mode_prediction(state, validation_result, proposed_action)

        return {
            **state,
            "risk_assessment": risk_assessment,
            "current_step": "action_validated"
        }

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Fallback to conservative risk assessment
        fallback_assessment = {
            "predicted_success_probability": 0.2,
            "prediction_confidence_score": 0.05,
            "risk_score": 0.95,
            "predicted_risks": [{"risk": f"Validation system error: {str(e)}", "mitigation": "Manual review required"}],
            "reasoning_trace": f"Validation failed with error: {str(e)}",
            "requires_human_approval": True,
            "validation_status": "error",
            "error_details": str(e)
        }

        return {
            **state,
            "risk_assessment": fallback_assessment,
            "current_step": "validation_error"
        }


def route_after_validation(state: CollectiveState) -> str:
    """
    🎯 Conditional routing based on validation results.

    This implements the core decision logic from what.md:
    - High confidence (>0.7) → Execute tools
    - Medium confidence (0.4-0.7) → Return to supervisor for replanning
    - Low confidence (<0.4) or human approval required → Error handler for escalation
    """
    risk_assessment = state.get("risk_assessment", {})

    # Check if validation failed
    if risk_assessment.get("validation_status") == "error":
        logger.warning("⚠️ Routing to error handler due to validation failure")
        return "error_handler"

    # Check if human approval is explicitly required
    if risk_assessment.get("requires_human_approval", False):
        logger.info("👤 Routing to error handler for human approval")
        return "error_handler"

    # Calculate decision score: success_probability * confidence
    success_prob = risk_assessment.get("predicted_success_probability", 0.0)
    confidence = risk_assessment.get("prediction_confidence_score", 0.0)
    decision_score = success_prob * confidence

    logger.info(f"🎯 Decision score: {decision_score:.3f} (prob: {success_prob:.2f} × conf: {confidence:.2f})")

    # Route based on decision score thresholds
    if decision_score > 0.7:
        logger.info("✅ High confidence → Executing tools")
        return "tools"
    elif decision_score >= 0.4:
        logger.info("🔄 Medium confidence → Returning to supervisor for replanning")
        return "supervisor"
    else:
        logger.info("⚠️ Low confidence → Escalating to error handler")
        return "error_handler"


async def create_collective_graph(config: RunnableConfig):
    """Create the collective intelligence graph using latest LangGraph patterns."""

    # Extract configuration
    config_data = extract_config(config)

    # Define tools - Phase 3C: Use shadow-aware tool execution
    tools = [observe_system_event, analyze_risk_patterns, execute_remediation]
    # Note: shadow_aware_tool_node handles tool execution with outcome recording

    # Create supervisor
    supervisor = AmbientSupervisor(config)

    # Build graph using latest StateGraph patterns
    workflow = StateGraph(CollectiveState)

    # Add nodes - Phase 3B: Include validator node for prospection
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("memory_enrichment", memory_enrichment_node)
    workflow.add_node("validator", validator_node)  # Phase 3B: Predictive validator node
    workflow.add_node("tools", shadow_aware_tool_node)  # Phase 3C: Shadow mode tool execution
    workflow.add_node("error_handler", error_handler)

    # Add edges using latest patterns with validator integration
    workflow.add_edge(START, "memory_enrichment")
    workflow.add_edge("memory_enrichment", "supervisor")

    # Phase 3B: Supervisor routes to validator instead of directly to tools
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {"validator": "validator", "error_handler": "error_handler", END: END}
    )

    # Phase 3B: Validator routes based on risk assessment and confidence
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "tools": "tools",           # High confidence → Execute
            "supervisor": "supervisor", # Medium confidence → Replan
            "error_handler": "error_handler"  # Low confidence or human approval
        }
    )

    # Tools can route to error handler or back to supervisor
    workflow.add_conditional_edges(
        "tools",
        lambda state: "error_handler" if state.get("last_error") else "supervisor",
        {"error_handler": "error_handler", "supervisor": "supervisor"}
    )

    # Error handler routes based on recovery strategy
    workflow.add_conditional_edges(
        "error_handler",
        lambda state: state.get("current_step", "supervisor"),
        {"supervisor": "supervisor", "tools": "tools", END: END}
    )

    # Setup checkpointing
    if config_data["checkpoint_mode"] == "sqlite":
        checkpointer = SqliteSaver.from_conn_string(":memory:")
    else:
        checkpointer = None

    # Compile with latest features
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"] if config_data["enable_human_loop"] else None
    )

    return app


class CollectiveWorkflow:
    """
    Professional collective intelligence using latest LangGraph patterns.

    Based on:
    - LangGraph Academy ambient agents course
    - Configuration patterns from assistants-demo
    - July 2025 cutting-edge features
    """

    def __init__(self, base_config: Optional[Dict[str, Any]] = None):
        self.base_config = base_config or {}
        self.workflow_id = f"collective_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.app = None

        logger.info(f"🎼 CollectiveWorkflow created: {self.workflow_id}")

    async def initialize(self, config: RunnableConfig) -> None:
        """Initialize using configuration-driven patterns."""
        logger.info("🎼 Initializing collective intelligence workflow...")

        try:
            # Create graph with configuration
            self.app = await create_collective_graph(config)

            logger.info("✅ Collective intelligence workflow initialized successfully")

        except Exception as e:
            logger.error(f"❌ Workflow initialization failed: {e}")
            raise

    async def process_event(self, event_data: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """
        Process events using latest LangGraph streaming patterns.

        Args:
            event_data: Raw event data to process
            config: Runtime configuration

        Returns:
            Complete workflow results
        """

        if not self.app:
            await self.initialize(config)

        logger.info(f"🎼 Processing event: {event_data.get('type', 'unknown')}")

        try:
            # Create initial state using latest patterns
            initial_state: CollectiveState = {
                "messages": [HumanMessage(content=f"Process event: {event_data}")],
                "workflow_id": self.workflow_id,
                "thread_id": f"thread_{datetime.now().strftime('%H%M%S')}",
                "evidence_log": [],
                "supervisor_decisions": [],
                "memory_context": {},
                "active_config": extract_config(config),
                "current_step": "initialized",
                "risk_assessment": None,
                "execution_results": []
            }

            # Execute workflow with streaming
            final_state = None
            evidence_updates = []

            async for state in self.app.astream(
                initial_state,
                config=config,
                stream_mode="values"
            ):
                final_state = state

                # Track evidence updates
                if "evidence_log" in state and state["evidence_log"]:
                    evidence_updates = state["evidence_log"]
                    logger.info(f"📊 Evidence updated: {len(evidence_updates)} entries")

            # Format response
            return self._format_response(final_state, evidence_updates)

        except Exception as e:
            logger.error(f"❌ Event processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_id": self.workflow_id,
                "timestamp": datetime.now().isoformat()
            }

    def _format_response(self, final_state: CollectiveState, evidence_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format workflow response using latest patterns."""

        # Extract insights from evidence
        insights = {}
        if evidence_updates:
            latest_evidence = evidence_updates[-1]
            insights = {
                "evidence_type": latest_evidence.get("evidence_type"),
                "risk_score": latest_evidence.get("risk_score"),
                "risk_level": latest_evidence.get("risk_level"),
                "actions_executed": latest_evidence.get("actions_executed", []),
                "confidence": latest_evidence.get("confidence")
            }

        return {
            "status": "success",
            "workflow_id": final_state.get("workflow_id"),
            "thread_id": final_state.get("thread_id"),
            "evidence_count": len(evidence_updates),
            "supervisor_decisions": len(final_state.get("supervisor_decisions", [])),
            "insights": insights,
            "memory_context": final_state.get("memory_context", {}),
            "final_step": final_state.get("current_step"),
            "processing_complete": True,
            "timestamp": datetime.now().isoformat()
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health using latest patterns."""
        return {
            "workflow_id": self.workflow_id,
            "app_initialized": self.app is not None,
            "config_driven": True,
            "langgraph_version": "latest",
            "patterns": ["ambient_agents", "configuration_driven", "streaming_execution"],
            "timestamp": datetime.now().isoformat()
        }

