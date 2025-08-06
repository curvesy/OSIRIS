"""
🎼 Advanced LangGraph Collective Intelligence Workflows

This module provides configuration-driven patterns using cutting-edge LangGraph features.
Based on LangGraph Academy ambient agents and professional configuration patterns.
"""

from .state import CollectiveState
from .config import extract_config, WorkflowConfig
from .tools import (
    observe_system_event,
    analyze_risk_patterns,
    execute_remediation
)
from .shadow_mode import (
    get_shadow_logger,
    log_shadow_mode_prediction,
    record_shadow_mode_outcome
)
# from .graph import create_collective_workflow  # TODO: Implement graph module
# Temporarily commented out due to import issues
# from .nodes import (
#     SupervisorNode,
#     create_supervisor_node,
#     ObserverNode,
#     create_observer_node,
#     AnalystNode,
#     create_analyst_node,
#     # executor_node,  # TODO: Implement
#     # error_handler_node  # TODO: Implement
# )

__all__ = [
    "CollectiveState",
    "extract_config",
    "WorkflowConfig",
    "observe_system_event",
    "analyze_risk_patterns",
    "execute_remediation",
    "get_shadow_logger",
    "log_shadow_mode_prediction",
    "record_shadow_mode_outcome"
    # Temporarily removed due to import issues
    # "create_collective_workflow",
    # "supervisor_node",
    # "observer_node",
    # "analyst_node",
    # "executor_node",
    # "error_handler_node"
]