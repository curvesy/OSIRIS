"""
Example usage of the refactored workflows module.

This demonstrates how the modular structure improves code organization
and makes the system more maintainable and testable.
"""

import asyncio
from typing import Dict, Any

# Import from the new modular structure
from aura_intelligence.orchestration.workflows import (
    CollectiveState,
    WorkflowConfig,
    extract_config,
    observe_system_event,
    analyze_risk_patterns,
    execute_remediation,
    log_shadow_mode_prediction,
    record_shadow_mode_outcome
)
from aura_intelligence.orchestration.workflows.state import (
    WorkflowStatus,
    RiskLevel,
    SystemHealth,
    ErrorContext
)


async def example_workflow_usage():
    """
    Example of using the refactored workflow components.
    
    Before refactoring:
    - All code was in a single 1215-line file
    - Mixed concerns (state, config, tools, shadow mode)
    - Difficult to test individual components
    - No clear separation of responsibilities
    
    After refactoring:
    - Modular structure with clear responsibilities
    - Each module is focused and testable
    - Modern Python 3.13+ features (pattern matching, Pydantic v2)
    - Production-grade error handling
    """
    
    # 1. Create workflow configuration using Pydantic v2
    config = WorkflowConfig(
        enable_streaming=True,
        enable_shadow_mode=True,
        context_window=10,
        risk_thresholds={
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
    )
    
    print(f"Configuration validated: {config.model_dump()}")
    
    # 2. Initialize workflow state with type safety
    state: CollectiveState = {
        "messages": [],
        "workflow_id": "example-workflow-001",
        "thread_id": "thread-123",
        "current_step": "initializing",
        "workflow_status": WorkflowStatus.INITIALIZING,
        "evidence_log": [],
        "supervisor_decisions": [],
        "execution_results": [],
        "memory_context": {},
        "active_config": config.to_dict(),
        "risk_assessment": None,
        "risk_level": None,
        "error_log": [],
        "error_recovery_attempts": 0,
        "last_error": None,
        "system_health": SystemHealth(
            cpu_usage=0.45,
            memory_usage=0.60,
            error_rate=0.02
        ),
        "shadow_mode_enabled": True,
        "shadow_predictions": []
    }
    
    # 3. Observe system event using refactored tool
    event_data = {
        "source": "api_gateway",
        "severity": "high",
        "message": "Increased error rate detected"
    }
    
    observation = await observe_system_event(
        event_data=json.dumps(event_data)
    )
    print(f"\nObservation result: {observation}")
    
    # Add to evidence log
    state["evidence_log"].append(observation)
    
    # 4. Analyze risk patterns with production-hardened logic
    risk_analysis = await analyze_risk_patterns(
        evidence_log=json.dumps(state["evidence_log"])
    )
    print(f"\nRisk analysis: {risk_analysis}")
    
    # Update state with risk assessment
    state["risk_assessment"] = risk_analysis
    
    # 5. Use pattern matching to determine risk level (Python 3.13+)
    risk_score = risk_analysis.get("risk_score", 0.5)
    
    match risk_score:
        case score if score >= 0.85:
            state["risk_level"] = RiskLevel.CRITICAL
        case score if score >= 0.65:
            state["risk_level"] = RiskLevel.HIGH
        case score if score >= 0.35:
            state["risk_level"] = RiskLevel.MEDIUM
        case _:
            state["risk_level"] = RiskLevel.LOW
    
    print(f"\nDetermined risk level: {state['risk_level'].value}")
    
    # 6. Log shadow mode prediction if enabled
    if state["shadow_mode_enabled"]:
        await log_shadow_mode_prediction(
            state=state,
            validation_result=risk_analysis,
            proposed_action="execute_remediation"
        )
        print("\nShadow mode prediction logged")
    
    # 7. Execute remediation if needed
    if state["risk_level"] in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
        action_plan = [
            {"type": "scale_up", "target": "api_gateway"},
            {"type": "enable_rate_limiting", "target": "api_gateway"}
        ]
        
        remediation_result = await execute_remediation(
            action_plan=json.dumps(action_plan)
        )
        print(f"\nRemediation result: {remediation_result}")
        
        state["execution_results"].append(remediation_result)
    
    # 8. Update workflow status
    state["workflow_status"] = WorkflowStatus.COMPLETED
    
    # 9. Record shadow mode outcome
    if state["shadow_mode_enabled"]:
        await record_shadow_mode_outcome(
            workflow_id=state["workflow_id"],
            outcome="success",
            execution_time=1.5  # seconds
        )
    
    return state


def demonstrate_error_handling():
    """
    Demonstrate improved error handling with the refactored code.
    
    The new structure provides:
    - Consistent error context tracking
    - Graceful degradation
    - Circuit breaker patterns
    - Structured error logging
    """
    
    # Create an error context with full tracking
    error = ErrorContext(
        error_type="APIError",
        message="External service unavailable",
        stack_trace="Full stack trace here...",
        recovery_attempted=True
    )
    
    # Convert to dict for logging/serialization
    error_dict = error.to_dict()
    print(f"Error context: {error_dict}")
    
    # System health check with property
    health = SystemHealth(
        cpu_usage=0.85,  # High CPU
        memory_usage=0.70,
        error_rate=0.05
    )
    
    print(f"System healthy: {health.is_healthy}")  # False due to high CPU


def demonstrate_config_validation():
    """
    Demonstrate Pydantic v2 configuration validation.
    
    The new config system provides:
    - Type validation
    - Value constraints
    - Automatic documentation
    - Serialization support
    """
    
    try:
        # This will fail validation
        invalid_config = WorkflowConfig(
            context_window=100,  # Exceeds maximum of 50
            risk_thresholds={
                "critical": 0.5,
                "high": 0.7,  # Invalid: high > critical
                "medium": 0.3,
                "low": 0.1
            }
        )
    except ValueError as e:
        print(f"Config validation error: {e}")
    
    # Valid configuration
    valid_config = WorkflowConfig(
        context_window=10,
        enable_human_loop=True,
        checkpoint_mode="postgres"
    )
    
    print(f"Valid config: {valid_config.model_dump_json(indent=2)}")


if __name__ == "__main__":
    import json
    
    print("=== Refactored Workflows Module Demo ===\n")
    
    # Run async workflow example
    print("1. Running workflow example...")
    final_state = asyncio.run(example_workflow_usage())
    print(f"\nFinal workflow status: {final_state['workflow_status'].name}")
    
    print("\n" + "="*50 + "\n")
    
    # Demonstrate error handling
    print("2. Error handling demonstration...")
    demonstrate_error_handling()
    
    print("\n" + "="*50 + "\n")
    
    # Demonstrate config validation
    print("3. Configuration validation demonstration...")
    demonstrate_config_validation()
    
    print("\n=== Demo Complete ===")


"""
Key Improvements in the Refactored Code:

1. **Modular Structure**:
   - state.py: State definitions with dataclasses and enums
   - config.py: Pydantic v2 configuration management
   - tools.py: Tool implementations with consistent error handling
   - shadow_mode.py: Shadow mode logging infrastructure
   
2. **Modern Python 3.13+ Features**:
   - Pattern matching for risk level determination
   - Structural type hints with TypedDict
   - Dataclasses for structured data
   - Enum with properties
   
3. **Production-Grade Patterns**:
   - Circuit breakers for external calls
   - Retry logic with exponential backoff
   - Consistent error handling decorators
   - Graceful degradation
   
4. **Improved Testability**:
   - Each module can be tested independently
   - Clear interfaces and responsibilities
   - Fixture-friendly design
   - Type safety throughout
   
5. **Better Documentation**:
   - Clear docstrings for all public functions
   - Type hints provide inline documentation
   - Examples show usage patterns
   - Comments explain design decisions
"""