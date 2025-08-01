#!/usr/bin/env python3
"""
🚀 AURA Intelligence: Advanced LangGraph Integration
Modern Enterprise AI Workflow with Guardrails & Shadow Mode

Following 2025 best practices from pgdo.md research:
- Modular, functional, object-oriented design
- Low-code configuration-driven approach
- Enterprise guardrails with circuit breakers
- Shadow mode deployment for safe rollout
- Comprehensive observability and monitoring
"""

import asyncio
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# Our proven components
from ..infrastructure.guardrails import EnterpriseGuardrails, GuardrailsConfig, get_guardrails
from ..observability.shadow_mode_logger import ShadowModeLogger, ShadowModeEntry

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 CONFIGURATION & STATE MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeploymentMode(Enum):
    """Deployment modes for safe rollout"""
    SHADOW = "shadow"      # Log predictions, don't enforce
    ACTIVE = "active"      # Enforce predictions
    CANARY = "canary"      # Partial rollout

@dataclass
class AURAIntegrationConfig:
    """Configuration for AURA Intelligence integration"""
    
    # Deployment settings
    deployment_mode: DeploymentMode = DeploymentMode.SHADOW
    shadow_sample_rate: float = 1.0  # Log 100% in shadow mode
    
    # Model settings
    primary_model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Guardrails settings
    enable_guardrails: bool = True
    cost_limit_per_hour: float = 50.0
    rate_limit_per_minute: int = 100
    
    # Validation settings
    enable_validator: bool = True
    validation_threshold: float = 0.7  # >0.7 = execute, 0.4-0.7 = replan, <0.4 = escalate
    
    # Shadow logging settings
    enable_shadow_logging: bool = True
    async_logging: bool = True
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

class AURAState(TypedDict):
    """Enhanced state for AURA Intelligence workflow"""
    
    # Core workflow state
    messages: List[BaseMessage]
    current_task: str
    proposed_action: Optional[Dict[str, Any]]
    
    # Validation state
    validation_result: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    decision_score: Optional[float]
    routing_decision: Optional[str]
    
    # Execution state
    execution_result: Optional[Dict[str, Any]]
    should_execute: bool
    requires_human_approval: bool
    
    # Shadow mode state
    shadow_logged: bool
    shadow_entry_id: Optional[str]
    
    # Monitoring state
    workflow_id: str
    trace_id: str
    performance_metrics: Dict[str, float]
    
    # Error handling
    error_details: Optional[Dict[str, Any]]
    failure_count: int

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧠 ADVANCED WORKFLOW NODES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AURAWorkflowNodes:
    """Enterprise workflow nodes with full integration"""
    
    def __init__(self, config: AURAIntegrationConfig):
        self.config = config
        
        # Initialize components
        self.guardrails = get_guardrails() if config.enable_guardrails else None
        self.shadow_logger = ShadowModeLogger() if config.enable_shadow_logging else None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.primary_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Circuit breaker state
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = 0.0
        
        logger.info(f"🚀 AURA Workflow initialized in {config.deployment_mode.value} mode")
    
    async def supervisor_node(self, state: AURAState) -> AURAState:
        """🎯 Supervisor node - analyzes task and proposes action"""
        
        start_time = time.time()
        
        try:
            # Extract task from messages
            if not state.get("messages"):
                state["error_details"] = {"error": "No messages provided"}
                return state
            
            last_message = state["messages"][-1]
            task = last_message.content if hasattr(last_message, 'content') else str(last_message)
            state["current_task"] = task
            
            # Prepare supervisor prompt
            supervisor_messages = [
                SystemMessage(content=self._get_supervisor_prompt()),
                HumanMessage(content=f"Task: {task}")
            ]
            
            # Execute with guardrails if enabled
            if self.guardrails:
                response = await self.guardrails.secure_ainvoke(
                    self.llm,
                    supervisor_messages,
                    model_name=self.config.primary_model
                )
            else:
                response = await self.llm.ainvoke(supervisor_messages)
            
            # Parse proposed action
            proposed_action = self._parse_supervisor_response(response.content)
            state["proposed_action"] = proposed_action
            
            # Update performance metrics
            state["performance_metrics"] = state.get("performance_metrics", {})
            state["performance_metrics"]["supervisor_latency"] = time.time() - start_time
            
            logger.info(f"🎯 Supervisor completed: {proposed_action.get('type', 'unknown')}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Supervisor failed: {e}")
            state["error_details"] = {"error": str(e), "node": "supervisor"}
            state["failure_count"] = state.get("failure_count", 0) + 1
            return state
    
    async def validator_node(self, state: AURAState) -> AURAState:
        """🛡️ Validator node - assesses risk and makes routing decision"""
        
        start_time = time.time()
        
        try:
            if not self.config.enable_validator:
                # Skip validation, proceed to execution
                state["routing_decision"] = "tools"
                state["should_execute"] = True
                return state
            
            proposed_action = state.get("proposed_action")
            if not proposed_action:
                state["error_details"] = {"error": "No proposed action to validate"}
                return state
            
            # Prepare validation prompt
            validation_messages = [
                SystemMessage(content=self._get_validator_prompt()),
                HumanMessage(content=f"Action: {json.dumps(proposed_action, indent=2)}")
            ]
            
            # Execute validation with guardrails
            if self.guardrails:
                response = await self.guardrails.secure_ainvoke(
                    self.llm,
                    validation_messages,
                    model_name=self.config.primary_model
                )
            else:
                response = await self.llm.ainvoke(validation_messages)
            
            # Parse validation result
            validation_result = self._parse_validation_response(response.content)
            state["validation_result"] = validation_result
            
            # Calculate decision score and routing
            success_prob = validation_result.get("success_probability", 0.5)
            confidence = validation_result.get("confidence_score", 0.8)
            decision_score = success_prob * confidence
            
            state["decision_score"] = decision_score
            
            # Determine routing based on thresholds
            if decision_score > self.config.validation_threshold:
                routing_decision = "tools"
                should_execute = True
            elif decision_score >= 0.4:
                routing_decision = "supervisor"  # Replan
                should_execute = False
            else:
                routing_decision = "error_handler"  # Escalate
                should_execute = False
                state["requires_human_approval"] = True
            
            state["routing_decision"] = routing_decision
            state["should_execute"] = should_execute
            
            # 🌙 Shadow mode logging
            if self.config.enable_shadow_logging and self.shadow_logger:
                await self._log_shadow_prediction(state, validation_result)
            
            # Update performance metrics
            state["performance_metrics"]["validator_latency"] = time.time() - start_time
            
            logger.info(f"🛡️ Validation complete: {routing_decision} (score: {decision_score:.3f})")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Validator failed: {e}")
            state["error_details"] = {"error": str(e), "node": "validator"}
            state["failure_count"] = state.get("failure_count", 0) + 1
            return state
    
    async def tools_node(self, state: AURAState) -> AURAState:
        """⚙️ Tools execution node with outcome recording"""
        
        start_time = time.time()
        
        try:
            proposed_action = state.get("proposed_action", {})
            action_type = proposed_action.get("type", "unknown")
            
            # Simulate tool execution (replace with real tools)
            execution_result = await self._execute_action(proposed_action)
            
            state["execution_result"] = execution_result
            
            # Determine outcome for shadow logging
            outcome = "success" if execution_result.get("success", False) else "failure"
            
            # Record shadow mode outcome
            if self.config.enable_shadow_logging and state.get("shadow_entry_id"):
                await self._record_shadow_outcome(state, outcome, time.time() - start_time)
            
            # Update performance metrics
            state["performance_metrics"]["tools_latency"] = time.time() - start_time
            
            logger.info(f"⚙️ Tools execution complete: {action_type} -> {outcome}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Tools execution failed: {e}")
            
            # Record failure outcome
            if self.config.enable_shadow_logging and state.get("shadow_entry_id"):
                await self._record_shadow_outcome(state, "failure", time.time() - start_time, {"error": str(e)})
            
            state["error_details"] = {"error": str(e), "node": "tools"}
            state["failure_count"] = state.get("failure_count", 0) + 1
            return state
    
    async def _log_shadow_prediction(self, state: AURAState, validation_result: Dict[str, Any]):
        """🌙 Log prediction in shadow mode"""
        
        try:
            if not self.shadow_logger:
                return
            
            # Create shadow mode entry
            entry = ShadowModeEntry(
                workflow_id=state["workflow_id"],
                thread_id=state.get("trace_id", "unknown"),
                timestamp=datetime.now(),
                evidence_log=state.get("messages", []),
                memory_context={},  # TODO: Add memory context
                supervisor_decision=state.get("proposed_action", {}),
                predicted_success_probability=validation_result.get("success_probability", 0.5),
                prediction_confidence_score=validation_result.get("confidence_score", 0.8),
                risk_score=validation_result.get("risk_score", 0.3),
                predicted_risks=validation_result.get("risks", []),
                reasoning_trace=validation_result.get("reasoning", ""),
                requires_human_approval=state.get("requires_human_approval", False),
                routing_decision=state.get("routing_decision", "unknown"),
                decision_score=state.get("decision_score", 0.0)
            )
            
            # Log prediction
            entry_id = await self.shadow_logger.log_prediction(entry)
            state["shadow_entry_id"] = entry_id
            state["shadow_logged"] = True
            
            logger.debug(f"🌙 Shadow prediction logged: {entry_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Shadow logging failed (non-blocking): {e}")
    
    async def _record_shadow_outcome(self, state: AURAState, outcome: str, execution_time: float, error_details: Optional[Dict] = None):
        """🌙 Record actual outcome for shadow analysis"""
        
        try:
            if not self.shadow_logger or not state.get("shadow_entry_id"):
                return
            
            await self.shadow_logger.record_outcome(
                workflow_id=state["workflow_id"],
                actual_outcome=outcome,
                execution_time=execution_time,
                error_details=error_details
            )
            
            logger.debug(f"🌙 Shadow outcome recorded: {outcome}")
            
        except Exception as e:
            logger.warning(f"⚠️ Shadow outcome recording failed (non-blocking): {e}")
    
    def _get_supervisor_prompt(self) -> str:
        """Get supervisor system prompt"""
        return """You are an AI supervisor that analyzes tasks and proposes specific actions.

For each task, respond with a JSON object containing:
{
    "type": "action_type",
    "description": "what this action does",
    "parameters": {"key": "value"},
    "priority": "high|medium|low",
    "estimated_duration": "time estimate"
}

Be specific and actionable in your proposals."""
    
    def _get_validator_prompt(self) -> str:
        """Get validator system prompt"""
        return """You are a professional risk validator that assesses proposed actions.

For each action, respond with a JSON object containing:
{
    "success_probability": 0.85,
    "confidence_score": 0.90,
    "risk_score": 0.15,
    "risks": [{"risk": "description", "mitigation": "how to handle"}],
    "reasoning": "detailed explanation of assessment"
}

Be thorough and conservative in your risk assessment."""
    
    def _parse_supervisor_response(self, content: str) -> Dict[str, Any]:
        """Parse supervisor response"""
        try:
            # Try to extract JSON from response
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to simple parsing
        return {
            "type": "generic_action",
            "description": content[:100],
            "priority": "medium"
        }
    
    def _parse_validation_response(self, content: str) -> Dict[str, Any]:
        """Parse validation response"""
        try:
            # Try to extract JSON from response
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to conservative defaults
        return {
            "success_probability": 0.5,
            "confidence_score": 0.7,
            "risk_score": 0.5,
            "risks": [{"risk": "parsing_failed", "mitigation": "manual_review"}],
            "reasoning": "Failed to parse validation response"
        }
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the proposed action (mock implementation)"""
        
        # Simulate execution based on action type
        action_type = action.get("type", "unknown")
        
        # Mock success rates based on action type
        success_rates = {
            "routine_task": 0.9,
            "data_analysis": 0.85,
            "system_restart": 0.8,
            "configuration_change": 0.7,
            "high_risk_operation": 0.3
        }
        
        success_rate = success_rates.get(action_type, 0.6)
        
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Determine success
        import random
        success = random.random() < success_rate
        
        return {
            "success": success,
            "action_type": action_type,
            "execution_time": 0.1,
            "details": f"Executed {action_type} with {'success' if success else 'failure'}"
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏗️ WORKFLOW BUILDER & ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AURAWorkflowBuilder:
    """🏗️ Builder for AURA Intelligence workflow with enterprise patterns"""

    def __init__(self, config: AURAIntegrationConfig = None):
        self.config = config or AURAIntegrationConfig()
        self.nodes = AURAWorkflowNodes(self.config)

    def build_workflow(self) -> StateGraph:
        """Build the complete AURA Intelligence workflow"""

        # Create state graph
        workflow = StateGraph(AURAState)

        # Add nodes
        workflow.add_node("supervisor", self.nodes.supervisor_node)
        workflow.add_node("validator", self.nodes.validator_node)
        workflow.add_node("tools", self.nodes.tools_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define routing logic
        workflow.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "validator": "validator",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "validator",
            self._route_after_validator,
            {
                "tools": "tools",
                "supervisor": "supervisor",  # Replan
                "error_handler": "error_handler",
                "end": END
            }
        )

        workflow.add_edge("tools", END)
        workflow.add_edge("error_handler", END)

        # Set entry point
        workflow.set_entry_point("supervisor")

        logger.info(f"🏗️ AURA workflow built in {self.config.deployment_mode.value} mode")

        return workflow

    def _route_after_supervisor(self, state: AURAState) -> str:
        """Route after supervisor node"""
        if state.get("error_details"):
            return "error"
        return "validator"

    def _route_after_validator(self, state: AURAState) -> str:
        """Route after validator node based on decision"""

        if state.get("error_details"):
            return "error_handler"

        routing_decision = state.get("routing_decision", "error_handler")

        # In shadow mode, always proceed to tools for data collection
        if self.config.deployment_mode == DeploymentMode.SHADOW:
            if routing_decision in ["tools", "supervisor"]:
                return "tools"
            else:
                return "error_handler"

        # In active mode, respect the routing decision
        elif self.config.deployment_mode == DeploymentMode.ACTIVE:
            return routing_decision

        # Default to error handler
        return "error_handler"

    async def _error_handler_node(self, state: AURAState) -> AURAState:
        """🚨 Error handler node"""

        error_details = state.get("error_details", {})
        failure_count = state.get("failure_count", 0)

        logger.error(f"🚨 Error handler activated: {error_details}")

        # Record error outcome in shadow mode
        if self.config.enable_shadow_logging and state.get("shadow_entry_id"):
            await self.nodes._record_shadow_outcome(
                state,
                "failure",
                0.0,
                error_details
            )

        # Update state
        state["execution_result"] = {
            "success": False,
            "error": error_details,
            "requires_human_intervention": failure_count >= 3
        }

        return state

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 MAIN INTEGRATION CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AURAIntelligenceIntegration:
    """🎯 Main integration class for AURA Intelligence"""

    def __init__(self, config: AURAIntegrationConfig = None):
        self.config = config or AURAIntegrationConfig()
        self.builder = AURAWorkflowBuilder(self.config)
        self.workflow = None
        self.shadow_logger = None

    async def initialize(self):
        """Initialize the AURA Intelligence system"""

        logger.info("🚀 Initializing AURA Intelligence Integration...")

        # Initialize shadow logger
        if self.config.enable_shadow_logging:
            self.shadow_logger = ShadowModeLogger()
            await self.shadow_logger.initialize()

        # Build workflow
        workflow_graph = self.builder.build_workflow()

        # Compile with checkpointer for state persistence
        checkpointer = MemorySaver()
        self.workflow = workflow_graph.compile(checkpointer=checkpointer)

        logger.info("✅ AURA Intelligence Integration initialized successfully")

    async def execute_workflow(self, task: str, thread_id: str = None) -> Dict[str, Any]:
        """Execute a workflow with the given task"""

        if not self.workflow:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        # Generate IDs
        import uuid
        workflow_id = f"aura_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        thread_id = thread_id or f"thread_{uuid.uuid4().hex[:8]}"

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_task": task,
            "workflow_id": workflow_id,
            "trace_id": thread_id,
            "performance_metrics": {},
            "failure_count": 0,
            "shadow_logged": False,
            "should_execute": False,
            "requires_human_approval": False
        }

        # Execute workflow
        start_time = time.time()

        try:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            result = await self.workflow.ainvoke(initial_state, config=config)

            execution_time = time.time() - start_time

            # Add execution metadata
            result["execution_metadata"] = {
                "total_execution_time": execution_time,
                "deployment_mode": self.config.deployment_mode.value,
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Workflow completed: {workflow_id} ({execution_time:.2f}s)")

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(f"❌ Workflow failed: {workflow_id} - {e}")

            return {
                "execution_result": {"success": False, "error": str(e)},
                "execution_metadata": {
                    "total_execution_time": execution_time,
                    "deployment_mode": self.config.deployment_mode.value,
                    "workflow_id": workflow_id,
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat(),
                    "failed": True
                }
            }

    async def get_shadow_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get shadow mode metrics for analysis"""

        if not self.shadow_logger:
            return {"error": "Shadow logging not enabled"}

        return await self.shadow_logger.get_accuracy_metrics(days=days)

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""

        health_status = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "config": {
                "deployment_mode": self.config.deployment_mode.value,
                "guardrails_enabled": self.config.enable_guardrails,
                "shadow_logging_enabled": self.config.enable_shadow_logging
            }
        }

        # Check workflow
        health_status["components"]["workflow"] = self.workflow is not None

        # Check shadow logger
        if self.shadow_logger:
            try:
                metrics = await self.shadow_logger.get_accuracy_metrics(days=1)
                health_status["components"]["shadow_logger"] = True
                health_status["shadow_metrics"] = metrics
            except Exception as e:
                health_status["components"]["shadow_logger"] = False
                health_status["healthy"] = False

        # Check guardrails
        if self.config.enable_guardrails:
            try:
                guardrails = get_guardrails()
                metrics = guardrails.get_metrics()
                health_status["components"]["guardrails"] = True
                health_status["guardrails_metrics"] = metrics
            except Exception as e:
                health_status["components"]["guardrails"] = False
                health_status["healthy"] = False

        return health_status

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def create_aura_intelligence(config: AURAIntegrationConfig = None) -> AURAIntelligenceIntegration:
    """🎯 Create and initialize AURA Intelligence system"""

    integration = AURAIntelligenceIntegration(config)
    await integration.initialize()
    return integration

async def quick_test_workflow(task: str = "Analyze system performance and recommend optimizations") -> Dict[str, Any]:
    """🧪 Quick test of the integrated workflow"""

    config = AURAIntegrationConfig(
        deployment_mode=DeploymentMode.SHADOW,
        enable_guardrails=True,
        enable_shadow_logging=True
    )

    aura = await create_aura_intelligence(config)
    result = await aura.execute_workflow(task)

    return result
