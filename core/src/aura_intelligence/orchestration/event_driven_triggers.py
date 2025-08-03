"""
Event-Driven Workflow Triggers for AURA Intelligence

This module implements concrete triggers that automatically connect
TDA-detected anomalies to agent voting and adaptive behavior in workflows.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import structlog

from langgraph.graph import StateGraph, END
from ..events.bus import EventBus, Event
from ..tda.algorithms import TDAResult
from ..agents.council.production_lnn_council import ProductionLNNCouncilAgent
from ..orchestration.langgraph_workflows import AURACollectiveIntelligence
from ..feature_flags import FeatureFlags

logger = structlog.get_logger(__name__)


@dataclass
class TriggerRule:
    """Rule for triggering workflows based on events."""
    name: str
    event_pattern: str
    condition: Callable[[Event], bool]
    action: str
    priority: int = 0
    enabled: bool = True


class EventDrivenOrchestrator:
    """
    Orchestrates event-driven workflows connecting TDA anomalies
    to agent decisions and adaptive behaviors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator with configuration."""
        self.config = config
        self.event_bus = EventBus()
        self.feature_flags = FeatureFlags(config.get("feature_source"))
        
        # Initialize components
        self.collective_intelligence = AURACollectiveIntelligence()
        self.lnn_council = ProductionLNNCouncilAgent(config.get("council", {}))
        
        # Trigger rules
        self.trigger_rules: List[TriggerRule] = []
        self._setup_default_triggers()
        
        # Active workflows
        self.active_workflows: Dict[str, Any] = {}
        
        # Metrics
        self.triggers_fired = 0
        self.workflows_started = 0
        
        logger.info("Event-driven orchestrator initialized")
        
    def _setup_default_triggers(self):
        """Set up default trigger rules."""
        # High anomaly trigger
        self.add_trigger(TriggerRule(
            name="high_anomaly_trigger",
            event_pattern="tda.anomaly_detected",
            condition=lambda e: e.data.get("anomaly_score", 0) > 0.8,
            action="urgent_investigation",
            priority=10
        ))
        
        # Cascade detection trigger
        self.add_trigger(TriggerRule(
            name="cascade_detection_trigger",
            event_pattern="tda.cascade_detected",
            condition=lambda e: e.data.get("cascade_probability", 0) > 0.7,
            action="cascade_mitigation",
            priority=20
        ))
        
        # Performance degradation trigger
        self.add_trigger(TriggerRule(
            name="performance_trigger",
            event_pattern="monitoring.performance_degraded",
            condition=lambda e: e.data.get("latency_increase", 0) > 2.0,
            action="performance_optimization",
            priority=5
        ))
        
        # Agent consensus failure trigger
        self.add_trigger(TriggerRule(
            name="consensus_failure_trigger",
            event_pattern="agent.consensus_failed",
            condition=lambda e: e.data.get("retry_count", 0) > 3,
            action="human_escalation",
            priority=15
        ))
        
    def add_trigger(self, rule: TriggerRule):
        """Add a trigger rule."""
        self.trigger_rules.append(rule)
        self.trigger_rules.sort(key=lambda r: r.priority, reverse=True)
        
    async def start(self):
        """Start the orchestrator and subscribe to events."""
        # Subscribe to all event patterns
        patterns = set(rule.event_pattern for rule in self.trigger_rules)
        
        for pattern in patterns:
            await self.event_bus.subscribe(pattern, self._handle_event)
            
        logger.info(f"Subscribed to {len(patterns)} event patterns")
        
    async def _handle_event(self, event: Event):
        """Handle incoming events and check triggers."""
        logger.info(f"Received event: {event.type}", event_data=event.data)
        
        # Check all matching triggers
        matching_triggers = []
        
        for rule in self.trigger_rules:
            if not rule.enabled:
                continue
                
            # Check pattern match
            if self._matches_pattern(event.type, rule.event_pattern):
                # Check condition
                if rule.condition(event):
                    matching_triggers.append(rule)
                    
        # Execute triggers by priority
        for trigger in matching_triggers:
            await self._execute_trigger(trigger, event)
            
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports wildcards)."""
        if pattern.endswith("*"):
            return event_type.startswith(pattern[:-1])
        return event_type == pattern
        
    async def _execute_trigger(self, trigger: TriggerRule, event: Event):
        """Execute a trigger action."""
        logger.info(f"Executing trigger: {trigger.name}", action=trigger.action)
        
        self.triggers_fired += 1
        
        # Create workflow based on action
        if trigger.action == "urgent_investigation":
            await self._start_urgent_investigation_workflow(event)
        elif trigger.action == "cascade_mitigation":
            await self._start_cascade_mitigation_workflow(event)
        elif trigger.action == "performance_optimization":
            await self._start_performance_optimization_workflow(event)
        elif trigger.action == "human_escalation":
            await self._start_human_escalation_workflow(event)
        else:
            logger.warning(f"Unknown trigger action: {trigger.action}")
            
    async def _start_urgent_investigation_workflow(self, event: Event):
        """Start urgent investigation workflow for high anomalies."""
        workflow_id = f"urgent_investigation_{datetime.utcnow().timestamp()}"
        
        logger.info(f"Starting urgent investigation workflow: {workflow_id}")
        
        # Create workflow state
        initial_state = {
            "workflow_id": workflow_id,
            "trigger_event": event.data,
            "anomaly_score": event.data.get("anomaly_score"),
            "tda_result_id": event.data.get("tda_result_id"),
            "status": "investigating",
            "start_time": datetime.utcnow()
        }
        
        # Build investigation workflow
        workflow = self._build_investigation_workflow()
        
        # Start workflow asynchronously
        self.active_workflows[workflow_id] = workflow
        self.workflows_started += 1
        
        # Execute workflow
        result = await workflow.ainvoke(initial_state)
        
        # Handle result
        await self._handle_workflow_result(workflow_id, result)
        
    def _build_investigation_workflow(self) -> StateGraph:
        """Build investigation workflow graph."""
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("analyze_anomaly", self._analyze_anomaly_node)
        workflow.add_node("gather_context", self._gather_context_node)
        workflow.add_node("agent_deliberation", self._agent_deliberation_node)
        workflow.add_node("execute_decision", self._execute_decision_node)
        workflow.add_node("monitor_outcome", self._monitor_outcome_node)
        
        # Define flow
        workflow.set_entry_point("analyze_anomaly")
        workflow.add_edge("analyze_anomaly", "gather_context")
        workflow.add_edge("gather_context", "agent_deliberation")
        
        # Conditional routing based on decision
        workflow.add_conditional_edges(
            "agent_deliberation",
            self._route_by_decision,
            {
                "mitigate": "execute_decision",
                "escalate": "execute_decision",
                "monitor": "monitor_outcome",
                "end": END
            }
        )
        
        workflow.add_edge("execute_decision", "monitor_outcome")
        workflow.add_edge("monitor_outcome", END)
        
        return workflow.compile()
        
    async def _analyze_anomaly_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the anomaly in detail."""
        logger.info("Analyzing anomaly", workflow_id=state["workflow_id"])
        
        # Fetch detailed TDA results
        tda_result_id = state.get("tda_result_id")
        
        # In production, fetch from storage
        # tda_details = await self.storage.get_tda_result(tda_result_id)
        
        # For demo, simulate analysis
        state["analysis"] = {
            "severity": "high" if state["anomaly_score"] > 0.9 else "medium",
            "affected_dimensions": [0, 1, 2],
            "persistence_change": 2.5,
            "similar_past_events": 3
        }
        
        return state
        
    async def _gather_context_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Gather context from various sources."""
        logger.info("Gathering context", workflow_id=state["workflow_id"])
        
        # Simulate context gathering
        state["context"] = {
            "system_load": 0.85,
            "recent_changes": ["deployment_123", "config_update_456"],
            "affected_services": ["api", "database"],
            "user_impact": "high"
        }
        
        # Publish context gathered event
        await self.event_bus.publish(Event(
            type="workflow.context_gathered",
            data={
                "workflow_id": state["workflow_id"],
                "context": state["context"]
            },
            source="investigation_workflow"
        ))
        
        return state
        
    async def _agent_deliberation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent council deliberation."""
        logger.info("Starting agent deliberation", workflow_id=state["workflow_id"])
        
        # Prepare task for council
        council_task = {
            "type": "anomaly_investigation",
            "analysis": state["analysis"],
            "context": state["context"],
            "urgency": "high"
        }
        
        # Get council decision
        decision = await self.lnn_council.deliberate(council_task)
        
        # Update state
        state["council_decision"] = decision
        state["decision_type"] = decision.get("vote", "monitor")
        state["confidence"] = decision.get("confidence", 0.5)
        
        # Publish decision event
        await self.event_bus.publish(Event(
            type="agent.decision_made",
            data={
                "workflow_id": state["workflow_id"],
                "decision": state["decision_type"],
                "confidence": state["confidence"]
            },
            source="lnn_council"
        ))
        
        return state
        
    async def _execute_decision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the council's decision."""
        logger.info(
            f"Executing decision: {state['decision_type']}", 
            workflow_id=state["workflow_id"]
        )
        
        decision_type = state["decision_type"]
        
        if decision_type == "mitigate":
            # Execute mitigation actions
            actions = [
                {"type": "scale_resources", "target": "compute", "factor": 2.0},
                {"type": "enable_circuit_breaker", "service": "api"},
                {"type": "notify_oncall", "severity": "high"}
            ]
            
        elif decision_type == "escalate":
            # Escalate to human
            actions = [
                {"type": "page_oncall", "priority": "P1"},
                {"type": "create_incident", "severity": "critical"},
                {"type": "gather_diagnostics", "full": True}
            ]
            
        else:
            actions = []
            
        state["executed_actions"] = actions
        
        # Execute actions (simulated)
        for action in actions:
            logger.info(f"Executing action: {action['type']}", action=action)
            # In production, execute real actions
            await asyncio.sleep(0.1)  # Simulate execution
            
        return state
        
    async def _monitor_outcome_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the outcome of actions."""
        logger.info("Monitoring outcome", workflow_id=state["workflow_id"])
        
        # Simulate monitoring
        await asyncio.sleep(1)
        
        # Check if anomaly resolved
        # In production, re-run TDA or check metrics
        resolved = state.get("anomaly_score", 1.0) < 0.5
        
        state["outcome"] = {
            "resolved": resolved,
            "duration": (datetime.utcnow() - state["start_time"]).total_seconds(),
            "actions_effective": resolved
        }
        
        # Publish outcome event
        await self.event_bus.publish(Event(
            type="workflow.completed",
            data={
                "workflow_id": state["workflow_id"],
                "outcome": state["outcome"]
            },
            source="investigation_workflow"
        ))
        
        return state
        
    def _route_by_decision(self, state: Dict[str, Any]) -> str:
        """Route based on agent decision."""
        decision = state.get("decision_type", "monitor")
        
        if decision in ["mitigate", "escalate"]:
            return decision
        elif decision == "monitor":
            return "monitor"
        else:
            return "end"
            
    async def _start_cascade_mitigation_workflow(self, event: Event):
        """Start cascade mitigation workflow."""
        workflow_id = f"cascade_mitigation_{datetime.utcnow().timestamp()}"
        
        logger.info(f"Starting cascade mitigation workflow: {workflow_id}")
        
        # Build specialized cascade workflow
        workflow = self._build_cascade_workflow()
        
        initial_state = {
            "workflow_id": workflow_id,
            "cascade_probability": event.data.get("cascade_probability"),
            "affected_components": event.data.get("affected_components", []),
            "status": "mitigating"
        }
        
        # Execute with urgency
        result = await workflow.ainvoke(initial_state)
        
        await self._handle_workflow_result(workflow_id, result)
        
    def _build_cascade_workflow(self) -> StateGraph:
        """Build cascade mitigation workflow."""
        workflow = StateGraph(dict)
        
        # Cascade-specific nodes
        workflow.add_node("isolate_components", self._isolate_components_node)
        workflow.add_node("redirect_traffic", self._redirect_traffic_node)
        workflow.add_node("scale_healthy", self._scale_healthy_node)
        workflow.add_node("verify_stability", self._verify_stability_node)
        
        # Linear flow for cascade mitigation
        workflow.set_entry_point("isolate_components")
        workflow.add_edge("isolate_components", "redirect_traffic")
        workflow.add_edge("redirect_traffic", "scale_healthy")
        workflow.add_edge("scale_healthy", "verify_stability")
        workflow.add_edge("verify_stability", END)
        
        return workflow.compile()
        
    async def _isolate_components_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate failing components."""
        components = state.get("affected_components", [])
        
        state["isolated"] = []
        for component in components:
            logger.info(f"Isolating component: {component}")
            # In production, actually isolate
            state["isolated"].append(component)
            
        return state
        
    async def _redirect_traffic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Redirect traffic away from isolated components."""
        state["traffic_redirected"] = True
        logger.info("Traffic redirected to healthy components")
        return state
        
    async def _scale_healthy_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Scale up healthy components."""
        state["scaled_components"] = ["healthy_1", "healthy_2"]
        logger.info("Scaled healthy components")
        return state
        
    async def _verify_stability_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify system stability."""
        state["stable"] = True
        state["cascade_mitigated"] = True
        logger.info("System stability verified")
        return state
        
    async def _start_performance_optimization_workflow(self, event: Event):
        """Start performance optimization workflow."""
        logger.info("Starting performance optimization workflow")
        
        # Use feature flags to select optimization strategy
        use_auto_tuning = await self.feature_flags.is_enabled("performance.auto_tuning")
        
        if use_auto_tuning:
            # Automatic optimization
            await self._auto_optimize_performance(event)
        else:
            # Manual optimization with agent input
            await self._manual_optimize_performance(event)
            
    async def _auto_optimize_performance(self, event: Event):
        """Automatically optimize performance."""
        optimizations = [
            "increase_cache_size",
            "enable_connection_pooling",
            "optimize_query_patterns",
            "adjust_gc_settings"
        ]
        
        for opt in optimizations:
            logger.info(f"Applying optimization: {opt}")
            await asyncio.sleep(0.1)
            
    async def _manual_optimize_performance(self, event: Event):
        """Manual optimization with agent recommendations."""
        # Get agent recommendations
        recommendations = await self.collective_intelligence.get_performance_recommendations()
        
        for rec in recommendations:
            logger.info(f"Agent recommendation: {rec}")
            
    async def _start_human_escalation_workflow(self, event: Event):
        """Start human escalation workflow."""
        logger.info("Starting human escalation workflow")
        
        # Create incident
        incident = {
            "id": f"incident_{datetime.utcnow().timestamp()}",
            "severity": "P1",
            "title": "Agent consensus failure - human intervention required",
            "context": event.data
        }
        
        # Notify humans
        await self._notify_humans(incident)
        
        # Gather diagnostics
        diagnostics = await self._gather_full_diagnostics()
        
        # Create runbook
        runbook = self._generate_runbook(incident, diagnostics)
        
        logger.info("Human escalation complete", incident_id=incident["id"])
        
    async def _notify_humans(self, incident: Dict[str, Any]):
        """Notify humans about incident."""
        channels = ["slack", "pagerduty", "email"]
        
        for channel in channels:
            logger.info(f"Notifying via {channel}", incident_id=incident["id"])
            # In production, send real notifications
            
    async def _gather_full_diagnostics(self) -> Dict[str, Any]:
        """Gather comprehensive diagnostics."""
        return {
            "system_state": "degraded",
            "active_workflows": len(self.active_workflows),
            "recent_events": [],  # Would fetch from event store
            "metrics_snapshot": {}  # Would fetch from monitoring
        }
        
    def _generate_runbook(self, incident: Dict[str, Any], diagnostics: Dict[str, Any]) -> str:
        """Generate runbook for incident."""
        return f"""
        Incident Runbook: {incident['id']}
        
        Severity: {incident['severity']}
        Title: {incident['title']}
        
        Current State:
        - System: {diagnostics['system_state']}
        - Active Workflows: {diagnostics['active_workflows']}
        
        Recommended Actions:
        1. Check agent logs for consensus failure details
        2. Review recent TDA anomaly scores
        3. Verify external service connectivity
        4. Consider manual override of agent decisions
        
        Escalation Path:
        - L1: On-call engineer
        - L2: Senior architect
        - L3: VP Engineering
        """
        
    async def _handle_workflow_result(self, workflow_id: str, result: Dict[str, Any]):
        """Handle workflow completion."""
        logger.info(
            f"Workflow completed: {workflow_id}",
            outcome=result.get("outcome", "unknown")
        )
        
        # Clean up
        self.active_workflows.pop(workflow_id, None)
        
        # Publish completion event
        await self.event_bus.publish(Event(
            type="orchestrator.workflow_completed",
            data={
                "workflow_id": workflow_id,
                "result": result
            },
            source="event_driven_orchestrator"
        ))
        
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "triggers_configured": len(self.trigger_rules),
            "triggers_fired": self.triggers_fired,
            "workflows_started": self.workflows_started,
            "active_workflows": len(self.active_workflows),
            "workflow_ids": list(self.active_workflows.keys())
        }


# Example usage
async def demonstrate_event_triggers():
    """Demonstrate event-driven triggers."""
    config = {
        "council": {"agent_id": "demo_council"},
        "feature_source": "memory"
    }
    
    # Create orchestrator
    orchestrator = EventDrivenOrchestrator(config)
    
    # Start orchestrator
    await orchestrator.start()
    
    # Simulate TDA anomaly event
    anomaly_event = Event(
        type="tda.anomaly_detected",
        data={
            "anomaly_score": 0.92,
            "algorithm": "neural_surveillance",
            "tda_result_id": "tda_123",
            "dimensions_affected": [0, 1, 2]
        },
        source="tda_engine"
    )
    
    # Publish event - will trigger workflow
    await orchestrator.event_bus.publish(anomaly_event)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check status
    status = await orchestrator.get_status()
    print(f"Orchestrator status: {status}")


if __name__ == "__main__":
    asyncio.run(demonstrate_event_triggers())