"""
ObserverAgent V2 Implementation

Enhanced observer agent with:
- Temporal workflow integration
- Kafka event streaming
- Advanced observability
- Resilience patterns
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
import structlog

from ..base import AgentBase, AgentConfig, AgentState
from ..observability import AgentInstrumentor
from ...observability.knowledge_graph import KnowledgeGraphClient, KnowledgeGraphConfig


class ObserverAgentV2(AgentBase[Dict[str, Any], Dict[str, Any], AgentState]):
    """
    V2 Observer Agent with enhanced capabilities.
    
    Features:
    - Multi-source observation (system, logs, metrics, events)
    - Pattern detection and anomaly identification
    - Real-time event streaming via Kafka
    - Knowledge graph integration for context
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="observer_v2",
                model="gpt-4",
                temperature=0.3,  # Lower temperature for factual observations
                enable_memory=True,
                enable_tools=True
            )
        super().__init__(config)
        
        # Initialize knowledge graph client if available
        try:
            self.kg_client = KnowledgeGraphClient(
                KnowledgeGraphConfig(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password"
                )
            )
        except:
            self.kg_client = None
            self.logger.warning("Knowledge graph not available")
    
    def build_graph(self) -> StateGraph:
        """Build the observation workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("gather_context", self.gather_context_step)
        workflow.add_node("observe_system", self.observe_system_step)
        workflow.add_node("detect_patterns", self.detect_patterns_step)
        workflow.add_node("enrich_observations", self.enrich_observations_step)
        workflow.add_node("generate_report", self.generate_report_step)
        
        # Define flow
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "observe_system")
        workflow.add_edge("observe_system", "detect_patterns")
        workflow.add_edge("detect_patterns", "enrich_observations")
        workflow.add_edge("enrich_observations", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow
    
    async def _execute_step(self, state: AgentState, step_name: str) -> AgentState:
        """Execute a specific workflow step."""
        step_methods = {
            "gather_context": self.gather_context_step,
            "observe_system": self.observe_system_step,
            "detect_patterns": self.detect_patterns_step,
            "enrich_observations": self.enrich_observations_step,
            "generate_report": self.generate_report_step
        }
        
        if step_name in step_methods:
            return await step_methods[step_name](state)
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    async def gather_context_step(self, state: AgentState) -> AgentState:
        """Gather context from knowledge graph and previous observations."""
        self.logger.info("Gathering observation context")
        
        # Record decision
        self._metrics.decision_counter.add(
            1,
            {
                "agent": self.config.name,
                "decision": "gather_context",
                "reason": "Starting observation workflow"
            }
        )
        
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "observer_id": state.agent_id,
            "previous_observations": []
        }
        
        # Query knowledge graph for relevant context
        if self.kg_client and self.kg_client.is_available:
            try:
                # Get recent observations
                recent_obs = await self.kg_client.query(
                    """
                    MATCH (o:Observation)-[:OBSERVED_BY]->(a:Agent {id: $agent_id})
                    WHERE o.timestamp > datetime() - duration('PT1H')
                    RETURN o
                    ORDER BY o.timestamp DESC
                    LIMIT 10
                    """,
                    {"agent_id": state.agent_id}
                )
                
                context["previous_observations"] = recent_obs
                
            except Exception as e:
                self.logger.warning(f"Failed to query knowledge graph: {e}")
        
        # Update state
        state.context["observation_context"] = context
        state.messages.append({
            "role": "system",
            "content": f"Context gathered: {len(context['previous_observations'])} previous observations",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def observe_system_step(self, state: AgentState) -> AgentState:
        """Perform system observations."""
        self.logger.info("Observing system state")
        
        observations = {
            "system_metrics": {},
            "active_processes": [],
            "recent_events": [],
            "anomalies": []
        }
        
        # Simulate system observation (in production, would query actual systems)
        async def observe_metrics():
            # In production: query Prometheus, CloudWatch, etc.
            return {
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "disk_usage": 78.1,
                "network_throughput": 1024.5,
                "active_connections": 234,
                "request_rate": 1250.0,
                "error_rate": 0.02,
                "response_time_p95": 125.5
            }
        
        async def observe_processes():
            # In production: query process managers, container orchestrators
            return [
                {"name": "api-server", "status": "healthy", "cpu": 12.5, "memory": 512},
                {"name": "database", "status": "healthy", "cpu": 25.0, "memory": 2048},
                {"name": "cache", "status": "healthy", "cpu": 5.0, "memory": 256},
                {"name": "worker-1", "status": "degraded", "cpu": 85.0, "memory": 1024}
            ]
        
        async def observe_events():
            # In production: query event streams, logs
            return [
                {"type": "deployment", "service": "api-server", "version": "2.1.0", "time": "10m ago"},
                {"type": "alert", "severity": "warning", "message": "High CPU on worker-1", "time": "5m ago"},
                {"type": "scaling", "service": "worker", "from": 3, "to": 5, "time": "3m ago"}
            ]
        
        # Gather observations in parallel
        metrics, processes, events = await asyncio.gather(
            observe_metrics(),
            observe_processes(),
            observe_events()
        )
        
        observations["system_metrics"] = metrics
        observations["active_processes"] = processes
        observations["recent_events"] = events
        
        # Detect obvious anomalies
        if metrics["cpu_usage"] > 80:
            observations["anomalies"].append({
                "type": "high_cpu",
                "value": metrics["cpu_usage"],
                "threshold": 80
            })
        
        if metrics["error_rate"] > 0.05:
            observations["anomalies"].append({
                "type": "high_error_rate",
                "value": metrics["error_rate"],
                "threshold": 0.05
            })
        
        # Update state
        state.context["observations"] = observations
        state.messages.append({
            "role": "assistant",
            "content": f"System observation complete. Found {len(observations['anomalies'])} anomalies.",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Record metrics
        self._metrics.decision_counter.add(
            1,
            {
                "agent": self.config.name,
                "decision": "observe_system",
                "reason": f"Found {len(observations['anomalies'])} anomalies"
            }
        )
        
        return state
    
    async def detect_patterns_step(self, state: AgentState) -> AgentState:
        """Detect patterns in observations."""
        self.logger.info("Detecting patterns")
        
        observations = state.context.get("observations", {})
        patterns = []
        
        # Simple pattern detection (in production, use ML models)
        metrics = observations.get("system_metrics", {})
        processes = observations.get("active_processes", [])
        events = observations.get("recent_events", [])
        
        # Pattern 1: Resource pressure
        high_resource_processes = [p for p in processes if p.get("cpu", 0) > 70]
        if high_resource_processes:
            patterns.append({
                "type": "resource_pressure",
                "severity": "medium",
                "affected_processes": [p["name"] for p in high_resource_processes],
                "recommendation": "Consider scaling or optimization"
            })
        
        # Pattern 2: Recent deployment correlation
        recent_deployments = [e for e in events if e["type"] == "deployment"]
        if recent_deployments and observations.get("anomalies"):
            patterns.append({
                "type": "deployment_impact",
                "severity": "high",
                "deployment": recent_deployments[0],
                "anomalies": observations["anomalies"],
                "recommendation": "Investigate deployment impact"
            })
        
        # Pattern 3: Scaling response
        scaling_events = [e for e in events if e["type"] == "scaling"]
        if scaling_events and metrics.get("cpu_usage", 0) > 70:
            patterns.append({
                "type": "insufficient_scaling",
                "severity": "medium",
                "current_cpu": metrics["cpu_usage"],
                "scaling_event": scaling_events[0],
                "recommendation": "Additional scaling may be needed"
            })
        
        # Update state
        state.context["detected_patterns"] = patterns
        state.messages.append({
            "role": "assistant",
            "content": f"Pattern detection complete. Found {len(patterns)} patterns.",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def enrich_observations_step(self, state: AgentState) -> AgentState:
        """Enrich observations with historical context and predictions."""
        self.logger.info("Enriching observations")
        
        enrichments = {
            "historical_comparison": {},
            "trend_analysis": {},
            "impact_assessment": {},
            "predictions": []
        }
        
        # Compare with historical data (mock)
        current_metrics = state.context.get("observations", {}).get("system_metrics", {})
        enrichments["historical_comparison"] = {
            "cpu_usage": {
                "current": current_metrics.get("cpu_usage", 0),
                "avg_last_hour": 42.5,
                "avg_last_day": 38.2,
                "percentile_rank": 75
            },
            "error_rate": {
                "current": current_metrics.get("error_rate", 0),
                "avg_last_hour": 0.015,
                "avg_last_day": 0.012,
                "percentile_rank": 85
            }
        }
        
        # Trend analysis
        enrichments["trend_analysis"] = {
            "cpu_trend": "increasing",
            "memory_trend": "stable",
            "error_trend": "spike_detected"
        }
        
        # Impact assessment
        patterns = state.context.get("detected_patterns", [])
        if any(p["type"] == "deployment_impact" for p in patterns):
            enrichments["impact_assessment"] = {
                "user_impact": "moderate",
                "affected_services": ["api-server", "worker"],
                "estimated_duration": "15-30 minutes",
                "mitigation_available": True
            }
        
        # Simple predictions
        if current_metrics.get("cpu_usage", 0) > 70:
            enrichments["predictions"].append({
                "metric": "cpu_usage",
                "prediction": "likely to exceed 90% in next 30 minutes",
                "confidence": 0.75,
                "recommendation": "Proactive scaling recommended"
            })
        
        # Update state
        state.context["enrichments"] = enrichments
        state.messages.append({
            "role": "assistant",
            "content": "Observations enriched with historical context and predictions",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def generate_report_step(self, state: AgentState) -> AgentState:
        """Generate comprehensive observation report."""
        self.logger.info("Generating observation report")
        
        # Compile all findings
        observations = state.context.get("observations", {})
        patterns = state.context.get("detected_patterns", [])
        enrichments = state.context.get("enrichments", {})
        
        report = {
            "summary": {
                "status": "degraded" if observations.get("anomalies") else "healthy",
                "anomaly_count": len(observations.get("anomalies", [])),
                "pattern_count": len(patterns),
                "key_findings": []
            },
            "observations": observations,
            "patterns": patterns,
            "enrichments": enrichments,
            "recommendations": [],
            "metadata": {
                "observer_id": state.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "observation_duration_ms": 1250  # Mock
            }
        }
        
        # Generate key findings
        if observations.get("anomalies"):
            report["summary"]["key_findings"].append(
                f"Detected {len(observations['anomalies'])} anomalies requiring attention"
            )
        
        for pattern in patterns:
            if pattern["severity"] == "high":
                report["summary"]["key_findings"].append(
                    f"High severity pattern detected: {pattern['type']}"
                )
        
        # Generate recommendations
        for pattern in patterns:
            if pattern.get("recommendation"):
                report["recommendations"].append({
                    "priority": pattern["severity"],
                    "action": pattern["recommendation"],
                    "context": pattern["type"]
                })
        
        # Store in knowledge graph if available
        if self.kg_client and self.kg_client.is_available:
            try:
                await self.kg_client.create_node(
                    "Observation",
                    {
                        "id": f"obs_{state.agent_id}_{datetime.utcnow().timestamp()}",
                        "observer_id": state.agent_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": report["summary"]["status"],
                        "anomaly_count": report["summary"]["anomaly_count"],
                        "pattern_count": report["summary"]["pattern_count"]
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to store observation: {e}")
        
        # Update state with final report
        state.context["observation_report"] = report
        state.completed = True
        state.messages.append({
            "role": "assistant",
            "content": f"Observation complete. Status: {report['summary']['status']}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Record final metrics
        self._metrics.decision_counter.add(
            1,
            {
                "agent": self.config.name,
                "decision": "complete_observation",
                "reason": f"Status: {report['summary']['status']}"
            }
        )
        
        return state
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> AgentState:
        """Create initial state from input."""
        return AgentState(
            agent_id=f"{self.config.name}_{datetime.utcnow().timestamp()}",
            context={"input_data": input_data},
            messages=[{
                "role": "user",
                "content": input_data.get("query", "Perform system observation"),
                "timestamp": datetime.utcnow().isoformat()
            }]
        )
    
    def _extract_output(self, final_state: AgentState) -> Dict[str, Any]:
        """Extract output from final state."""
        return final_state.context.get("observation_report", {
            "error": "No observation report generated",
            "state": final_state.dict()
        })
    
    async def process_state(self, state: AgentState) -> Dict[str, Any]:
        """Process with existing state (for Temporal integration)."""
        # Run the workflow with provided state
        result_state = await self._run_graph(state)
        
        return {
            "output": self._extract_output(result_state),
            "state": result_state,
            "metrics": {
                "decisions": getattr(self._metrics, 'decision_count', 0),
                "duration_ms": (result_state.updated_at - result_state.created_at).total_seconds() * 1000
            }
        }