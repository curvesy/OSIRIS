"""
Unified Monitoring Dashboard - 2025 Production Standard
Real-time observability for TDA, agents, workflows, and system health
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import structlog
from prometheus_client import Counter, Histogram, Gauge, Info
import aiohttp

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    INFO = "info"


@dataclass
class DashboardPanel:
    """Dashboard panel configuration"""
    id: str
    title: str
    panel_type: str  # graph, table, gauge, heatmap, etc.
    metrics: List[str]
    query: Optional[str] = None
    refresh_interval: int = 10  # seconds
    thresholds: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration"""
    name: str
    condition: str
    severity: str  # critical, warning, info
    channels: List[str]
    cooldown: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedDashboard:
    """
    Unified monitoring dashboard with:
    - Real-time metrics visualization
    - Alert management
    - System health monitoring
    - Performance tracking
    - Anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.panels: Dict[str, DashboardPanel] = {}
        self.alerts: Dict[str, Alert] = {}
        self.metrics_store: Dict[str, Any] = {}
        self._alert_history: Dict[str, datetime] = {}
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Register default panels
        self._register_default_panels()
        
        # Register default alerts
        self._register_default_alerts()
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # TDA Metrics
        self.tda_computations = Counter(
            'tda_computations_total',
            'Total TDA computations',
            ['algorithm', 'status']
        )
        
        self.tda_duration = Histogram(
            'tda_computation_duration_seconds',
            'TDA computation duration',
            ['algorithm'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.tda_anomaly_score = Histogram(
            'tda_anomaly_score',
            'TDA anomaly scores distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Agent Metrics
        self.agent_decisions = Counter(
            'agent_decisions_total',
            'Total agent decisions',
            ['agent_id', 'decision_type']
        )
        
        self.agent_decision_duration = Histogram(
            'agent_decision_duration_seconds',
            'Agent decision duration',
            ['agent_id']
        )
        
        self.agent_confidence = Gauge(
            'agent_decision_confidence',
            'Agent decision confidence score',
            ['agent_id']
        )
        
        # Workflow Metrics
        self.workflow_executions = Counter(
            'workflow_executions_total',
            'Total workflow executions',
            ['status']
        )
        
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        )
        
        self.workflow_state = Gauge(
            'workflow_current_state',
            'Current workflow state',
            ['workflow_id']
        )
        
        # System Metrics
        self.system_health = Gauge(
            'system_health_score',
            'Overall system health score (0-100)'
        )
        
        self.active_agents = Gauge(
            'active_agents_count',
            'Number of active agents'
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage by component',
            ['component']
        )
        
    def _register_default_panels(self):
        """Register default dashboard panels"""
        # Executive Summary Panel
        self.register_panel(DashboardPanel(
            id="exec_summary",
            title="Executive Summary",
            panel_type="stat_row",
            metrics=[
                "system_health_score",
                "active_agents_count",
                "tda_throughput_rate",
                "anomaly_detection_rate"
            ],
            refresh_interval=5
        ))
        
        # TDA Analytics Panel
        self.register_panel(DashboardPanel(
            id="tda_performance",
            title="TDA Algorithm Performance",
            panel_type="stacked_bar",
            metrics=[
                "tda_computation_duration_seconds"
            ],
            query="""
                histogram_quantile(0.95,
                    sum(rate(tda_computation_duration_seconds_bucket[5m])) 
                    by (algorithm, le)
                )
            """,
            refresh_interval=10
        ))
        
        # Agent Performance Panel
        self.register_panel(DashboardPanel(
            id="agent_decisions",
            title="Agent Decision Latency",
            panel_type="box_plot",
            metrics=["agent_decision_duration_seconds"],
            query="""
                histogram_quantile(0.5,
                    sum(rate(agent_decision_duration_seconds_bucket[5m]))
                    by (agent_id, le)
                )
            """
        ))
        
        # Workflow Execution Panel
        self.register_panel(DashboardPanel(
            id="workflow_state",
            title="Workflow State Transitions",
            panel_type="state_diagram",
            metrics=["workflow_current_state"],
            refresh_interval=5
        ))
        
        # System Resources Panel
        self.register_panel(DashboardPanel(
            id="system_resources",
            title="System Resource Utilization",
            panel_type="area_chart",
            metrics=[
                "memory_usage_bytes",
                "cpu_usage_percent",
                "gpu_utilization_percent"
            ],
            thresholds={
                "memory_warning": 0.8,
                "memory_critical": 0.95,
                "cpu_warning": 0.7,
                "cpu_critical": 0.9
            }
        ))
        
        # Anomaly Detection Panel
        self.register_panel(DashboardPanel(
            id="anomaly_trends",
            title="Anomaly Score Trends",
            panel_type="time_series",
            metrics=["tda_anomaly_score"],
            query="""
                histogram_quantile(0.95,
                    sum(rate(tda_anomaly_score_bucket[5m]))
                    by (le)
                )
            """,
            thresholds={
                "warning": 0.7,
                "critical": 0.9
            }
        ))
        
    def _register_default_alerts(self):
        """Register default alerts"""
        # Critical Alerts
        self.register_alert(Alert(
            name="tda_pipeline_failure",
            condition="rate(tda_computations_total{status='failed'}[5m]) > 0.1",
            severity="critical",
            channels=["pagerduty", "slack"],
            cooldown=300
        ))
        
        self.register_alert(Alert(
            name="agent_council_deadlock",
            condition="agent_decision_duration_seconds > 30",
            severity="critical",
            channels=["pagerduty"],
            cooldown=600
        ))
        
        self.register_alert(Alert(
            name="memory_leak_detection",
            condition="deriv(memory_usage_bytes[10m]) > 100000000",  # 100MB/10min
            severity="critical",
            channels=["pagerduty", "email"],
            cooldown=900
        ))
        
        # Warning Alerts
        self.register_alert(Alert(
            name="high_anomaly_rate",
            condition="rate(tda_anomaly_score > 0.7[15m]) > 0.5",
            severity="warning",
            channels=["slack"],
            cooldown=600
        ))
        
        self.register_alert(Alert(
            name="workflow_slowdown",
            condition="histogram_quantile(0.95, workflow_duration_seconds) > 60",
            severity="warning",
            channels=["slack", "email"],
            cooldown=300
        ))
        
    def register_panel(self, panel: DashboardPanel):
        """Register a dashboard panel"""
        self.panels[panel.id] = panel
        logger.info(f"Registered dashboard panel: {panel.id}")
        
    def register_alert(self, alert: Alert):
        """Register an alert"""
        self.alerts[alert.name] = alert
        logger.info(f"Registered alert: {alert.name}")
        
    async def update_metrics(self):
        """Update all dashboard metrics"""
        for panel in self.panels.values():
            try:
                await self._update_panel_metrics(panel)
            except Exception as e:
                logger.error(f"Error updating panel {panel.id}: {e}")
                
    async def _update_panel_metrics(self, panel: DashboardPanel):
        """Update metrics for a specific panel"""
        # Fetch metrics based on panel configuration
        if panel.query:
            # Execute Prometheus query
            result = await self._execute_prometheus_query(panel.query)
            self.metrics_store[panel.id] = result
        else:
            # Fetch individual metrics
            metrics_data = {}
            for metric in panel.metrics:
                value = await self._fetch_metric(metric)
                metrics_data[metric] = value
            self.metrics_store[panel.id] = metrics_data
            
    async def _execute_prometheus_query(self, query: str) -> Any:
        """Execute Prometheus query"""
        # Implementation depends on your Prometheus setup
        # This is a placeholder
        return {"query": query, "result": []}
        
    async def _fetch_metric(self, metric_name: str) -> Any:
        """Fetch individual metric value"""
        # Implementation depends on your metrics backend
        # This is a placeholder
        return {"metric": metric_name, "value": 0}
        
    async def check_alerts(self):
        """Check all alert conditions"""
        for alert in self.alerts.values():
            try:
                if await self._should_fire_alert(alert):
                    await self._fire_alert(alert)
            except Exception as e:
                logger.error(f"Error checking alert {alert.name}: {e}")
                
    async def _should_fire_alert(self, alert: Alert) -> bool:
        """Check if alert should fire"""
        # Check cooldown
        if alert.name in self._alert_history:
            last_fired = self._alert_history[alert.name]
            if (datetime.utcnow() - last_fired).seconds < alert.cooldown:
                return False
                
        # Evaluate condition
        # This would integrate with your metrics backend
        # Placeholder implementation
        return False
        
    async def _fire_alert(self, alert: Alert):
        """Fire an alert"""
        logger.warning(f"Alert fired: {alert.name} - {alert.severity}")
        
        # Record alert
        self._alert_history[alert.name] = datetime.utcnow()
        
        # Send to channels
        for channel in alert.channels:
            await self._send_alert_to_channel(alert, channel)
            
    async def _send_alert_to_channel(self, alert: Alert, channel: str):
        """Send alert to specific channel"""
        if channel == "slack":
            await self._send_slack_alert(alert)
        elif channel == "pagerduty":
            await self._send_pagerduty_alert(alert)
        elif channel == "email":
            await self._send_email_alert(alert)
            
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        webhook_url = self.config.get("slack_webhook_url")
        if not webhook_url:
            return
            
        payload = {
            "text": f"🚨 *{alert.severity.upper()} Alert*: {alert.name}",
            "attachments": [{
                "color": "danger" if alert.severity == "critical" else "warning",
                "fields": [
                    {"title": "Condition", "value": alert.condition},
                    {"title": "Time", "value": datetime.utcnow().isoformat()}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
            
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        # PagerDuty integration
        pass
        
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        # Email integration
        pass
        
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration for frontend"""
        return {
            "panels": [
                {
                    "id": panel.id,
                    "title": panel.title,
                    "type": panel.panel_type,
                    "metrics": panel.metrics,
                    "query": panel.query,
                    "refresh_interval": panel.refresh_interval,
                    "thresholds": panel.thresholds,
                    "data": self.metrics_store.get(panel.id, {})
                }
                for panel in self.panels.values()
            ],
            "alerts": [
                {
                    "name": alert.name,
                    "condition": alert.condition,
                    "severity": alert.severity,
                    "channels": alert.channels,
                    "last_fired": self._alert_history.get(alert.name)
                }
                for alert in self.alerts.values()
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    async def export_grafana_dashboard(self) -> Dict[str, Any]:
        """Export dashboard as Grafana JSON"""
        return {
            "dashboard": {
                "title": "AURA Intelligence - Unified Operations",
                "uid": "aura-ops-main",
                "tags": ["aura", "tda", "agents", "production"],
                "timezone": "utc",
                "refresh": "10s",
                "panels": self._convert_to_grafana_panels(),
                "time": {
                    "from": "now-6h",
                    "to": "now"
                }
            }
        }
        
    def _convert_to_grafana_panels(self) -> List[Dict[str, Any]]:
        """Convert panels to Grafana format"""
        grafana_panels = []
        
        for i, panel in enumerate(self.panels.values()):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": self._map_panel_type(panel.panel_type),
                "gridPos": self._calculate_grid_position(i),
                "targets": self._create_grafana_targets(panel)
            }
            
            if panel.thresholds:
                grafana_panel["thresholds"] = self._convert_thresholds(panel.thresholds)
                
            grafana_panels.append(grafana_panel)
            
        return grafana_panels
        
    def _map_panel_type(self, panel_type: str) -> str:
        """Map internal panel type to Grafana panel type"""
        mapping = {
            "stat_row": "stat",
            "stacked_bar": "graph",
            "box_plot": "graph",
            "state_diagram": "state",
            "area_chart": "graph",
            "time_series": "timeseries"
        }
        return mapping.get(panel_type, "graph")
        
    def _calculate_grid_position(self, index: int) -> Dict[str, int]:
        """Calculate Grafana grid position"""
        cols_per_row = 2
        row = index // cols_per_row
        col = index % cols_per_row
        
        return {
            "h": 8,
            "w": 12,
            "x": col * 12,
            "y": row * 8
        }
        
    def _create_grafana_targets(self, panel: DashboardPanel) -> List[Dict[str, Any]]:
        """Create Grafana query targets"""
        if panel.query:
            return [{
                "expr": panel.query,
                "refId": "A"
            }]
        else:
            return [
                {
                    "expr": metric,
                    "refId": chr(65 + i)  # A, B, C, ...
                }
                for i, metric in enumerate(panel.metrics)
            ]
            
    def _convert_thresholds(self, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Convert thresholds to Grafana format"""
        return [
            {
                "value": value,
                "color": "red" if "critical" in key else "yellow"
            }
            for key, value in thresholds.items()
        ]
        
    async def start_monitoring_loop(self):
        """Start the monitoring loop"""
        while True:
            try:
                # Update metrics
                await self.update_metrics()
                
                # Check alerts
                await self.check_alerts()
                
                # Calculate system health
                await self._update_system_health()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Longer wait on error
                
    async def _update_system_health(self):
        """Calculate and update system health score"""
        # Composite health score based on multiple factors
        factors = {
            "uptime": 1.0,  # Would calculate actual uptime
            "error_rate": 0.95,  # 1 - error_rate
            "resource_usage": 0.8,  # 1 - avg(cpu, memory, gpu)
            "agent_health": 0.9,  # Active agents / total agents
            "workflow_success": 0.85  # Success rate
        }
        
        # Weighted average
        weights = {
            "uptime": 0.2,
            "error_rate": 0.3,
            "resource_usage": 0.2,
            "agent_health": 0.2,
            "workflow_success": 0.1
        }
        
        health_score = sum(
            factors[key] * weights[key] 
            for key in factors
        ) * 100
        
        self.system_health.set(health_score)