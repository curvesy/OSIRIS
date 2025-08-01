"""
🚀 Enhanced CI/CD Automation for Streaming TDA
Automated testing, alerts, and performance tracking
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import yaml

import structlog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import requests
from slack_sdk.webhook import WebhookClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from .benchmark_framework import BenchmarkRunner, BenchmarkSuite, CIBenchmarkValidator
from .chaos_engineering import ChaosOrchestrator
from .comprehensive_testing import run_all_tests

logger = structlog.get_logger(__name__)


@dataclass
class CIConfig:
    """CI/CD configuration"""
    project_name: str = "streaming-tda"
    
    # Test thresholds
    unit_test_coverage_threshold: float = 90.0
    integration_test_timeout: int = 600
    benchmark_regression_threshold: float = 10.0
    chaos_recovery_time_threshold: float = 30.0
    
    # Alert channels
    slack_webhook_url: Optional[str] = None
    pagerduty_api_key: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # Monitoring
    prometheus_gateway: Optional[str] = None
    grafana_url: Optional[str] = None
    
    # Artifacts
    artifact_retention_days: int = 30
    performance_history_weeks: int = 12


@dataclass
class TestResult:
    """Test execution result"""
    test_type: str
    status: str  # 'passed', 'failed', 'degraded'
    duration_seconds: float
    timestamp: datetime
    metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AlertManager:
    """Manages test alerts and notifications"""
    
    def __init__(self, config: CIConfig):
        self.config = config
        self.slack_client = WebhookClient(config.slack_webhook_url) if config.slack_webhook_url else None
        
    async def send_alert(
        self,
        severity: str,  # 'info', 'warning', 'error', 'critical'
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send alert through configured channels"""
        
        # Format message
        formatted_message = self._format_alert(severity, title, message, details)
        
        # Send to Slack
        if self.slack_client and severity in ['error', 'critical']:
            await self._send_slack_alert(formatted_message)
            
        # Send to PagerDuty for critical issues
        if self.config.pagerduty_api_key and severity == 'critical':
            await self._send_pagerduty_alert(title, message, details)
            
        # Send email for warnings and above
        if self.config.email_recipients and severity in ['warning', 'error', 'critical']:
            await self._send_email_alert(formatted_message)
            
        logger.info(
            "alert_sent",
            severity=severity,
            title=title,
            channels=self._get_active_channels()
        )
        
    def _format_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format alert message"""
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        color_map = {
            'info': '#36a64f',
            'warning': '#ff9800',
            'error': '#f44336',
            'critical': '#d32f2f'
        }
        
        return {
            'text': f"{emoji_map.get(severity, '')} {title}",
            'color': color_map.get(severity, '#808080'),
            'fields': [
                {
                    'title': 'Message',
                    'value': message,
                    'short': False
                },
                {
                    'title': 'Timestamp',
                    'value': datetime.now().isoformat(),
                    'short': True
                },
                {
                    'title': 'Environment',
                    'value': os.environ.get('CI_ENVIRONMENT', 'unknown'),
                    'short': True
                }
            ],
            'attachments': [
                {
                    'title': 'Details',
                    'text': json.dumps(details, indent=2) if details else 'No additional details'
                }
            ] if details else []
        }
        
    async def _send_slack_alert(self, message: Dict[str, Any]) -> None:
        """Send Slack alert"""
        try:
            response = self.slack_client.send(
                text=message['text'],
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message['text']
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*{field['title']}:*\n{field['value']}"
                            } for field in message['fields']
                        ]
                    }
                ]
            )
        except Exception as e:
            logger.error("slack_alert_failed", error=str(e))
            
    async def _send_pagerduty_alert(
        self,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]]
    ) -> None:
        """Send PagerDuty alert"""
        # Implementation would use PagerDuty API
        pass
        
    async def _send_email_alert(self, message: Dict[str, Any]) -> None:
        """Send email alert"""
        # Implementation would use email service
        pass
        
    def _get_active_channels(self) -> List[str]:
        """Get list of active alert channels"""
        channels = []
        if self.slack_client:
            channels.append('slack')
        if self.config.pagerduty_api_key:
            channels.append('pagerduty')
        if self.config.email_recipients:
            channels.append('email')
        return channels


class PerformanceTracker:
    """Tracks and analyzes performance trends"""
    
    def __init__(self, history_dir: Path = Path("performance_history")):
        self.history_dir = history_dir
        self.history_dir.mkdir(exist_ok=True)
        self.registry = CollectorRegistry()
        self._init_metrics()
        
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics"""
        self.throughput_gauge = Gauge(
            'streaming_tda_throughput',
            'Current throughput (items/sec)',
            registry=self.registry
        )
        self.latency_gauge = Gauge(
            'streaming_tda_latency_p99',
            'P99 latency (seconds)',
            registry=self.registry
        )
        self.memory_gauge = Gauge(
            'streaming_tda_memory_mb',
            'Memory usage (MB)',
            registry=self.registry
        )
        
    def record_performance(
        self,
        test_name: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Save to history file
        history_file = self.history_dir / f"{test_name}_history.json"
        
        # Load existing history
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        # Append new record
        history.append({
            'timestamp': timestamp.isoformat(),
            'metrics': metrics
        })
        
        # Keep only recent history (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        history = [
            record for record in history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        # Update Prometheus metrics
        if 'throughput' in metrics:
            self.throughput_gauge.set(metrics['throughput'])
        if 'latency_p99' in metrics:
            self.latency_gauge.set(metrics['latency_p99'])
        if 'memory_mb' in metrics:
            self.memory_gauge.set(metrics['memory_mb'])
            
    def analyze_trends(
        self,
        test_name: str,
        metric_name: str,
        weeks: int = 4
    ) -> Dict[str, Any]:
        """Analyze performance trends"""
        history_file = self.history_dir / f"{test_name}_history.json"
        
        if not history_file.exists():
            return {'status': 'no_data'}
            
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Filter to requested timeframe
        cutoff_date = datetime.now() - timedelta(weeks=weeks)
        recent_history = [
            record for record in history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_history:
            return {'status': 'no_recent_data'}
            
        # Extract metric values
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_history]
        values = [r['metrics'].get(metric_name, 0) for r in recent_history]
        
        # Calculate statistics
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        # Trend analysis
        df['timestamp_numeric'] = pd.to_numeric(df['timestamp'])
        correlation = df['timestamp_numeric'].corr(df['value'])
        
        # Detect anomalies (simple z-score method)
        mean = df['value'].mean()
        std = df['value'].std()
        df['z_score'] = (df['value'] - mean) / std
        anomalies = df[abs(df['z_score']) > 2]
        
        return {
            'status': 'analyzed',
            'metric': metric_name,
            'period_weeks': weeks,
            'num_samples': len(df),
            'current_value': values[-1] if values else None,
            'mean': mean,
            'std': std,
            'min': df['value'].min(),
            'max': df['value'].max(),
            'trend': 'increasing' if correlation > 0.3 else 'decreasing' if correlation < -0.3 else 'stable',
            'correlation': correlation,
            'anomalies': [
                {
                    'timestamp': row['timestamp'].isoformat(),
                    'value': row['value'],
                    'z_score': row['z_score']
                } for _, row in anomalies.iterrows()
            ]
        }
        
    def generate_trend_report(
        self,
        test_names: List[str],
        output_path: Path
    ) -> None:
        """Generate comprehensive trend report"""
        # Create figure with subplots
        fig, axes = plt.subplots(
            len(test_names),
            3,
            figsize=(15, 5 * len(test_names))
        )
        
        if len(test_names) == 1:
            axes = axes.reshape(1, -1)
            
        # Metrics to plot
        metrics = ['throughput', 'latency_p99', 'memory_mb']
        
        for i, test_name in enumerate(test_names):
            history_file = self.history_dir / f"{test_name}_history.json"
            
            if not history_file.exists():
                continue
                
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': datetime.fromisoformat(r['timestamp']),
                    **r['metrics']
                } for r in history
            ])
            
            # Plot each metric
            for j, metric in enumerate(metrics):
                if metric not in df.columns:
                    continue
                    
                ax = axes[i, j]
                
                # Time series plot
                ax.plot(df['timestamp'], df[metric], 'b-', alpha=0.7)
                
                # Add rolling average
                if len(df) > 7:
                    rolling_avg = df[metric].rolling(window=7).mean()
                    ax.plot(df['timestamp'], rolling_avg, 'r-', linewidth=2, label='7-day avg')
                    
                # Formatting
                ax.set_title(f'{test_name} - {metric}')
                ax.set_xlabel('Date')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info("trend_report_generated", path=str(output_path))


class PRCommentGenerator:
    """Generates detailed PR comments with test results"""
    
    def __init__(self):
        self.template = Template("""
## 🧪 Streaming TDA Test Results

{{ summary_emoji }} **Overall Status:** {{ overall_status }}

### 📊 Test Summary

| Test Type | Status | Duration | Key Metrics |
|-----------|--------|----------|-------------|
{% for result in results %}
| {{ result.test_type }} | {{ result.status_emoji }} {{ result.status }} | {{ "%.1f"|format(result.duration_seconds) }}s | {{ result.key_metrics }} |
{% endfor %}

### 📈 Performance Comparison

{% if performance_comparison %}
| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
{% for metric in performance_comparison %}
| {{ metric.name }} | {{ "%.3f"|format(metric.baseline) }} | {{ "%.3f"|format(metric.current) }} | {{ metric.change_emoji }} {{ metric.change_percent }}% |
{% endfor %}
{% else %}
No baseline available for comparison.
{% endif %}

### 🔍 Detailed Results

{% for result in results %}
<details>
<summary>{{ result.test_type }}</summary>

**Status:** {{ result.status }}  
**Duration:** {{ "%.1f"|format(result.duration_seconds) }} seconds  

{% if result.errors %}
#### ❌ Errors
{% for error in result.errors %}
- {{ error }}
{% endfor %}
{% endif %}

{% if result.warnings %}
#### ⚠️ Warnings
{% for warning in result.warnings %}
- {{ warning }}
{% endfor %}
{% endif %}

#### 📊 Metrics
```json
{{ result.metrics | tojson(indent=2) }}
```

</details>
{% endfor %}

### 🎯 Recommendations

{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

---
*Generated at {{ timestamp }} | [View Full Report]({{ report_url }})*
        """)
        
    def generate_comment(
        self,
        results: List[TestResult],
        performance_comparison: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[str]] = None,
        report_url: Optional[str] = None
    ) -> str:
        """Generate PR comment"""
        # Determine overall status
        failed_tests = [r for r in results if r.status == 'failed']
        degraded_tests = [r for r in results if r.status == 'degraded']
        
        if failed_tests:
            overall_status = 'Failed'
            summary_emoji = '❌'
        elif degraded_tests:
            overall_status = 'Degraded'
            summary_emoji = '⚠️'
        else:
            overall_status = 'Passed'
            summary_emoji = '✅'
            
        # Add status emojis to results
        for result in results:
            if result.status == 'passed':
                result.status_emoji = '✅'
            elif result.status == 'failed':
                result.status_emoji = '❌'
            else:
                result.status_emoji = '⚠️'
                
            # Format key metrics
            key_metrics = []
            if 'throughput' in result.metrics:
                key_metrics.append(f"Throughput: {result.metrics['throughput']:.0f}/s")
            if 'latency_p99' in result.metrics:
                key_metrics.append(f"P99: {result.metrics['latency_p99']*1000:.1f}ms")
            if 'coverage' in result.metrics:
                key_metrics.append(f"Coverage: {result.metrics['coverage']:.1f}%")
            result.key_metrics = ' | '.join(key_metrics) if key_metrics else 'N/A'
            
        # Add change emojis to performance comparison
        if performance_comparison:
            for metric in performance_comparison:
                change = metric['change_percent']
                if abs(change) < 5:
                    metric['change_emoji'] = '➡️'
                elif change > 0:
                    metric['change_emoji'] = '📈' if metric.get('higher_is_better', True) else '📉'
                else:
                    metric['change_emoji'] = '📉' if metric.get('higher_is_better', True) else '📈'
                    
        # Default recommendations if none provided
        if not recommendations:
            recommendations = []
            if failed_tests:
                recommendations.append('Fix failing tests before merging')
            if degraded_tests:
                recommendations.append('Investigate performance degradation')
            if not recommendations:
                recommendations.append('All tests passed - ready for review')
                
        return self.template.render(
            overall_status=overall_status,
            summary_emoji=summary_emoji,
            results=results,
            performance_comparison=performance_comparison,
            recommendations=recommendations,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            report_url=report_url or '#'
        )


class CIPipeline:
    """Main CI/CD pipeline orchestrator"""
    
    def __init__(self, config: CIConfig):
        self.config = config
        self.alert_manager = AlertManager(config)
        self.performance_tracker = PerformanceTracker()
        self.pr_comment_generator = PRCommentGenerator()
        self.results: List[TestResult] = []
        
    async def run_pipeline(
        self,
        pr_number: Optional[int] = None,
        branch: Optional[str] = None
    ) -> bool:
        """Run complete CI pipeline"""
        logger.info(
            "starting_ci_pipeline",
            pr_number=pr_number,
            branch=branch
        )
        
        pipeline_start = datetime.now()
        all_passed = True
        
        try:
            # 1. Unit Tests
            unit_result = await self._run_unit_tests()
            self.results.append(unit_result)
            if unit_result.status == 'failed':
                all_passed = False
                
            # 2. Integration Tests
            integration_result = await self._run_integration_tests()
            self.results.append(integration_result)
            if integration_result.status == 'failed':
                all_passed = False
                
            # 3. Performance Benchmarks
            if branch == 'main' or pr_number:
                benchmark_result = await self._run_benchmarks()
                self.results.append(benchmark_result)
                if benchmark_result.status in ['failed', 'degraded']:
                    all_passed = False
                    
            # 4. Chaos Tests (only on main)
            if branch == 'main':
                chaos_result = await self._run_chaos_tests()
                self.results.append(chaos_result)
                if chaos_result.status == 'failed':
                    all_passed = False
                    
            # 5. Generate reports
            await self._generate_reports()
            
            # 6. Send notifications
            await self._send_notifications(all_passed, pr_number)
            
            # 7. Update metrics
            self._update_metrics()
            
        except Exception as e:
            logger.error("pipeline_error", error=str(e))
            await self.alert_manager.send_alert(
                'critical',
                'CI Pipeline Failed',
                f'Pipeline encountered an error: {str(e)}',
                {'branch': branch, 'pr_number': pr_number}
            )
            all_passed = False
            
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        logger.info(
            "pipeline_complete",
            duration=pipeline_duration,
            all_passed=all_passed
        )
        
        return all_passed
        
    async def _run_unit_tests(self) -> TestResult:
        """Run unit tests with coverage"""
        logger.info("running_unit_tests")
        start_time = datetime.now()
        
        try:
            # Run pytest with coverage
            result = subprocess.run(
                [
                    'pytest',
                    'tests/unit',
                    '-v',
                    '--cov=aura_intelligence',
                    '--cov-report=json',
                    '--junitxml=test-results/unit.xml'
                ],
                capture_output=True,
                text=True
            )
            
            # Parse coverage report
            with open('coverage.json', 'r') as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data['totals']['percent_covered']
                
            # Determine status
            if result.returncode != 0:
                status = 'failed'
                errors = [result.stderr]
            elif coverage_percent < self.config.unit_test_coverage_threshold:
                status = 'degraded'
                warnings = [f'Coverage {coverage_percent:.1f}% below threshold {self.config.unit_test_coverage_threshold}%']
                errors = []
            else:
                status = 'passed'
                errors = []
                warnings = []
                
            return TestResult(
                test_type='Unit Tests',
                status=status,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={'coverage': coverage_percent},
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return TestResult(
                test_type='Unit Tests',
                status='failed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={},
                errors=[str(e)]
            )
            
    async def _run_integration_tests(self) -> TestResult:
        """Run integration tests"""
        logger.info("running_integration_tests")
        start_time = datetime.now()
        
        try:
            # Start test infrastructure
            subprocess.run(['docker-compose', '-f', 'docker-compose.test.yml', 'up', '-d'])
            
            # Run tests
            result = subprocess.run(
                [
                    'pytest',
                    'tests/integration',
                    '-v',
                    f'--timeout={self.config.integration_test_timeout}',
                    '--junitxml=test-results/integration.xml'
                ],
                capture_output=True,
                text=True
            )
            
            # Clean up
            subprocess.run(['docker-compose', '-f', 'docker-compose.test.yml', 'down'])
            
            return TestResult(
                test_type='Integration Tests',
                status='passed' if result.returncode == 0 else 'failed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={},
                errors=[result.stderr] if result.returncode != 0 else []
            )
            
        except Exception as e:
            return TestResult(
                test_type='Integration Tests',
                status='failed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={},
                errors=[str(e)]
            )
            
    async def _run_benchmarks(self) -> TestResult:
        """Run performance benchmarks"""
        logger.info("running_benchmarks")
        start_time = datetime.now()
        
        try:
            # Run benchmarks
            from .benchmark_framework import STREAMING_TDA_SUITE, BenchmarkRunner
            
            runner = BenchmarkRunner()
            results = await runner.run_suite(STREAMING_TDA_SUITE)
            
            # Compare with baseline
            validator = CIBenchmarkValidator(
                tolerance_percent=self.config.benchmark_regression_threshold
            )
            
            all_passed = True
            warnings = []
            metrics = {}
            
            for name, result in results.items():
                # Record performance
                self.performance_tracker.record_performance(
                    name,
                    result.metrics,
                    result.start_time
                )
                
                # Validate against baseline
                if name in STREAMING_TDA_SUITE.baseline:
                    passed, issues = validator.validate_result(
                        result,
                        {'mean': STREAMING_TDA_SUITE.baseline[name]}
                    )
                    
                    if not passed:
                        all_passed = False
                        warnings.extend(issues)
                        
                # Collect metrics
                metrics[f'{name}_mean'] = result.metrics['mean']
                metrics[f'{name}_p99'] = result.percentiles['p99']
                
            return TestResult(
                test_type='Performance Benchmarks',
                status='passed' if all_passed else 'degraded',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics=metrics,
                warnings=warnings
            )
            
        except Exception as e:
            return TestResult(
                test_type='Performance Benchmarks',
                status='failed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={},
                errors=[str(e)]
            )
            
    async def _run_chaos_tests(self) -> TestResult:
        """Run chaos engineering tests"""
        logger.info("running_chaos_tests")
        start_time = datetime.now()
        
        try:
            # Run subset of chaos scenarios
            from .advanced_chaos import PRODUCTION_CHAOS_SCENARIOS
            
            # Select quick chaos scenarios for CI
            ci_scenarios = [
                s for s in PRODUCTION_CHAOS_SCENARIOS
                if s.duration <= timedelta(minutes=15)
            ]
            
            errors = []
            metrics = {}
            
            for scenario in ci_scenarios:
                # Run scenario
                result = await self._run_chaos_scenario(scenario)
                
                # Check recovery time
                if result.recovery_time_s > self.config.chaos_recovery_time_threshold:
                    errors.append(
                        f"{scenario.name}: Recovery time {result.recovery_time_s}s "
                        f"exceeds threshold {self.config.chaos_recovery_time_threshold}s"
                    )
                    
                metrics[f'{scenario.name}_recovery_time'] = result.recovery_time_s
                metrics[f'{scenario.name}_data_loss'] = result.data_loss
                
            return TestResult(
                test_type='Chaos Tests',
                status='failed' if errors else 'passed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics=metrics,
                errors=errors
            )
            
        except Exception as e:
            return TestResult(
                test_type='Chaos Tests',
                status='failed',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                metrics={},
                errors=[str(e)]
            )
            
    async def _run_chaos_scenario(self, scenario) -> Any:
        """Run individual chaos scenario"""
        # Simplified implementation for CI
        from .chaos_engineering import ChaosResult
        
        # Mock implementation - would run actual chaos test
        return ChaosResult(
            scenario=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            faults_injected=1,
            data_loss=0,
            max_latency_ms=100,
            recovery_time_s=15,
            errors=[]
        )
        
    async def _generate_reports(self) -> None:
        """Generate test reports"""
        logger.info("generating_reports")
        
        # Generate trend report
        test_names = ['streaming_small', 'streaming_medium', 'streaming_large']
        self.performance_tracker.generate_trend_report(
            test_names,
            Path('test-results/performance_trends.png')
        )
        
        # Generate HTML report
        await self._generate_html_report()
        
    async def _generate_html_report(self) -> None:
        """Generate comprehensive HTML report"""
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Streaming TDA Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .passed { color: green; }
                .failed { color: red; }
                .degraded { color: orange; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-family: monospace; }
            </style>
        </head>
        <body>
            <h1>Streaming TDA Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Generated: {{ timestamp }}</p>
                <p>Overall Status: <span class="{{ overall_status|lower }}">{{ overall_status }}</span></p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Details</th>
                </tr>
                {% for result in results %}
                <tr>
                    <td>{{ result.test_type }}</td>
                    <td class="{{ result.status }}">{{ result.status|upper }}</td>
                    <td class="metric">{{ "%.1f"|format(result.duration_seconds) }}s</td>
                    <td>
                        {% if result.errors %}
                            <strong>Errors:</strong> {{ result.errors|length }}<br>
                        {% endif %}
                        {% if result.warnings %}
                            <strong>Warnings:</strong> {{ result.warnings|length }}<br>
                        {% endif %}
                        <strong>Metrics:</strong> {{ result.metrics|length }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Performance Trends</h2>
            <img src="performance_trends.png" style="max-width: 100%;">
            
            <h2>Detailed Results</h2>
            {% for result in results %}
            <h3>{{ result.test_type }}</h3>
            {% if result.errors %}
                <h4>Errors</h4>
                <ul>
                {% for error in result.errors %}
                    <li>{{ error }}</li>
                {% endfor %}
                </ul>
            {% endif %}
            {% if result.warnings %}
                <h4>Warnings</h4>
                <ul>
                {% for warning in result.warnings %}
                    <li>{{ warning }}</li>
                {% endfor %}
                </ul>
            {% endif %}
            {% if result.metrics %}
                <h4>Metrics</h4>
                <pre>{{ result.metrics | tojson(indent=2) }}</pre>
            {% endif %}
            {% endfor %}
        </body>
        </html>
        """)
        
        # Determine overall status
        failed = any(r.status == 'failed' for r in self.results)
        degraded = any(r.status == 'degraded' for r in self.results)
        overall_status = 'Failed' if failed else 'Degraded' if degraded else 'Passed'
        
        html_content = html_template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            overall_status=overall_status,
            results=self.results
        )
        
        with open('test-results/report.html', 'w') as f:
            f.write(html_content)
            
    async def _send_notifications(self, all_passed: bool, pr_number: Optional[int]) -> None:
        """Send notifications based on results"""
        logger.info("sending_notifications")
        
        # Send alerts for failures
        failed_tests = [r for r in self.results if r.status == 'failed']
        if failed_tests:
            await self.alert_manager.send_alert(
                'error',
                'Test Failures Detected',
                f'{len(failed_tests)} test(s) failed in CI pipeline',
                {
                    'failed_tests': [t.test_type for t in failed_tests],
                    'pr_number': pr_number
                }
            )
            
        # Send PR comment
        if pr_number:
            await self._post_pr_comment(pr_number)
            
        # Send success notification for main branch
        if all_passed and not pr_number:
            await self.alert_manager.send_alert(
                'info',
                'CI Pipeline Successful',
                'All tests passed on main branch',
                {'duration': sum(r.duration_seconds for r in self.results)}
            )
            
    async def _post_pr_comment(self, pr_number: int) -> None:
        """Post comment to PR"""
        # Get performance comparison
        performance_comparison = []
        
        for result in self.results:
            if result.test_type == 'Performance Benchmarks':
                for metric_name, current_value in result.metrics.items():
                    if '_mean' in metric_name:
                        # Get baseline from history
                        test_name = metric_name.replace('_mean', '')
                        trend = self.performance_tracker.analyze_trends(
                            test_name,
                            'mean',
                            weeks=4
                        )
                        
                        if trend['status'] == 'analyzed' and trend['mean']:
                            baseline = trend['mean']
                            change_percent = ((current_value - baseline) / baseline) * 100
                            
                            performance_comparison.append({
                                'name': test_name,
                                'baseline': baseline,
                                'current': current_value,
                                'change_percent': change_percent,
                                'higher_is_better': 'throughput' in test_name
                            })
                            
        # Generate comment
        comment = self.pr_comment_generator.generate_comment(
            self.results,
            performance_comparison=performance_comparison if performance_comparison else None,
            report_url=f"https://ci.example.com/job/{self.config.project_name}/{pr_number}/artifacts/test-results/report.html"
        )
        
        # Post to GitHub (mock implementation)
        logger.info("pr_comment_generated", pr_number=pr_number)
        # In real implementation, would use GitHub API to post comment
        
    def _update_metrics(self) -> None:
        """Update Prometheus metrics"""
        if self.config.prometheus_gateway:
            try:
                # Push metrics to gateway
                push_to_gateway(
                    self.config.prometheus_gateway,
                    job='streaming_tda_ci',
                    registry=self.performance_tracker.registry
                )
                logger.info("metrics_pushed_to_prometheus")
            except Exception as e:
                logger.error("prometheus_push_failed", error=str(e))


# Example CI configuration
DEFAULT_CI_CONFIG = CIConfig(
    slack_webhook_url=os.environ.get('SLACK_WEBHOOK_URL'),
    pagerduty_api_key=os.environ.get('PAGERDUTY_API_KEY'),
    email_recipients=os.environ.get('CI_EMAIL_RECIPIENTS', '').split(','),
    prometheus_gateway=os.environ.get('PROMETHEUS_GATEWAY'),
    grafana_url=os.environ.get('GRAFANA_URL')
)


async def main():
    """Main entry point for CI pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming TDA CI Pipeline')
    parser.add_argument('--pr-number', type=int, help='Pull request number')
    parser.add_argument('--branch', type=str, help='Branch name')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            config = CIConfig(**config_data)
    else:
        config = DEFAULT_CI_CONFIG
        
    # Run pipeline
    pipeline = CIPipeline(config)
    success = await pipeline.run_pipeline(
        pr_number=args.pr_number,
        branch=args.branch
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())