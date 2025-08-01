"""
Metrics and Observability Validation Suite for AURA Intelligence.

This module validates that all operational metrics, dashboards, and
observability features are functioning correctly in production conditions.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from structlog import get_logger
from prometheus_client import REGISTRY, Counter, Histogram, Gauge
from prometheus_client.parser import text_string_to_metric_families
import aiohttp

from ..observability.metrics import (
    # Event metrics
    event_processing_duration,
    event_processing_errors,
    event_store_size,
    event_replay_progress,
    
    # Projection metrics
    projection_lag,
    projection_errors,
    projection_processing_time,
    projection_health,
    
    # Debate metrics
    active_debates,
    debate_duration,
    debate_consensus_rate,
    agent_response_time,
    
    # System metrics
    system_health_score,
    component_health,
    
    # Chaos metrics
    chaos_experiments_run,
    chaos_experiments_failed,
    chaos_injection_active
)

logger = get_logger(__name__)

# Validation metrics
metrics_validation_total = Counter(
    "metrics_validation_total",
    "Total metrics validation attempts",
    ["category", "status"]
)

metrics_accuracy = Gauge(
    "metrics_accuracy_percent",
    "Accuracy of metrics compared to actual values",
    ["metric_name"]
)

dashboard_response_time = Histogram(
    "dashboard_response_time_seconds",
    "Time taken for dashboard queries to respond",
    ["dashboard"]
)


class MetricsValidator:
    """Validates all operational metrics and dashboards"""
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        grafana_url: str = "http://localhost:3000",
        grafana_api_key: Optional[str] = None
    ):
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key
        self.validation_results: Dict[str, Any] = {}
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Execute all metrics validation scenarios"""
        logger.info("Starting metrics validation suite")
        
        validations = [
            ("prometheus_metrics", self.validate_prometheus_metrics),
            ("grafana_dashboards", self.validate_grafana_dashboards),
            ("metric_accuracy", self.validate_metric_accuracy),
            ("alert_rules", self.validate_alert_rules),
            ("metric_cardinality", self.validate_metric_cardinality),
            ("query_performance", self.validate_query_performance)
        ]
        
        for name, validator in validations:
            try:
                result = await validator()
                self.validation_results[name] = result
                
                metrics_validation_total.labels(
                    category=name,
                    status="success" if result.get("status") == "passed" else "failure"
                ).inc()
                
            except Exception as e:
                logger.error(
                    "Metrics validation failed",
                    validation=name,
                    error=str(e)
                )
                self.validation_results[name] = {
                    "status": "failed",
                    "error": str(e)
                }
                
                metrics_validation_total.labels(
                    category=name,
                    status="error"
                ).inc()
        
        # Calculate overall status
        total_validations = len(validations)
        passed_validations = sum(
            1 for r in self.validation_results.values()
            if r.get("status") == "passed"
        )
        
        self.validation_results["summary"] = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "success_rate": (passed_validations / total_validations * 100) if total_validations > 0 else 0,
            "overall_status": "passed" if passed_validations == total_validations else "failed"
        }
        
        return self.validation_results
    
    async def validate_prometheus_metrics(self) -> Dict[str, Any]:
        """Validate all Prometheus metrics are being collected"""
        logger.info("Validating Prometheus metrics")
        
        # Expected metrics grouped by category
        expected_metrics = {
            "event_processing": [
                "event_processing_duration_seconds",
                "event_processing_errors_total",
                "event_store_size_bytes",
                "event_replay_progress_ratio"
            ],
            "projections": [
                "projection_lag_seconds",
                "projection_errors_total",
                "projection_processing_time_seconds",
                "projection_health_status"
            ],
            "debates": [
                "active_debates_gauge",
                "debate_duration_seconds",
                "debate_consensus_rate",
                "agent_response_time_seconds"
            ],
            "system": [
                "system_health_score",
                "component_health_status"
            ],
            "chaos": [
                "chaos_experiments_run_total",
                "chaos_experiments_failed_total",
                "chaos_injection_active"
            ]
        }
        
        missing_metrics = []
        found_metrics = []
        metric_samples = {}
        
        async with aiohttp.ClientSession() as session:
            # Query Prometheus for current metrics
            async with session.get(f"{self.prometheus_url}/api/v1/label/__name__/values") as resp:
                if resp.status != 200:
                    return {
                        "status": "failed",
                        "error": f"Failed to query Prometheus: {resp.status}"
                    }
                
                data = await resp.json()
                available_metrics = set(data.get("data", []))
            
            # Check each expected metric
            for category, metrics in expected_metrics.items():
                for metric in metrics:
                    if metric in available_metrics:
                        found_metrics.append(metric)
                        
                        # Get sample value
                        query_url = f"{self.prometheus_url}/api/v1/query"
                        params = {"query": metric}
                        
                        async with session.get(query_url, params=params) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                if result.get("data", {}).get("result"):
                                    metric_samples[metric] = result["data"]["result"][0]
                    else:
                        missing_metrics.append(f"{category}/{metric}")
        
        # Calculate completeness
        total_expected = sum(len(metrics) for metrics in expected_metrics.values())
        completeness = (len(found_metrics) / total_expected * 100) if total_expected > 0 else 0
        
        return {
            "status": "passed" if len(missing_metrics) == 0 else "failed",
            "total_expected_metrics": total_expected,
            "found_metrics": len(found_metrics),
            "missing_metrics": missing_metrics,
            "completeness_percent": completeness,
            "metric_samples": metric_samples
        }
    
    async def validate_grafana_dashboards(self) -> Dict[str, Any]:
        """Validate Grafana dashboards are accessible and functional"""
        logger.info("Validating Grafana dashboards")
        
        expected_dashboards = [
            "aura-system-overview",
            "event-processing",
            "debate-analytics",
            "agent-performance",
            "chaos-engineering"
        ]
        
        dashboard_results = []
        headers = {}
        if self.grafana_api_key:
            headers["Authorization"] = f"Bearer {self.grafana_api_key}"
        
        async with aiohttp.ClientSession() as session:
            # Get list of dashboards
            async with session.get(
                f"{self.grafana_url}/api/search",
                headers=headers
            ) as resp:
                if resp.status != 200:
                    return {
                        "status": "failed",
                        "error": f"Failed to query Grafana: {resp.status}"
                    }
                
                dashboards = await resp.json()
                available_uids = {d.get("uid", ""): d for d in dashboards}
            
            # Check each expected dashboard
            for expected_uid in expected_dashboards:
                start_time = time.time()
                
                if expected_uid in available_uids:
                    # Test dashboard loading
                    dashboard_url = f"{self.grafana_url}/api/dashboards/uid/{expected_uid}"
                    
                    async with session.get(dashboard_url, headers=headers) as resp:
                        load_time = time.time() - start_time
                        dashboard_response_time.labels(dashboard=expected_uid).observe(load_time)
                        
                        if resp.status == 200:
                            dashboard_data = await resp.json()
                            panel_count = len(
                                dashboard_data.get("dashboard", {}).get("panels", [])
                            )
                            
                            dashboard_results.append({
                                "uid": expected_uid,
                                "status": "found",
                                "load_time": load_time,
                                "panel_count": panel_count,
                                "title": dashboard_data.get("dashboard", {}).get("title", "")
                            })
                        else:
                            dashboard_results.append({
                                "uid": expected_uid,
                                "status": "error",
                                "error": f"HTTP {resp.status}"
                            })
                else:
                    dashboard_results.append({
                        "uid": expected_uid,
                        "status": "missing"
                    })
        
        # Calculate success rate
        found_dashboards = sum(
            1 for d in dashboard_results if d.get("status") == "found"
        )
        success_rate = (
            found_dashboards / len(expected_dashboards) * 100
            if expected_dashboards else 0
        )
        
        return {
            "status": "passed" if found_dashboards == len(expected_dashboards) else "failed",
            "expected_dashboards": len(expected_dashboards),
            "found_dashboards": found_dashboards,
            "success_rate": success_rate,
            "dashboard_results": dashboard_results,
            "average_load_time": sum(
                d.get("load_time", 0) for d in dashboard_results if d.get("load_time")
            ) / found_dashboards if found_dashboards > 0 else 0
        }
    
    async def validate_metric_accuracy(self) -> Dict[str, Any]:
        """Validate metrics are accurate by comparing with actual values"""
        logger.info("Validating metric accuracy")
        
        accuracy_tests = []
        
        # Test 1: Event counter accuracy
        # Generate known number of events and verify counter
        test_event_count = 100
        initial_count = await self._get_metric_value("event_processing_duration_seconds_count")
        
        # Simulate events (in real implementation, actually generate events)
        for _ in range(test_event_count):
            event_processing_duration.labels(
                event_type="test_event",
                status="success"
            ).observe(0.1)
        
        # Wait for metric to update
        await asyncio.sleep(2)
        
        final_count = await self._get_metric_value("event_processing_duration_seconds_count")
        actual_increment = final_count - initial_count if final_count and initial_count else 0
        
        accuracy = (actual_increment / test_event_count * 100) if test_event_count > 0 else 0
        metrics_accuracy.labels(metric_name="event_counter").set(accuracy)
        
        accuracy_tests.append({
            "test": "event_counter_accuracy",
            "expected": test_event_count,
            "actual": actual_increment,
            "accuracy_percent": accuracy,
            "passed": accuracy > 95
        })
        
        # Test 2: Gauge accuracy
        # Set known gauge value and verify
        test_gauge_value = 42
        active_debates.set(test_gauge_value)
        
        await asyncio.sleep(1)
        
        measured_value = await self._get_metric_value("active_debates_gauge")
        gauge_accuracy = (
            100 - abs(measured_value - test_gauge_value) / test_gauge_value * 100
            if measured_value and test_gauge_value > 0 else 0
        )
        
        metrics_accuracy.labels(metric_name="gauge").set(gauge_accuracy)
        
        accuracy_tests.append({
            "test": "gauge_accuracy",
            "expected": test_gauge_value,
            "actual": measured_value,
            "accuracy_percent": gauge_accuracy,
            "passed": gauge_accuracy > 99
        })
        
        # Test 3: Histogram accuracy
        # Record known values and verify percentiles
        test_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for value in test_values:
            debate_duration.observe(value)
        
        await asyncio.sleep(1)
        
        p50_value = await self._get_metric_value(
            'debate_duration_seconds{quantile="0.5"}'
        )
        expected_p50 = 0.3  # Median of test values
        
        histogram_accuracy = (
            100 - abs(p50_value - expected_p50) / expected_p50 * 100
            if p50_value and expected_p50 > 0 else 0
        )
        
        metrics_accuracy.labels(metric_name="histogram").set(histogram_accuracy)
        
        accuracy_tests.append({
            "test": "histogram_accuracy",
            "expected_p50": expected_p50,
            "actual_p50": p50_value,
            "accuracy_percent": histogram_accuracy,
            "passed": histogram_accuracy > 90
        })
        
        # Overall accuracy
        all_passed = all(test.get("passed", False) for test in accuracy_tests)
        avg_accuracy = (
            sum(test.get("accuracy_percent", 0) for test in accuracy_tests) /
            len(accuracy_tests) if accuracy_tests else 0
        )
        
        return {
            "status": "passed" if all_passed else "failed",
            "accuracy_tests": accuracy_tests,
            "average_accuracy": avg_accuracy,
            "all_tests_passed": all_passed
        }
    
    async def validate_alert_rules(self) -> Dict[str, Any]:
        """Validate Prometheus alert rules are configured correctly"""
        logger.info("Validating alert rules")
        
        expected_alerts = [
            {
                "name": "HighEventProcessingErrors",
                "severity": "critical",
                "for": "5m"
            },
            {
                "name": "ProjectionLagHigh",
                "severity": "warning",
                "for": "10m"
            },
            {
                "name": "LowDebateConsensusRate",
                "severity": "warning",
                "for": "15m"
            },
            {
                "name": "SystemHealthDegraded",
                "severity": "critical",
                "for": "5m"
            },
            {
                "name": "ChaosExperimentFailed",
                "severity": "info",
                "for": "1m"
            }
        ]
        
        alert_results = []
        
        async with aiohttp.ClientSession() as session:
            # Get configured rules
            async with session.get(f"{self.prometheus_url}/api/v1/rules") as resp:
                if resp.status != 200:
                    return {
                        "status": "failed",
                        "error": f"Failed to query alert rules: {resp.status}"
                    }
                
                data = await resp.json()
                groups = data.get("data", {}).get("groups", [])
                
                # Flatten all rules
                all_rules = {}
                for group in groups:
                    for rule in group.get("rules", []):
                        if rule.get("type") == "alerting":
                            all_rules[rule.get("name")] = rule
            
            # Check each expected alert
            for expected in expected_alerts:
                alert_name = expected["name"]
                
                if alert_name in all_rules:
                    rule = all_rules[alert_name]
                    
                    # Validate configuration
                    configured_for = rule.get("for", "")
                    configured_severity = rule.get("labels", {}).get("severity", "")
                    
                    alert_results.append({
                        "name": alert_name,
                        "status": "configured",
                        "expected_severity": expected["severity"],
                        "actual_severity": configured_severity,
                        "expected_for": expected["for"],
                        "actual_for": configured_for,
                        "matches": (
                            configured_severity == expected["severity"] and
                            configured_for == expected["for"]
                        )
                    })
                else:
                    alert_results.append({
                        "name": alert_name,
                        "status": "missing"
                    })
            
            # Check for any firing alerts
            async with session.get(f"{self.prometheus_url}/api/v1/alerts") as resp:
                if resp.status == 200:
                    alerts_data = await resp.json()
                    firing_alerts = [
                        alert for alert in alerts_data.get("data", {}).get("alerts", [])
                        if alert.get("state") == "firing"
                    ]
                else:
                    firing_alerts = []
        
        # Calculate results
        configured_correctly = sum(
            1 for alert in alert_results
            if alert.get("status") == "configured" and alert.get("matches", False)
        )
        
        return {
            "status": "passed" if configured_correctly == len(expected_alerts) else "failed",
            "expected_alerts": len(expected_alerts),
            "configured_correctly": configured_correctly,
            "alert_results": alert_results,
            "firing_alerts": len(firing_alerts),
            "firing_alert_names": [a.get("labels", {}).get("alertname") for a in firing_alerts]
        }
    
    async def validate_metric_cardinality(self) -> Dict[str, Any]:
        """Validate metric cardinality is within acceptable limits"""
        logger.info("Validating metric cardinality")
        
        cardinality_limits = {
            "event_processing_duration_seconds": 1000,
            "projection_lag_seconds": 100,
            "debate_duration_seconds": 500,
            "agent_response_time_seconds": 200
        }
        
        cardinality_results = []
        total_series = 0
        
        async with aiohttp.ClientSession() as session:
            for metric, limit in cardinality_limits.items():
                # Query for series count
                query = f"count by (__name__) (group by (__name__, {{{metric}}}){{{metric}}})"
                params = {"query": query}
                
                async with session.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params=params
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("data", {}).get("result", [])
                        
                        if results:
                            series_count = int(float(results[0].get("value", [0, "0"])[1]))
                        else:
                            series_count = 0
                        
                        total_series += series_count
                        
                        cardinality_results.append({
                            "metric": metric,
                            "limit": limit,
                            "actual": series_count,
                            "within_limit": series_count <= limit,
                            "usage_percent": (series_count / limit * 100) if limit > 0 else 0
                        })
        
        # Check total cardinality
        total_limit = 10000  # Example total limit
        
        all_within_limits = all(r.get("within_limit", False) for r in cardinality_results)
        total_within_limit = total_series <= total_limit
        
        return {
            "status": "passed" if all_within_limits and total_within_limit else "failed",
            "cardinality_results": cardinality_results,
            "total_series": total_series,
            "total_limit": total_limit,
            "total_usage_percent": (total_series / total_limit * 100) if total_limit > 0 else 0,
            "all_within_limits": all_within_limits
        }
    
    async def validate_query_performance(self) -> Dict[str, Any]:
        """Validate metric query performance"""
        logger.info("Validating query performance")
        
        test_queries = [
            {
                "name": "instant_query",
                "query": "up",
                "expected_latency": 0.1
            },
            {
                "name": "simple_aggregation",
                "query": "sum(rate(event_processing_duration_seconds_count[5m]))",
                "expected_latency": 0.5
            },
            {
                "name": "complex_aggregation",
                "query": """
                    sum by (event_type) (
                        rate(event_processing_duration_seconds_count[5m])
                    ) / 
                    sum by (event_type) (
                        rate(event_processing_duration_seconds_sum[5m])
                    )
                """,
                "expected_latency": 1.0
            },
            {
                "name": "range_query",
                "query": "active_debates_gauge[1h]",
                "expected_latency": 2.0,
                "is_range": True
            }
        ]
        
        query_results = []
        
        async with aiohttp.ClientSession() as session:
            for test in test_queries:
                start_time = time.time()
                
                if test.get("is_range"):
                    url = f"{self.prometheus_url}/api/v1/query_range"
                    params = {
                        "query": test["query"],
                        "start": int((datetime.utcnow() - timedelta(hours=1)).timestamp()),
                        "end": int(datetime.utcnow().timestamp()),
                        "step": "60s"
                    }
                else:
                    url = f"{self.prometheus_url}/api/v1/query"
                    params = {"query": test["query"]}
                
                async with session.get(url, params=params) as resp:
                    latency = time.time() - start_time
                    
                    if resp.status == 200:
                        data = await resp.json()
                        result_count = len(data.get("data", {}).get("result", []))
                        
                        query_results.append({
                            "name": test["name"],
                            "status": "success",
                            "latency": latency,
                            "expected_latency": test["expected_latency"],
                            "within_sla": latency <= test["expected_latency"],
                            "result_count": result_count
                        })
                    else:
                        query_results.append({
                            "name": test["name"],
                            "status": "failed",
                            "error": f"HTTP {resp.status}",
                            "latency": latency
                        })
        
        # Calculate performance metrics
        successful_queries = [q for q in query_results if q.get("status") == "success"]
        all_within_sla = all(q.get("within_sla", False) for q in successful_queries)
        avg_latency = (
            sum(q.get("latency", 0) for q in successful_queries) / len(successful_queries)
            if successful_queries else 0
        )
        
        return {
            "status": "passed" if all_within_sla and len(successful_queries) == len(test_queries) else "failed",
            "query_results": query_results,
            "successful_queries": len(successful_queries),
            "total_queries": len(test_queries),
            "all_within_sla": all_within_sla,
            "average_latency": avg_latency
        }
    
    async def _get_metric_value(self, metric_query: str) -> Optional[float]:
        """Helper to get current value of a metric"""
        async with aiohttp.ClientSession() as session:
            params = {"query": metric_query}
            async with session.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("data", {}).get("result", [])
                    if results and len(results) > 0:
                        # Return the first result's value
                        value = results[0].get("value", [0, "0"])[1]
                        try:
                            return float(value)
                        except ValueError:
                            return None
        return None


async def main():
    """Run metrics validation suite"""
    # Create validator
    validator = MetricsValidator(
        prometheus_url="http://localhost:9090",
        grafana_url="http://localhost:3000"
    )
    
    # Run validations
    results = await validator.run_all_validations()
    
    # Print results
    print("\n=== Metrics Validation Results ===\n")
    
    summary = results.pop("summary", {})
    
    for validation, result in results.items():
        print(f"{validation}: {result['status'].upper()}")
        if result['status'] != 'passed':
            print(f"  Details: {json.dumps(result, indent=2)}")
        print()
    
    print("\n=== Overall Summary ===")
    print(f"Total Validations: {summary.get('total_validations', 0)}")
    print(f"Passed: {summary.get('passed_validations', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    print(f"Overall Status: {summary.get('overall_status', 'N/A').upper()}")


if __name__ == "__main__":
    asyncio.run(main())