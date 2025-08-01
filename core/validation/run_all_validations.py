#!/usr/bin/env python3
"""
Master validation runner for AURA Intelligence production validation.

This script orchestrates all validation suites and generates a comprehensive
validation report following the disciplined approach outlined in the
production validation execution plan.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from structlog import get_logger

# Import all validators
from event_replay_validator import EventReplayValidator
from resilience_validator import ResilienceValidator, ProductionTrafficSimulator
from metrics_validator import MetricsValidator

# Import system components
from ..event_store.hardened_store import HardenedNATSEventStore
from ..event_store.robust_projections import ProjectionManager
from ..areopagus.debate_graph import DebateGraph

logger = get_logger(__name__)


class ValidationOrchestrator:
    """Orchestrates all validation suites for production readiness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Any] = {
            "validation_id": str(datetime.utcnow().timestamp()),
            "start_time": datetime.utcnow().isoformat(),
            "config": config
        }
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Execute all validation suites in order"""
        logger.info(
            "Starting AURA Intelligence production validation",
            validation_id=self.results["validation_id"]
        )
        
        try:
            # Initialize shared components
            components = await self._initialize_components()
            
            # Run validation suites
            validation_results = {}
            
            # 1. Event Replay Validation
            logger.info("Starting Event Replay Validation")
            replay_results = await self._run_event_replay_validation(components)
            validation_results["event_replay"] = replay_results
            
            # 2. Resilience Validation
            logger.info("Starting Resilience Validation")
            resilience_results = await self._run_resilience_validation(components)
            validation_results["resilience"] = resilience_results
            
            # 3. Metrics & Observability Validation
            logger.info("Starting Metrics Validation")
            metrics_results = await self._run_metrics_validation()
            validation_results["metrics"] = metrics_results
            
            # 4. Performance Baseline
            logger.info("Establishing Performance Baseline")
            performance_baseline = await self._establish_performance_baseline(components)
            validation_results["performance_baseline"] = performance_baseline
            
            # Compile results
            self.results["validation_results"] = validation_results
            self.results["end_time"] = datetime.utcnow().isoformat()
            self.results["duration_seconds"] = (
                datetime.fromisoformat(self.results["end_time"]) -
                datetime.fromisoformat(self.results["start_time"])
            ).total_seconds()
            
            # Generate summary
            self.results["summary"] = self._generate_summary(validation_results)
            
            # Save results
            await self._save_results()
            
            # Cleanup
            await self._cleanup_components(components)
            
            return self.results
            
        except Exception as e:
            logger.error(
                "Validation orchestration failed",
                error=str(e),
                validation_id=self.results["validation_id"]
            )
            self.results["error"] = str(e)
            self.results["status"] = "failed"
            return self.results
    
    async def _initialize_components(self) -> Dict[str, Any]:
        """Initialize shared system components"""
        logger.info("Initializing system components")
        
        # Event Store
        event_store = HardenedNATSEventStore(
            nats_url=self.config.get("nats_url", "nats://localhost:4222"),
            stream_name="AURA_VALIDATION"
        )
        await event_store.connect()
        
        # Projection Manager
        projections = []  # Would be initialized with actual projections
        projection_manager = ProjectionManager(event_store, projections)
        
        # Debate System
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=self.config.get("llm_model", "gpt-4"),
            temperature=0.7
        )
        debate_system = DebateGraph(llm=llm, event_store=event_store)
        
        return {
            "event_store": event_store,
            "projection_manager": projection_manager,
            "debate_system": debate_system
        }
    
    async def _run_event_replay_validation(
        self,
        components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run event replay validation suite"""
        validator = EventReplayValidator(
            event_store=components["event_store"],
            projection_manager=components["projection_manager"],
            test_event_count=self.config.get("test_event_count", 1000)
        )
        
        results = await validator.run_all_validations()
        
        # Determine pass/fail
        all_passed = all(
            r.get("status") == "passed"
            for r in results.values()
            if isinstance(r, dict) and "status" in r
        )
        
        return {
            "status": "passed" if all_passed else "failed",
            "scenarios": results,
            "summary": {
                "total_scenarios": len(results),
                "passed_scenarios": sum(
                    1 for r in results.values()
                    if isinstance(r, dict) and r.get("status") == "passed"
                ),
                "critical_failures": [
                    k for k, v in results.items()
                    if isinstance(v, dict) and v.get("status") == "failed"
                ]
            }
        }
    
    async def _run_resilience_validation(
        self,
        components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run resilience validation suite"""
        traffic_simulator = ProductionTrafficSimulator(
            components["debate_system"]
        )
        
        validator = ResilienceValidator(
            event_store=components["event_store"],
            debate_system=components["debate_system"],
            traffic_simulator=traffic_simulator
        )
        
        results = await validator.run_all_validations()
        
        return {
            "status": results.get("summary", {}).get("overall_status", "failed"),
            "experiments": results,
            "resilience_score": results.get("summary", {}).get("overall_resilience_score", 0),
            "recommendation": results.get("summary", {}).get("recommendation", "")
        }
    
    async def _run_metrics_validation(self) -> Dict[str, Any]:
        """Run metrics and observability validation"""
        validator = MetricsValidator(
            prometheus_url=self.config.get("prometheus_url", "http://localhost:9090"),
            grafana_url=self.config.get("grafana_url", "http://localhost:3000"),
            grafana_api_key=self.config.get("grafana_api_key")
        )
        
        results = await validator.run_all_validations()
        
        return {
            "status": results.get("summary", {}).get("overall_status", "failed"),
            "validations": results,
            "metrics_completeness": next(
                (r.get("completeness_percent", 0) for k, r in results.items()
                 if k == "prometheus_metrics" and isinstance(r, dict)),
                0
            ),
            "dashboards_available": next(
                (r.get("success_rate", 0) for k, r in results.items()
                 if k == "grafana_dashboards" and isinstance(r, dict)),
                0
            )
        }
    
    async def _establish_performance_baseline(
        self,
        components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Establish performance baseline under production load"""
        logger.info("Establishing performance baseline")
        
        traffic_simulator = ProductionTrafficSimulator(
            components["debate_system"]
        )
        
        # Run different load scenarios
        load_scenarios = [
            {"name": "light_load", "duration": 300, "debates_per_minute": 10},
            {"name": "normal_load", "duration": 300, "debates_per_minute": 30},
            {"name": "peak_load", "duration": 300, "debates_per_minute": 60},
            {"name": "stress_test", "duration": 60, "debates_per_minute": 120}
        ]
        
        baseline_results = []
        
        for scenario in load_scenarios:
            logger.info(
                f"Running {scenario['name']} scenario",
                debates_per_minute=scenario["debates_per_minute"]
            )
            
            metrics = await traffic_simulator.generate_traffic(
                duration_seconds=scenario["duration"],
                debates_per_minute=scenario["debates_per_minute"]
            )
            
            baseline_results.append({
                **scenario,
                "results": metrics
            })
            
            # Brief pause between scenarios
            await asyncio.sleep(10)
        
        # Calculate baseline metrics
        normal_load_metrics = next(
            (r["results"] for r in baseline_results if r["name"] == "normal_load"),
            {}
        )
        
        return {
            "status": "established",
            "scenarios": baseline_results,
            "recommended_baseline": {
                "normal_load_debates_per_minute": 30,
                "expected_success_rate": normal_load_metrics.get("success_rate", 0),
                "expected_average_latency": normal_load_metrics.get("average_latency", 0),
                "peak_capacity_debates_per_minute": 60
            }
        }
    
    def _generate_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary"""
        # Check critical validations
        event_replay_passed = (
            validation_results.get("event_replay", {}).get("status") == "passed"
        )
        resilience_passed = (
            validation_results.get("resilience", {}).get("resilience_score", 0) >= 70
        )
        metrics_passed = (
            validation_results.get("metrics", {}).get("status") == "passed"
        )
        
        all_critical_passed = all([
            event_replay_passed,
            resilience_passed,
            metrics_passed
        ])
        
        # Calculate overall score
        scores = [
            validation_results.get("resilience", {}).get("resilience_score", 0),
            validation_results.get("metrics", {}).get("metrics_completeness", 0),
            100 if event_replay_passed else 0
        ]
        overall_score = sum(scores) / len(scores)
        
        # Determine go/no-go decision
        go_decision = all_critical_passed and overall_score >= 80
        
        return {
            "overall_status": "passed" if go_decision else "failed",
            "overall_score": overall_score,
            "critical_validations": {
                "event_replay": "passed" if event_replay_passed else "failed",
                "resilience": "passed" if resilience_passed else "failed",
                "metrics": "passed" if metrics_passed else "failed"
            },
            "go_no_go_decision": "GO" if go_decision else "NO-GO",
            "recommendations": self._generate_recommendations(
                validation_results,
                go_decision
            )
        }
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        go_decision: bool
    ) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        if not go_decision:
            recommendations.append(
                "⚠️ System is NOT ready for production deployment. "
                "Address critical failures before proceeding."
            )
        
        # Check event replay
        replay_failures = (
            results.get("event_replay", {})
            .get("summary", {})
            .get("critical_failures", [])
        )
        if replay_failures:
            recommendations.append(
                f"🔧 Fix event replay issues in: {', '.join(replay_failures)}"
            )
        
        # Check resilience
        resilience_score = results.get("resilience", {}).get("resilience_score", 0)
        if resilience_score < 90:
            recommendations.append(
                f"📈 Improve resilience score from {resilience_score:.1f} to 90+"
            )
        
        # Check metrics
        metrics_completeness = (
            results.get("metrics", {})
            .get("metrics_completeness", 0)
        )
        if metrics_completeness < 100:
            recommendations.append(
                f"📊 Implement missing metrics (currently {metrics_completeness:.1f}% complete)"
            )
        
        if go_decision:
            recommendations.append(
                "✅ System is ready for production deployment. "
                "Proceed with staged rollout as per deployment plan."
            )
        
        return recommendations
    
    async def _save_results(self) -> None:
        """Save validation results to file"""
        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"validation_{self.results['validation_id']}.json"
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(
            "Validation results saved",
            filepath=str(filepath)
        )
        
        # Also save a summary report
        summary_path = output_dir / f"summary_{self.results['validation_id']}.md"
        await self._generate_markdown_report(summary_path)
    
    async def _generate_markdown_report(self, filepath: Path) -> None:
        """Generate human-readable markdown report"""
        summary = self.results.get("summary", {})
        
        report = f"""# AURA Intelligence Production Validation Report

**Validation ID**: {self.results['validation_id']}  
**Date**: {self.results['start_time']}  
**Duration**: {self.results.get('duration_seconds', 0):.1f} seconds  

## Overall Result: {summary.get('go_no_go_decision', 'NO-GO')}

**Overall Score**: {summary.get('overall_score', 0):.1f}/100

### Critical Validations
- Event Replay: {summary.get('critical_validations', {}).get('event_replay', 'N/A')}
- Resilience: {summary.get('critical_validations', {}).get('resilience', 'N/A')}
- Metrics: {summary.get('critical_validations', {}).get('metrics', 'N/A')}

### Recommendations
"""
        
        for rec in summary.get('recommendations', []):
            report += f"- {rec}\n"
        
        report += """
## Detailed Results

### Event Replay Validation
"""
        replay_summary = (
            self.results.get('validation_results', {})
            .get('event_replay', {})
            .get('summary', {})
        )
        report += f"- Total Scenarios: {replay_summary.get('total_scenarios', 0)}\n"
        report += f"- Passed: {replay_summary.get('passed_scenarios', 0)}\n"
        
        report += """
### Resilience Validation
"""
        resilience = self.results.get('validation_results', {}).get('resilience', {})
        report += f"- Resilience Score: {resilience.get('resilience_score', 0):.1f}/100\n"
        report += f"- Recommendation: {resilience.get('recommendation', 'N/A')}\n"
        
        report += """
### Metrics & Observability
"""
        metrics = self.results.get('validation_results', {}).get('metrics', {})
        report += f"- Metrics Completeness: {metrics.get('metrics_completeness', 0):.1f}%\n"
        report += f"- Dashboards Available: {metrics.get('dashboards_available', 0):.1f}%\n"
        
        with open(filepath, "w") as f:
            f.write(report)
    
    async def _cleanup_components(self, components: Dict[str, Any]) -> None:
        """Clean up system components"""
        if "event_store" in components:
            await components["event_store"].disconnect()


async def main():
    """Run production validation suite"""
    # Load configuration
    config = {
        "nats_url": "nats://localhost:4222",
        "prometheus_url": "http://localhost:9090",
        "grafana_url": "http://localhost:3000",
        "llm_model": "gpt-4",
        "test_event_count": 1000
    }
    
    # Create orchestrator
    orchestrator = ValidationOrchestrator(config)
    
    # Run validations
    print("🚀 Starting AURA Intelligence Production Validation Suite")
    print("=" * 60)
    
    results = await orchestrator.run_all_validations()
    
    # Print summary
    summary = results.get("summary", {})
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION COMPLETE - {summary.get('go_no_go_decision', 'NO-GO')}")
    print("=" * 60)
    print(f"Overall Score: {summary.get('overall_score', 0):.1f}/100")
    print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
    print("\nRecommendations:")
    for rec in summary.get("recommendations", []):
        print(f"  {rec}")
    
    # Exit with appropriate code
    if summary.get("go_no_go_decision") == "GO":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())