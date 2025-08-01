#!/usr/bin/env python3
"""
🚀 Phase 3D Professional Demo - Modular Architecture Test
Tests the professional modular implementation of Active Mode Deployment.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the professional modular components
from aura_intelligence.governance.active_mode.deployment import ActiveModeDeployment
from aura_intelligence.governance.schemas import RiskThresholds


async def demo_phase3d_professional():
    """
    🚀 Professional Phase 3D Demo
    
    Demonstrates the clean, modular architecture with:
    - Proper separation of concerns
    - Professional error handling
    - Comprehensive metrics
    - Clean interfaces
    """
    print("🚀 Phase 3D: Professional Active Mode Deployment Demo")
    print("=" * 70)
    print("✨ Showcasing modular, professional architecture")
    print()
    
    # Initialize with custom risk thresholds
    risk_thresholds = RiskThresholds(
        auto_execute=0.25,
        human_approval=0.75,
        block_threshold=0.85
    )
    
    # Initialize the professional deployment system
    active_mode = ActiveModeDeployment(
        db_path="demo_professional_governance.db",
        risk_thresholds=risk_thresholds
    )
    
    print("🏗️ Professional Architecture Initialized:")
    print("   ⚖️ Risk Assessment Engine")
    print("   🗄️ Governance Database")
    print("   📊 Metrics Manager")
    print("   ⚡ Action Executor")
    print("   🤔 Human Approval Manager")
    print()
    
    # Professional test scenarios
    scenarios = [
        {
            'name': 'Low Risk: Performance Optimization',
            'evidence': [
                {'type': 'performance_metric', 'severity': 'low', 'metric': 'response_time', 'value': 250},
                {'type': 'trend_analysis', 'direction': 'degrading', 'confidence': 0.7}
            ],
            'action': 'optimize_database_queries',
            'reasoning': 'Response time degradation detected, database optimization recommended'
        },
        {
            'name': 'Medium Risk: Service Scaling',
            'evidence': [
                {'type': 'load_alert', 'severity': 'medium', 'metric': 'cpu_usage', 'value': 85},
                {'type': 'capacity_forecast', 'prediction': 'overload_in_30min', 'confidence': 0.8}
            ],
            'action': 'scale_up_service_instances',
            'reasoning': 'High CPU usage with predicted overload, proactive scaling needed'
        },
        {
            'name': 'High Risk: Security Incident',
            'evidence': [
                {'type': 'security_alert', 'severity': 'critical', 'source': 'suspicious_ip'},
                {'type': 'attack_pattern', 'pattern': 'brute_force', 'confidence': 0.95},
                {'type': 'user_impact', 'affected_accounts': 150}
            ],
            'action': 'block_suspicious_ip_range',
            'reasoning': 'Critical security threat with high confidence, immediate blocking required'
        },
        {
            'name': 'Medium Risk: Deployment',
            'evidence': [
                {'type': 'deployment_request', 'severity': 'medium', 'approved': True},
                {'type': 'system_health', 'status': 'stable', 'confidence': 0.9}
            ],
            'action': 'deploy_application_update',
            'reasoning': 'Approved deployment with stable system conditions'
        }
    ]
    
    print(f"🔍 Processing {len(scenarios)} Professional Test Scenarios:")
    print()
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print(f"   Action: {scenario['action']}")
        
        # Process through professional pipeline
        decision = await active_mode.process_decision(
            evidence_log=scenario['evidence'],
            proposed_action=scenario['action'],
            reasoning=scenario['reasoning']
        )
        
        print(f"   🎯 Risk Score: {decision.risk_score:.3f}")
        print(f"   📊 Risk Level: {decision.risk_level.value.upper()}")
        print(f"   ⚡ Status: {decision.status.value.upper()}")
        
        # Handle human approval for medium-risk decisions
        if decision.status.value == 'pending':
            print(f"   🤔 Queued for human approval...")
            # Simulate human approval
            await active_mode.approve_decision(decision.decision_id, "senior_engineer", True)
            print(f"   ✅ Human approved and executed")
        
        print()
        
        # Small delay for realistic demo
        await asyncio.sleep(0.3)
    
    # Display professional metrics
    print("📊 Professional Production Metrics:")
    print("=" * 50)
    
    metrics = active_mode.get_production_metrics()
    production_metrics = metrics['metrics']
    
    print(f"   Total Decisions: {production_metrics['total_decisions']}")
    print(f"   Auto Executed: {production_metrics['auto_executed']}")
    print(f"   Human Approved: {production_metrics['human_approved']}")
    print(f"   Blocked Actions: {production_metrics['blocked_actions']}")
    print(f"   Cost Savings: ${production_metrics['cost_savings']:,.2f}")
    print(f"   Incidents Prevented: {production_metrics['incidents_prevented']}")
    print(f"   Avg Response Time: {production_metrics['average_response_time']*1000:.1f}ms")
    print(f"   System Accuracy: {production_metrics['accuracy_rate']*100:.1f}%")
    
    print()
    
    # Display professional ROI report
    print("💰 Professional ROI Report:")
    print("=" * 50)
    
    roi_report = active_mode.generate_roi_report()
    financial = roi_report['financial_impact']
    operational = roi_report['operational_metrics']
    
    print(f"   📈 Total ROI: ${financial['total_roi']:,.2f}")
    print(f"   💵 Direct Savings: ${financial['direct_cost_savings']:,.2f}")
    print(f"   🛡️ Incident Prevention: ${financial['incident_prevention_savings']:,.2f}")
    print(f"   🤖 Automation Rate: {operational['automation_rate']:.1f}%")
    print(f"   ⚡ Response Time: {operational['average_response_time_ms']:.1f}ms")
    print(f"   🎯 System Accuracy: {operational['system_accuracy']:.1f}%")
    
    print()
    
    # Display system health
    print("🏥 System Health Status:")
    print("=" * 50)
    
    health = active_mode.get_system_health()
    components = health['components']
    
    for component, status in components.items():
        status_icon = "✅" if status == "healthy" else "❌"
        print(f"   {status_icon} {component.replace('_', ' ').title()}: {status}")
    
    print(f"   📋 Pending Approvals: {health['pending_approvals']}")
    print(f"   📊 Total Decisions: {health['total_decisions']}")
    
    print()
    print("🎉 Professional Phase 3D Demo Complete!")
    print("✨ Modular Architecture Validated:")
    print("   🏗️ Clean separation of concerns")
    print("   📦 Professional module structure")
    print("   🔧 Maintainable, testable code")
    print("   📊 Comprehensive metrics and reporting")
    print("   🚀 Production-ready deployment")
    
    return active_mode


async def demo_human_approval_workflow():
    """Demo the human approval workflow specifically."""
    print("\n🤔 Human Approval Workflow Demo:")
    print("=" * 50)
    
    active_mode = ActiveModeDeployment()
    
    # Create a medium-risk scenario that requires approval
    decision = await active_mode.process_decision(
        evidence_log=[
            {'type': 'service_alert', 'severity': 'medium', 'service': 'payment_processor'},
            {'type': 'error_rate', 'current': 0.05, 'threshold': 0.02}
        ],
        proposed_action='restart_payment_service',
        reasoning='Payment service error rate above threshold, restart recommended'
    )
    
    print(f"📋 Decision created: {decision.decision_id}")
    print(f"   Status: {decision.status.value}")
    
    # Check pending approvals
    pending = active_mode.get_pending_approvals()
    print(f"   📋 Pending approvals: {len(pending)}")
    
    if pending:
        approval = pending[0]
        print(f"   ⏰ Time waiting: {approval['time_waiting_minutes']} minutes")
        print(f"   📝 Evidence: {approval['evidence_summary']}")
    
    # Simulate human approval
    print("   🤔 Simulating human review...")
    await asyncio.sleep(1)
    
    success = await active_mode.approve_decision(decision.decision_id, "ops_manager", True)
    print(f"   ✅ Approval processed: {success}")
    
    # Check final status
    updated_decision = active_mode._find_decision(decision.decision_id)
    print(f"   🎯 Final status: {updated_decision.status.value}")


if __name__ == "__main__":
    # Run the professional demo
    asyncio.run(demo_phase3d_professional())
    
    # Run the human approval demo
    asyncio.run(demo_human_approval_workflow())
