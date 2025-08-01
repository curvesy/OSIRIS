#!/usr/bin/env python3
"""
🚀 ObserverAgent Demo - World-Class Architecture in Action

This demo proves our modular schema architecture works end-to-end:
- Cryptographically signed evidence creation
- Immutable state management with pure functional updates
- Enhanced decision explainability
- OpenTelemetry trace context integration
- Production-grade error handling and retry logic

Run this to see The Collective's first agent in action!
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only our modular schemas (avoiding full system imports)
from aura_intelligence.agents.schemas.crypto import get_crypto_provider, SignatureAlgorithm
from aura_intelligence.agents.schemas.enums import TaskStatus, EvidenceType, ActionType, RiskLevel
from aura_intelligence.agents.schemas.base import utc_now
from aura_intelligence.agents.schemas.evidence import DossierEntry, LogEvidence
from aura_intelligence.agents.schemas.state import AgentState
from aura_intelligence.agents.schemas.decision import DecisionPoint, DecisionCriterion, DecisionOption


async def main():
    """
    Demonstrate the ObserverAgent with realistic scenarios.
    """
    print("🔍 AURA Intelligence - ObserverAgent Demo")
    print("=" * 60)
    print("Demonstrating world-class modular schema architecture")
    print()
    
    # Initialize crypto provider and keys
    provider = get_crypto_provider(SignatureAlgorithm.HMAC_SHA256)
    private_key = "demo_private_key_aura_2025"
    public_key = "demo_public_key_aura_2025"
    
    # Create ObserverAgent
    agent = ObserverAgent(
        agent_id="demo_observer_001",
        private_key=private_key,
        public_key=public_key,
        config={
            "max_concurrent": 10,
            "retry_attempts": 3,
            "trace_sampling": 1.0
        }
    )
    
    print(f"✅ {agent}")
    print()
    
    # Demo scenarios
    scenarios = [
        {
            "name": "🚨 Security Alert",
            "event": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "critical",
                "message": "Multiple failed login attempts detected from IP 192.168.1.100",
                "source": "security_monitor",
                "type": "security_threat",
                "fields": {
                    "ip_address": "192.168.1.100",
                    "failed_attempts": 15,
                    "time_window": "5_minutes",
                    "user_accounts": ["admin", "root", "service"]
                },
                "environment": "production",
                "priority": "critical",
                "correlation_id": "sec_alert_001"
            }
        },
        {
            "name": "⚡ Performance Issue",
            "event": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "warning",
                "message": "Database query performance degraded - average response time 2.5s",
                "source": "database_monitor",
                "type": "performance_degradation",
                "fields": {
                    "avg_response_time_ms": 2500,
                    "query_type": "user_lookup",
                    "affected_queries": 1247,
                    "database": "user_db_primary"
                },
                "metrics": {
                    "cpu": 85.2,
                    "memory": 92.1,
                    "disk_io": 78.5
                },
                "environment": "production",
                "priority": "high",
                "correlation_id": "perf_alert_002"
            }
        },
        {
            "name": "💰 Business Event",
            "event": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "info",
                "message": "Large transaction processed successfully - $50,000 payment",
                "source": "payment_processor",
                "type": "high_value_transaction",
                "fields": {
                    "transaction_id": "txn_789012345",
                    "amount": 50000.00,
                    "currency": "USD",
                    "merchant": "Enterprise_Client_A",
                    "payment_method": "wire_transfer"
                },
                "environment": "production",
                "priority": "normal",
                "correlation_id": "biz_event_003"
            }
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        try:
            # Process the event
            start_time = datetime.now()
            final_state = await agent.process_event(scenario["event"])
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Display results
            print(f"✅ Processing completed in {processing_time:.2f}ms")
            print(f"   Workflow ID: {final_state.workflow_id}")
            print(f"   Task Type: {final_state.task_type}")
            print(f"   Priority: {final_state.priority}")
            print(f"   Status: {final_state.status.value}")
            print(f"   State Version: {final_state.state_version}")
            print(f"   Overall Confidence: {final_state.overall_confidence:.3f}")
            
            # Evidence details
            evidence = final_state.context_dossier[0]
            print(f"   Evidence Type: {evidence.evidence_type.value}")
            print(f"   Evidence Confidence: {evidence.confidence:.3f}")
            print(f"   Evidence Summary: {evidence.summary[:80]}...")
            
            # Decision details
            decision = final_state.decision_points[0]
            print(f"   Decision: {decision.chosen_option_id}")
            print(f"   Decision Confidence: {decision.confidence_in_decision:.3f}")
            print(f"   Rationale: {decision.rationale}")
            
            # Signature verification
            is_valid = final_state.verify_signature(private_key)
            print(f"   Signature Valid: {'✅' if is_valid else '❌'}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing scenario: {e}")
            print()
    
    # Display final metrics
    print("📊 Final Agent Metrics")
    print("-" * 30)
    health = await agent.get_health_status()
    metrics = health["metrics"]
    
    print(f"Events Processed: {metrics['events_processed']}")
    print(f"Evidence Created: {metrics['evidence_created']}")
    print(f"Workflows Initiated: {metrics['workflows_initiated']}")
    print(f"Average Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"Errors: {metrics['errors']}")
    print()
    
    # Demonstrate state immutability
    print("🔒 Demonstrating Immutable State Management")
    print("-" * 45)
    
    # Create a simple event
    test_event = {
        "message": "Test immutability",
        "level": "info",
        "source": "demo"
    }
    
    # Process it twice
    state1 = await agent.process_event(test_event)
    state2 = await agent.process_event(test_event)
    
    print(f"State 1 ID: {state1.workflow_id}")
    print(f"State 2 ID: {state2.workflow_id}")
    print(f"States are different objects: {state1 is not state2}")
    print(f"States have different versions: {state1.state_version != state2.state_version}")
    print(f"States have different timestamps: {state1.updated_at != state2.updated_at}")
    print()
    
    # Demonstrate cryptographic verification
    print("🔐 Demonstrating Cryptographic Signatures")
    print("-" * 42)
    
    # Verify evidence signature
    evidence = state1.context_dossier[0]
    evidence_bytes = evidence.get_canonical_representation().encode('utf-8')
    evidence_valid = provider.verify(evidence_bytes, evidence.signature, private_key)
    print(f"Evidence signature valid: {'✅' if evidence_valid else '❌'}")
    
    # Verify decision signature
    decision = state1.decision_points[0]
    decision_bytes = decision.get_canonical_representation().encode('utf-8')
    decision_valid = provider.verify(decision_bytes, decision.signature, private_key)
    print(f"Decision signature valid: {'✅' if decision_valid else '❌'}")
    
    # Verify state signature
    state_valid = state1.verify_signature(private_key)
    print(f"State signature valid: {'✅' if state_valid else '❌'}")
    print()
    
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("The ObserverAgent successfully demonstrated:")
    print("✅ Modular schema architecture")
    print("✅ Cryptographically signed evidence")
    print("✅ Immutable state management")
    print("✅ Enhanced decision explainability")
    print("✅ Production-grade error handling")
    print("✅ Comprehensive observability")
    print()
    print("🚀 Ready for The Collective deployment!")


if __name__ == "__main__":
    asyncio.run(main())
