#!/usr/bin/env python3
"""
Example: Using the Modular LNN Council Architecture

This example demonstrates how to use the modular LNN council agent
with clean separation of concerns and dependency injection.
"""

import asyncio
from uuid import uuid4

# Import contracts
from aura_intelligence.agents.council.lnn import (
    CouncilRequest,
    CouncilResponse,
    AgentCapability,
    VoteDecision,
    ContextScope
)

# Import factory
from aura_intelligence.agents.council.lnn import CouncilAgentFactory


async def basic_example():
    """Basic example using the default agent."""
    print("=== Basic Example ===\n")
    
    # Create a default agent using the factory
    agent = CouncilAgentFactory.create_default_agent(
        agent_id="example_agent_001",
        capabilities=[
            AgentCapability.GPU_ALLOCATION,
            AgentCapability.RESOURCE_MANAGEMENT
        ],
        neural_config={
            "input_size": 64,
            "hidden_sizes": [32, 16],
            "output_size": 4,
            "device": "cpu"
        }
    )
    
    # Initialize the agent
    await agent.initialize()
    
    # Create a request
    request = CouncilRequest(
        request_type="gpu_allocation",
        payload={
            "resource_type": "GPU",
            "quantity": 2,
            "duration_hours": 24,
            "purpose": "model_training"
        },
        requester_id="user_123",
        priority=7,
        context_scope=ContextScope.LOCAL
    )
    
    # Process the request
    response = await agent.process_request(request)
    
    # Display results
    print(f"Request ID: {request.request_id}")
    print(f"Decision: {response.decision}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Processing Time: {response.processing_time_ms:.2f}ms")
    print()
    
    # Get agent metrics
    metrics = await agent.get_metrics()
    print(f"Agent Metrics: {metrics}")
    
    # Cleanup
    await agent.cleanup()


async def specialized_agents_example():
    """Example using specialized agents."""
    print("\n=== Specialized Agents Example ===\n")
    
    # Create different specialized agents
    gpu_specialist = CouncilAgentFactory.create_specialized_agent(
        agent_type="gpu_specialist",
        agent_id="gpu_expert_001"
    )
    
    risk_assessor = CouncilAgentFactory.create_specialized_agent(
        agent_type="risk_assessor",
        agent_id="risk_expert_001"
    )
    
    cost_optimizer = CouncilAgentFactory.create_specialized_agent(
        agent_type="cost_optimizer",
        agent_id="cost_expert_001"
    )
    
    agents = [gpu_specialist, risk_assessor, cost_optimizer]
    
    # Initialize all agents
    for agent in agents:
        await agent.initialize()
    
    # Create a complex request requiring multiple perspectives
    request = CouncilRequest(
        request_type="complex_resource_allocation",
        payload={
            "resource_type": "GPU_CLUSTER",
            "quantity": 10,
            "duration_hours": 168,  # 1 week
            "budget_limit": 50000,
            "compliance_requirements": ["SOC2", "HIPAA"],
            "purpose": "production_deployment"
        },
        requester_id="enterprise_customer",
        priority=9,
        context_scope=ContextScope.GLOBAL,
        capabilities_required=[
            AgentCapability.GPU_ALLOCATION,
            AgentCapability.RISK_ASSESSMENT,
            AgentCapability.COST_OPTIMIZATION
        ]
    )
    
    # Get responses from all agents
    print("Gathering council opinions...\n")
    
    for agent in agents:
        response = await agent.process_request(request)
        print(f"Agent: {response.agent_id}")
        print(f"  Decision: {response.decision}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Reasoning: {response.reasoning[:100]}...")
        print()
    
    # Cleanup all agents
    for agent in agents:
        await agent.cleanup()


async def multi_agent_council_example():
    """Example using a multi-agent council."""
    print("\n=== Multi-Agent Council Example ===\n")
    
    # Create a council of agents
    council = CouncilAgentFactory.create_multi_agent_council(
        council_size=5,
        agent_types=["gpu_specialist", "risk_assessor", "cost_optimizer"]
    )
    
    # Initialize all council members
    for agent in council:
        await agent.initialize()
    
    # Create a request
    request = CouncilRequest(
        request_type="strategic_decision",
        payload={
            "decision_type": "infrastructure_expansion",
            "options": [
                {"provider": "AWS", "cost": 100000, "gpus": 50},
                {"provider": "GCP", "cost": 95000, "gpus": 45},
                {"provider": "Azure", "cost": 105000, "gpus": 55}
            ],
            "timeline": "Q2 2025",
            "risk_tolerance": "medium"
        },
        requester_id="cto_office",
        priority=10,
        context_scope=ContextScope.GLOBAL
    )
    
    # Collect votes from council
    votes = []
    decisions = {}
    
    print("Council voting in progress...\n")
    
    for agent in council:
        response = await agent.process_request(request)
        votes.append(response)
        
        # Count decisions
        decision = response.decision.value
        decisions[decision] = decisions.get(decision, 0) + 1
        
        print(f"Agent {response.agent_id} voted: {response.decision} "
              f"(confidence: {response.confidence:.2f})")
    
    # Determine consensus
    print(f"\nCouncil Decision Summary:")
    for decision, count in decisions.items():
        percentage = (count / len(council)) * 100
        print(f"  {decision}: {count} votes ({percentage:.0f}%)")
    
    # Find majority decision
    majority_decision = max(decisions.items(), key=lambda x: x[1])[0]
    print(f"\nMajority Decision: {majority_decision}")
    
    # Calculate average confidence
    avg_confidence = sum(v.confidence for v in votes) / len(votes)
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    # Cleanup
    for agent in council:
        await agent.cleanup()


async def health_check_example():
    """Example showing health check and monitoring."""
    print("\n=== Health Check Example ===\n")
    
    # Create an agent
    agent = CouncilAgentFactory.create_default_agent(
        agent_id="monitored_agent",
        neural_config={
            "input_size": 128,
            "hidden_sizes": [64, 32],
            "mixed_precision": True
        }
    )
    
    await agent.initialize()
    
    # Perform health check
    health = await agent.health_check()
    
    print("Agent Health Status:")
    print(f"  Status: {health['status']}")
    print(f"  Uptime: {health['uptime_seconds']:.2f} seconds")
    print(f"  Capabilities: {health['capabilities']}")
    
    print("\nComponent Health:")
    for component, status in health['components'].items():
        print(f"  {component}: {status.get('status', 'unknown')}")
        if 'metrics' in status:
            print(f"    - Parameters: {status['metrics'].get('parameter_count', 0):,}")
            print(f"    - Device: {status['metrics'].get('device', 'unknown')}")
    
    # Process some requests to generate metrics
    print("\nProcessing test requests...")
    for i in range(5):
        request = CouncilRequest(
            request_type="test_request",
            payload={"test_id": i},
            requester_id="test_system",
            priority=5
        )
        await agent.process_request(request)
    
    # Get updated metrics
    metrics = await agent.get_metrics()
    print(f"\nAgent Performance Metrics:")
    print(f"  Total Decisions: {metrics.total_decisions}")
    print(f"  Approval Rate: {metrics.approval_rate:.2%}")
    print(f"  Average Confidence: {metrics.average_confidence:.2f}")
    print(f"  Average Processing Time: {metrics.average_processing_time_ms:.2f}ms")
    
    await agent.cleanup()


async def main():
    """Run all examples."""
    print("🚀 Modular LNN Council Architecture Examples\n")
    
    # Run examples
    await basic_example()
    await specialized_agents_example()
    await multi_agent_council_example()
    await health_check_example()
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())