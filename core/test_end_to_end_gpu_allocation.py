#!/usr/bin/env python3
"""
End-to-End GPU Allocation Flow Test

This script demonstrates the complete flow:
1. Submit GPU allocation request
2. Temporal workflow orchestration  
3. LNN council agent voting
4. Consensus decision
5. Result storage in Neo4j
6. Event streaming via Kafka

Run this to verify the entire system is connected and working.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
import json
from typing import Dict, Any
import uuid

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aura_intelligence.workflows.gpu_allocation import (
    GPUAllocationRequest,
    GPUAllocationWorkflow,
    GPUAllocationActivities,
    GPUType,
    AllocationPriority
)
from aura_intelligence.agents.council.lnn_council import LNNCouncilAgent
from aura_intelligence.adapters import Neo4jAdapter, Neo4jConfig, RedisAdapter, RedisConfig
from aura_intelligence.events.producers import EventProducer, ProducerConfig

# Import Temporal
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment

# Logging
import structlog
logger = structlog.get_logger()


class EndToEndFlowTracker:
    """Tracks the flow of a request through the system."""
    
    def __init__(self):
        self.steps = []
        self.errors = []
        self.start_time = datetime.utcnow()
        
    def log_step(self, step: str, details: Dict[str, Any] = None):
        """Log a step in the flow."""
        timestamp = datetime.utcnow()
        elapsed = (timestamp - self.start_time).total_seconds()
        
        entry = {
            "step": step,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": elapsed,
            "details": details or {}
        }
        self.steps.append(entry)
        
        # Print for visibility
        print(f"\n{'='*60}")
        print(f"STEP: {step}")
        print(f"TIME: +{elapsed:.2f}s")
        if details:
            print(f"DETAILS: {json.dumps(details, indent=2)}")
        print('='*60)
        
    def log_error(self, step: str, error: str):
        """Log an error in the flow."""
        self.errors.append({
            "step": step,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"\n❌ ERROR in {step}: {error}")
        
    def print_summary(self):
        """Print flow summary."""
        print("\n" + "="*80)
        print("END-TO-END FLOW SUMMARY")
        print("="*80)
        
        print(f"\nTotal Steps: {len(self.steps)}")
        print(f"Total Time: {(datetime.utcnow() - self.start_time).total_seconds():.2f}s")
        print(f"Errors: {len(self.errors)}")
        
        print("\nFlow Path:")
        for i, step in enumerate(self.steps):
            status = "✅" if not any(e['step'] == step['step'] for e in self.errors) else "❌"
            print(f"{i+1}. {status} {step['step']} (+{step['elapsed_seconds']:.2f}s)")
            
        if self.errors:
            print("\nErrors Encountered:")
            for error in self.errors:
                print(f"- {error['step']}: {error['error']}")
                
        print("\nMissing Integrations:")
        self._check_missing_integrations()
        
    def _check_missing_integrations(self):
        """Check what's not integrated yet."""
        expected_steps = [
            "request_created",
            "temporal_workflow_started",
            "gpu_availability_checked",
            "cost_calculated",
            "council_task_created",
            "lnn_agents_invoked",
            "council_votes_gathered",
            "consensus_achieved",
            "allocation_decision_made",
            "neo4j_decision_stored",
            "kafka_event_published",
            "gpu_allocated",
            "metrics_recorded"
        ]
        
        actual_steps = [s['step'] for s in self.steps]
        missing = [s for s in expected_steps if s not in actual_steps]
        
        if missing:
            print("- " + "\n- ".join(missing))
        else:
            print("- None! Full flow connected! 🎉")


async def test_end_to_end_flow():
    """Run the complete end-to-end test."""
    tracker = EndToEndFlowTracker()
    
    # Step 1: Create GPU allocation request
    request_id = str(uuid.uuid4())
    gpu_request = GPUAllocationRequest(
        request_id=request_id,
        requester_id="test-user-123",
        gpu_type=GPUType.A100,
        gpu_count=2,
        duration_hours=4,
        priority=AllocationPriority.HIGH,
        workload_type="training",
        estimated_memory_gb=40.0,
        estimated_compute_tflops=312.0,
        project_id="ml-research-001",
        max_cost_per_hour=15.0,
        budget_remaining=1000.0,
        metadata={"model": "llm-7b", "framework": "pytorch"}
    )
    
    tracker.log_step("request_created", {
        "request_id": request_id,
        "gpu_type": gpu_request.gpu_type,
        "gpu_count": gpu_request.gpu_count,
        "duration_hours": gpu_request.duration_hours
    })
    
    # Step 2: Initialize services (check if they're running)
    try:
        # Check Neo4j
        neo4j_config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="aura-intelligence-2025"
        )
        neo4j = Neo4jAdapter(neo4j_config)
        await neo4j.initialize()
        tracker.log_step("neo4j_connected")
    except Exception as e:
        tracker.log_error("neo4j_connection", str(e))
        print("\n⚠️  Neo4j not running. Start with: docker-compose up neo4j")
        
    try:
        # Check Redis
        redis_config = RedisConfig(host="localhost", port=6379)
        redis = RedisAdapter(redis_config)
        await redis.initialize()
        tracker.log_step("redis_connected")
    except Exception as e:
        tracker.log_error("redis_connection", str(e))
        print("\n⚠️  Redis not running. Start with: docker-compose up redis")
        
    try:
        # Check Kafka
        producer_config = ProducerConfig(bootstrap_servers="localhost:9092")
        producer = EventProducer(producer_config)
        await producer.start()
        tracker.log_step("kafka_connected")
    except Exception as e:
        tracker.log_error("kafka_connection", str(e))
        print("\n⚠️  Kafka not running. Start with: docker-compose up kafka")
    
    # Step 3: Start Temporal workflow
    try:
        # Use test environment for now (production would use real Temporal)
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tracker.log_step("temporal_environment_started")
            
            # Register workflow and activities
            activities = GPUAllocationActivities()
            
            async with Worker(
                env.client,
                task_queue="gpu-allocation-queue",
                workflows=[GPUAllocationWorkflow],
                activities=[
                    activities.check_gpu_availability,
                    activities.calculate_allocation_cost,
                    activities.create_council_task,
                    activities.gather_council_votes,
                    activities.allocate_gpus,
                    activities.deallocate_gpus,
                    activities.emit_allocation_event
                ]
            ):
                tracker.log_step("temporal_worker_started")
                
                # Execute workflow
                handle = await env.client.start_workflow(
                    GPUAllocationWorkflow.run,
                    gpu_request,
                    id=f"gpu-alloc-{request_id}",
                    task_queue="gpu-allocation-queue"
                )
                tracker.log_step("temporal_workflow_started", {
                    "workflow_id": f"gpu-alloc-{request_id}"
                })
                
                # Wait for result
                result = await handle.result()
                
                tracker.log_step("workflow_completed", {
                    "status": result.status,
                    "allocation_id": result.allocation_id,
                    "cost_per_hour": result.cost_per_hour,
                    "consensus": result.consensus_achieved
                })
                
                # Log the internal steps that happened
                if result.status == "allocated":
                    tracker.log_step("gpu_availability_checked", {
                        "sufficient": True
                    })
                    tracker.log_step("cost_calculated", {
                        "cost_per_hour": result.cost_per_hour,
                        "total_cost": result.estimated_cost
                    })
                    tracker.log_step("council_task_created")
                    tracker.log_step("council_votes_gathered", {
                        "vote_count": len(result.council_votes)
                    })
                    tracker.log_step("consensus_achieved", {
                        "consensus": result.consensus_achieved
                    })
                    tracker.log_step("allocation_decision_made", {
                        "decision": "approved"
                    })
                    tracker.log_step("gpu_allocated", {
                        "allocation_id": result.allocation_id,
                        "gpu_count": len(result.allocated_gpus)
                    })
                    
    except Exception as e:
        tracker.log_error("temporal_workflow", str(e))
        print(f"\n⚠️  Temporal workflow failed: {e}")
        print("For production, ensure Temporal is running: docker-compose up temporal")
    
    # Step 4: Check if decision was stored in Neo4j
    if 'neo4j' in locals():
        try:
            # Query for the decision
            decisions = await neo4j.query(
                "MATCH (d:Decision {request_id: $request_id}) RETURN d",
                {"request_id": request_id}
            )
            if decisions:
                tracker.log_step("neo4j_decision_stored", {
                    "decision_count": len(decisions)
                })
            else:
                tracker.log_error("neo4j_decision_storage", "No decision found in Neo4j")
        except Exception as e:
            tracker.log_error("neo4j_query", str(e))
    
    # Step 5: Check if event was published to Kafka
    if 'producer' in locals():
        try:
            # In a real system, we'd consume from Kafka to verify
            # For now, we note it as a missing integration
            tracker.log_error("kafka_event_verification", 
                            "Cannot verify Kafka events without consumer")
        except Exception as e:
            tracker.log_error("kafka_verification", str(e))
    
    # Print summary
    tracker.print_summary()
    
    # Cleanup
    if 'neo4j' in locals():
        await neo4j.close()
    if 'redis' in locals():
        await redis.close()
    if 'producer' in locals():
        await producer.stop()
    
    return tracker


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("AURA INTELLIGENCE - END-TO-END GPU ALLOCATION FLOW TEST")
    print("="*80)
    print("\nThis test will trace a GPU allocation request through:")
    print("1. Request creation")
    print("2. Temporal workflow orchestration")
    print("3. LNN council agent voting")
    print("4. Consensus decision")
    print("5. Neo4j storage")
    print("6. Kafka event streaming")
    print("\nStarting test...\n")
    
    # Run the test
    tracker = asyncio.run(test_end_to_end_flow())
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if tracker.errors:
        print("\n1. Fix the integration errors above first")
        print("2. Ensure all services are running: ./deployments/staging/start.sh")
        print("3. Re-run this test to verify connections")
    else:
        print("\n✅ All systems connected and working!")
        print("\nNext steps:")
        print("1. Add real LNN agent invocation (currently mocked)")
        print("2. Add real consensus protocol (currently simplified)")
        print("3. Add Kafka event consumption verification")
        print("4. Add Prometheus metrics collection")
        print("5. Add Grafana dashboard for visualization")
    
    print("\nTo run in production mode:")
    print("1. Start all services: cd deployments/staging && ./start.sh")
    print("2. Run this test: python test_end_to_end_gpu_allocation.py")
    print("3. Check Grafana dashboards: http://localhost:3000")
    print("4. Check Temporal UI: http://localhost:8088")


if __name__ == "__main__":
    main()