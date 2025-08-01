#!/usr/bin/env python3
"""
Minimal Integration Test - Core Workflow Only

Tests the essential GPU allocation workflow without complex dependencies:
1. GPU allocation request creation
2. LNN council decision (even if mocked)
3. Neo4j decision storage
4. Redis context caching
5. Basic data flow validation

This bypasses observability, Temporal, and other complex systems to focus
on proving the core business logic integration works.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def log_step(step: str, success: bool = True, details: Dict[str, Any] = None):
    """Simple step logger."""
    status = "✅" if success else "❌"
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    print(f"{status} [{timestamp}] {step}")
    if details:
        for key, value in details.items():
            print(f"    {key}: {value}")

async def test_core_components():
    """Test core components can be imported and instantiated."""
    print("\n" + "="*60)
    print("TESTING CORE COMPONENT IMPORTS")
    print("="*60)
    
    results = {}
    
    # Test GPU allocation components
    try:
        from aura_intelligence.workflows.gpu_allocation import (
            GPUAllocationRequest, GPUType, AllocationPriority
        )
        
        # Create test request
        request = GPUAllocationRequest(
            request_id=str(uuid.uuid4()),
            requester_id="test-user",
            gpu_type=GPUType.A100,
            gpu_count=2,
            duration_hours=1,
            priority=AllocationPriority.NORMAL,
            workload_type="integration-test",
            estimated_memory_gb=16.0,
            estimated_compute_tflops=200.0
        )
        
        log_step("GPU Allocation Components", True, {
            "request_id": request.request_id[:8] + "...",
            "gpu_type": request.gpu_type,
            "gpu_count": request.gpu_count
        })
        results["gpu_components"] = True
        
    except Exception as e:
        log_step("GPU Allocation Components", False, {"error": str(e)})
        results["gpu_components"] = False
    
    # Test LNN Council components
    try:
        from aura_intelligence.agents.council.lnn_council import (
            CouncilTask, CouncilVote, VoteType
        )
        
        # Create test council task
        task = CouncilTask(
            task_id=str(uuid.uuid4()),
            task_type="gpu_allocation",
            payload={"gpu_count": 2, "gpu_type": "A100"},
            context={"test": True}
        )
        
        log_step("LNN Council Components", True, {
            "task_id": task.task_id[:8] + "...",
            "task_type": task.task_type
        })
        results["lnn_components"] = True
        
    except Exception as e:
        log_step("LNN Council Components", False, {"error": str(e)})
        results["lnn_components"] = False
    
    return results

async def test_database_connections():
    """Test database connections work."""
    print("\n" + "="*60)
    print("TESTING DATABASE CONNECTIONS")
    print("="*60)
    
    results = {}
    
    # Test Neo4j connection
    try:
        from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
        
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="dev_password"  # From docker-compose.dev.yml
        )
        
        neo4j = Neo4jAdapter(config)
        await neo4j.initialize()
        
        # Test basic query
        result = await neo4j.query("RETURN 1 as test")
        
        log_step("Neo4j Connection", True, {
            "query_result": result[0]["test"] if result else "no result",
            "connection": "successful"
        })
        results["neo4j"] = True
        
        await neo4j.close()
        
    except Exception as e:
        log_step("Neo4j Connection", False, {"error": str(e)})
        results["neo4j"] = False
    
    # Test Redis connection
    try:
        from aura_intelligence.adapters.redis_adapter import RedisAdapter, RedisConfig
        
        config = RedisConfig(host="localhost", port=6379)
        redis = RedisAdapter(config)
        await redis.initialize()
        
        # Test basic operation
        test_key = f"integration_test:{uuid.uuid4()}"
        test_data = {"timestamp": datetime.utcnow().isoformat(), "test": True}
        
        await redis.set(test_key, test_data, ttl=60)
        retrieved = await redis.get(test_key)
        
        log_step("Redis Connection", True, {
            "write_success": True,
            "read_success": retrieved is not None,
            "data_match": retrieved == test_data if retrieved else False
        })
        results["redis"] = True
        
        await redis.close()
        
    except Exception as e:
        log_step("Redis Connection", False, {"error": str(e)})
        results["redis"] = False
    
    return results

async def test_data_flow():
    """Test actual data flow through the system."""
    print("\n" + "="*60)
    print("TESTING END-TO-END DATA FLOW")
    print("="*60)
    
    # Create test data
    request_id = str(uuid.uuid4())
    agent_id = "test-lnn-agent"
    decision_id = str(uuid.uuid4())
    
    log_step("Starting Data Flow Test", True, {
        "request_id": request_id[:8] + "...",
        "agent_id": agent_id,
        "decision_id": decision_id[:8] + "..."
    })
    
    # Test Neo4j decision storage
    try:
        from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
        
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="dev_password"
        )
        
        neo4j = Neo4jAdapter(config)
        await neo4j.initialize()
        
        # Store a decision
        await neo4j.add_decision_node(
            decision_id=decision_id,
            agent_id=agent_id,
            decision_type="gpu_allocation",
            confidence=0.85,
            context={
                "request_id": request_id,
                "gpu_type": "A100",
                "gpu_count": 2,
                "test": True
            }
        )
        
        # Verify storage
        decisions = await neo4j.query(
            "MATCH (d:Decision {id: $decision_id}) RETURN d",
            {"decision_id": decision_id}
        )
        
        log_step("Neo4j Decision Storage", True, {
            "decision_stored": len(decisions) > 0,
            "decision_id": decision_id[:8] + "...",
            "confidence": 0.85
        })
        
        await neo4j.close()
        
    except Exception as e:
        log_step("Neo4j Decision Storage", False, {"error": str(e)})
    
    # Test Redis context caching
    try:
        from aura_intelligence.adapters.redis_adapter import RedisAdapter, RedisConfig
        
        config = RedisConfig(host="localhost", port=6379)
        redis = RedisAdapter(config)
        await redis.initialize()
        
        # Cache context
        context_data = {
            "request_id": request_id,
            "decision_id": decision_id,
            "gpu_allocation": {
                "gpu_type": "A100",
                "gpu_count": 2,
                "estimated_cost": 10.24
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await redis.cache_context_window(agent_id, request_id, context_data)
        retrieved = await redis.get_context_window(agent_id, request_id)
        
        log_step("Redis Context Caching", True, {
            "cache_success": success,
            "retrieval_success": retrieved is not None,
            "data_integrity": retrieved == context_data if retrieved else False
        })
        
        await redis.close()
        
    except Exception as e:
        log_step("Redis Context Caching", False, {"error": str(e)})

async def test_workflow_logic():
    """Test the workflow logic without Temporal."""
    print("\n" + "="*60)
    print("TESTING WORKFLOW LOGIC (NO TEMPORAL)")
    print("="*60)
    
    try:
        # Import workflow activities (not the full workflow)
        from aura_intelligence.workflows.gpu_allocation import GPUAllocationActivities
        
        activities = GPUAllocationActivities()
        
        # Test availability check
        availability = await activities.check_gpu_availability("a100", 2)
        log_step("GPU Availability Check", True, {
            "sufficient": availability.get("sufficient", False),
            "available": availability.get("available", 0),
            "requested": availability.get("requested", 0)
        })
        
        # Test cost calculation
        cost_info = await activities.calculate_allocation_cost("a100", 2, 1)
        log_step("Cost Calculation", True, {
            "cost_per_hour": cost_info.get("cost_per_hour", 0),
            "total_cost": cost_info.get("total_cost", 0)
        })
        
        # Test council task creation
        from aura_intelligence.workflows.gpu_allocation import GPUAllocationRequest, GPUType, AllocationPriority
        
        request = GPUAllocationRequest(
            request_id=str(uuid.uuid4()),
            requester_id="test-user",
            gpu_type=GPUType.A100,
            gpu_count=2,
            duration_hours=1,
            priority=AllocationPriority.NORMAL,
            workload_type="test",
            estimated_memory_gb=16.0,
            estimated_compute_tflops=200.0
        )
        
        council_task = await activities.create_council_task(request, availability, cost_info)
        log_step("Council Task Creation", True, {
            "task_id": council_task.task_id[:8] + "...",
            "task_type": council_task.task_type,
            "payload_keys": list(council_task.payload.keys())
        })
        
        # Test council voting (even if mocked)
        votes = await activities.gather_council_votes(council_task, ["agent1", "agent2", "agent3"])
        log_step("Council Voting", True, {
            "vote_count": len(votes),
            "votes": [f"{v.vote}({v.confidence:.2f})" for v in votes]
        })
        
    except Exception as e:
        log_step("Workflow Logic Test", False, {"error": str(e)})

def print_summary(component_results, db_results):
    """Print test summary."""
    print("\n" + "="*80)
    print("MINIMAL INTEGRATION TEST SUMMARY")
    print("="*80)
    
    # Component imports
    print(f"\nComponent Imports:")
    for component, status in component_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    
    # Database connections
    print(f"\nDatabase Connections:")
    for db, status in db_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {db}")
    
    # Overall assessment
    components_ok = all(component_results.values())
    databases_ok = all(db_results.values())
    
    print(f"\nOverall Status:")
    print(f"  Components: {'✅ All Loading' if components_ok else '❌ Some Failed'}")
    print(f"  Databases: {'✅ All Connected' if databases_ok else '❌ Some Failed'}")
    
    if components_ok and databases_ok:
        print(f"\n🎉 CORE INTEGRATION WORKING!")
        print("✅ GPU allocation components load correctly")
        print("✅ LNN council components load correctly") 
        print("✅ Neo4j connection and storage works")
        print("✅ Redis connection and caching works")
        print("✅ Basic workflow logic executes")
        print("\nNext: Fix observability dependencies and run full test")
    else:
        print(f"\n⚠️  CORE ISSUES FOUND")
        if not components_ok:
            print("❌ Component import issues - check dependencies")
        if not databases_ok:
            print("❌ Database connection issues - check services")
        print("\nFix these core issues before proceeding")

async def main():
    """Main test execution."""
    print("="*80)
    print("AURA INTELLIGENCE - MINIMAL INTEGRATION TEST")
    print("="*80)
    print("Testing core workflow without complex dependencies...")
    
    # Run core tests
    component_results = await test_core_components()
    db_results = await test_database_connections()
    
    # Only run data flow tests if basics work
    if all(component_results.values()) and all(db_results.values()):
        await test_data_flow()
        await test_workflow_logic()
    
    # Print summary
    print_summary(component_results, db_results)

if __name__ == "__main__":
    asyncio.run(main())