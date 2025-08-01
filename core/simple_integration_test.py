#!/usr/bin/env python3
"""
Simple Integration Test for AURA Intelligence

This test directly checks the integration points without loading the full system.
It focuses on the core workflow components we identified in the system status check.
"""

import asyncio
import os
import sys
from datetime import datetime
import json
import uuid

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Simple logging
def log_step(step: str, details: dict = None, success: bool = True):
    """Simple step logger."""
    status = "✅" if success else "❌"
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    print(f"\n{status} [{timestamp}] {step}")
    if details:
        for key, value in details.items():
            print(f"    {key}: {value}")

def log_error(step: str, error: str):
    """Log an error."""
    log_step(step, {"error": str(error)}, success=False)

async def test_service_connections():
    """Test basic service connectivity."""
    print("\n" + "="*60)
    print("TESTING SERVICE CONNECTIONS")
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
        
        log_step("Neo4j Connection", {
            "status": "connected",
            "query_result": result[0]["test"] if result else "no result"
        })
        results["neo4j"] = True
        
        await neo4j.close()
        
    except Exception as e:
        log_error("Neo4j Connection", str(e))
        results["neo4j"] = False
    
    # Test Redis connection
    try:
        from aura_intelligence.adapters.redis_adapter import RedisAdapter, RedisConfig
        
        config = RedisConfig(host="localhost", port=6379)
        redis = RedisAdapter(config)
        await redis.initialize()
        
        # Test basic operation
        test_key = f"test:{uuid.uuid4()}"
        await redis.set(test_key, {"test": "data"}, ttl=60)
        result = await redis.get(test_key)
        
        log_step("Redis Connection", {
            "status": "connected",
            "test_write": "success" if result else "failed"
        })
        results["redis"] = True
        
        await redis.close()
        
    except Exception as e:
        log_error("Redis Connection", str(e))
        results["redis"] = False
    
    return results

async def test_workflow_components():
    """Test workflow component loading."""
    print("\n" + "="*60)
    print("TESTING WORKFLOW COMPONENTS")
    print("="*60)
    
    results = {}
    
    # Test GPU Allocation Workflow loading
    try:
        from aura_intelligence.workflows.gpu_allocation import (
            GPUAllocationRequest, GPUType, AllocationPriority
        )
        
        # Create a test request
        request = GPUAllocationRequest(
            request_id=str(uuid.uuid4()),
            requester_id="test-user",
            gpu_type=GPUType.A100,
            gpu_count=1,
            duration_hours=1,
            priority=AllocationPriority.NORMAL,
            workload_type="test",
            estimated_memory_gb=10.0,
            estimated_compute_tflops=100.0
        )
        
        log_step("GPU Allocation Request", {
            "request_id": request.request_id,
            "gpu_type": request.gpu_type,
            "gpu_count": request.gpu_count
        })
        results["gpu_workflow"] = True
        
    except Exception as e:
        log_error("GPU Allocation Workflow", str(e))
        results["gpu_workflow"] = False
    
    # Test LNN Council Agent loading
    try:
        from aura_intelligence.agents.council.lnn_council import (
            CouncilTask, CouncilVote, VoteType
        )
        
        # Create a test council task
        task = CouncilTask(
            task_id=str(uuid.uuid4()),
            task_type="test_task",
            payload={"action": "test"},
            context={"test": True}
        )
        
        log_step("LNN Council Components", {
            "task_id": task.task_id,
            "task_type": task.task_type
        })
        results["lnn_council"] = True
        
    except Exception as e:
        log_error("LNN Council Agent", str(e))
        results["lnn_council"] = False
    
    return results

async def test_integration_gaps():
    """Test the specific integration gaps identified."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION GAPS")
    print("="*60)
    
    gaps = []
    
    # Gap 1: Check if LNN agents are mocked
    try:
        # Read the GPU allocation workflow file
        workflow_file = "core/src/aura_intelligence/workflows/gpu_allocation.py"
        with open(workflow_file, 'r') as f:
            content = f.read()
            
        if "# In production, this would actually invoke the agents" in content:
            gaps.append("LNN Agent votes are still mocked")
            log_step("LNN Agent Integration", {"status": "mocked"}, success=False)
        else:
            log_step("LNN Agent Integration", {"status": "real implementation"})
            
    except Exception as e:
        log_error("LNN Agent Integration Check", str(e))
    
    # Gap 2: Check if Neo4j storage is implemented
    try:
        workflow_file = "core/src/aura_intelligence/workflows/gpu_allocation.py"
        with open(workflow_file, 'r') as f:
            content = f.read()
            
        if "# In production, publish to Kafka" in content:
            gaps.append("Neo4j decision storage not implemented")
            log_step("Neo4j Decision Storage", {"status": "not implemented"}, success=False)
        else:
            log_step("Neo4j Decision Storage", {"status": "implemented"})
            
    except Exception as e:
        log_error("Neo4j Storage Check", str(e))
    
    # Gap 3: Check if Kafka events are verified
    try:
        test_file = "core/test_end_to_end_gpu_allocation.py"
        with open(test_file, 'r') as f:
            content = f.read()
            
        if "Cannot verify Kafka events without consumer" in content:
            gaps.append("Kafka event verification missing")
            log_step("Kafka Event Verification", {"status": "missing consumer"}, success=False)
        else:
            log_step("Kafka Event Verification", {"status": "implemented"})
            
    except Exception as e:
        log_error("Kafka Verification Check", str(e))
    
    return gaps

async def test_data_flow():
    """Test actual data flow through connected components."""
    print("\n" + "="*60)
    print("TESTING DATA FLOW")
    print("="*60)
    
    # Test Neo4j data storage and retrieval
    try:
        from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
        
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="dev_password"
        )
        
        neo4j = Neo4jAdapter(config)
        await neo4j.initialize()
        
        # Test decision storage
        decision_id = str(uuid.uuid4())
        await neo4j.add_decision_node(
            decision_id=decision_id,
            agent_id="test-agent",
            decision_type="gpu_allocation",
            confidence=0.85,
            context={"test": True}
        )
        
        # Verify storage
        decisions = await neo4j.query(
            "MATCH (d:Decision {id: $decision_id}) RETURN d",
            {"decision_id": decision_id}
        )
        
        if decisions:
            log_step("Neo4j Data Flow", {
                "decision_stored": True,
                "decision_id": decision_id
            })
        else:
            log_step("Neo4j Data Flow", {"decision_stored": False}, success=False)
        
        await neo4j.close()
        
    except Exception as e:
        log_error("Neo4j Data Flow", str(e))
    
    # Test Redis caching
    try:
        from aura_intelligence.adapters.redis_adapter import RedisAdapter, RedisConfig
        
        config = RedisConfig(host="localhost", port=6379)
        redis = RedisAdapter(config)
        await redis.initialize()
        
        # Test context caching
        agent_id = "test-agent"
        context_id = str(uuid.uuid4())
        context_data = {
            "request_id": str(uuid.uuid4()),
            "gpu_type": "A100",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await redis.cache_context_window(agent_id, context_id, context_data)
        retrieved = await redis.get_context_window(agent_id, context_id)
        
        log_step("Redis Data Flow", {
            "cache_success": success,
            "retrieval_success": retrieved is not None,
            "data_match": retrieved == context_data if retrieved else False
        })
        
        await redis.close()
        
    except Exception as e:
        log_error("Redis Data Flow", str(e))

def print_summary(service_results, component_results, integration_gaps):
    """Print test summary."""
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    # Service connectivity
    print(f"\nService Connectivity:")
    for service, status in service_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {service}")
    
    # Component loading
    print(f"\nComponent Loading:")
    for component, status in component_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    
    # Integration gaps
    print(f"\nIntegration Gaps Found: {len(integration_gaps)}")
    for gap in integration_gaps:
        print(f"  ❌ {gap}")
    
    # Overall status
    services_ok = all(service_results.values())
    components_ok = all(component_results.values())
    no_gaps = len(integration_gaps) == 0
    
    print(f"\nOverall Status:")
    print(f"  Services: {'✅ All Connected' if services_ok else '❌ Some Failed'}")
    print(f"  Components: {'✅ All Loading' if components_ok else '❌ Some Failed'}")
    print(f"  Integration: {'✅ Fully Connected' if no_gaps else f'❌ {len(integration_gaps)} Gaps'}")
    
    if services_ok and components_ok and no_gaps:
        print(f"\n🎉 SYSTEM FULLY INTEGRATED!")
        print("Ready for production workloads.")
    else:
        print(f"\n⚠️  INTEGRATION WORK NEEDED")
        print("Fix the issues above before proceeding.")
    
    # Next steps
    print(f"\nNext Steps:")
    if not services_ok:
        print("1. Fix service connectivity issues")
        print("   - Check Docker containers are running")
        print("   - Verify network connectivity")
        print("   - Check service configurations")
    
    if not components_ok:
        print("2. Fix component loading issues")
        print("   - Install missing dependencies")
        print("   - Check import paths")
        print("   - Verify module structure")
    
    if integration_gaps:
        print("3. Fix integration gaps:")
        for i, gap in enumerate(integration_gaps, 1):
            print(f"   {i}. {gap}")
    
    print("\n4. Re-run this test to verify fixes")
    print("5. Run full end-to-end test when all issues resolved")

async def main():
    """Main test execution."""
    print("="*80)
    print("AURA INTELLIGENCE - SIMPLE INTEGRATION TEST")
    print("="*80)
    print("Testing core integration points without full system load...")
    
    # Run tests
    service_results = await test_service_connections()
    component_results = await test_workflow_components()
    integration_gaps = await test_integration_gaps()
    
    # Test data flow if services are connected
    if service_results.get("neo4j") and service_results.get("redis"):
        await test_data_flow()
    
    # Print summary
    print_summary(service_results, component_results, integration_gaps)

if __name__ == "__main__":
    asyncio.run(main())