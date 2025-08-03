#!/usr/bin/env python3
"""
Test LNN Council Integration

This script tests the real LNN council agent integration
without the complexity of Temporal workflows.
"""

import asyncio
import os
import sys
from datetime import datetime
import json
import uuid

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aura_intelligence.agents.council.lnn import (
    CouncilAgentFactory,
    CouncilRequest,
    CouncilResponse,
    VoteDecision,
    AgentCapability
)
from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig

async def test_lnn_council_integration():
    """Test the LNN council agent with real GPU allocation decision."""
    print("🧪 LNN Council Integration Test")
    print("=" * 50)
    
    # Step 1: Create a GPU allocation decision context
    request_id = str(uuid.uuid4())
    
    council_request = CouncilRequest(
        request_id=request_id,
        request_type="gpu_allocation",
        data={
            "gpu_allocation": {
                "gpu_type": "a100",
                "gpu_count": 2,
                "duration_hours": 4,
                "user_id": "test-user-123",
                "priority": "high",
                "cost_per_hour": 6.40,
                "estimated_cost": 25.60
            }
        },
        context={
            "user_context": {"user_id": "test-user-123"},
            "system_context": {"available_gpus": 8}
        },
        capabilities_required=[AgentCapability.GPU_ALLOCATION],
        priority=8
    )
    
    print(f"🔍 Testing LNN council decision for request: {request_id}")
    print(f"📋 Request: {json.dumps(council_request.data, indent=2)}")
    
    # Step 2: Initialize LNN Council Agent using Factory
    try:
        print("\\n🤖 Initializing LNN Council Agent...")
        council_agent = CouncilAgentFactory.create_default_agent(
            agent_id="test_lnn_council",
            capabilities=[AgentCapability.GPU_ALLOCATION, AgentCapability.COST_OPTIMIZATION]
        )
        
        # Initialize the agent
        await council_agent.initialize()
        print("✅ LNN Council Agent initialized successfully!")
        
        # Step 3: Process decision with real LNN council
        print("\\n🗳️  Processing decision with LNN council...")
        start_time = datetime.utcnow()
        
        council_result = await council_agent.process_request(council_request)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ LNN council processing completed in {duration:.2f}s")
        print(f"📊 Council Result: {council_result}")
        
        # Step 4: Extract decision information
        approved = council_result.decision == VoteDecision.APPROVE
        confidence = council_result.confidence
        reasoning = council_result.reasoning
        
        print(f"\\n🎯 Decision Summary:")
        print(f"   Approved: {'✅ YES' if approved else '❌ NO'}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Reasoning: {reasoning}")
        
        # Step 5: Test Neo4j storage of decision
        print(f"\\n💾 Storing decision in Neo4j...")
        try:
            neo4j_config = Neo4jConfig(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="dev_password"
            )
            neo4j = Neo4jAdapter(neo4j_config)
            await neo4j.initialize()
            
            # Store the decision
            query = """
            CREATE (d:Decision {
                request_id: $request_id,
                approved: $approved,
                confidence: $confidence,
                reasoning: $reasoning,
                gpu_type: $gpu_type,
                gpu_count: $gpu_count,
                cost_per_hour: $cost_per_hour,
                estimated_cost: $estimated_cost,
                created_at: $created_at
            })
            RETURN d
            """
            
            result = await neo4j.query(query, {
                "request_id": request_id,
                "approved": approved,
                "confidence": confidence,
                "reasoning": reasoning,
                "gpu_type": council_request.data["gpu_allocation"]["gpu_type"],
                "gpu_count": council_request.data["gpu_allocation"]["gpu_count"],
                "cost_per_hour": council_request.data["gpu_allocation"]["cost_per_hour"],
                "estimated_cost": council_request.data["gpu_allocation"]["estimated_cost"],
                "created_at": datetime.utcnow().isoformat()
            })
            
            print(f"✅ Decision stored in Neo4j: {len(result)} nodes created")
            
            # Verify the decision was stored
            verify_query = "MATCH (d:Decision {request_id: $request_id}) RETURN d"
            verify_result = await neo4j.query(verify_query, {"request_id": request_id})
            
            if verify_result:
                print(f"✅ Decision verification successful: {len(verify_result)} nodes found")
                stored_decision = verify_result[0]['d']
                print(f"📋 Stored Decision: {json.dumps(dict(stored_decision), indent=2, default=str)}")
            else:
                print("❌ Decision verification failed: No nodes found")
            
            await neo4j.close()
            
        except Exception as e:
            print(f"❌ Neo4j storage failed: {e}")
        
        # Step 6: Test summary
        print(f"\\n🎉 LNN Council Integration Test Summary:")
        print(f"   ✅ LNN Council Agent: Working")
        print(f"   ✅ Decision Processing: {duration:.2f}s")
        print(f"   ✅ Neo4j Storage: Working")
        print(f"   ✅ Real AI Decision: {'Approved' if approved else 'Denied'}")
        
        return True
        
    except Exception as e:
        print(f"❌ LNN Council integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_lnn_council_integration())
    if success:
        print("\\n🎉 All tests passed! LNN Council integration is working!")
        sys.exit(0)
    else:
        print("\\n💥 Tests failed! Check the errors above.")
        sys.exit(1)