"""
🚀 Enhanced Hybrid Orchestration Example

Demonstrates the power of combining LangGraph v0.2+ features with Temporal.io
for enterprise-grade AI orchestration with cross-thread memory and durability.

This example shows:
- PostgresSaver for persistent checkpointing
- Cross-thread memory with Store interface
- Hybrid checkpoint management
- TDA-aware recovery strategies
- Enterprise-grade reliability
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

# Enhanced orchestration imports
from aura_intelligence.orchestration.semantic import (
    LangGraphSemanticOrchestrator,
    SemanticWorkflowConfig,
    LANGGRAPH_AVAILABLE,
    POSTGRES_CHECKPOINTING_AVAILABLE
)

from aura_intelligence.orchestration.durable import (
    HybridCheckpointManager,
    HybridCheckpointConfig,
    CheckpointLevel,
    RecoveryMode,
    TemporalDurableOrchestrator,
    DurableWorkflowConfig,
    CompensationStrategy
)

from aura_intelligence.orchestration.semantic.base_interfaces import (
    AgentState,
    OrchestrationStrategy
)

class EnhancedAuraOrchestrationDemo:
    """
    Demonstrates the enhanced AURA orchestration capabilities
    """
    
    def __init__(self):
        # Configuration
        self.postgres_url = "postgresql://aura:aura@localhost:5432/aura_dev"
        self.redis_url = "redis://localhost:6379/0"
        
        # Initialize enhanced LangGraph orchestrator
        self.semantic_orchestrator = LangGraphSemanticOrchestrator(
            postgres_url=self.postgres_url,
            redis_url=self.redis_url,
            enable_cross_thread_memory=True
        )
        
        # Initialize hybrid checkpoint manager
        hybrid_config = HybridCheckpointConfig(
            postgres_url=self.postgres_url,
            checkpoint_level=CheckpointLevel.HYBRID,
            recovery_mode=RecoveryMode.INTELLIGENT_RECOVERY,
            enable_tda_optimization=True
        )
        self.hybrid_checkpointer = HybridCheckpointManager(
            config=hybrid_config
        )
        
        # Initialize Temporal.io durable orchestrator
        self.temporal_orchestrator = TemporalDurableOrchestrator()
    
    async def demonstrate_enhanced_features(self):
        """Demonstrate all enhanced features"""
        print("🚀 Enhanced AURA Orchestration Demo")
        print("=" * 50)
        
        # Check feature availability
        await self._check_feature_availability()
        
        # Demonstrate semantic orchestration with memory
        await self._demo_semantic_orchestration_with_memory()
        
        # Demonstrate hybrid checkpointing
        await self._demo_hybrid_checkpointing()
        
        # Demonstrate durable workflow with compensation
        await self._demo_durable_workflow_with_compensation()
        
        # Demonstrate intelligent recovery
        await self._demo_intelligent_recovery()
        
        print("\n✅ Demo completed successfully!")
    
    async def _check_feature_availability(self):
        """Check which enhanced features are available"""
        print("\n📋 Feature Availability Check:")
        print(f"   LangGraph Available: {LANGGRAPH_AVAILABLE}")
        print(f"   PostgresSaver Available: {POSTGRES_CHECKPOINTING_AVAILABLE}")
        print(f"   Cross-Thread Memory: {self.semantic_orchestrator.memory_store is not None}")
        print(f"   Hybrid Checkpointing: {self.hybrid_checkpointer.langgraph_checkpointer is not None}")
    
    async def _demo_semantic_orchestration_with_memory(self):
        """Demonstrate semantic orchestration with cross-thread memory"""
        print("\n🧠 Semantic Orchestration with Cross-Thread Memory:")
        
        # Create workflow configuration
        workflow_config = SemanticWorkflowConfig(
            workflow_id="enhanced-demo-001",
            orchestrator_agent="semantic_coordinator",
            worker_agents=["observer", "analyst", "supervisor"],
            routing_strategy=OrchestrationStrategy.SEMANTIC
        )
        
        # Create enhanced workflow
        if LANGGRAPH_AVAILABLE:
            workflow = await self.semantic_orchestrator.create_orchestrator_worker_graph(
                workflow_config
            )
            
            if workflow:
                print("   ✅ Enhanced StateGraph created with memory support")
                print(f"   📊 Memory Store: {type(self.semantic_orchestrator.memory_store).__name__ if self.semantic_orchestrator.memory_store else 'None'}")
                print(f"   💾 Checkpointer: {type(self.semantic_orchestrator.checkpointer).__name__ if self.semantic_orchestrator.checkpointer else 'None'}")
            else:
                print("   ⚠️  Fallback orchestration mode")
        else:
            print("   ❌ LangGraph not available")
    
    async def _demo_hybrid_checkpointing(self):
        """Demonstrate hybrid checkpointing capabilities"""
        print("\n🔄 Hybrid Checkpointing Demo:")
        
        # Sample conversation state
        conversation_state = {
            "messages": [
                {"role": "user", "content": "Analyze the recent system anomalies"},
                {"role": "assistant", "content": "I'll analyze the anomalies using our multi-agent system..."}
            ],
            "context": {
                "user_id": "demo-user",
                "session_id": "demo-session-001",
                "workflow_type": "anomaly_analysis"
            },
            "agent_memory": {
                "previous_analyses": 3,
                "success_rate": 0.95,
                "learned_patterns": ["network_spike", "memory_leak", "disk_io_bottleneck"]
            }
        }
        
        # Sample workflow state
        workflow_state = {
            "current_step": "deep_analysis",
            "completed_steps": ["observation", "data_collection", "initial_analysis"],
            "step_results": {
                "observation": {"anomalies_detected": 5, "severity": "medium"},
                "data_collection": {"data_points": 2500, "time_range": "24h"},
                "initial_analysis": {"patterns_identified": 3, "confidence": 0.87}
            },
            "workflow_metadata": {
                "workflow_id": "anomaly-analysis-001",
                "start_time": datetime.utcnow().isoformat(),
                "expected_completion": "2025-01-31T15:30:00Z"
            }
        }
        
        try:
            # Create hybrid checkpoint
            checkpoint_result = await self.hybrid_checkpointer.create_hybrid_checkpoint(
                workflow_id="anomaly-analysis-001",
                conversation_state=conversation_state,
                workflow_state=workflow_state,
                tda_correlation_id="demo-tda-correlation-001"
            )
            
            print(f"   ✅ Hybrid checkpoint created: {checkpoint_result.checkpoint_id}")
            print(f"   📊 Checkpoint level: {checkpoint_result.checkpoint_level.value}")
            print(f"   💾 Size: {checkpoint_result.size_bytes} bytes")
            print(f"   🔗 TDA correlation: {checkpoint_result.tda_correlation_id}")
            
            if checkpoint_result.conversation_checkpoint_id:
                print(f"   💬 Conversation checkpoint: {checkpoint_result.conversation_checkpoint_id}")
            if checkpoint_result.workflow_checkpoint_id:
                print(f"   ⚙️  Workflow checkpoint: {checkpoint_result.workflow_checkpoint_id}")
            
            return checkpoint_result.checkpoint_id
            
        except Exception as e:
            print(f"   ❌ Checkpoint creation failed: {e}")
            return None
    
    async def _demo_durable_workflow_with_compensation(self):
        """Demonstrate durable workflow with saga pattern compensation"""
        print("\n⏱️  Durable Workflow with Compensation:")
        
        # Create durable workflow configuration
        workflow_config = DurableWorkflowConfig(
            workflow_id="durable-demo-001",
            workflow_type="multi_agent_analysis",
            steps=[
                {
                    "name": "enhanced_observation",
                    "agent_type": "observer",
                    "timeout": 300,
                    "parameters": {
                        "data_sources": ["metrics", "logs", "traces"],
                        "memory_enhanced": True
                    }
                },
                {
                    "name": "intelligent_analysis", 
                    "agent_type": "analyst",
                    "timeout": 600,
                    "parameters": {
                        "analysis_type": "pattern_detection_with_memory",
                        "use_historical_patterns": True
                    }
                },
                {
                    "name": "contextual_decision",
                    "agent_type": "supervisor",
                    "timeout": 300,
                    "parameters": {
                        "decision_criteria": "confidence > 0.8",
                        "consider_past_decisions": True
                    }
                }
            ],
            retry_policy={
                "max_attempts": 3,
                "initial_interval": 1,
                "max_interval": 30
            },
            compensation_strategy=CompensationStrategy.ROLLBACK_ALL,
            tda_correlation_id="demo-tda-correlation-002"
        )
        
        try:
            # Execute durable workflow
            result = await self.temporal_orchestrator.execute_durable_workflow(
                workflow_config,
                {
                    "analysis_request": "Comprehensive system health analysis",
                    "priority": "high",
                    "memory_context": "Use learned patterns from similar analyses"
                }
            )
            
            print(f"   ✅ Durable workflow completed: {result.workflow_id}")
            print(f"   📊 Status: {result.status.value}")
            print(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
            print(f"   🔄 Checkpoints created: {len(result.checkpoints)}")
            print(f"   🛡️  Compensation actions: {len(result.compensation_actions)}")
            
            if result.error_details:
                print(f"   ⚠️  Errors handled: {result.error_details}")
            
        except Exception as e:
            print(f"   ❌ Durable workflow failed: {e}")
    
    async def _demo_intelligent_recovery(self):
        """Demonstrate intelligent recovery strategies"""
        print("\n🧠 Intelligent Recovery Demo:")
        
        # Simulate a checkpoint that needs recovery
        checkpoint_id = "demo-checkpoint-recovery-001"
        
        # Create a mock checkpoint for recovery demo
        from aura_intelligence.orchestration.durable.hybrid_checkpointer import HybridCheckpointResult
        
        mock_checkpoint = HybridCheckpointResult(
            checkpoint_id=checkpoint_id,
            conversation_checkpoint_id="conv_recovery_001",
            workflow_checkpoint_id="wf_recovery_001",
            checkpoint_level=CheckpointLevel.HYBRID,
            timestamp=datetime.utcnow(),
            size_bytes=1024,
            tda_correlation_id="demo-recovery-tda-001"
        )
        
        # Add to active checkpoints for demo
        self.hybrid_checkpointer.active_checkpoints[checkpoint_id] = mock_checkpoint
        
        # Mock the recovery methods for demo
        async def mock_conv_recovery(checkpoint_id):
            await asyncio.sleep(0.1)  # Simulate recovery time
            return {
                "type": "conversation",
                "checkpoint_id": checkpoint_id,
                "status": "recovered",
                "state": {
                    "messages": ["Recovered conversation state"],
                    "context": {"user_id": "demo-user"},
                    "memory": {"learned_patterns": ["pattern1", "pattern2"]}
                }
            }
        
        async def mock_wf_recovery(checkpoint_id):
            await asyncio.sleep(0.1)  # Simulate recovery time
            return {
                "type": "workflow",
                "checkpoint_id": checkpoint_id,
                "status": "recovered",
                "state": {
                    "current_step": "analysis",
                    "completed_steps": ["observation", "data_collection"],
                    "results": {"observation": "completed"}
                }
            }
        
        self.hybrid_checkpointer._recover_conversation_state = mock_conv_recovery
        self.hybrid_checkpointer._recover_workflow_state = mock_wf_recovery
        
        try:
            # Test different recovery modes
            recovery_modes = [
                RecoveryMode.PARALLEL_RECOVERY,
                RecoveryMode.CONVERSATION_FIRST,
                RecoveryMode.WORKFLOW_FIRST,
                RecoveryMode.INTELLIGENT_RECOVERY
            ]
            
            for mode in recovery_modes:
                print(f"\n   🔄 Testing {mode.value} recovery:")
                
                recovery_result = await self.hybrid_checkpointer.recover_from_hybrid_checkpoint(
                    checkpoint_id=checkpoint_id,
                    recovery_mode=mode,
                    tda_correlation_id="demo-recovery-tda-001"
                )
                
                if recovery_result["status"] == "success":
                    print(f"      ✅ Recovery successful in {recovery_result['recovery_time']:.3f}s")
                    print(f"      📊 Mode used: {recovery_result['recovery_mode']}")
                    print(f"      🔧 Components recovered: {len(recovery_result.get('recovery_results', []))}")
                else:
                    print(f"      ❌ Recovery failed: {recovery_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"   ❌ Recovery demo failed: {e}")
    
    async def get_system_metrics(self):
        """Get comprehensive system metrics"""
        print("\n📊 System Metrics:")
        
        # Hybrid checkpoint metrics
        checkpoint_metrics = self.hybrid_checkpointer.get_checkpoint_metrics()
        print("   Hybrid Checkpointing:")
        for key, value in checkpoint_metrics.items():
            print(f"      {key}: {value}")
        
        # Temporal orchestrator metrics
        temporal_metrics = self.temporal_orchestrator.get_execution_metrics()
        print("   Temporal Orchestration:")
        for key, value in temporal_metrics.items():
            print(f"      {key}: {value}")

async def main():
    """Run the enhanced orchestration demo"""
    demo = EnhancedAuraOrchestrationDemo()
    
    try:
        await demo.demonstrate_enhanced_features()
        await demo.get_system_metrics()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Enhanced AURA Orchestration Demo...")
    asyncio.run(main())