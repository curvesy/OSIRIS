"""
🌐 Distributed Orchestration Example

Demonstrates the power of AURA's distributed orchestration system combining
Ray Serve, CrewAI Flows, and hybrid checkpointing for enterprise-scale
AI agent coordination.

This example shows:
- Ray Serve agent ensemble deployments
- CrewAI Flows hierarchical coordination
- Distributed coordination across systems
- Cross-service checkpointing
- TDA-aware distributed decision making
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

# Distributed orchestration imports
from aura_intelligence.orchestration.distributed import (
    DistributedCoordinator,
    RayServeOrchestrator,
    CrewAIFlowOrchestrator,
    DistributedExecutionPlan,
    DistributedExecutionMode,
    DistributedRecoveryStrategy,
    AgentType,
    DistributedAgentConfig,
    HierarchicalFlowConfig,
    FlowLevel,
    RAY_SERVE_AVAILABLE,
    CREWAI_FLOWS_AVAILABLE
)

from aura_intelligence.orchestration.durable import (
    HybridCheckpointManager,
    HybridCheckpointConfig,
    CheckpointLevel,
    RecoveryMode
)

class DistributedOrchestrationDemo:
    """
    Demonstrates the distributed orchestration capabilities
    """
    
    def __init__(self):
        # Initialize hybrid checkpointer
        hybrid_config = HybridCheckpointConfig(
            postgres_url="postgresql://aura:aura@localhost:5432/aura_dev",
            checkpoint_level=CheckpointLevel.HYBRID,
            recovery_mode=RecoveryMode.INTELLIGENT_RECOVERY,
            enable_tda_optimization=True
        )
        self.hybrid_checkpointer = HybridCheckpointManager(config=hybrid_config)
        
        # Initialize distributed coordinator
        self.distributed_coordinator = DistributedCoordinator(
            hybrid_checkpointer=self.hybrid_checkpointer
        )
    
    async def demonstrate_distributed_orchestration(self):
        """Demonstrate all distributed orchestration features"""
        print("🌐 Distributed Orchestration Demo")
        print("=" * 50)
        
        # Check feature availability
        await self._check_feature_availability()
        
        # Initialize distributed systems
        await self._initialize_systems()
        
        # Demonstrate Ray Serve agent ensembles
        await self._demo_ray_serve_ensembles()
        
        # Demonstrate CrewAI Flows coordination
        await self._demo_crewai_flows_coordination()
        
        # Demonstrate distributed coordination
        await self._demo_distributed_coordination()
        
        # Demonstrate cross-service checkpointing
        await self._demo_cross_service_checkpointing()
        
        # Demonstrate intelligent execution mode selection
        await self._demo_intelligent_execution_modes()
        
        # Show comprehensive metrics
        await self._show_comprehensive_metrics()
        
        print("\n✅ Distributed orchestration demo completed successfully!")
    
    async def _check_feature_availability(self):
        """Check which distributed features are available"""
        print("\n📋 Distributed Feature Availability:")
        print(f"   Ray Serve Available: {RAY_SERVE_AVAILABLE}")
        print(f"   CrewAI Flows Available: {CREWAI_FLOWS_AVAILABLE}")
        print(f"   Hybrid Checkpointing: {self.hybrid_checkpointer is not None}")
        print(f"   Distributed Coordination: ✅ Available")
    
    async def _initialize_systems(self):
        """Initialize all distributed systems"""
        print("\n🚀 Initializing Distributed Systems:")
        
        try:
            await self.distributed_coordinator.initialize_distributed_systems()
            print("   ✅ All distributed systems initialized successfully")
        except Exception as e:
            print(f"   ⚠️  Initialization completed with fallbacks: {e}")
    
    async def _demo_ray_serve_ensembles(self):
        """Demonstrate Ray Serve agent ensemble deployments"""
        print("\n🎯 Ray Serve Agent Ensembles Demo:")
        
        # Create agent configurations
        agent_configs = [
            DistributedAgentConfig(
                agent_type=AgentType.OBSERVER,
                num_replicas=2,
                min_replicas=1,
                max_replicas=5,
                cpu_per_replica=2.0,
                enable_tda_integration=True
            ),
            DistributedAgentConfig(
                agent_type=AgentType.ANALYST,
                num_replicas=3,
                min_replicas=2,
                max_replicas=8,
                cpu_per_replica=4.0,
                gpu_per_replica=0.5,
                enable_tda_integration=True
            ),
            DistributedAgentConfig(
                agent_type=AgentType.SUPERVISOR,
                num_replicas=1,
                min_replicas=1,
                max_replicas=3,
                cpu_per_replica=2.0,
                enable_tda_integration=True
            )
        ]
        
        # Deploy agent ensembles
        for i, config in enumerate(agent_configs):
            deployment_name = f"demo_{config.agent_type.value}_ensemble_{i}"
            
            try:
                result = await self.distributed_coordinator.ray_orchestrator.deploy_agent_ensemble(
                    deployment_name, config
                )
                print(f"   ✅ Deployed {config.agent_type.value} ensemble: {result}")
                print(f"      📊 Replicas: {config.min_replicas}-{config.max_replicas}")
                print(f"      💻 Resources: {config.cpu_per_replica} CPU, {config.gpu_per_replica} GPU")
            except Exception as e:
                print(f"   ⚠️  Deployment {deployment_name} failed: {e}")
        
        # Show cluster metrics
        metrics = self.distributed_coordinator.ray_orchestrator.get_cluster_metrics()
        print(f"   📈 Cluster Metrics:")
        print(f"      Active Deployments: {metrics['active_deployments']}")
        print(f"      Total Agents: {metrics['total_agents']}")
        print(f"      Total Requests: {metrics['total_requests']}")
    
    async def _demo_crewai_flows_coordination(self):
        """Demonstrate CrewAI Flows hierarchical coordination"""
        print("\n🤖 CrewAI Flows Hierarchical Coordination Demo:")
        
        # Create hierarchical flow configurations
        flow_configs = [
            {
                "level": FlowLevel.STRATEGIC,
                "name": "Strategic Planning Flow",
                "agents": ["supervisor", "coordinator"],
                "execution_mode": "sequential"
            },
            {
                "level": FlowLevel.TACTICAL,
                "name": "Tactical Coordination Flow",
                "agents": ["coordinator", "analyst", "observer"],
                "execution_mode": "parallel"
            },
            {
                "level": FlowLevel.OPERATIONAL,
                "name": "Operational Execution Flow",
                "agents": ["observer", "analyst", "executor"],
                "execution_mode": "hybrid"
            }
        ]
        
        # Execute flows at different levels
        for i, config in enumerate(flow_configs):
            flow_config = HierarchicalFlowConfig(
                flow_id=f"demo_flow_{config['level'].value}_{i}",
                flow_name=config["name"],
                flow_level=config["level"],
                execution_mode=config["execution_mode"],
                agents=config["agents"],
                flow_steps=[
                    {"name": "initialization", "type": "setup"},
                    {"name": "execution", "type": "process"},
                    {"name": "completion", "type": "finalize"}
                ],
                tda_correlation_id=f"demo_tda_correlation_{i}"
            )
            
            try:
                # Create and execute flow
                flow = await self.distributed_coordinator.crewai_orchestrator.create_hierarchical_flow(
                    flow_config
                )
                
                result = await flow.execute_flow({
                    "demo_input": f"Hierarchical flow demo data for {config['level'].value}",
                    "complexity": "medium",
                    "priority": "high"
                })
                
                print(f"   ✅ {config['level'].value.title()} Flow Completed:")
                print(f"      📊 Status: {result.status.value}")
                print(f"      ⏱️  Execution Time: {result.execution_time:.2f}s")
                print(f"      🎯 Steps Executed: {len(result.steps_executed)}")
                print(f"      🔗 TDA Enhanced: {result.tda_correlation_id is not None}")
                
            except Exception as e:
                print(f"   ❌ {config['level'].value.title()} Flow Failed: {e}")
        
        # Show orchestration metrics
        metrics = self.distributed_coordinator.crewai_orchestrator.get_orchestration_metrics()
        print(f"   📈 Flow Orchestration Metrics:")
        print(f"      Total Flows: {metrics['total_flows']}")
        print(f"      Success Rate: {metrics['success_rate']:.2%}")
        print(f"      Average Execution Time: {metrics['average_execution_time']:.2f}s")
    
    async def _demo_distributed_coordination(self):
        """Demonstrate distributed coordination across systems"""
        print("\n🌐 Distributed Coordination Demo:")
        
        # Test different execution modes
        execution_modes = [
            DistributedExecutionMode.RAY_SERVE_ONLY,
            DistributedExecutionMode.CREWAI_FLOWS_ONLY,
            DistributedExecutionMode.HYBRID_COORDINATION,
            DistributedExecutionMode.INTELLIGENT_SELECTION
        ]
        
        for mode in execution_modes:
            print(f"\n   🔄 Testing {mode.value.replace('_', ' ').title()} Mode:")
            
            try:
                # Create execution plan
                request_data = {
                    "task_type": "comprehensive_analysis",
                    "required_agents": ["observer", "analyst", "supervisor"],
                    "complexity": "high",
                    "urgency": "medium",
                    "tda_correlation_id": f"demo_coordination_{mode.value}"
                }
                
                plan = await self.distributed_coordinator.create_distributed_execution_plan(
                    request_data, execution_mode=mode
                )
                
                print(f"      ✅ Execution Plan Created: {plan.plan_id}")
                print(f"      📊 Ray Deployments: {len(plan.ray_serve_deployments)}")
                print(f"      🤖 CrewAI Flows: {len(plan.crewai_flows)}")
                print(f"      🔧 Coordination Strategy: {plan.coordination_strategy}")
                print(f"      💾 Checkpoint Strategy: {plan.checkpoint_strategy.value}")
                
                # Execute the plan
                result = await self.distributed_coordinator.execute_distributed_plan(plan)
                
                print(f"      ✅ Execution Completed:")
                print(f"         Status: {result['status']}")
                print(f"         Execution Time: {result['execution_time']:.2f}s")
                print(f"         Checkpoint ID: {result.get('checkpoint_id', 'None')}")
                
            except Exception as e:
                print(f"      ❌ Execution Mode {mode.value} Failed: {e}")
    
    async def _demo_cross_service_checkpointing(self):
        """Demonstrate cross-service checkpointing"""
        print("\n💾 Cross-Service Checkpointing Demo:")
        
        # Create a complex distributed plan
        request_data = {
            "task_type": "multi_system_analysis",
            "required_agents": ["observer", "analyst", "supervisor", "coordinator"],
            "complexity": "very_high",
            "urgency": "critical",
            "tda_correlation_id": "demo_checkpoint_correlation"
        }
        
        try:
            # Create execution plan
            plan = await self.distributed_coordinator.create_distributed_execution_plan(
                request_data, execution_mode=DistributedExecutionMode.HYBRID_COORDINATION
            )
            
            print(f"   📋 Created Distributed Plan: {plan.plan_id}")
            print(f"      Services Involved: Ray Serve + CrewAI Flows + Hybrid Checkpointer")
            
            # Execute with checkpointing
            result = await self.distributed_coordinator.execute_distributed_plan(plan)
            
            print(f"   ✅ Execution with Cross-Service Checkpointing:")
            print(f"      Status: {result['status']}")
            print(f"      Execution Time: {result['execution_time']:.2f}s")
            print(f"      Cross-Service Checkpoint: {result.get('checkpoint_id', 'None')}")
            
            # Show checkpoint metrics
            if self.hybrid_checkpointer:
                checkpoint_metrics = self.hybrid_checkpointer.get_checkpoint_metrics()
                print(f"   📊 Checkpoint Metrics:")
                print(f"      Total Checkpoints: {checkpoint_metrics['total_checkpoints']}")
                print(f"      Hybrid Checkpoints: {checkpoint_metrics.get('hybrid_checkpoints', 0)}")
                print(f"      Recovery Success Rate: {checkpoint_metrics.get('recovery_success_rate', 0):.2%}")
            
        except Exception as e:
            print(f"   ❌ Cross-Service Checkpointing Demo Failed: {e}")
    
    async def _demo_intelligent_execution_modes(self):
        """Demonstrate intelligent execution mode selection"""
        print("\n🧠 Intelligent Execution Mode Selection Demo:")
        
        # Test different scenarios that should trigger different modes
        scenarios = [
            {
                "name": "Simple Request",
                "data": {
                    "task_type": "basic_monitoring",
                    "required_agents": ["observer"],
                    "complexity": "low",
                    "urgency": "low",
                    "tda_correlation_id": "demo_simple"
                },
                "expected_mode": "Ray Serve Only"
            },
            {
                "name": "Complex Analysis",
                "data": {
                    "task_type": "deep_pattern_analysis",
                    "required_agents": ["observer", "analyst", "supervisor"],
                    "complexity": "very_high",
                    "urgency": "medium",
                    "tda_correlation_id": "demo_complex"
                },
                "expected_mode": "CrewAI Flows or Hybrid"
            },
            {
                "name": "Critical Incident",
                "data": {
                    "task_type": "incident_response",
                    "required_agents": ["observer", "analyst", "supervisor", "executor"],
                    "complexity": "high",
                    "urgency": "critical",
                    "anomaly_severity": 0.95,
                    "tda_correlation_id": "demo_critical"
                },
                "expected_mode": "Hybrid Coordination"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n   🎯 Scenario: {scenario['name']}")
            print(f"      Expected Mode: {scenario['expected_mode']}")
            
            try:
                # Let the system intelligently select execution mode
                plan = await self.distributed_coordinator.create_distributed_execution_plan(
                    scenario["data"]  # No execution_mode specified - let it choose
                )
                
                print(f"      ✅ Selected Mode: {plan.execution_mode.value.replace('_', ' ').title()}")
                print(f"      🔧 Coordination Strategy: {plan.coordination_strategy}")
                print(f"      📊 Ray Deployments: {len(plan.ray_serve_deployments)}")
                print(f"      🤖 CrewAI Flows: {len(plan.crewai_flows)}")
                print(f"      🛡️  Recovery Strategy: {plan.recovery_strategy.value}")
                
                # Quick execution to test the mode
                result = await self.distributed_coordinator.execute_distributed_plan(plan)
                print(f"      ✅ Execution Result: {result['status']} ({result['execution_time']:.2f}s)")
                
            except Exception as e:
                print(f"      ❌ Scenario {scenario['name']} Failed: {e}")
    
    async def _show_comprehensive_metrics(self):
        """Show comprehensive metrics from all systems"""
        print("\n📊 Comprehensive System Metrics:")
        
        # Distributed coordination metrics
        coord_metrics = self.distributed_coordinator.get_coordination_metrics()
        print(f"\n   🌐 Distributed Coordination:")
        print(f"      Total Executions: {coord_metrics['total_executions']}")
        print(f"      Success Rate: {coord_metrics['success_rate']:.2%}")
        print(f"      Average Execution Time: {coord_metrics['average_execution_time']:.2f}s")
        print(f"      Cross-Service Checkpoints: {coord_metrics['cross_service_checkpoints']}")
        print(f"      Recovery Operations: {coord_metrics['recovery_operations']}")
        
        # Ray Serve metrics
        ray_metrics = coord_metrics.get('ray_serve_metrics', {})
        if ray_metrics:
            print(f"\n   🎯 Ray Serve Cluster:")
            print(f"      Active Deployments: {ray_metrics.get('active_deployments', 0)}")
            print(f"      Total Agents: {ray_metrics.get('total_agents', 0)}")
            print(f"      Total Requests: {ray_metrics.get('total_requests', 0)}")
            print(f"      Total Errors: {ray_metrics.get('total_errors', 0)}")
        
        # CrewAI Flows metrics
        crewai_metrics = coord_metrics.get('crewai_flows_metrics', {})
        if crewai_metrics:
            print(f"\n   🤖 CrewAI Flows:")
            print(f"      Total Flows: {crewai_metrics.get('total_flows', 0)}")
            print(f"      Successful Flows: {crewai_metrics.get('successful_flows', 0)}")
            print(f"      Failed Flows: {crewai_metrics.get('failed_flows', 0)}")
            print(f"      Average Execution Time: {crewai_metrics.get('average_execution_time', 0):.2f}s")
        
        # Hybrid checkpointing metrics
        if self.hybrid_checkpointer:
            checkpoint_metrics = self.hybrid_checkpointer.get_checkpoint_metrics()
            print(f"\n   💾 Hybrid Checkpointing:")
            print(f"      Total Checkpoints: {checkpoint_metrics['total_checkpoints']}")
            print(f"      Active Checkpoints: {checkpoint_metrics['active_checkpoints']}")
            print(f"      Recovery Operations: {checkpoint_metrics['recovery_operations']}")
            print(f"      TDA Optimizations: {checkpoint_metrics['tda_optimizations']}")
    
    async def cleanup_demo(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo resources...")
        
        try:
            await self.distributed_coordinator.shutdown_distributed_systems()
            print("   ✅ All systems shutdown successfully")
        except Exception as e:
            print(f"   ⚠️  Cleanup completed with warnings: {e}")

async def main():
    """Run the distributed orchestration demo"""
    demo = DistributedOrchestrationDemo()
    
    try:
        await demo.demonstrate_distributed_orchestration()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await demo.cleanup_demo()

if __name__ == "__main__":
    print("🌐 Starting Distributed Orchestration Demo...")
    asyncio.run(main())