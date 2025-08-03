"""
Council Agent Factory

Factory for creating LNN council agents with various configurations.
"""

from typing import Dict, Any, Optional, List
from uuid import uuid4

from .agent import LNNCouncilAgent
from .contracts import AgentCapability
from .neural import LiquidNeuralEngine, NeuralConfig
from .implementations import (
    DefaultContextProvider,
    DefaultFeatureExtractor,
    DefaultDecisionMaker,
    DefaultEvidenceCollector,
    DefaultReasoningEngine,
    Neo4jStorageAdapter,
    KafkaEventPublisher,
    Mem0MemoryManager,
    DefaultResourceManager
)
from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter
from aura_intelligence.events.producers import EventProducer
from aura_intelligence.adapters.mem0_adapter import Mem0Adapter as Mem0Manager


class CouncilAgentFactory:
    """Factory for creating LNN council agents."""
    
    @staticmethod
    def create_default_agent(
        agent_id: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        neural_config: Optional[Dict[str, Any]] = None
    ) -> LNNCouncilAgent:
        """
        Create a default LNN council agent with basic configuration.
        
        Args:
            agent_id: Optional agent ID (generated if not provided)
            capabilities: Optional list of capabilities
            neural_config: Optional neural network configuration
            
        Returns:
            Configured LNN council agent
        """
        # Default values
        if agent_id is None:
            agent_id = f"council_agent_{uuid4().hex[:8]}"
        
        if capabilities is None:
            capabilities = [
                AgentCapability.GPU_ALLOCATION,
                AgentCapability.RESOURCE_MANAGEMENT
            ]
        
        # Create neural engine with proper defaults
        default_neural_config = {
            "input_size": 256,
            "hidden_sizes": [128, 64],
            "output_size": 4,
            "time_constant": 1.0,
            "learning_rate": 0.001
        }
        if neural_config:
            default_neural_config.update(neural_config)
        
        config = NeuralConfig(**default_neural_config)
        neural_engine = LiquidNeuralEngine(config)
        
        # Create core components
        context_provider = DefaultContextProvider()
        feature_extractor = DefaultFeatureExtractor(target_size=default_neural_config["input_size"])
        decision_maker = DefaultDecisionMaker()
        evidence_collector = DefaultEvidenceCollector()
        reasoning_engine = DefaultReasoningEngine()
        
        # Create agent
        return LNNCouncilAgent(
            agent_id=agent_id,
            capabilities=capabilities,
            neural_engine=neural_engine,
            context_provider=context_provider,
            feature_extractor=feature_extractor,
            decision_maker=decision_maker,
            evidence_collector=evidence_collector,
            reasoning_engine=reasoning_engine
        )
    
    @staticmethod
    def create_production_agent(
        agent_id: str,
        capabilities: List[AgentCapability],
        neural_config: Dict[str, Any],
        neo4j_adapter: Neo4jAdapter,
        event_producer: EventProducer,
        memory_manager: Mem0Manager,
        enable_resource_management: bool = True
    ) -> LNNCouncilAgent:
        """
        Create a production-ready LNN council agent with full integration.
        
        Args:
            agent_id: Agent identifier
            capabilities: List of agent capabilities
            neural_config: Neural network configuration
            neo4j_adapter: Neo4j adapter for storage
            event_producer: Event producer for publishing
            memory_manager: Memory manager for experience storage
            enable_resource_management: Whether to enable resource management
            
        Returns:
            Production-configured LNN council agent
        """
        # Create neural engine
        config = NeuralConfig(**neural_config)
        neural_engine = LiquidNeuralEngine(config)
        
        # Create core components with production implementations
        context_provider = DefaultContextProvider(neo4j_adapter)
        feature_extractor = DefaultFeatureExtractor()
        decision_maker = DefaultDecisionMaker()
        evidence_collector = DefaultEvidenceCollector()
        reasoning_engine = DefaultReasoningEngine()
        
        # Create storage and event adapters
        storage_adapter = Neo4jStorageAdapter(neo4j_adapter)
        event_publisher = KafkaEventPublisher(event_producer)
        memory_adapter = Mem0MemoryManager(memory_manager)
        
        # Optional resource manager
        resource_manager = DefaultResourceManager() if enable_resource_management else None
        
        # Create agent
        return LNNCouncilAgent(
            agent_id=agent_id,
            capabilities=capabilities,
            neural_engine=neural_engine,
            context_provider=context_provider,
            feature_extractor=feature_extractor,
            decision_maker=decision_maker,
            evidence_collector=evidence_collector,
            reasoning_engine=reasoning_engine,
            storage_adapter=storage_adapter,
            event_publisher=event_publisher,
            memory_manager=memory_adapter,
            resource_manager=resource_manager
        )
    
    @staticmethod
    def create_specialized_agent(
        agent_type: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> LNNCouncilAgent:
        """
        Create a specialized agent for specific use cases.
        
        Args:
            agent_type: Type of specialized agent
            agent_id: Optional agent ID
            **kwargs: Additional configuration parameters
            
        Returns:
            Specialized LNN council agent
        """
        specialized_configs = {
            "gpu_specialist": {
                "capabilities": [
                    AgentCapability.GPU_ALLOCATION,
                    AgentCapability.COST_OPTIMIZATION
                ],
                "neural_config": {
                    "input_size": 128,
                    "hidden_sizes": [64, 32],
                    "output_size": 4,
                    "time_constant": 0.8,
                    "sparsity": 0.6
                }
            },
            "risk_assessor": {
                "capabilities": [
                    AgentCapability.RISK_ASSESSMENT,
                    AgentCapability.COMPLIANCE_CHECK
                ],
                "neural_config": {
                    "input_size": 256,
                    "hidden_sizes": [128, 64, 32],
                    "output_size": 4,
                    "time_constant": 1.2,
                    "sparsity": 0.8,
                    "dropout_rate": 0.2
                }
            },
            "cost_optimizer": {
                "capabilities": [
                    AgentCapability.COST_OPTIMIZATION,
                    AgentCapability.RESOURCE_MANAGEMENT
                ],
                "neural_config": {
                    "input_size": 192,
                    "hidden_sizes": [96, 48],
                    "output_size": 4,
                    "time_constant": 1.0,
                    "learning_rate": 0.0005
                }
            }
        }
        
        if agent_type not in specialized_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        config = specialized_configs[agent_type]
        
        # Override with kwargs
        if "capabilities" in kwargs:
            config["capabilities"] = kwargs["capabilities"]
        if "neural_config" in kwargs:
            config["neural_config"].update(kwargs["neural_config"])
        
        return CouncilAgentFactory.create_default_agent(
            agent_id=agent_id or f"{agent_type}_{uuid4().hex[:8]}",
            capabilities=config["capabilities"],
            neural_config=config["neural_config"]
        )
    
    @staticmethod
    def create_multi_agent_council(
        council_size: int = 3,
        agent_types: Optional[List[str]] = None,
        shared_memory: Optional[Mem0Manager] = None
    ) -> List[LNNCouncilAgent]:
        """
        Create a council of multiple agents for consensus decision making.
        
        Args:
            council_size: Number of agents in the council
            agent_types: Optional list of agent types
            shared_memory: Optional shared memory manager
            
        Returns:
            List of configured agents
        """
        if agent_types is None:
            agent_types = ["gpu_specialist", "risk_assessor", "cost_optimizer"]
        
        agents = []
        
        for i in range(council_size):
            agent_type = agent_types[i % len(agent_types)]
            agent = CouncilAgentFactory.create_specialized_agent(
                agent_type=agent_type,
                agent_id=f"{agent_type}_{i}"
            )
            
            # Set shared memory if provided
            if shared_memory and hasattr(agent, 'memory_manager'):
                agent.memory_manager = Mem0MemoryManager(shared_memory)
            
            agents.append(agent)
        
        return agents