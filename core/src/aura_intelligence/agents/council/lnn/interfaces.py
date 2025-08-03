"""
Council Agent Interfaces

Defines clean interfaces following SOLID principles.
Each interface has a single responsibility and clear contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime

import torch

from .contracts import (
    CouncilRequest,
    CouncilResponse,
    ContextSnapshot,
    NeuralFeatures,
    DecisionEvidence,
    VoteDecision,
    VoteConfidence,
    AgentMetrics
)


class ICouncilAgent(ABC):
    """Core interface for council agents."""
    
    @abstractmethod
    async def process_request(self, request: CouncilRequest) -> CouncilResponse:
        """Process a council request and return a response."""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> AgentMetrics:
        """Get agent performance metrics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass


class INeuralEngine(ABC):
    """Interface for neural network operations."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the neural engine."""
        pass
    
    @abstractmethod
    async def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    async def adapt(self, feedback: Dict[str, Any]) -> None:
        """Adapt network based on feedback."""
        pass
    
    @abstractmethod
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get current network state."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load network state."""
        pass


class IContextProvider(ABC):
    """Interface for context retrieval and management."""
    
    @abstractmethod
    async def gather_context(
        self,
        request: CouncilRequest,
        scope: str = "local"
    ) -> ContextSnapshot:
        """Gather relevant context for the request."""
        pass
    
    @abstractmethod
    async def query_historical(
        self,
        entity_id: str,
        time_window: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query historical data."""
        pass
    
    @abstractmethod
    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get entity relationships."""
        pass
    
    @abstractmethod
    async def store_context(
        self,
        context: ContextSnapshot,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Store context for future use."""
        pass


class IFeatureExtractor(ABC):
    """Interface for feature extraction."""
    
    @abstractmethod
    async def extract_features(
        self,
        request: CouncilRequest,
        context: ContextSnapshot
    ) -> NeuralFeatures:
        """Extract neural features from request and context."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class IDecisionMaker(ABC):
    """Interface for decision making logic."""
    
    @abstractmethod
    async def make_decision(
        self,
        neural_output: torch.Tensor,
        features: NeuralFeatures,
        context: ContextSnapshot
    ) -> tuple[VoteDecision, VoteConfidence]:
        """Make a decision based on neural output."""
        pass
    
    @abstractmethod
    def get_decision_threshold(self) -> Dict[str, float]:
        """Get decision thresholds."""
        pass
    
    @abstractmethod
    def set_decision_threshold(self, thresholds: Dict[str, float]) -> None:
        """Set decision thresholds."""
        pass


class IEvidenceCollector(ABC):
    """Interface for evidence collection."""
    
    @abstractmethod
    async def collect_evidence(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        context: ContextSnapshot
    ) -> List[DecisionEvidence]:
        """Collect evidence supporting the decision."""
        pass
    
    @abstractmethod
    async def validate_evidence(
        self,
        evidence: List[DecisionEvidence]
    ) -> bool:
        """Validate evidence quality."""
        pass


class IReasoningEngine(ABC):
    """Interface for generating human-readable reasoning."""
    
    @abstractmethod
    async def generate_reasoning(
        self,
        request: CouncilRequest,
        decision: VoteDecision,
        confidence: VoteConfidence,
        evidence: List[DecisionEvidence],
        context: ContextSnapshot
    ) -> str:
        """Generate human-readable reasoning."""
        pass
    
    @abstractmethod
    async def explain_decision(
        self,
        response: CouncilResponse
    ) -> Dict[str, Any]:
        """Generate detailed explanation of decision."""
        pass


class IStorageAdapter(ABC):
    """Interface for storage operations."""
    
    @abstractmethod
    async def store_decision(
        self,
        response: CouncilResponse,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store decision and return storage ID."""
        pass
    
    @abstractmethod
    async def retrieve_decision(
        self,
        decision_id: str
    ) -> Optional[CouncilResponse]:
        """Retrieve stored decision."""
        pass
    
    @abstractmethod
    async def query_decisions(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[CouncilResponse]:
        """Query stored decisions."""
        pass


class IEventPublisher(ABC):
    """Interface for event publishing."""
    
    @abstractmethod
    async def publish_decision(
        self,
        response: CouncilResponse,
        topic: Optional[str] = None
    ) -> None:
        """Publish decision event."""
        pass
    
    @abstractmethod
    async def publish_metrics(
        self,
        metrics: AgentMetrics,
        topic: Optional[str] = None
    ) -> None:
        """Publish metrics event."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        topics: List[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to event topics."""
        pass


class IMemoryManager(ABC):
    """Interface for memory management."""
    
    @abstractmethod
    async def store_experience(
        self,
        request: CouncilRequest,
        response: CouncilResponse,
        outcome: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store experience for learning."""
        pass
    
    @abstractmethod
    async def recall_similar(
        self,
        request: CouncilRequest,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Recall similar past experiences."""
        pass
    
    @abstractmethod
    async def update_outcome(
        self,
        request_id: str,
        outcome: Dict[str, Any]
    ) -> None:
        """Update outcome of past decision."""
        pass


class IResourceManager(ABC):
    """Interface for resource lifecycle management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize resources."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, bool]:
        """Check health of resources."""
        pass
    
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        pass