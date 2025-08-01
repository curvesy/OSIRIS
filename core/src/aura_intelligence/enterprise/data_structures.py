"""
🏗️ AURA Intelligence Enterprise Data Structures

Core data structures for the Topological Search & Memory Layer.
Based on kiki.md and ppdd.md research - the "soul" that transforms 
raw computational power into true intelligence.

This implements the professional system design from your research:
- TopologicalSignature for persistence diagrams
- SystemEvent for contextual relationships
- AgentAction for decision tracking
- Outcome for learning feedback loops
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import hashlib
import json
import numpy as np


@dataclass
class TopologicalSignature:
    """
    Core topological signature from TDA analysis.
    
    This is the fundamental unit of topological intelligence - each signature
    represents a unique structural pattern discovered by our Mojo TDA engine.
    """
    betti_numbers: List[int]
    persistence_diagram: Dict[str, Any]
    agent_context: Dict[str, Any]
    timestamp: datetime
    signature_hash: str
    consciousness_level: float = 0.5
    quantum_coherence: float = 0.0
    algorithm_used: str = "unknown"
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate signature hash if not provided."""
        if not self.signature_hash:
            self.signature_hash = self.generate_hash()
    
    def generate_hash(self) -> str:
        """Generate unique hash for this topological signature."""
        content = {
            "betti_numbers": self.betti_numbers,
            "persistence_diagram": self.persistence_diagram,
            "timestamp": self.timestamp.isoformat(),
            "consciousness_level": round(self.consciousness_level, 3)
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def to_vector(self) -> List[float]:
        """Convert signature to vector for similarity search."""
        # Vectorize Betti numbers (normalized)
        betti_vector = [float(b) for b in self.betti_numbers[:3]]  # Take first 3
        while len(betti_vector) < 3:
            betti_vector.append(0.0)
        
        # Add consciousness and quantum features
        consciousness_vector = [
            self.consciousness_level,
            self.quantum_coherence,
            len(self.agent_context.get("agents", [])) / 7.0  # Normalized agent count
        ]
        
        # Add persistence features (simplified)
        persistence_features = []
        if "birth_death_pairs" in self.persistence_diagram:
            pairs = self.persistence_diagram["birth_death_pairs"][:5]  # Take first 5
            for pair in pairs:
                persistence_features.extend([pair.get("birth", 0.0), pair.get("death", 1.0)])
        
        # Pad to fixed size
        while len(persistence_features) < 10:
            persistence_features.append(0.0)
        
        return betti_vector + consciousness_vector + persistence_features[:10]


@dataclass
class SystemEvent:
    """
    System event that triggered TDA analysis.
    
    Represents the contextual situation that led to a topological signature.
    This enables causal reasoning and pattern understanding.
    """
    event_id: str
    event_type: str
    timestamp: datetime
    system_state: Dict[str, Any]
    triggering_agents: List[str]
    consciousness_state: Dict[str, Any]
    severity_level: str = "normal"  # normal, warning, critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = f"evt_{int(self.timestamp.timestamp())}_{hash(self.event_type) % 10000}"


@dataclass
class AgentAction:
    """
    Action taken by an agent in response to a topological signature.
    
    This captures the decision-making process and enables learning
    from successful and unsuccessful actions.
    """
    action_id: str
    agent_id: str
    action_type: str
    timestamp: datetime
    input_signature: str  # Reference to TopologicalSignature hash
    decision_context: Dict[str, Any]
    action_parameters: Dict[str, Any]
    confidence_score: float = 0.5
    reasoning: str = ""
    
    def __post_init__(self):
        """Generate action ID if not provided."""
        if not self.action_id:
            self.action_id = f"act_{self.agent_id}_{int(self.timestamp.timestamp())}"


@dataclass
class Outcome:
    """
    Outcome of an agent action.
    
    This completes the learning loop by capturing the results of actions,
    enabling the system to learn from experience and improve over time.
    """
    outcome_id: str
    action_id: str  # Reference to AgentAction
    timestamp: datetime
    success: bool
    impact_score: float  # -1.0 to 1.0
    metrics: Dict[str, Any]
    learned_insights: List[str] = field(default_factory=list)
    follow_up_actions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate outcome ID if not provided."""
        if not self.outcome_id:
            self.outcome_id = f"out_{self.action_id}_{int(self.timestamp.timestamp())}"


# Pydantic models for API serialization
class TopologicalSignatureAPI(BaseModel):
    """API model for topological signatures."""
    betti_numbers: List[int] = Field(..., description="Betti numbers from TDA analysis")
    persistence_diagram: Dict[str, Any] = Field(..., description="Persistence diagram data")
    agent_context: Dict[str, Any] = Field(..., description="Agent context information")
    timestamp: datetime = Field(default_factory=datetime.now)
    consciousness_level: float = Field(0.5, ge=0.0, le=1.0)
    quantum_coherence: float = Field(0.0, ge=0.0, le=1.0)
    algorithm_used: str = Field("unknown", description="TDA algorithm used")
    
    def to_dataclass(self) -> TopologicalSignature:
        """Convert to dataclass for internal processing."""
        return TopologicalSignature(
            betti_numbers=self.betti_numbers,
            persistence_diagram=self.persistence_diagram,
            agent_context=self.agent_context,
            timestamp=self.timestamp,
            signature_hash="",  # Will be generated
            consciousness_level=self.consciousness_level,
            quantum_coherence=self.quantum_coherence,
            algorithm_used=self.algorithm_used
        )


class SearchResult(BaseModel):
    """Result from topological search."""
    query_signature: str = Field(..., description="Hash of query signature")
    similar_signatures: List[Dict[str, Any]] = Field(..., description="Similar signatures found")
    contextual_insights: List[Dict[str, Any]] = Field(..., description="Contextual information")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Search confidence")
    search_time_ms: float = Field(..., description="Search execution time")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")


class SystemEventAPI(BaseModel):
    """API model for system events."""
    event_type: str = Field(..., description="Type of system event")
    system_state: Dict[str, Any] = Field(..., description="Current system state")
    triggering_agents: List[str] = Field(..., description="Agents that triggered this event")
    consciousness_state: Dict[str, Any] = Field(..., description="Consciousness state")
    severity_level: str = Field("normal", description="Event severity")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentActionAPI(BaseModel):
    """API model for agent actions."""
    agent_id: str = Field(..., description="ID of the acting agent")
    action_type: str = Field(..., description="Type of action taken")
    input_signature: str = Field(..., description="Input topological signature hash")
    decision_context: Dict[str, Any] = Field(..., description="Decision context")
    action_parameters: Dict[str, Any] = Field(..., description="Action parameters")
    confidence_score: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = Field("", description="Reasoning for the action")


class OutcomeAPI(BaseModel):
    """API model for outcomes."""
    action_id: str = Field(..., description="ID of the related action")
    success: bool = Field(..., description="Whether the action was successful")
    impact_score: float = Field(..., ge=-1.0, le=1.0, description="Impact score")
    metrics: Dict[str, Any] = Field(..., description="Outcome metrics")
    learned_insights: List[str] = Field(default_factory=list)
    follow_up_actions: List[str] = Field(default_factory=list)


# Utility functions for data processing
def vectorize_persistence_diagram(diagram: Dict[str, Any]) -> List[float]:
    """
    Convert persistence diagram to vector representation.
    
    This implements the vectorization strategy from your research for
    similarity search in the vector database.
    """
    vector = []
    
    # Extract birth-death pairs
    if "birth_death_pairs" in diagram:
        pairs = diagram["birth_death_pairs"][:10]  # Limit to first 10 pairs
        for pair in pairs:
            birth = pair.get("birth", 0.0)
            death = pair.get("death", 1.0)
            persistence = death - birth
            vector.extend([birth, death, persistence])
    
    # Pad to fixed size (30 dimensions: 10 pairs * 3 values)
    while len(vector) < 30:
        vector.append(0.0)
    
    return vector[:30]


def calculate_signature_similarity(sig1: TopologicalSignature, sig2: TopologicalSignature) -> float:
    """
    Calculate similarity between two topological signatures.
    
    Uses a combination of Betti number similarity and consciousness correlation.
    """
    # Betti number similarity (Jaccard-like)
    betti1 = np.array(sig1.betti_numbers[:3])
    betti2 = np.array(sig2.betti_numbers[:3])
    
    # Pad to same length
    max_len = max(len(betti1), len(betti2))
    betti1 = np.pad(betti1, (0, max_len - len(betti1)))
    betti2 = np.pad(betti2, (0, max_len - len(betti2)))
    
    # Calculate similarity
    if np.sum(betti1) + np.sum(betti2) == 0:
        betti_similarity = 1.0
    else:
        intersection = np.minimum(betti1, betti2)
        union = np.maximum(betti1, betti2)
        betti_similarity = np.sum(intersection) / np.sum(union)
    
    # Consciousness similarity
    consciousness_similarity = 1.0 - abs(sig1.consciousness_level - sig2.consciousness_level)
    
    # Quantum coherence similarity
    quantum_similarity = 1.0 - abs(sig1.quantum_coherence - sig2.quantum_coherence)
    
    # Weighted combination
    total_similarity = (
        betti_similarity * 0.5 +
        consciousness_similarity * 0.3 +
        quantum_similarity * 0.2
    )
    
    return min(1.0, max(0.0, total_similarity))


# Export all models for API use
__all__ = [
    "TopologicalSignature",
    "SystemEvent", 
    "AgentAction",
    "Outcome",
    "TopologicalSignatureAPI",
    "SearchResult",
    "SystemEventAPI",
    "AgentActionAPI",
    "OutcomeAPI",
    "vectorize_persistence_diagram",
    "calculate_signature_similarity"
]
