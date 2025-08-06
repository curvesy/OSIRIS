"""
Causal Store for AURA Intelligence
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CausalRelation:
    """Represents a causal relationship."""
    cause: str
    effect: str
    confidence: float
    timestamp: datetime


@dataclass
class CausalPattern:
    """Represents a causal pattern with multiple relationships."""
    pattern_id: str
    name: str
    description: str
    relations: List[CausalRelation]
    confidence: float
    evidence: List[Dict[str, Any]]
    created_at: datetime
    metadata: Dict[str, Any]


class CausalPatternStore:
    """Store for causal patterns (alias for CausalStore)."""
    def __init__(self):
        self.store = CausalStore()
        self.patterns: List[CausalPattern] = []
        
    def add_pattern(self, pattern: Dict[str, Any]):
        if 'cause' in pattern and 'effect' in pattern:
            self.store.add_relation(
                pattern['cause'], 
                pattern['effect'], 
                pattern.get('confidence', 0.8)
            )
        
        # Also store as CausalPattern if it has pattern structure
        if 'pattern_id' in pattern:
            causal_pattern = CausalPattern(
                pattern_id=pattern['pattern_id'],
                name=pattern.get('name', ''),
                description=pattern.get('description', ''),
                relations=[],
                confidence=pattern.get('confidence', 0.8),
                evidence=pattern.get('evidence', []),
                created_at=datetime.now(),
                metadata=pattern.get('metadata', {})
            )
            self.patterns.append(causal_pattern)
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        patterns = []
        
        # Get patterns from relations
        for r in self.store.relations:
            patterns.append({
                'cause': r.cause,
                'effect': r.effect,
                'confidence': r.confidence,
                'timestamp': r.timestamp
            })
        
        # Add stored CausalPatterns
        for p in self.patterns:
            patterns.append({
                'pattern_id': p.pattern_id,
                'name': p.name,
                'description': p.description,
                'confidence': p.confidence,
                'evidence': p.evidence,
                'created_at': p.created_at,
                'metadata': p.metadata
            })
        
        return patterns


class CausalStore:
    """Store for causal relationships and patterns."""
    def __init__(self):
        self.relations: List[CausalRelation] = []
        
    def add_relation(self, cause: str, effect: str, confidence: float = 0.8):
        self.relations.append(CausalRelation(
            cause=cause,
            effect=effect,
            confidence=confidence,
            timestamp=datetime.now()
        ))
    
    def get_causes(self, effect: str) -> List[CausalRelation]:
        return [r for r in self.relations if r.effect == effect]
    
    def get_effects(self, cause: str) -> List[CausalRelation]:
        return [r for r in self.relations if r.cause == cause]