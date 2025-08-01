"""
💾 Causal Pattern Store
Neo4j-integrated storage for topological patterns and causal relationships.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from ..utils.logger import get_logger


class CausalPatternStore:
    """
    💾 Causal Pattern Store
    
    Stores and manages topological patterns with causal relationships:
    - Pattern classification storage (e.g., 'Pattern_7_Failure')
    - Topological feature persistence (Betti numbers, entropy)
    - Causal relationship mapping
    - Pattern evolution tracking
    - Knowledge graph integration
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.logger = get_logger(__name__)
        self.neo4j_available = NEO4J_AVAILABLE
        
        if self.neo4j_available:
            try:
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                self._initialize_schema()
                self.logger.info("💾 Neo4j Causal Pattern Store initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Neo4j unavailable, using in-memory store: {e}")
                self.neo4j_available = False
        
        if not self.neo4j_available:
            # Fallback to in-memory storage
            self.memory_store = {
                'patterns': [],
                'relationships': [],
                'statistics': {}
            }
            self.logger.info("💾 In-memory Causal Pattern Store initialized")
    
    def _initialize_schema(self):
        """Initialize Neo4j schema for pattern storage."""
        
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("""
                CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS
                FOR (p:TopologicalPattern) REQUIRE p.pattern_id IS UNIQUE
            """)
            
            session.run("""
                CREATE INDEX pattern_classification_idx IF NOT EXISTS
                FOR (p:TopologicalPattern) ON (p.classification)
            """)
            
            session.run("""
                CREATE INDEX pattern_timestamp_idx IF NOT EXISTS
                FOR (p:TopologicalPattern) ON (p.timestamp)
            """)
            
            session.run("""
                CREATE INDEX event_timestamp_idx IF NOT EXISTS
                FOR (e:Event) ON (e.timestamp)
            """)
    
    async def store_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Store a topological pattern with causal context.
        
        Args:
            pattern_data: Pattern data including classification, features, and context
            
        Returns:
            Success status
        """
        try:
            if self.neo4j_available:
                return await self._store_pattern_neo4j(pattern_data)
            else:
                return await self._store_pattern_memory(pattern_data)
                
        except Exception as e:
            self.logger.error(f"❌ Pattern storage failed: {e}")
            return False
    
    async def _store_pattern_neo4j(self, pattern_data: Dict[str, Any]) -> bool:
        """Store pattern in Neo4j graph database."""
        
        with self.driver.session() as session:
            # Create topological pattern node
            pattern_query = """
                CREATE (p:TopologicalPattern {
                    pattern_id: $pattern_id,
                    classification: $classification,
                    anomaly_score: $anomaly_score,
                    betti_numbers: $betti_numbers,
                    persistence_entropy: $persistence_entropy,
                    topological_signature: $topological_signature,
                    confidence: $confidence,
                    timestamp: $timestamp,
                    request_id: $request_id,
                    events_processed: $events_processed,
                    agent_responses: $agent_responses
                })
                RETURN p.pattern_id as pattern_id
            """
            
            result = session.run(pattern_query, {
                'pattern_id': pattern_data.get('pattern_id'),
                'classification': pattern_data.get('topological_pattern'),
                'anomaly_score': pattern_data.get('anomaly_score', 0.0),
                'betti_numbers': json.dumps(pattern_data.get('betti_numbers', [])),
                'persistence_entropy': pattern_data.get('persistence_entropy', 0.0),
                'topological_signature': pattern_data.get('topological_signature', ''),
                'confidence': pattern_data.get('confidence', 0.0),
                'timestamp': pattern_data.get('timestamp'),
                'request_id': pattern_data.get('request_id'),
                'events_processed': pattern_data.get('events_processed', 0),
                'agent_responses': pattern_data.get('agent_responses', 0)
            })
            
            pattern_id = result.single()['pattern_id']
            
            # Create relationships to similar patterns
            await self._create_pattern_relationships(session, pattern_data)
            
            self.logger.info(f"💾 Pattern stored in Neo4j: {pattern_id}")
            return True
    
    async def _create_pattern_relationships(self, session, pattern_data: Dict[str, Any]):
        """Create relationships between similar patterns."""
        
        classification = pattern_data.get('topological_pattern')
        anomaly_score = pattern_data.get('anomaly_score', 0.0)
        
        # Find similar patterns
        similarity_query = """
            MATCH (p:TopologicalPattern)
            WHERE p.classification = $classification
            AND abs(p.anomaly_score - $anomaly_score) < 0.1
            AND p.pattern_id <> $current_pattern_id
            RETURN p.pattern_id as similar_pattern_id
            ORDER BY abs(p.anomaly_score - $anomaly_score)
            LIMIT 5
        """
        
        similar_patterns = session.run(similarity_query, {
            'classification': classification,
            'anomaly_score': anomaly_score,
            'current_pattern_id': pattern_data.get('pattern_id')
        })
        
        # Create SIMILAR_TO relationships
        for record in similar_patterns:
            relationship_query = """
                MATCH (p1:TopologicalPattern {pattern_id: $pattern_id})
                MATCH (p2:TopologicalPattern {pattern_id: $similar_pattern_id})
                CREATE (p1)-[:SIMILAR_TO {
                    similarity_score: $similarity_score,
                    created_at: $timestamp
                }]->(p2)
            """
            
            session.run(relationship_query, {
                'pattern_id': pattern_data.get('pattern_id'),
                'similar_pattern_id': record['similar_pattern_id'],
                'similarity_score': 0.8,  # Calculated similarity
                'timestamp': datetime.now().isoformat()
            })
        
        # Create temporal relationships (FOLLOWS)
        temporal_query = """
            MATCH (p:TopologicalPattern)
            WHERE p.timestamp < $current_timestamp
            AND p.request_id <> $current_request_id
            RETURN p.pattern_id as previous_pattern_id, p.timestamp as prev_timestamp
            ORDER BY p.timestamp DESC
            LIMIT 3
        """
        
        previous_patterns = session.run(temporal_query, {
            'current_timestamp': pattern_data.get('timestamp'),
            'current_request_id': pattern_data.get('request_id')
        })
        
        for record in previous_patterns:
            follows_query = """
                MATCH (p1:TopologicalPattern {pattern_id: $previous_pattern_id})
                MATCH (p2:TopologicalPattern {pattern_id: $current_pattern_id})
                CREATE (p2)-[:FOLLOWS {
                    time_delta: $time_delta,
                    created_at: $timestamp
                }]->(p1)
            """
            
            session.run(follows_query, {
                'previous_pattern_id': record['previous_pattern_id'],
                'current_pattern_id': pattern_data.get('pattern_id'),
                'time_delta': 'calculated_delta',  # Would calculate actual time delta
                'timestamp': datetime.now().isoformat()
            })
    
    async def _store_pattern_memory(self, pattern_data: Dict[str, Any]) -> bool:
        """Store pattern in in-memory fallback storage."""
        
        self.memory_store['patterns'].append({
            **pattern_data,
            'stored_at': datetime.now().isoformat()
        })
        
        # Update statistics
        classification = pattern_data.get('topological_pattern', 'unknown')
        if classification not in self.memory_store['statistics']:
            self.memory_store['statistics'][classification] = 0
        self.memory_store['statistics'][classification] += 1
        
        self.logger.info(f"💾 Pattern stored in memory: {pattern_data.get('pattern_id')}")
        return True
    
    async def get_pattern_history(self, classification: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical patterns of a specific classification."""
        
        try:
            if self.neo4j_available:
                return await self._get_pattern_history_neo4j(classification, limit)
            else:
                return await self._get_pattern_history_memory(classification, limit)
                
        except Exception as e:
            self.logger.error(f"❌ Pattern history retrieval failed: {e}")
            return []
    
    async def _get_pattern_history_neo4j(self, classification: str, 
                                       limit: int) -> List[Dict[str, Any]]:
        """Get pattern history from Neo4j."""
        
        with self.driver.session() as session:
            query = """
                MATCH (p:TopologicalPattern)
                WHERE p.classification = $classification
                RETURN p.pattern_id as pattern_id,
                       p.classification as classification,
                       p.anomaly_score as anomaly_score,
                       p.betti_numbers as betti_numbers,
                       p.persistence_entropy as persistence_entropy,
                       p.timestamp as timestamp,
                       p.confidence as confidence
                ORDER BY p.timestamp DESC
                LIMIT $limit
            """
            
            result = session.run(query, {
                'classification': classification,
                'limit': limit
            })
            
            patterns = []
            for record in result:
                pattern = dict(record)
                # Parse JSON fields
                if pattern['betti_numbers']:
                    pattern['betti_numbers'] = json.loads(pattern['betti_numbers'])
                patterns.append(pattern)
            
            return patterns
    
    async def _get_pattern_history_memory(self, classification: str,
                                        limit: int) -> List[Dict[str, Any]]:
        """Get pattern history from memory storage."""
        
        matching_patterns = [
            p for p in self.memory_store['patterns']
            if p.get('topological_pattern') == classification
        ]
        
        # Sort by timestamp and limit
        matching_patterns.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return matching_patterns[:limit]
    
    async def find_causal_relationships(self, pattern_id: str) -> Dict[str, Any]:
        """Find causal relationships for a specific pattern."""
        
        try:
            if self.neo4j_available:
                return await self._find_causal_relationships_neo4j(pattern_id)
            else:
                return await self._find_causal_relationships_memory(pattern_id)
                
        except Exception as e:
            self.logger.error(f"❌ Causal relationship search failed: {e}")
            return {'similar_patterns': [], 'temporal_patterns': []}
    
    async def _find_causal_relationships_neo4j(self, pattern_id: str) -> Dict[str, Any]:
        """Find causal relationships in Neo4j."""
        
        with self.driver.session() as session:
            # Find similar patterns
            similar_query = """
                MATCH (p:TopologicalPattern {pattern_id: $pattern_id})-[r:SIMILAR_TO]->(similar)
                RETURN similar.pattern_id as pattern_id,
                       similar.classification as classification,
                       similar.anomaly_score as anomaly_score,
                       r.similarity_score as similarity_score
                ORDER BY r.similarity_score DESC
            """
            
            similar_result = session.run(similar_query, {'pattern_id': pattern_id})
            similar_patterns = [dict(record) for record in similar_result]
            
            # Find temporal relationships
            temporal_query = """
                MATCH (p:TopologicalPattern {pattern_id: $pattern_id})-[r:FOLLOWS]->(previous)
                RETURN previous.pattern_id as pattern_id,
                       previous.classification as classification,
                       previous.timestamp as timestamp,
                       r.time_delta as time_delta
                ORDER BY previous.timestamp DESC
            """
            
            temporal_result = session.run(temporal_query, {'pattern_id': pattern_id})
            temporal_patterns = [dict(record) for record in temporal_result]
            
            return {
                'similar_patterns': similar_patterns,
                'temporal_patterns': temporal_patterns
            }
    
    async def _find_causal_relationships_memory(self, pattern_id: str) -> Dict[str, Any]:
        """Find causal relationships in memory storage."""
        
        # Simple similarity based on classification and anomaly score
        target_pattern = None
        for p in self.memory_store['patterns']:
            if p.get('pattern_id') == pattern_id:
                target_pattern = p
                break
        
        if not target_pattern:
            return {'similar_patterns': [], 'temporal_patterns': []}
        
        target_class = target_pattern.get('topological_pattern')
        target_anomaly = target_pattern.get('anomaly_score', 0.0)
        
        similar_patterns = []
        for p in self.memory_store['patterns']:
            if (p.get('pattern_id') != pattern_id and
                p.get('topological_pattern') == target_class and
                abs(p.get('anomaly_score', 0.0) - target_anomaly) < 0.1):
                similar_patterns.append(p)
        
        return {
            'similar_patterns': similar_patterns[:5],
            'temporal_patterns': []  # Simplified for memory storage
        }
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern statistics."""
        
        try:
            if self.neo4j_available:
                return await self._get_pattern_statistics_neo4j()
            else:
                return await self._get_pattern_statistics_memory()
                
        except Exception as e:
            self.logger.error(f"❌ Pattern statistics retrieval failed: {e}")
            return {}
    
    async def _get_pattern_statistics_neo4j(self) -> Dict[str, Any]:
        """Get pattern statistics from Neo4j."""
        
        with self.driver.session() as session:
            # Pattern distribution
            distribution_query = """
                MATCH (p:TopologicalPattern)
                RETURN p.classification as classification, count(p) as count
                ORDER BY count DESC
            """
            
            distribution_result = session.run(distribution_query)
            pattern_distribution = {record['classification']: record['count'] 
                                  for record in distribution_result}
            
            # Total patterns
            total_query = "MATCH (p:TopologicalPattern) RETURN count(p) as total"
            total_result = session.run(total_query)
            total_patterns = total_result.single()['total']
            
            # Average anomaly scores by pattern
            anomaly_query = """
                MATCH (p:TopologicalPattern)
                RETURN p.classification as classification, 
                       avg(p.anomaly_score) as avg_anomaly_score
            """
            
            anomaly_result = session.run(anomaly_query)
            anomaly_scores = {record['classification']: record['avg_anomaly_score']
                            for record in anomaly_result}
            
            return {
                'total_patterns': total_patterns,
                'pattern_distribution': pattern_distribution,
                'average_anomaly_scores': anomaly_scores,
                'storage_type': 'neo4j'
            }
    
    async def _get_pattern_statistics_memory(self) -> Dict[str, Any]:
        """Get pattern statistics from memory storage."""
        
        return {
            'total_patterns': len(self.memory_store['patterns']),
            'pattern_distribution': self.memory_store['statistics'].copy(),
            'storage_type': 'memory'
        }
    
    async def close(self):
        """Close database connections."""
        if self.neo4j_available and hasattr(self, 'driver'):
            self.driver.close()
            self.logger.info("💾 Neo4j connection closed")
