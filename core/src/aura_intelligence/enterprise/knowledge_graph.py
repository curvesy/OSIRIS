"""
🧠 AURA Intelligence Knowledge Graph Service

Neo4j integration for causal reasoning and contextual relationships.
This enables "Why did this happen?" queries by building causal chains
between topological signatures, events, actions, and outcomes.

Based on kiki.md and ppdd.md research for professional graph reasoning.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, SystemEvent, AgentAction, Outcome
)
from aura_intelligence.utils.logger import get_logger


class KnowledgeGraphService:
    """
    🧠 Knowledge Graph Service for Causal Intelligence
    
    Provides causal reasoning capabilities by storing and querying relationships
    between topological signatures, system events, agent actions, and outcomes.
    This answers "Why did this happen?" through graph traversal.
    
    Features:
    - Causal chain tracking and analysis
    - Pattern relationship discovery
    - Temporal graph analysis
    - Graph ML for pattern prediction
    - Production-ready with connection pooling
    """
    
    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 database: str = "aura_intelligence",
                 enable_monitoring: bool = True):
        """
        Initialize Knowledge Graph Service.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
            enable_monitoring: Enable performance monitoring
        """
        self.logger = get_logger(__name__)
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.enable_monitoring = enable_monitoring
        
        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.avg_query_time = 0.0
        
        # Initialize Neo4j driver
        self.driver = None
        self.initialized = False
        
        self.logger.info(f"🧠 Knowledge Graph Service initialized (Neo4j: {uri})")
    
    async def initialize(self) -> bool:
        """Initialize Neo4j connection and create schema."""
        try:
            # Initialize async Neo4j driver
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
                # Use default database instead of specific name
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            
            # Create schema and constraints
            await self._create_schema()
            
            self.initialized = True
            self.logger.info("✅ Knowledge Graph Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Knowledge Graph: {e}")
            return False
    
    async def _create_schema(self):
        """Create Neo4j schema with constraints and indexes."""
        schema_queries = [
            # Create constraints for unique identifiers
            "CREATE CONSTRAINT signature_hash_unique IF NOT EXISTS FOR (s:Signature) REQUIRE s.hash IS UNIQUE",
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            "CREATE CONSTRAINT action_id_unique IF NOT EXISTS FOR (a:Action) REQUIRE a.action_id IS UNIQUE",
            "CREATE CONSTRAINT outcome_id_unique IF NOT EXISTS FOR (o:Outcome) REQUIRE o.outcome_id IS UNIQUE",
            
            # Create indexes for performance
            "CREATE INDEX signature_timestamp IF NOT EXISTS FOR (s:Signature) ON (s.timestamp)",
            "CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            "CREATE INDEX action_timestamp IF NOT EXISTS FOR (a:Action) ON (a.timestamp)",
            "CREATE INDEX outcome_timestamp IF NOT EXISTS FOR (o:Outcome) ON (o.timestamp)",
            "CREATE INDEX signature_consciousness IF NOT EXISTS FOR (s:Signature) ON (s.consciousness_level)",
            "CREATE INDEX event_severity IF NOT EXISTS FOR (e:Event) ON (e.severity_level)",
            "CREATE INDEX action_agent IF NOT EXISTS FOR (a:Action) ON (a.agent_id)",
            "CREATE INDEX outcome_success IF NOT EXISTS FOR (o:Outcome) ON (o.success)"
        ]
        
        async with self.driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Ignore constraint/index already exists errors
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Schema query failed: {query} - {e}")
        
        self.logger.info("✅ Neo4j schema created successfully")
    
    async def store_signature(self, signature: TopologicalSignature) -> bool:
        """
        Store a topological signature in the knowledge graph.
        
        Args:
            signature: TopologicalSignature to store
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            query = """
            MERGE (s:Signature {hash: $hash})
            SET s.betti_numbers = $betti_numbers,
                s.consciousness_level = $consciousness_level,
                s.quantum_coherence = $quantum_coherence,
                s.algorithm_used = $algorithm_used,
                s.timestamp = datetime($timestamp),
                s.agent_context = $agent_context,
                s.performance_metrics = $performance_metrics
            RETURN s.hash as hash
            """
            
            parameters = {
                "hash": signature.signature_hash,
                "betti_numbers": signature.betti_numbers,
                "consciousness_level": signature.consciousness_level,
                "quantum_coherence": signature.quantum_coherence,
                "algorithm_used": signature.algorithm_used,
                "timestamp": signature.timestamp.isoformat(),
                "agent_context": signature.agent_context,
                "performance_metrics": signature.performance_metrics
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                record = await result.single()
                
                if record:
                    storage_time = (time.time() - start_time) * 1000
                    self.logger.debug(f"📦 Stored signature {signature.signature_hash[:8]}... in {storage_time:.2f}ms")
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store signature: {e}")
            return False
    
    async def store_event_chain(self,
                              signature: TopologicalSignature,
                              event: SystemEvent,
                              action: Optional[AgentAction] = None,
                              outcome: Optional[Outcome] = None) -> bool:
        """
        Store a complete event chain: Signature → Event → Action → Outcome.
        
        This creates the causal relationships that enable reasoning.
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Build the chain creation query
            query = """
            // Create or update signature
            MERGE (s:Signature {hash: $sig_hash})
            SET s.betti_numbers = $betti_numbers,
                s.consciousness_level = $consciousness_level,
                s.timestamp = datetime($sig_timestamp)
            
            // Create event
            MERGE (e:Event {event_id: $event_id})
            SET e.event_type = $event_type,
                e.timestamp = datetime($event_timestamp),
                e.system_state = $system_state,
                e.triggering_agents = $triggering_agents,
                e.consciousness_state = $consciousness_state,
                e.severity_level = $severity_level
            
            // Create relationship: Signature generated by Event
            MERGE (s)-[:GENERATED_BY]->(e)
            """
            
            parameters = {
                "sig_hash": signature.signature_hash,
                "betti_numbers": signature.betti_numbers,
                "consciousness_level": signature.consciousness_level,
                "sig_timestamp": signature.timestamp.isoformat(),
                "event_id": event.event_id,
                "event_type": event.event_type,
                "event_timestamp": event.timestamp.isoformat(),
                "system_state": event.system_state,
                "triggering_agents": event.triggering_agents,
                "consciousness_state": event.consciousness_state,
                "severity_level": event.severity_level
            }
            
            # Add action if provided
            if action:
                query += """
                // Create action
                MERGE (a:Action {action_id: $action_id})
                SET a.agent_id = $agent_id,
                    a.action_type = $action_type,
                    a.timestamp = datetime($action_timestamp),
                    a.decision_context = $decision_context,
                    a.action_parameters = $action_parameters,
                    a.confidence_score = $confidence_score,
                    a.reasoning = $reasoning
                
                // Create relationship: Event triggered Action
                MERGE (e)<-[:TRIGGERED_BY]-(a)
                """
                
                parameters.update({
                    "action_id": action.action_id,
                    "agent_id": action.agent_id,
                    "action_type": action.action_type,
                    "action_timestamp": action.timestamp.isoformat(),
                    "decision_context": action.decision_context,
                    "action_parameters": action.action_parameters,
                    "confidence_score": action.confidence_score,
                    "reasoning": action.reasoning
                })
            
            # Add outcome if provided
            if outcome and action:
                query += """
                // Create outcome
                MERGE (o:Outcome {outcome_id: $outcome_id})
                SET o.timestamp = datetime($outcome_timestamp),
                    o.success = $success,
                    o.impact_score = $impact_score,
                    o.metrics = $metrics,
                    o.learned_insights = $learned_insights,
                    o.follow_up_actions = $follow_up_actions
                
                // Create relationship: Action led to Outcome
                MERGE (a)-[:LED_TO]->(o)
                
                // Create feedback loop: Outcome influences future Events
                MERGE (o)-[:INFLUENCES {weight: $impact_score}]->(e)
                """
                
                parameters.update({
                    "outcome_id": outcome.outcome_id,
                    "outcome_timestamp": outcome.timestamp.isoformat(),
                    "success": outcome.success,
                    "impact_score": outcome.impact_score,
                    "metrics": outcome.metrics,
                    "learned_insights": outcome.learned_insights,
                    "follow_up_actions": outcome.follow_up_actions
                })
            
            query += " RETURN s.hash as signature_hash"
            
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                record = await result.single()
                
                if record:
                    storage_time = (time.time() - start_time) * 1000
                    self.logger.debug(f"🔗 Stored event chain in {storage_time:.2f}ms")
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store event chain: {e}")
            return False
    
    async def get_causal_context(self, signature_hash: str, depth: int = 3) -> Dict[str, Any]:
        """
        Get causal context for a signature by traversing the graph.
        
        Args:
            signature_hash: Hash of the signature to analyze
            depth: Maximum traversal depth
            
        Returns:
            Causal context with related events, actions, and outcomes
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            query = f"""
            MATCH (s:Signature {{hash: $signature_hash}})
            OPTIONAL MATCH path1 = (s)-[:GENERATED_BY]->(e:Event)
            OPTIONAL MATCH path2 = (e)<-[:TRIGGERED_BY]-(a:Action)
            OPTIONAL MATCH path3 = (a)-[:LED_TO]->(o:Outcome)
            OPTIONAL MATCH path4 = (o)-[:INFLUENCES]->(future_e:Event)
            
            // Find similar signatures (same Betti numbers)
            OPTIONAL MATCH (similar:Signature)
            WHERE similar.betti_numbers = s.betti_numbers 
              AND similar.hash <> s.hash
            LIMIT 5
            
            RETURN s, e, a, o, future_e,
                   collect(DISTINCT similar) as similar_signatures,
                   [path1, path2, path3, path4] as paths
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, {"signature_hash": signature_hash})
                record = await result.single()
                
                if not record:
                    return {"error": "Signature not found"}
                
                # Process the results
                context = {
                    "signature": dict(record["s"]) if record["s"] else None,
                    "event": dict(record["e"]) if record["e"] else None,
                    "action": dict(record["a"]) if record["a"] else None,
                    "outcome": dict(record["o"]) if record["o"] else None,
                    "future_event": dict(record["future_e"]) if record["future_e"] else None,
                    "similar_signatures": [dict(sig) for sig in record["similar_signatures"]],
                    "causal_chain_length": sum(1 for path in record["paths"] if path),
                    "query_time_ms": (time.time() - start_time) * 1000
                }
                
                # Update performance metrics
                query_time = (time.time() - start_time) * 1000
                self.query_count += 1
                self.total_query_time += query_time
                self.avg_query_time = self.total_query_time / self.query_count
                
                self.logger.debug(f"🔍 Retrieved causal context in {query_time:.2f}ms")
                
                return context
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get causal context: {e}")
            return {"error": str(e)}
    
    async def find_pattern_relationships(self, pattern_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find relationships between patterns of a specific type.
        
        Args:
            pattern_type: Type of pattern to analyze
            limit: Maximum number of results
            
        Returns:
            List of pattern relationships
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            query = """
            MATCH (s1:Signature)-[:GENERATED_BY]->(e1:Event)
            MATCH (s2:Signature)-[:GENERATED_BY]->(e2:Event)
            WHERE e1.event_type = $pattern_type 
              AND e2.event_type = $pattern_type
              AND s1.hash <> s2.hash
            
            // Calculate similarity based on Betti numbers
            WITH s1, s2, e1, e2,
                 reduce(sim = 0, i IN range(0, size(s1.betti_numbers)-1) | 
                   sim + abs(s1.betti_numbers[i] - s2.betti_numbers[i])) as betti_diff
            
            WHERE betti_diff <= 2  // Similar topological patterns
            
            RETURN s1.hash as signature1,
                   s2.hash as signature2,
                   s1.betti_numbers as betti1,
                   s2.betti_numbers as betti2,
                   betti_diff,
                   e1.timestamp as timestamp1,
                   e2.timestamp as timestamp2
            ORDER BY betti_diff ASC
            LIMIT $limit
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, {"pattern_type": pattern_type, "limit": limit})
                
                relationships = []
                async for record in result:
                    relationships.append({
                        "signature1": record["signature1"],
                        "signature2": record["signature2"],
                        "betti_numbers1": record["betti1"],
                        "betti_numbers2": record["betti2"],
                        "similarity_score": 1.0 / (1.0 + record["betti_diff"]),
                        "timestamp1": record["timestamp1"],
                        "timestamp2": record["timestamp2"]
                    })
                
                return relationships
                
        except Exception as e:
            self.logger.error(f"❌ Failed to find pattern relationships: {e}")
            return []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.initialized:
            await self.initialize()
        
        try:
            query = """
            MATCH (s:Signature) WITH count(s) as signatures
            MATCH (e:Event) WITH signatures, count(e) as events
            MATCH (a:Action) WITH signatures, events, count(a) as actions
            MATCH (o:Outcome) WITH signatures, events, actions, count(o) as outcomes
            MATCH ()-[r]->() WITH signatures, events, actions, outcomes, count(r) as relationships
            RETURN signatures, events, actions, outcomes, relationships
            """
            
            async with self.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                
                if record:
                    return {
                        "signatures": record["signatures"],
                        "events": record["events"],
                        "actions": record["actions"],
                        "outcomes": record["outcomes"],
                        "relationships": record["relationships"],
                        "avg_query_time_ms": round(self.avg_query_time, 2),
                        "total_queries": self.query_count
                    }
                
            return {"error": "No data found"}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get graph stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the knowledge graph."""
        try:
            if not self.initialized:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Test connectivity and performance
            start_time = time.time()
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            query_time = (time.time() - start_time) * 1000
            status = "healthy" if query_time < 100.0 else "degraded"
            
            return {
                "status": status,
                "query_time_ms": round(query_time, 2),
                "avg_query_time_ms": round(self.avg_query_time, 2),
                "total_queries": self.query_count,
                "database": self.database
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }
    
    async def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            self.logger.info("🔌 Knowledge Graph Service connection closed")
