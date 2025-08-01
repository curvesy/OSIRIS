"""
Neo4j Adapter for AURA Intelligence.

Provides async interface to Neo4j knowledge graph with:
- Connection pooling
- Automatic retry logic
- Query optimization
- Observability
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
import structlog
from opentelemetry import trace

from ..resilience import resilient, ResilienceLevel
from ..observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("neo4j_adapter")


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
    # Connection pool settings
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 30.0
    connection_timeout: float = 30.0
    
    # Retry settings
    max_retry_time: float = 30.0
    initial_retry_delay: float = 1.0
    retry_delay_multiplier: float = 2.0
    retry_delay_jitter_factor: float = 0.2
    
    # Query settings
    query_timeout: float = 30.0
    fetch_size: int = 1000


class Neo4jAdapter:
    """Async adapter for Neo4j operations."""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the Neo4j driver."""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("neo4j_initialize") as span:
            span.set_attribute("neo4j.uri", self.config.uri)
            span.set_attribute("neo4j.database", self.config.database)
            
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password),
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                    connection_timeout=self.config.connection_timeout,
                    max_retry_time=self.config.max_retry_time,
                    initial_retry_delay=self.config.initial_retry_delay,
                    retry_delay_multiplier=self.config.retry_delay_multiplier,
                    retry_delay_jitter_factor=self.config.retry_delay_jitter_factor
                )
                
                # Verify connectivity
                await self._driver.verify_connectivity()
                self._initialized = True
                logger.info("Neo4j adapter initialized", uri=self.config.uri)
                
            except Exception as e:
                logger.error("Failed to initialize Neo4j", error=str(e))
                raise
                
    async def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._initialized = False
            logger.info("Neo4j adapter closed")
            
    @asynccontextmanager
    async def session(self, database: Optional[str] = None):
        """Create a Neo4j session."""
        if not self._initialized:
            await self.initialize()
            
        async with self._driver.session(
            database=database or self.config.database,
            fetch_size=self.config.fetch_size
        ) as session:
            yield session
            
    @resilient(level=ResilienceLevel.CRITICAL)
    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a read query."""
        with tracer.start_as_current_span("neo4j_query") as span:
            span.set_attribute("neo4j.query", cypher[:100])  # First 100 chars
            span.set_attribute("neo4j.database", database or self.config.database)
            
            async with self.session(database) as session:
                try:
                    result = await session.run(
                        cypher,
                        params or {},
                        timeout=self.config.query_timeout
                    )
                    records = await result.data()
                    
                    span.set_attribute("neo4j.record_count", len(records))
                    return records
                    
                except Neo4jError as e:
                    logger.error("Neo4j query failed", 
                               query=cypher[:100], 
                               error=str(e))
                    raise
                    
    @resilient(level=ResilienceLevel.CRITICAL)
    async def write(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a write query."""
        with tracer.start_as_current_span("neo4j_write") as span:
            span.set_attribute("neo4j.query", cypher[:100])
            span.set_attribute("neo4j.database", database or self.config.database)
            
            async with self.session(database) as session:
                try:
                    result = await session.run(
                        cypher,
                        params or {},
                        timeout=self.config.query_timeout
                    )
                    summary = await result.consume()
                    
                    stats = {
                        "nodes_created": summary.counters.nodes_created,
                        "nodes_deleted": summary.counters.nodes_deleted,
                        "relationships_created": summary.counters.relationships_created,
                        "relationships_deleted": summary.counters.relationships_deleted,
                        "properties_set": summary.counters.properties_set
                    }
                    
                    span.set_attribute("neo4j.nodes_created", stats["nodes_created"])
                    span.set_attribute("neo4j.relationships_created", stats["relationships_created"])
                    
                    return stats
                    
                except Neo4jError as e:
                    logger.error("Neo4j write failed", 
                               query=cypher[:100], 
                               error=str(e))
                    raise
                    
    async def transaction(
        self,
        queries: List[tuple[str, Dict[str, Any]]],
        database: Optional[str] = None
    ) -> List[Any]:
        """Execute multiple queries in a transaction."""
        with tracer.start_as_current_span("neo4j_transaction") as span:
            span.set_attribute("neo4j.query_count", len(queries))
            span.set_attribute("neo4j.database", database or self.config.database)
            
            async with self.session(database) as session:
                async with session.begin_transaction() as tx:
                    results = []
                    try:
                        for cypher, params in queries:
                            result = await tx.run(cypher, params)
                            data = await result.data()
                            results.append(data)
                            
                        await tx.commit()
                        return results
                        
                    except Exception as e:
                        await tx.rollback()
                        logger.error("Neo4j transaction failed", error=str(e))
                        raise
                        
    # Knowledge graph specific methods
    
    async def find_similar_patterns(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find similar patterns using vector similarity."""
        query = """
        MATCH (p:Pattern)
        WHERE gds.similarity.cosine(p.embedding, $embedding) > $threshold
        RETURN p, gds.similarity.cosine(p.embedding, $embedding) as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        return await self.query(
            query,
            {
                "embedding": embedding,
                "threshold": threshold,
                "limit": limit
            }
        )
        
    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> Dict[str, List[str]]:
        """Get relationships for an entity."""
        if relationship_types:
            rel_filter = f"[r:{' | '.join(relationship_types)}]"
        else:
            rel_filter = "[r]"
            
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})-{rel_filter}-(related)
        RETURN type(r) as relationship, collect(distinct related.id) as related_ids
        """
        
        results = await self.query(query, {"entity_id": entity_id})
        
        relationships = {}
        for record in results:
            relationships[record["relationship"]] = record["related_ids"]
            
        return relationships
        
    async def add_decision_node(
        self,
        decision_id: str,
        agent_id: str,
        decision_type: str,
        confidence: float,
        context: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> str:
        """Add a decision node to the graph."""
        query = """
        CREATE (d:Decision {
            id: $decision_id,
            agent_id: $agent_id,
            type: $decision_type,
            confidence: $confidence,
            context: $context,
            embedding: $embedding,
            timestamp: datetime()
        })
        RETURN d.id as id
        """
        
        result = await self.query(
            query,
            {
                "decision_id": decision_id,
                "agent_id": agent_id,
                "decision_type": decision_type,
                "confidence": confidence,
                "context": context,
                "embedding": embedding
            }
        )
        
        return result[0]["id"] if result else decision_id
        
    async def link_decision_to_context(
        self,
        decision_id: str,
        context_ids: List[str],
        relationship_type: str = "INFLUENCED_BY"
    ):
        """Link a decision to its context nodes."""
        query = f"""
        MATCH (d:Decision {{id: $decision_id}})
        MATCH (c:Context) WHERE c.id IN $context_ids
        CREATE (d)-[:{relationship_type}]->(c)
        """
        
        await self.write(
            query,
            {
                "decision_id": decision_id,
                "context_ids": context_ids
            }
        )
        
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ):
        """Update entity properties."""
        set_clause = ", ".join([f"e.{k} = ${k}" for k in properties.keys()])
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        SET {set_clause}, e.updated_at = datetime()
        RETURN e
        """
        
        params = {"entity_id": entity_id}
        params.update(properties)
        
        await self.write(query, params)