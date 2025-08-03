"""
TDA Neo4j Storage Adapter

This adapter provides a production-ready interface for storing TDA results
in Neo4j knowledge graph, enabling semantic queries and relationship analysis.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
import json
import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError

from ..config.base import get_config

logger = structlog.get_logger(__name__)


@dataclass
class TDANode:
    """Standardized TDA result node for Neo4j storage."""
    id: str
    algorithm: str
    timestamp: datetime
    anomaly_score: float
    betti_numbers: List[int]
    persistence_features: Dict[str, Any]
    metadata: Dict[str, Any]


class TDANeo4jAdapter:
    """
    Production adapter for storing TDA results in Neo4j.
    
    Features:
    - Async operations for high throughput
    - Automatic relationship creation
    - Query optimization with indexes
    - Transaction safety
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, 
                 password: Optional[str] = None, database: Optional[str] = None):
        """Initialize Neo4j connection with configuration."""
        config = get_config()
        
        self.uri = uri or config.database.neo4j_uri
        self.user = user or config.database.neo4j_user
        self.password = password or config.database.neo4j_password
        self.database = database or config.database.neo4j_database
        
        self.driver: Optional[AsyncDriver] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize connection and create indexes."""
        if self._initialized:
            return
            
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            
            # Create indexes for performance
            await self._create_indexes()
            
            self._initialized = True
            logger.info("Neo4j adapter initialized", uri=self.uri)
            
        except Exception as e:
            logger.error("Failed to initialize Neo4j adapter", error=str(e))
            raise
            
    async def _create_indexes(self):
        """Create indexes for optimal query performance."""
        async with self.driver.session(database=self.database) as session:
            # Index on TDA node ID
            await session.run(
                "CREATE INDEX tda_id IF NOT EXISTS FOR (t:TDAResult) ON (t.id)"
            )
            
            # Index on timestamp for time-based queries
            await session.run(
                "CREATE INDEX tda_timestamp IF NOT EXISTS FOR (t:TDAResult) ON (t.timestamp)"
            )
            
            # Index on anomaly score for anomaly queries
            await session.run(
                "CREATE INDEX tda_anomaly IF NOT EXISTS FOR (t:TDAResult) ON (t.anomaly_score)"
            )
            
            # Index on algorithm for filtering
            await session.run(
                "CREATE INDEX tda_algorithm IF NOT EXISTS FOR (t:TDAResult) ON (t.algorithm)"
            )
            
    async def store_tda_result(
        self,
        result: Dict[str, Any],
        parent_data_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store TDA computation result in Neo4j.
        
        Args:
            result: TDA computation result
            parent_data_id: ID of the source data
            context: Additional context
            
        Returns:
            Node ID of stored result
        """
        if not self._initialized:
            await self.initialize()
            
        # Create TDA node
        node = TDANode(
            id=f"tda_{datetime.utcnow().timestamp()}_{result.get('algorithm', 'unknown')}",
            algorithm=result.get("algorithm", "unknown"),
            timestamp=datetime.utcnow(),
            anomaly_score=result.get("anomaly_score", 0.0),
            betti_numbers=result.get("betti_numbers", []),
            persistence_features=self._extract_persistence_features(result),
            metadata=result.get("metadata", {})
        )
        
        async with self.driver.session(database=self.database) as session:
            try:
                # Store node
                result = await session.run(
                    """
                    CREATE (t:TDAResult {
                        id: $id,
                        algorithm: $algorithm,
                        timestamp: datetime($timestamp),
                        anomaly_score: $anomaly_score,
                        betti_numbers: $betti_numbers,
                        persistence_features: $persistence_features,
                        metadata: $metadata
                    })
                    RETURN t.id as id
                    """,
                    id=node.id,
                    algorithm=node.algorithm,
                    timestamp=node.timestamp.isoformat(),
                    anomaly_score=node.anomaly_score,
                    betti_numbers=node.betti_numbers,
                    persistence_features=json.dumps(node.persistence_features),
                    metadata=json.dumps(node.metadata)
                )
                
                # Create relationships
                if parent_data_id:
                    await self._create_data_relationship(session, node.id, parent_data_id)
                    
                # Store persistence diagrams
                if "persistence_diagrams" in result:
                    await self._store_persistence_diagrams(
                        session, node.id, result["persistence_diagrams"]
                    )
                    
                # Create anomaly relationships if high score
                if node.anomaly_score > 0.7:
                    await self._create_anomaly_relationships(session, node.id, node.anomaly_score)
                    
                logger.info(
                    "Stored TDA result in Neo4j",
                    node_id=node.id,
                    anomaly_score=node.anomaly_score
                )
                
                return node.id
                
            except Neo4jError as e:
                logger.error("Failed to store TDA result", error=str(e))
                raise
                
    def _extract_persistence_features(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key persistence features from TDA result."""
        features = {}
        
        if "persistence_diagrams" in result:
            diagrams = result["persistence_diagrams"]
            
            # Extract key statistics
            features["num_components"] = len(diagrams[0].intervals) if diagrams else 0
            features["max_persistence"] = max(
                (d[1] - d[0] for diagram in diagrams for d in diagram.intervals),
                default=0
            )
            features["total_persistence"] = sum(
                (d[1] - d[0] for diagram in diagrams for d in diagram.intervals)
            )
            
        return features
        
    async def _create_data_relationship(
        self, session, tda_id: str, data_id: str
    ):
        """Create relationship between TDA result and source data."""
        await session.run(
            """
            MATCH (t:TDAResult {id: $tda_id})
            MERGE (d:Data {id: $data_id})
            CREATE (d)-[:ANALYZED_BY]->(t)
            """,
            tda_id=tda_id,
            data_id=data_id
        )
        
    async def _store_persistence_diagrams(
        self, session, tda_id: str, diagrams: List[Any]
    ):
        """Store individual persistence diagrams."""
        for i, diagram in enumerate(diagrams):
            await session.run(
                """
                MATCH (t:TDAResult {id: $tda_id})
                CREATE (p:PersistenceDiagram {
                    dimension: $dimension,
                    intervals: $intervals
                })
                CREATE (t)-[:HAS_DIAGRAM]->(p)
                """,
                tda_id=tda_id,
                dimension=i,
                intervals=json.dumps(diagram.intervals)
            )
            
    async def _create_anomaly_relationships(
        self, session, tda_id: str, score: float
    ):
        """Create relationships to similar anomalies."""
        # Find similar anomalies
        result = await session.run(
            """
            MATCH (t:TDAResult {id: $tda_id})
            MATCH (other:TDAResult)
            WHERE other.id <> $tda_id
            AND abs(other.anomaly_score - $score) < 0.1
            AND other.timestamp > datetime() - duration('P7D')
            WITH t, other, abs(other.anomaly_score - $score) as similarity
            ORDER BY similarity
            LIMIT 5
            CREATE (t)-[:SIMILAR_TO {score: 1 - similarity}]->(other)
            """,
            tda_id=tda_id,
            score=score
        )
        
    async def query_tda_results(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query TDA results with filters.
        
        Args:
            filters: Query filters (algorithm, time_range, anomaly_threshold)
            limit: Maximum results
            
        Returns:
            List of TDA results
        """
        if not self._initialized:
            await self.initialize()
            
        # Build query
        where_clauses = []
        params = {"limit": limit}
        
        if filters:
            if "algorithm" in filters:
                where_clauses.append("t.algorithm = $algorithm")
                params["algorithm"] = filters["algorithm"]
                
            if "anomaly_threshold" in filters:
                where_clauses.append("t.anomaly_score >= $threshold")
                params["threshold"] = filters["anomaly_threshold"]
                
            if "time_range" in filters:
                start, end = filters["time_range"]
                where_clauses.append("t.timestamp >= datetime($start)")
                where_clauses.append("t.timestamp <= datetime($end)")
                params["start"] = start.isoformat()
                params["end"] = end.isoformat()
                
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        MATCH (t:TDAResult)
        WHERE {where_clause}
        RETURN t
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            
            results = []
            async for record in result:
                node = record["t"]
                results.append({
                    "id": node["id"],
                    "algorithm": node["algorithm"],
                    "timestamp": node["timestamp"],
                    "anomaly_score": node["anomaly_score"],
                    "betti_numbers": node["betti_numbers"],
                    "persistence_features": json.loads(node["persistence_features"]),
                    "metadata": json.loads(node["metadata"])
                })
                
            return results
            
    async def get_topological_context(
        self, data_id: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get topological context for a data point.
        
        Args:
            data_id: Source data ID
            depth: Graph traversal depth
            
        Returns:
            Topological context including related TDA results
        """
        if not self._initialized:
            await self.initialize()
            
        query = """
        MATCH (d:Data {id: $data_id})
        OPTIONAL MATCH path = (d)-[*1..$depth]-(related)
        WHERE related:TDAResult OR related:Data
        WITH d, collect(distinct related) as related_nodes,
             collect(distinct path) as paths
        RETURN d, related_nodes, 
               [p in paths | [n in nodes(p) | n.id]] as path_ids
        """
        
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                data_id=data_id,
                depth=depth
            )
            
            record = await result.single()
            if not record:
                return {}
                
            # Process results
            context = {
                "data_id": data_id,
                "related_tda_results": [],
                "topological_patterns": [],
                "anomaly_connections": []
            }
            
            for node in record["related_nodes"]:
                if "TDAResult" in node.labels:
                    context["related_tda_results"].append({
                        "id": node["id"],
                        "algorithm": node["algorithm"],
                        "anomaly_score": node["anomaly_score"]
                    })
                    
            return context
            
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self._initialized = False
            logger.info("Neo4j adapter closed")