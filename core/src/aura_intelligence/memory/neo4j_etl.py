"""
Neo4j ETL Pipeline for Shape Memory V2
=====================================

Nightly ETL job that syncs topological shapes from Redis to Neo4j
for advanced graph analytics and danger-ring detection.

Based on the architecture from nowlookatthispart.md.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import numpy as np
from neo4j import AsyncGraphDatabase, AsyncSession
import redis
import schedule

from .observability import observability, traced
from .redis_store import RedisVectorStore, KEY_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    """Configuration for Neo4j ETL pipeline."""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    redis_url: str = "redis://localhost:6379"
    
    batch_size: int = 1000
    similarity_threshold: float = 0.8
    max_similar_edges: int = 10
    
    # Graph algorithms
    run_fastrp: bool = True
    run_community_detection: bool = True
    run_danger_analysis: bool = True


class Neo4jETL:
    """
    Production ETL pipeline for shape graph analytics.
    
    Features:
    - Incremental updates (only new/changed shapes)
    - Batch processing for efficiency
    - Graph algorithm execution
    - Danger-ring detection
    """
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.driver = AsyncGraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        self.redis = redis.Redis.from_url(config.redis_url)
        
    async def initialize_schema(self):
        """Create Neo4j schema and indexes."""
        async with self.driver.session() as session:
            # Create indexes
            await session.run("""
                CREATE INDEX shape_id IF NOT EXISTS FOR (s:Shape) ON (s.id)
            """)
            
            await session.run("""
                CREATE INDEX shape_status IF NOT EXISTS FOR (s:Shape) ON (s.status)
            """)
            
            await session.run("""
                CREATE INDEX shape_created IF NOT EXISTS FOR (s:Shape) ON (s.created_at)
            """)
            
            # Create constraints
            await session.run("""
                CREATE CONSTRAINT shape_unique IF NOT EXISTS 
                FOR (s:Shape) REQUIRE s.id IS UNIQUE
            """)
            
            logger.info("Neo4j schema initialized")
    
    @traced("etl_sync_shapes")
    async def sync_shapes(self) -> int:
        """
        Sync shapes from Redis to Neo4j.
        
        Returns number of shapes synced.
        """
        start_time = time.time()
        shapes_synced = 0
        
        async with self.driver.session() as session:
            # Get all shape keys from Redis
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor, 
                    match=f"{KEY_PREFIX}*",
                    count=self.config.batch_size
                )
                
                if keys:
                    shapes_synced += await self._process_batch(session, keys)
                
                if cursor == 0:
                    break
            
            # Update metrics
            duration = time.time() - start_time
            observability.record_latency("etl_sync", "neo4j", duration)
            logger.info(f"Synced {shapes_synced} shapes in {duration:.2f}s")
            
        return shapes_synced
    
    async def _process_batch(
        self, 
        session: AsyncSession, 
        keys: List[bytes]
    ) -> int:
        """Process a batch of Redis keys."""
        shapes = []
        
        # Fetch data from Redis
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.hgetall(key)
        
        results = pipe.execute()
        
        # Parse shapes
        for key, data in zip(keys, results):
            if data:
                shape_id = key.decode().replace(KEY_PREFIX, "")
                
                # Parse stored data
                content = json.loads(data.get(b"content", b"{}"))
                metadata = json.loads(data.get(b"metadata", b"{}"))
                
                # Extract Betti numbers
                betti = content.get("betti_numbers", {})
                
                shapes.append({
                    "id": shape_id,
                    "b0": betti.get("b0", 0),
                    "b1": betti.get("b1", 0),
                    "b2": betti.get("b2", 0),
                    "created_at": int(data.get(b"created_at", 0)),
                    "context_type": data.get(b"context_type", b"general").decode(),
                    "status": metadata.get("status", "normal"),
                    "embedding": np.frombuffer(
                        data.get(b"embedding", b""), 
                        dtype=np.float32
                    ).tolist()
                })
        
        # Batch insert/update in Neo4j
        if shapes:
            await session.run("""
                UNWIND $shapes AS shape
                MERGE (s:Shape {id: shape.id})
                SET s.b0 = shape.b0,
                    s.b1 = shape.b1,
                    s.b2 = shape.b2,
                    s.created_at = shape.created_at,
                    s.context_type = shape.context_type,
                    s.status = shape.status,
                    s.embedding = shape.embedding,
                    s.updated_at = timestamp()
            """, shapes=shapes)
        
        return len(shapes)
    
    @traced("etl_compute_similarities")
    async def compute_similarities(self):
        """
        Compute similarity edges between shapes.
        
        Uses Neo4j GDS for efficient k-NN computation.
        """
        async with self.driver.session() as session:
            # Create in-memory graph projection
            await session.run("""
                CALL gds.graph.project.cypher(
                    'shape-similarity',
                    'MATCH (s:Shape) RETURN id(s) AS id, s.embedding AS embedding',
                    'RETURN null AS source, null AS target'
                )
            """)
            
            # Run k-NN algorithm
            await session.run(f"""
                CALL gds.knn.write('shape-similarity', {{
                    topK: {self.config.max_similar_edges},
                    nodeProperties: ['embedding'],
                    writeRelationshipType: 'SIMILAR',
                    writeProperty: 'score',
                    similarityCutoff: {self.config.similarity_threshold}
                }})
            """)
            
            # Drop projection
            await session.run("""
                CALL gds.graph.drop('shape-similarity')
            """)
            
            logger.info("Similarity computation complete")
    
    @traced("etl_detect_danger_rings")
    async def detect_danger_rings(self) -> List[Dict[str, Any]]:
        """
        Detect danger rings - clusters of shapes near failures.
        
        Returns list of danger zones with risk scores.
        """
        danger_zones = []
        
        async with self.driver.session() as session:
            # Find shapes within 2 hops of failures
            result = await session.run("""
                MATCH (danger:Shape {status: 'failed'})
                MATCH (s:Shape)-[:SIMILAR*1..2]-(danger)
                WHERE s.status <> 'failed'
                WITH s, count(DISTINCT danger) as danger_count,
                     avg(similarity.score) as avg_similarity
                WHERE danger_count >= 2
                RETURN s.id as shape_id,
                       danger_count,
                       avg_similarity,
                       s.b0, s.b1, s.b2
                ORDER BY danger_count DESC, avg_similarity DESC
                LIMIT 100
            """)
            
            async for record in result:
                danger_zones.append({
                    "shape_id": record["shape_id"],
                    "danger_count": record["danger_count"],
                    "avg_similarity": record["avg_similarity"],
                    "risk_score": record["danger_count"] * record["avg_similarity"],
                    "topology": {
                        "b0": record["b0"],
                        "b1": record["b1"],
                        "b2": record["b2"]
                    }
                })
            
            # Mark high-risk shapes
            if danger_zones:
                high_risk_ids = [
                    z["shape_id"] for z in danger_zones 
                    if z["risk_score"] > 0.8
                ]
                
                await session.run("""
                    UNWIND $ids AS id
                    MATCH (s:Shape {id: id})
                    SET s.risk_status = 'high_risk',
                        s.risk_updated_at = timestamp()
                """, ids=high_risk_ids)
                
                logger.warning(f"Marked {len(high_risk_ids)} shapes as high risk")
        
        return danger_zones
    
    @traced("etl_run_graph_algorithms")
    async def run_graph_algorithms(self):
        """Run advanced graph algorithms for pattern detection."""
        async with self.driver.session() as session:
            # Create graph projection
            await session.run("""
                CALL gds.graph.project(
                    'shape-analysis',
                    'Shape',
                    {
                        SIMILAR: {
                            properties: 'score'
                        }
                    }
                )
            """)
            
            try:
                # Run FastRP for new embeddings
                if self.config.run_fastrp:
                    await session.run("""
                        CALL gds.fastRP.write('shape-analysis', {
                            embeddingDimension: 128,
                            iterationWeights: [0.0, 1.0, 1.0],
                            writeProperty: 'fastrp_embedding'
                        })
                    """)
                
                # Run community detection
                if self.config.run_community_detection:
                    await session.run("""
                        CALL gds.louvain.write('shape-analysis', {
                            writeProperty: 'community'
                        })
                    """)
                    
                    # Analyze communities
                    result = await session.run("""
                        MATCH (s:Shape)
                        WITH s.community as community, 
                             collect(s.status) as statuses
                        WITH community, 
                             size([s IN statuses WHERE s = 'failed']) as failed_count,
                             size(statuses) as total_count
                        WHERE failed_count > 0
                        RETURN community,
                               failed_count,
                               total_count,
                               toFloat(failed_count) / total_count as failure_rate
                        ORDER BY failure_rate DESC
                    """)
                    
                    async for record in result:
                        if record["failure_rate"] > 0.3:
                            logger.warning(
                                f"High-risk community {record['community']}: "
                                f"{record['failure_rate']:.1%} failure rate"
                            )
                
            finally:
                # Clean up projection
                await session.run("""
                    CALL gds.graph.drop('shape-analysis', false)
                """)
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate ETL summary report."""
        async with self.driver.session() as session:
            # Get statistics
            stats_result = await session.run("""
                MATCH (s:Shape)
                WITH count(s) as total_shapes,
                     count(CASE WHEN s.status = 'failed' THEN 1 END) as failed_shapes,
                     count(CASE WHEN s.risk_status = 'high_risk' THEN 1 END) as high_risk_shapes
                MATCH ()-[r:SIMILAR]->()
                WITH total_shapes, failed_shapes, high_risk_shapes, count(r) as similarity_edges
                RETURN total_shapes, failed_shapes, high_risk_shapes, similarity_edges
            """)
            
            stats = await stats_result.single()
            
            # Get community statistics
            community_result = await session.run("""
                MATCH (s:Shape)
                WHERE s.community IS NOT NULL
                RETURN count(DISTINCT s.community) as num_communities
            """)
            
            community_stats = await community_result.single()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_shapes": stats["total_shapes"],
                "failed_shapes": stats["failed_shapes"],
                "high_risk_shapes": stats["high_risk_shapes"],
                "similarity_edges": stats["similarity_edges"],
                "num_communities": community_stats["num_communities"] if community_stats else 0,
                "danger_zones": len(await self.detect_danger_rings())
            }
    
    async def run_full_pipeline(self):
        """Run the complete ETL pipeline."""
        logger.info("Starting Neo4j ETL pipeline")
        
        try:
            # Initialize schema
            await self.initialize_schema()
            
            # Sync shapes
            shapes_synced = await self.sync_shapes()
            
            if shapes_synced > 0:
                # Compute similarities
                await self.compute_similarities()
                
                # Run graph algorithms
                await self.run_graph_algorithms()
                
                # Detect danger rings
                danger_zones = await self.detect_danger_rings()
                
                # Generate report
                report = await self.generate_report()
                
                logger.info(f"ETL complete: {report}")
                
                # Update metrics
                observability.update_memory_count("neo4j", report["total_shapes"])
                
                return report
            else:
                logger.info("No shapes to process")
                return None
                
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            observability.increment_counter("etl_pipeline", "error")
            raise
    
    async def close(self):
        """Clean up resources."""
        await self.driver.close()


def schedule_etl(config: ETLConfig):
    """Schedule nightly ETL runs."""
    etl = Neo4jETL(config)
    
    async def run_etl():
        """Async wrapper for ETL."""
        try:
            await etl.run_full_pipeline()
        finally:
            await etl.close()
    
    # Schedule nightly at 2 AM
    schedule.every().day.at("02:00").do(
        lambda: asyncio.run(run_etl())
    )
    
    logger.info("ETL scheduled for nightly execution at 02:00")
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    # Example usage
    config = ETLConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379"
    )
    
    # Run once
    asyncio.run(Neo4jETL(config).run_full_pipeline())