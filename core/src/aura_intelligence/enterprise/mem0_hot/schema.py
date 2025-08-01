"""
📊 DuckDB Schema Management

CREATE TABLE definitions, hourly partitions, and indexing
for the recent_activity table following partab.md blueprint.
"""

import duckdb
from typing import Optional
from aura_intelligence.utils.logger import get_logger

logger = get_logger(__name__)

# Table name constant
RECENT_ACTIVITY_TABLE = "recent_activity"


def create_schema(conn: duckdb.DuckDBPyConnection, 
                 vector_dimension: int = 128) -> bool:
    """
    Create the recent_activity table with hourly partitioning.
    
    Schema based on partab.md specification:
    - timestamp, signature_hash, betti_0/1/2, agent_id, event_type
    - agent_meta, full_event JSON columns
    - signature_vector for similarity search
    - hour_bucket for automated partitioning
    """
    
    try:
        # Create main table with partitioning support
        schema_sql = f"""
        CREATE TABLE IF NOT EXISTS {RECENT_ACTIVITY_TABLE} (
            timestamp TIMESTAMP NOT NULL,
            signature_hash VARCHAR PRIMARY KEY,
            betti_0 INTEGER,
            betti_1 INTEGER, 
            betti_2 INTEGER,
            agent_id VARCHAR,
            event_type VARCHAR,
            agent_meta JSON,
            full_event JSON,
            signature_vector FLOAT[{vector_dimension}],
            retention_flag BOOLEAN DEFAULT FALSE,
            hour_bucket INTEGER GENERATED ALWAYS AS (
                CAST(EXTRACT(EPOCH FROM timestamp) / 3600 AS INTEGER)
            )
        )
        """
        
        conn.execute(schema_sql)
        logger.info(f"✅ Created {RECENT_ACTIVITY_TABLE} table with hourly partitioning")
        
        # Create indexes for optimal performance
        _create_indexes(conn)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create schema: {e}")
        return False


def _create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create performance indexes for the recent_activity table."""
    
    indexes = [
        # Primary index on signature_hash
        f"CREATE INDEX IF NOT EXISTS idx_signature_hash ON {RECENT_ACTIVITY_TABLE}(signature_hash)",
        
        # Time-based index for retention queries
        f"CREATE INDEX IF NOT EXISTS idx_hour_bucket ON {RECENT_ACTIVITY_TABLE}(hour_bucket)",
        
        # Agent-based index for agent-specific queries
        f"CREATE INDEX IF NOT EXISTS idx_agent_id ON {RECENT_ACTIVITY_TABLE}(agent_id)",
        
        # Event type index for filtering
        f"CREATE INDEX IF NOT EXISTS idx_event_type ON {RECENT_ACTIVITY_TABLE}(event_type)",
        
        # Timestamp index for time-range queries
        f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {RECENT_ACTIVITY_TABLE}(timestamp)",
        
        # Betti numbers composite index for topological similarity
        f"CREATE INDEX IF NOT EXISTS idx_betti_numbers ON {RECENT_ACTIVITY_TABLE}(betti_0, betti_1, betti_2)"
    ]
    
    for index_sql in indexes:
        try:
            conn.execute(index_sql)
            logger.debug(f"✅ Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
        except Exception as e:
            logger.warning(f"⚠️ Index creation failed: {e}")


def create_vector_index(conn: duckdb.DuckDBPyConnection, 
                       metric: str = "cosine") -> bool:
    """
    Create vector similarity index using DuckDB VSS extension.
    
    Args:
        conn: DuckDB connection
        metric: Similarity metric ('cosine', 'euclidean', 'dot_product')
    """
    
    try:
        # Install and load VSS extension
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        # Create vector similarity index
        vector_index_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_signature_vector 
        ON {RECENT_ACTIVITY_TABLE} 
        USING vss(signature_vector) 
        WITH (metric = '{metric}')
        """
        
        conn.execute(vector_index_sql)
        logger.info(f"✅ Created vector similarity index with {metric} metric")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Vector index creation failed: {e}")
        return False


def get_table_info(conn: duckdb.DuckDBPyConnection) -> dict:
    """Get information about the recent_activity table."""
    
    try:
        # Get table schema
        schema_result = conn.execute(f"DESCRIBE {RECENT_ACTIVITY_TABLE}").fetchall()
        
        # Get row count
        count_result = conn.execute(f"SELECT COUNT(*) FROM {RECENT_ACTIVITY_TABLE}").fetchone()
        row_count = count_result[0] if count_result else 0
        
        # Get partition info (hour buckets)
        partition_result = conn.execute(f"""
            SELECT hour_bucket, COUNT(*) as row_count
            FROM {RECENT_ACTIVITY_TABLE}
            GROUP BY hour_bucket
            ORDER BY hour_bucket DESC
            LIMIT 10
        """).fetchall()
        
        # Get index info
        index_result = conn.execute(f"""
            SELECT index_name, is_unique, is_primary
            FROM duckdb_indexes()
            WHERE table_name = '{RECENT_ACTIVITY_TABLE}'
        """).fetchall()
        
        return {
            "table_name": RECENT_ACTIVITY_TABLE,
            "schema": schema_result,
            "row_count": row_count,
            "recent_partitions": partition_result,
            "indexes": index_result
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get table info: {e}")
        return {"error": str(e)}


def cleanup_old_partitions(conn: duckdb.DuckDBPyConnection, 
                          retention_hours: int = 24) -> int:
    """
    Clean up old partitions beyond retention period.
    
    Returns:
        Number of rows deleted
    """
    
    try:
        # Calculate cutoff hour bucket
        cutoff_sql = f"""
        SELECT CAST(EXTRACT(EPOCH FROM (NOW() - INTERVAL '{retention_hours} hours')) / 3600 AS INTEGER) as cutoff_bucket
        """
        
        cutoff_result = conn.execute(cutoff_sql).fetchone()
        cutoff_bucket = cutoff_result[0] if cutoff_result else 0
        
        # Delete old partitions
        delete_sql = f"""
        DELETE FROM {RECENT_ACTIVITY_TABLE}
        WHERE hour_bucket < {cutoff_bucket}
        """
        
        result = conn.execute(delete_sql)
        deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
        
        logger.info(f"🗑️ Cleaned up {deleted_count} old records (bucket < {cutoff_bucket})")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup old partitions: {e}")
        return 0
