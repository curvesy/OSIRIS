#!/usr/bin/env python3
"""
Simple Neo4j connection test to isolate authentication issues.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig

async def test_neo4j_connection():
    """Test Neo4j connection with detailed logging."""
    print("🔍 Testing Neo4j connection...")
    
    # Create configuration
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j", 
        password="dev_password",
        database="neo4j"
    )
    
    print(f"📋 Configuration:")
    print(f"   URI: {config.uri}")
    print(f"   Username: {config.username}")
    print(f"   Password: {'*' * len(config.password)}")
    print(f"   Database: {config.database}")
    
    # Create adapter
    adapter = Neo4jAdapter(config)
    
    try:
        print("🔌 Initializing Neo4j adapter...")
        await adapter.initialize()
        print("✅ Neo4j adapter initialized successfully!")
        
        print("🔍 Testing simple query...")
        result = await adapter.query("RETURN 'Hello from Python' as message")
        print(f"✅ Query result: {result}")
        
        print("🔍 Testing write operation...")
        await adapter.write(
            "CREATE (n:TestNode {message: $message, timestamp: datetime()}) RETURN n",
            {"message": "Test from Python adapter"}
        )
        print("✅ Write operation successful!")
        
        print("🔍 Testing read back...")
        nodes = await adapter.query("MATCH (n:TestNode) RETURN n.message as message LIMIT 1")
        print(f"✅ Read back result: {nodes}")
        
        print("🧹 Cleaning up test data...")
        await adapter.write("MATCH (n:TestNode) DELETE n")
        print("✅ Cleanup successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("🔌 Closing Neo4j adapter...")
        await adapter.close()
        print("✅ Neo4j adapter closed!")
    
    return True

if __name__ == "__main__":
    print("🧪 Neo4j Connection Test")
    print("=" * 50)
    
    success = asyncio.run(test_neo4j_connection())
    
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Tests failed!")
        sys.exit(1)