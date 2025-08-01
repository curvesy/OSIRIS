"""
🧠 Knowledge Graph Manager - Latest 2025 Patterns
Professional Neo4j integration for memory consolidation and learning loops.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver not available - install with: pip install neo4j")

try:
    from .config import ObservabilityConfig
    from .context_managers import ObservabilityContext
except ImportError:
    # Fallback for direct import
    from config import ObservabilityConfig
    from context_managers import ObservabilityContext


class KnowledgeGraphManager:
    """
    Latest 2025 Neo4j integration for memory consolidation.
    
    Features:
    - Workflow execution graph recording
    - Agent interaction patterns
    - Decision tree visualization
    - Learning pattern extraction
    - Performance correlation analysis
    - Anomaly detection through graph patterns
    - Memory consolidation for continuous learning
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize knowledge graph manager.
        
        Args:
            config: Observability configuration
        """
        
        self.config = config
        self.driver = None
        self.is_available = NEO4J_AVAILABLE and bool(config.neo4j_uri)
        
        # Graph schema version
        self.schema_version = "2025.7.27"
        
        # Batch processing for performance
        self._event_queue: List[Dict[str, Any]] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._batch_size = 50
        self._batch_interval = 10.0  # seconds
    
    async def initialize(self) -> None:
        """
        Initialize Neo4j connection and schema.
        """
        
        if not self.is_available:
            print("⚠️ Neo4j not available - skipping knowledge graph initialization")
            return
        
        try:
            # Create async driver
            self.driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                database=self.config.neo4j_database,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60.0,
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize schema
            await self._initialize_schema()
            
            # Start batch processing
            self._batch_task = asyncio.create_task(self._batch_processor())
            
            print(f"✅ Knowledge graph initialized - Database: {self.config.neo4j_database}")
            
        except Exception as e:
            print(f"⚠️ Knowledge graph initialization failed: {e}")
            self.is_available = False
    
    async def _test_connection(self) -> None:
        """Test Neo4j connection."""
        
        if not self.driver:
            return
        
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                if record["test"] != 1:
                    raise Exception("Connection test failed")
                
            print("✅ Neo4j connection verified")
            
        except Exception as e:
            raise Exception(f"Neo4j connection test failed: {e}")
    
    async def _initialize_schema(self) -> None:
        """Initialize knowledge graph schema with latest 2025 patterns."""
        
        if not self.driver:
            return
        
        schema_queries = [
            # Organism nodes
            """
            CREATE CONSTRAINT organism_id_unique IF NOT EXISTS
            FOR (o:Organism) REQUIRE o.organism_id IS UNIQUE
            """,
            
            # Workflow nodes
            """
            CREATE CONSTRAINT workflow_id_unique IF NOT EXISTS
            FOR (w:Workflow) REQUIRE w.workflow_id IS UNIQUE
            """,
            
            # Agent nodes
            """
            CREATE CONSTRAINT agent_name_unique IF NOT EXISTS
            FOR (a:Agent) REQUIRE a.name IS UNIQUE
            """,
            
            # Evidence nodes
            """
            CREATE INDEX evidence_timestamp IF NOT EXISTS
            FOR (e:Evidence) ON (e.timestamp)
            """,
            
            # Decision nodes
            """
            CREATE INDEX decision_timestamp IF NOT EXISTS
            FOR (d:Decision) ON (d.timestamp)
            """,
            
            # Error nodes
            """
            CREATE INDEX error_timestamp IF NOT EXISTS
            FOR (err:Error) ON (err.timestamp)
            """,
            
            # Performance indexes
            """
            CREATE INDEX workflow_performance IF NOT EXISTS
            FOR (w:Workflow) ON (w.duration, w.status)
            """,
        ]
        
        try:
            async with self.driver.session() as session:
                for query in schema_queries:
                    await session.run(query)
                
                # Create organism node if not exists
                await session.run(
                    """
                    MERGE (o:Organism {organism_id: $organism_id})
                    SET o.generation = $generation,
                        o.environment = $environment,
                        o.version = $version,
                        o.schema_version = $schema_version,
                        o.created_at = datetime(),
                        o.last_updated = datetime()
                    """,
                    organism_id=self.config.organism_id,
                    generation=self.config.organism_generation,
                    environment=self.config.deployment_environment,
                    version=self.config.service_version,
                    schema_version=self.schema_version
                )
            
            print("✅ Knowledge graph schema initialized")
            
        except Exception as e:
            print(f"⚠️ Schema initialization failed: {e}")
    
    async def record_workflow_event(self, context: ObservabilityContext, state: Dict[str, Any]) -> None:
        """
        Record workflow execution in knowledge graph.
        
        Args:
            context: Observability context
            state: Final workflow state
        """
        
        if not self.is_available:
            return
        
        # Create workflow event for batch processing
        workflow_event = {
            "type": "workflow",
            "workflow_id": context.workflow_id,
            "workflow_type": context.workflow_type,
            "status": context.status,
            "duration": context.duration,
            "error": context.error,
            "evidence_count": len(state.get("evidence_log", [])),
            "error_count": len(state.get("error_log", [])),
            "recovery_attempts": state.get("error_recovery_attempts", 0),
            "system_health_score": state.get("system_health", {}).get("health_score", 0.0),
            "agents_involved": context.metadata.get("agents_involved", []),
            "trace_id": context.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organism_id": self.config.organism_id,
        }
        
        # Add to batch queue
        self._event_queue.append(workflow_event)
        
        # Process evidence log
        evidence_log = state.get("evidence_log", [])
        for evidence in evidence_log:
            evidence_event = {
                "type": "evidence",
                "workflow_id": context.workflow_id,
                "evidence_type": getattr(evidence, 'evidence_type', 'unknown'),
                "confidence": getattr(evidence, 'confidence', 0.0),
                "timestamp": getattr(evidence, 'timestamp', datetime.now(timezone.utc).isoformat()),
                "organism_id": self.config.organism_id,
            }
            self._event_queue.append(evidence_event)
        
        # Process error log
        error_log = state.get("error_log", [])
        for error in error_log:
            error_event = {
                "type": "error",
                "workflow_id": context.workflow_id,
                "error_type": error.get("error_type", "unknown"),
                "severity": error.get("severity", "medium"),
                "recovery_strategy": error.get("recovery_strategy", "none"),
                "timestamp": error.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "organism_id": self.config.organism_id,
            }
            self._event_queue.append(error_event)
    
    async def record_agent_interaction(
        self, 
        agent_name: str, 
        tool_name: str, 
        workflow_id: str,
        duration: float, 
        success: bool
    ) -> None:
        """Record agent interaction patterns."""
        
        if not self.is_available:
            return
        
        agent_event = {
            "type": "agent_interaction",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "workflow_id": workflow_id,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organism_id": self.config.organism_id,
        }
        
        self._event_queue.append(agent_event)
    
    async def record_decision_point(
        self, 
        workflow_id: str, 
        decision_type: str, 
        options: List[str],
        chosen_option: str, 
        confidence: float, 
        rationale: str
    ) -> None:
        """Record decision points for learning analysis."""
        
        if not self.is_available:
            return
        
        decision_event = {
            "type": "decision",
            "workflow_id": workflow_id,
            "decision_type": decision_type,
            "options": options,
            "chosen_option": chosen_option,
            "confidence": confidence,
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organism_id": self.config.organism_id,
        }
        
        self._event_queue.append(decision_event)
    
    async def _batch_processor(self) -> None:
        """Process event queue in batches for performance."""
        
        while True:
            try:
                if len(self._event_queue) >= self._batch_size:
                    # Process full batch
                    batch = self._event_queue[:self._batch_size]
                    self._event_queue = self._event_queue[self._batch_size:]
                    await self._process_batch(batch)
                elif self._event_queue:
                    # Process remaining events after interval
                    await asyncio.sleep(self._batch_interval)
                    if self._event_queue:
                        batch = self._event_queue.copy()
                        self._event_queue.clear()
                        await self._process_batch(batch)
                else:
                    # Wait for events
                    await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Batch processing error: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of events."""
        
        if not self.driver or not batch:
            return
        
        try:
            async with self.driver.session() as session:
                # Group events by type for efficient processing
                workflows = [e for e in batch if e["type"] == "workflow"]
                evidence = [e for e in batch if e["type"] == "evidence"]
                errors = [e for e in batch if e["type"] == "error"]
                agents = [e for e in batch if e["type"] == "agent_interaction"]
                decisions = [e for e in batch if e["type"] == "decision"]
                
                # Process workflows
                if workflows:
                    await self._process_workflow_batch(session, workflows)
                
                # Process evidence
                if evidence:
                    await self._process_evidence_batch(session, evidence)
                
                # Process errors
                if errors:
                    await self._process_error_batch(session, errors)
                
                # Process agent interactions
                if agents:
                    await self._process_agent_batch(session, agents)
                
                # Process decisions
                if decisions:
                    await self._process_decision_batch(session, decisions)
            
            print(f"📊 Processed knowledge graph batch: {len(batch)} events")
            
        except Exception as e:
            print(f"⚠️ Batch processing failed: {e}")
            # Re-queue events for retry
            self._event_queue.extend(batch)
    
    async def _process_workflow_batch(self, session, workflows: List[Dict[str, Any]]) -> None:
        """Process workflow events batch."""
        
        query = """
        UNWIND $workflows as workflow
        MATCH (o:Organism {organism_id: workflow.organism_id})
        MERGE (w:Workflow {workflow_id: workflow.workflow_id})
        SET w.workflow_type = workflow.workflow_type,
            w.status = workflow.status,
            w.duration = workflow.duration,
            w.error = workflow.error,
            w.evidence_count = workflow.evidence_count,
            w.error_count = workflow.error_count,
            w.recovery_attempts = workflow.recovery_attempts,
            w.system_health_score = workflow.system_health_score,
            w.trace_id = workflow.trace_id,
            w.timestamp = datetime(workflow.timestamp),
            w.last_updated = datetime()
        MERGE (o)-[:EXECUTED]->(w)
        """
        
        await session.run(query, workflows=workflows)
    
    async def _process_evidence_batch(self, session, evidence: List[Dict[str, Any]]) -> None:
        """Process evidence events batch."""
        
        query = """
        UNWIND $evidence as ev
        MATCH (w:Workflow {workflow_id: ev.workflow_id})
        CREATE (e:Evidence {
            evidence_type: ev.evidence_type,
            confidence: ev.confidence,
            timestamp: datetime(ev.timestamp),
            organism_id: ev.organism_id
        })
        CREATE (w)-[:GENERATED]->(e)
        """
        
        await session.run(query, evidence=evidence)
    
    async def _process_error_batch(self, session, errors: List[Dict[str, Any]]) -> None:
        """Process error events batch."""
        
        query = """
        UNWIND $errors as err
        MATCH (w:Workflow {workflow_id: err.workflow_id})
        CREATE (e:Error {
            error_type: err.error_type,
            severity: err.severity,
            recovery_strategy: err.recovery_strategy,
            timestamp: datetime(err.timestamp),
            organism_id: err.organism_id
        })
        CREATE (w)-[:ENCOUNTERED]->(e)
        """
        
        await session.run(query, errors=errors)
    
    async def _process_agent_batch(self, session, agents: List[Dict[str, Any]]) -> None:
        """Process agent interaction events batch."""
        
        query = """
        UNWIND $agents as agent
        MATCH (w:Workflow {workflow_id: agent.workflow_id})
        MERGE (a:Agent {name: agent.agent_name})
        CREATE (i:Interaction {
            tool_name: agent.tool_name,
            duration: agent.duration,
            success: agent.success,
            timestamp: datetime(agent.timestamp),
            organism_id: agent.organism_id
        })
        CREATE (a)-[:PERFORMED]->(i)
        CREATE (w)-[:INVOLVED]->(a)
        """
        
        await session.run(query, agents=agents)
    
    async def _process_decision_batch(self, session, decisions: List[Dict[str, Any]]) -> None:
        """Process decision events batch."""
        
        query = """
        UNWIND $decisions as dec
        MATCH (w:Workflow {workflow_id: dec.workflow_id})
        CREATE (d:Decision {
            decision_type: dec.decision_type,
            options: dec.options,
            chosen_option: dec.chosen_option,
            confidence: dec.confidence,
            rationale: dec.rationale,
            timestamp: datetime(dec.timestamp),
            organism_id: dec.organism_id
        })
        CREATE (w)-[:MADE]->(d)
        """
        
        await session.run(query, decisions=decisions)
    
    async def get_historical_context(self, current_evidence: List[dict], top_k: int = 3) -> List[dict]:
        """
        Queries the knowledge graph for similar past workflows to provide historical context.
        This is the core of the organism's memory retrieval.

        Args:
            current_evidence: List of current evidence dictionaries
            top_k: Maximum number of historical contexts to return

        Returns:
            List of historical context dictionaries with workflow patterns
        """
        if not self.driver or not current_evidence:
            return []

        # Create a unique signature for the current situation based on evidence types
        current_evidence_types = [e.get("evidence_type", "unknown") for e in current_evidence]

        # A Cypher query to find similar, successfully completed workflows
        # and retrieve the actions that were taken.
        cypher_query = """
        // Find workflows with similar evidence patterns
        MATCH (w:Workflow)-[:GENERATED]->(e:Evidence)
        WHERE w.status = 'success'
        AND w.timestamp >= datetime() - duration({days: 30})  // Within 30 days

        // Group by workflow and collect evidence types
        WITH w, collect(e.evidence_type) as workflow_evidence_types

        // Calculate similarity based on evidence type overlap
        WITH w, workflow_evidence_types,
             [type IN $current_evidence_types WHERE type IN workflow_evidence_types] as common_types
        WHERE size(common_types) > 0

        // Get agent interactions for successful workflows
        OPTIONAL MATCH (w)-[:INVOLVED]->(a:Agent)-[:PERFORMED]->(i:Interaction)
        WHERE i.success = true

        RETURN w.workflow_id as workflowId,
               w.duration as duration,
               w.workflow_type as workflowType,
               size(common_types) as similarityScore,
               collect(DISTINCT i.tool_name) as successfulActions,
               w.timestamp as completedAt
        ORDER BY similarityScore DESC, w.timestamp DESC
        LIMIT $top_k
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    cypher_query,
                    current_evidence_types=current_evidence_types,
                    top_k=top_k
                )
                historical_context = [dict(record) async for record in result]
                return historical_context

        except Exception as e:
            print(f"ERROR: Knowledge graph query for historical context failed: {e}")
            return []

    async def get_learning_insights(self, days: int = 7) -> Dict[str, Any]:
        """
        Extract learning insights from knowledge graph.

        Args:
            days: Number of days to analyze

        Returns:
            Dict containing learning insights
        """

        if not self.driver:
            return {}

        try:
            async with self.driver.session() as session:
                # Get workflow performance trends
                performance_query = """
                MATCH (w:Workflow)
                WHERE w.timestamp >= datetime() - duration({days: $days})
                RETURN w.workflow_type as type,
                       avg(w.duration) as avg_duration,
                       count(w) as total_count,
                       sum(case when w.status = 'success' then 1 else 0 end) as success_count
                """

                performance_result = await session.run(performance_query, days=days)
                performance_data = [dict(record) async for record in performance_result]

                # Get error patterns
                error_query = """
                MATCH (e:Error)
                WHERE e.timestamp >= datetime() - duration({days: $days})
                RETURN e.error_type as error_type,
                       e.recovery_strategy as recovery_strategy,
                       count(e) as frequency
                ORDER BY frequency DESC
                LIMIT 10
                """

                error_result = await session.run(error_query, days=days)
                error_data = [dict(record) async for record in error_result]

                # Get agent performance
                agent_query = """
                MATCH (a:Agent)-[:PERFORMED]->(i:Interaction)
                WHERE i.timestamp >= datetime() - duration({days: $days})
                RETURN a.name as agent_name,
                       avg(i.duration) as avg_duration,
                       sum(case when i.success then 1 else 0 end) as success_count,
                       count(i) as total_interactions
                """

                agent_result = await session.run(agent_query, days=days)
                agent_data = [dict(record) async for record in agent_result]

                return {
                    "performance_trends": performance_data,
                    "error_patterns": error_data,
                    "agent_performance": agent_data,
                    "analysis_period_days": days,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            print(f"⚠️ Learning insights extraction failed: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown knowledge graph manager."""
        
        # Cancel batch processing
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining events
        if self._event_queue:
            await self._process_batch(self._event_queue)
            self._event_queue.clear()
        
        # Close driver
        if self.driver:
            await self.driver.close()
        
        print("✅ Knowledge graph shutdown complete")
