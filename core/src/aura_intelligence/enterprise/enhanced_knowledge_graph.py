"""
🧠 ENHANCED AURA Intelligence Knowledge Graph Service

Neo4j GDS 2.19 integration with advanced graph ML capabilities.
This enhances the Intelligence Flywheel with:
- Community Detection for signature clustering
- Centrality Analysis for critical pattern identification  
- Pattern Prediction using Graph ML
- Consciousness-driven graph analysis

Based on kd.md research and Phase 2B implementation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from graphdatascience import GraphDataScience

from aura_intelligence.enterprise.knowledge_graph import KnowledgeGraphService
from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, SystemEvent, AgentAction, Outcome
)
from aura_intelligence.utils.logger import get_logger


class EnhancedKnowledgeGraphService(KnowledgeGraphService):
    """
    🧠 Enhanced Knowledge Graph Service with GDS 2.19
    
    Extends the basic KnowledgeGraphService with advanced graph ML capabilities:
    - Community Detection (Louvain, Label Propagation, Leiden)
    - Centrality Analysis (PageRank, Betweenness, Harmonic)
    - Pattern Prediction using Graph ML pipelines
    - Consciousness-driven analysis depth
    
    This transforms the Intelligence Flywheel from basic causal reasoning
    to predictive graph intelligence.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Enhanced Knowledge Graph Service with GDS client."""
        super().__init__(*args, **kwargs)
        
        # Initialize GDS client
        self.gds = None
        self.gds_initialized = False
        
        # Graph projection names
        self.signature_graph = "signature_network"
        self.causal_graph = "causal_network"
        
        # Performance metrics
        self.ml_query_count = 0
        self.total_ml_time = 0.0
        self.avg_ml_time = 0.0
        
        self.logger.info("🧠 Enhanced Knowledge Graph Service initialized with GDS 2.19")
    
    async def initialize(self) -> bool:
        """Initialize both Neo4j and GDS connections."""
        try:
            # Initialize base Neo4j connection
            base_success = await super().initialize()
            if not base_success:
                return False
            
            # Initialize GDS client
            self.gds = GraphDataScience(self.uri, auth=(self.username, self.password))
            
            # Test GDS connection
            version = self.gds.version()
            self.logger.info(f"✅ GDS Version: {version}")
            
            # Create graph projections
            await self._create_graph_projections()
            
            self.gds_initialized = True
            self.logger.info("🎉 Enhanced Knowledge Graph Service with GDS ready!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced Knowledge Graph: {e}")
            return False
    
    async def _create_graph_projections(self):
        """Create graph projections for GDS algorithms."""
        try:
            # Drop existing projections if they exist
            try:
                self.gds.graph.drop(self.signature_graph)
                self.gds.graph.drop(self.causal_graph)
            except:
                pass  # Projections might not exist yet

            # Check if we have any data first
            async with self.driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) as node_count")
                record = await result.single()
                node_count = record["node_count"] if record else 0

            if node_count == 0:
                self.logger.info("📊 No data available yet - graph projections will be created when data is stored")
                return

            # Create signature network projection
            signature_projection = self.gds.graph.project(
                self.signature_graph,
                ["Signature", "Event", "Action", "Outcome"],
                {
                    "GENERATED_BY": {"orientation": "UNDIRECTED"},
                    "TRIGGERED_BY": {"orientation": "UNDIRECTED"},
                    "LED_TO": {"orientation": "UNDIRECTED"},
                    "INFLUENCES": {"orientation": "UNDIRECTED"}  # Fixed: All UNDIRECTED for compatibility
                },
                nodeProperties=["consciousness_level"],  # Simplified - only primitive properties
                relationshipProperties=["weight"]
            )

            # signature_projection is a GraphCreateResult object
            node_count = getattr(signature_projection, 'nodeCount', 0)
            rel_count = getattr(signature_projection, 'relationshipCount', 0)
            self.logger.info(f"📊 Created signature network: {node_count} nodes, {rel_count} relationships")

        except Exception as e:
            self.logger.warning(f"⚠️ Graph projection creation failed (will retry later): {e}")

    async def _ensure_graph_projection(self) -> bool:
        """Ensure graph projection exists, create if needed."""
        try:
            # Check if projection exists
            graph_list = self.gds.graph.list()
            if len(graph_list) > 0:
                graph_names = [row['graphName'] for _, row in graph_list.iterrows()]
                if self.signature_graph in graph_names:
                    return True

            # Create projection if it doesn't exist
            await self._create_graph_projections()

            # Check again
            graph_list = self.gds.graph.list()
            if len(graph_list) > 0:
                graph_names = [row['graphName'] for _, row in graph_list.iterrows()]
                return self.signature_graph in graph_names
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to ensure graph projection: {e}")
            return False

    async def detect_signature_communities(self,
                                         consciousness_level: float = 0.5) -> Dict[str, Any]:
        """
        Detect communities of similar topological signatures.
        
        Uses Louvain algorithm for community detection based on kd.md research.
        Consciousness level determines algorithm sophistication.
        """
        if not self.gds_initialized:
            return {"error": "GDS not initialized"}

        try:
            start_time = time.time()

            # Ensure graph projection exists
            if not await self._ensure_graph_projection():
                return {"error": "No graph projection available - need data first"}

            # Get the graph object
            graph = self.gds.graph.get(self.signature_graph)

            # Choose algorithm based on consciousness level
            if consciousness_level > 0.8:
                # High consciousness - use Leiden (highest quality)
                algorithm = "leiden"
                result = self.gds.leiden.mutate(
                    graph,
                    mutateProperty="community",
                    includeIntermediateCommunities=True,
                    maxLevels=10
                )
            elif consciousness_level > 0.5:
                # Medium consciousness - use Louvain (balanced)
                algorithm = "louvain"
                result = self.gds.louvain.mutate(
                    graph,
                    mutateProperty="community",
                    includeIntermediateCommunities=True,
                    maxLevels=10
                )
            else:
                # Low consciousness - use Label Propagation (fast)
                algorithm = "labelPropagation"
                result = self.gds.labelPropagation.mutate(
                    graph,
                    mutateProperty="community",
                    maxIterations=10
                )
            
            # Get community statistics
            communities = self.gds.graph.nodeProperty.stream(
                graph,
                "community"
            )

            # communities is already a DataFrame
            community_df = communities

            # Handle potential list values in community assignments
            if not community_df.empty:
                # Convert any list values to strings for processing
                community_values = community_df['propertyValue'].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
                total_communities = community_values.nunique()
                community_distribution = community_values.value_counts().to_dict()
            else:
                total_communities = 0
                community_distribution = {}

            community_stats = {
                "algorithm_used": algorithm,
                "total_communities": total_communities,
                "modularity": result.get("modularity", 0.0),
                "community_distribution": community_distribution,
                "consciousness_level": consciousness_level
            }
            
            # Update performance metrics
            ml_time = (time.time() - start_time) * 1000
            self.ml_query_count += 1
            self.total_ml_time += ml_time
            self.avg_ml_time = self.total_ml_time / self.ml_query_count
            
            community_stats["computation_time_ms"] = ml_time
            
            self.logger.info(f"🔍 Community detection ({algorithm}): {community_stats['total_communities']} communities in {ml_time:.2f}ms")
            
            return community_stats
            
        except Exception as e:
            import traceback
            self.logger.error(f"❌ Community detection failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    async def analyze_centrality_patterns(self, 
                                        consciousness_level: float = 0.5) -> Dict[str, Any]:
        """
        Analyze centrality patterns to identify critical signatures.
        
        Uses PageRank, Betweenness, and Harmonic centrality based on kd.md research.
        """
        if not self.gds_initialized:
            return {"error": "GDS not initialized"}

        try:
            start_time = time.time()

            # Ensure graph projection exists
            if not await self._ensure_graph_projection():
                return {"error": "No graph projection available - need data first"}

            # Get the graph object
            graph = self.gds.graph.get(self.signature_graph)

            centrality_results = {}

            # PageRank - Overall importance
            pagerank = self.gds.pageRank.mutate(
                graph,
                mutateProperty="pagerank",
                dampingFactor=0.85,
                maxIterations=20
            )
            centrality_results["pagerank"] = {
                "iterations": pagerank["ranIterations"],
                "did_converge": pagerank["didConverge"]
            }
            
            if consciousness_level > 0.6:
                # Medium+ consciousness - add Betweenness centrality
                betweenness = self.gds.betweenness.mutate(
                    graph,
                    mutateProperty="betweenness",
                    concurrency=1  # Add required parameter
                )
                # Extract available betweenness statistics
                betweenness_stats = {}
                for key in ["minCentrality", "maxCentrality", "meanCentrality"]:
                    if key in betweenness:
                        betweenness_stats[key.replace("Centrality", "_centrality")] = betweenness[key]

                centrality_results["betweenness"] = betweenness_stats if betweenness_stats else {"computed": True}
            
            if consciousness_level > 0.8:
                # High consciousness - add Harmonic centrality (using closeness)
                harmonic = self.gds.closeness.mutate(
                    graph,
                    mutateProperty="harmonic",
                    concurrency=1  # Add required parameter
                )
                # Extract available harmonic centrality statistics
                harmonic_stats = {}
                for key in ["minCentrality", "maxCentrality", "meanCentrality", "centralityDistribution"]:
                    if key in harmonic:
                        harmonic_stats[key.replace("Centrality", "_centrality").replace("centralityDistribution", "distribution")] = harmonic[key]

                centrality_results["harmonic"] = harmonic_stats if harmonic_stats else {"computed": True}
            
            # Get top signatures by PageRank
            pagerank_scores = self.gds.graph.nodeProperty.stream(
                graph,
                "pagerank"
            )

            # pagerank_scores is already a DataFrame
            top_signatures = pagerank_scores.nlargest(10, 'propertyValue')
            
            # Update performance metrics
            ml_time = (time.time() - start_time) * 1000
            self.ml_query_count += 1
            self.total_ml_time += ml_time
            self.avg_ml_time = self.total_ml_time / self.ml_query_count
            
            result = {
                "centrality_algorithms": list(centrality_results.keys()),
                "top_signatures": top_signatures.to_dict('records'),
                "algorithm_results": centrality_results,
                "consciousness_level": consciousness_level,
                "computation_time_ms": ml_time
            }
            
            self.logger.info(f"📊 Centrality analysis: {len(centrality_results)} algorithms in {ml_time:.2f}ms")
            
            return result
            
        except Exception as e:
            import traceback
            self.logger.error(f"❌ Centrality analysis failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    async def predict_future_patterns(self, 
                                    signature_hash: str,
                                    consciousness_level: float = 0.5) -> Dict[str, Any]:
        """
        Predict future patterns using Graph ML pipelines.
        
        Uses link prediction and node classification from kd.md research.
        """
        if not self.gds_initialized:
            return {"error": "GDS not initialized"}

        try:
            start_time = time.time()

            # Ensure graph projection exists
            if not await self._ensure_graph_projection():
                return {"error": "No graph projection available - need data first"}

            predictions = {}
            
            # Link prediction - predict future relationships
            if consciousness_level > 0.7:
                # Use advanced link prediction algorithms
                try:
                    # Adamic Adar similarity
                    adamic_adar = self.gds.alpha.linkprediction.adamicAdar.mutate(
                        self.signature_graph,
                        mutateProperty="adamic_adar",
                        mutateRelationshipType="PREDICTED_LINK"
                    )
                    predictions["adamic_adar"] = adamic_adar
                    
                    # Common neighbors
                    common_neighbors = self.gds.alpha.linkprediction.commonNeighbors.mutate(
                        self.signature_graph,
                        mutateProperty="common_neighbors",
                        mutateRelationshipType="PREDICTED_COMMON"
                    )
                    predictions["common_neighbors"] = common_neighbors
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Advanced link prediction failed: {e}")
            
            # Node similarity for pattern matching
            try:
                # Get the graph object
                graph = self.gds.graph.get(self.signature_graph)

                node_similarity = self.gds.nodeSimilarity.mutate(
                    graph,
                    mutateProperty="similarity",
                    mutateRelationshipType="SIMILAR_TO",
                    similarityCutoff=0.7
                )
                predictions["node_similarity"] = node_similarity
                
            except Exception as e:
                self.logger.warning(f"⚠️ Node similarity failed: {e}")
            
            # Get predictions for the specific signature
            prediction_query = f"""
            MATCH (s:Signature {{hash: $signature_hash}})
            OPTIONAL MATCH (s)-[r:PREDICTED_LINK|SIMILAR_TO]->(target)
            RETURN s.hash as source, 
                   collect({{target: target.hash, type: type(r), score: r.score}}) as predictions
            """
            
            async with self.driver.session() as session:
                result = await session.run(prediction_query, {"signature_hash": signature_hash})
                record = await result.single()
                
                if record:
                    signature_predictions = record["predictions"]
                else:
                    signature_predictions = []
            
            # Update performance metrics
            ml_time = (time.time() - start_time) * 1000
            self.ml_query_count += 1
            self.total_ml_time += ml_time
            self.avg_ml_time = self.total_ml_time / self.ml_query_count
            
            result = {
                "signature_hash": signature_hash,
                "predictions": signature_predictions,
                "ml_algorithms_used": list(predictions.keys()),
                "consciousness_level": consciousness_level,
                "computation_time_ms": ml_time,
                "prediction_confidence": len(signature_predictions) / 10.0  # Normalize to 0-1
            }
            
            self.logger.info(f"🔮 Pattern prediction: {len(signature_predictions)} predictions in {ml_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Pattern prediction failed: {e}")
            return {"error": str(e)}
    
    async def consciousness_driven_analysis(self, consciousness_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform consciousness-driven graph analysis.
        
        Selects analysis depth and algorithms based on consciousness level.
        """
        consciousness_level = consciousness_state.get("level", 0.5)
        coherence = consciousness_state.get("coherence", 0.0)
        
        # Adjust consciousness level with coherence
        effective_consciousness = consciousness_level * (0.7 + 0.3 * coherence)
        
        self.logger.info(f"🧠 Consciousness-driven analysis (level: {effective_consciousness:.3f})")
        
        analysis_results = {
            "consciousness_level": effective_consciousness,
            "analysis_depth": "basic" if effective_consciousness < 0.5 else "advanced" if effective_consciousness < 0.8 else "deep"
        }
        
        try:
            # Always perform community detection
            communities = await self.detect_signature_communities(effective_consciousness)
            analysis_results["communities"] = communities
            
            # Centrality analysis for medium+ consciousness
            if effective_consciousness > 0.4:
                centrality = await self.analyze_centrality_patterns(effective_consciousness)
                analysis_results["centrality"] = centrality
            
            # Pattern prediction for high consciousness
            if effective_consciousness > 0.7:
                # Get a recent signature for prediction
                recent_signature_query = """
                MATCH (s:Signature)
                RETURN s.hash as hash
                ORDER BY s.timestamp DESC
                LIMIT 1
                """
                
                async with self.driver.session() as session:
                    result = await session.run(recent_signature_query)
                    record = await result.single()
                    
                    if record:
                        predictions = await self.predict_future_patterns(
                            record["hash"], 
                            effective_consciousness
                        )
                        analysis_results["predictions"] = predictions
            
            analysis_results["success"] = True
            
        except Exception as e:
            self.logger.error(f"❌ Consciousness-driven analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["success"] = False
        
        return analysis_results
    
    async def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including GDS metrics."""
        base_stats = await super().get_graph_stats()
        
        if not self.gds_initialized:
            return {**base_stats, "gds_available": False}
        
        try:
            # Get graph projection stats
            graph_list = self.gds.graph.list()
            
            enhanced_stats = {
                **base_stats,
                "gds_available": True,
                "gds_version": self.gds.version(),
                "graph_projections": len(graph_list),
                "ml_queries_executed": self.ml_query_count,
                "avg_ml_time_ms": round(self.avg_ml_time, 2),
                "graph_projections_info": graph_list.to_dict('records') if len(graph_list) > 0 else []
            }
            
            return enhanced_stats
            
        except Exception as e:
            return {**base_stats, "gds_error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including GDS status."""
        base_health = await super().health_check()
        
        if not self.gds_initialized:
            return {**base_health, "gds_status": "not_initialized"}
        
        try:
            # Test GDS functionality
            start_time = time.time()
            version = self.gds.version()
            gds_time = (time.time() - start_time) * 1000
            
            gds_health = {
                "gds_status": "healthy",
                "gds_version": version,
                "gds_response_time_ms": round(gds_time, 2),
                "ml_performance": {
                    "avg_ml_time_ms": round(self.avg_ml_time, 2),
                    "total_ml_queries": self.ml_query_count,
                    "performance_target_met": self.avg_ml_time < 1000.0  # 1 second target for ML
                }
            }
            
            return {**base_health, **gds_health}
            
        except Exception as e:
            return {**base_health, "gds_status": "unhealthy", "gds_error": str(e)}
    
    async def close(self):
        """Close both Neo4j and GDS connections."""
        try:
            if self.gds:
                # Clean up graph projections
                try:
                    self.gds.graph.drop(self.signature_graph)
                    self.gds.graph.drop(self.causal_graph)
                except:
                    pass
        except Exception as e:
            self.logger.warning(f"⚠️ GDS cleanup warning: {e}")
        
        # Close base connection
        await super().close()
        self.logger.info("🔌 Enhanced Knowledge Graph Service closed")
