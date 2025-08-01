"""
🚀 AURA Intelligence Search API Service

FastAPI service providing unified intelligence interface for the 7-agent system.
This is the core API that combines vector similarity search with causal reasoning
to provide actionable intelligence.

Based on kiki.md and ppdd.md research for professional 2025 architecture.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, TopologicalSignatureAPI, SearchResult,
    SystemEventAPI, AgentActionAPI, OutcomeAPI
)
from aura_intelligence.enterprise.vector_database import VectorDatabaseService
from aura_intelligence.enterprise.enhanced_knowledge_graph import EnhancedKnowledgeGraphService
from aura_intelligence.utils.logger import get_logger


class SearchAPIService:
    """
    🚀 Search API Service for Topological Intelligence
    
    Provides the unified intelligence interface that combines:
    - Vector similarity search (Qdrant) - "Have we seen this shape before?"
    - Causal reasoning (Neo4j) - "Why did this happen?"
    - Response enrichment - Actionable recommendations
    
    This is the core API that enables the intelligence flywheel.
    """
    
    def __init__(self,
                 vector_db_host: str = "localhost",
                 vector_db_port: int = 6333,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j",
                 neo4j_password: str = "password"):
        """Initialize Search API Service with database connections."""
        self.logger = get_logger(__name__)
        
        # Initialize database services
        self.vector_db = VectorDatabaseService(
            host=vector_db_host,
            port=vector_db_port
        )
        
        self.knowledge_graph = EnhancedKnowledgeGraphService(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Performance metrics
        self.search_count = 0
        self.total_search_time = 0.0
        self.avg_search_time = 0.0
        
        # Initialize FastAPI app
        self.app = self._create_app()
        
        self.logger.info("🚀 Search API Service initialized")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="AURA Intelligence Search API",
            description="Unified intelligence interface for topological search and causal reasoning",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)

        # Add Phase 2C Intelligence Flywheel routes
        self._add_phase2c_routes(app)

        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes to the FastAPI application."""
        
        @app.on_event("startup")
        async def startup_event():
            """Initialize database connections on startup."""
            await self.vector_db.initialize()
            await self.knowledge_graph.initialize()
            self.logger.info("✅ Search API Service started successfully")
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Clean up database connections on shutdown."""
            await self.knowledge_graph.close()
            self.logger.info("🔌 Search API Service shut down")
        
        @app.post("/search/topology", response_model=SearchResult)
        async def search_topology(signature: TopologicalSignatureAPI) -> SearchResult:
            """
            🔍 Main search endpoint - The core of the intelligence flywheel.
            
            This endpoint combines vector similarity search with causal reasoning
            to provide actionable intelligence for the 7-agent system.
            """
            try:
                start_time = time.time()
                
                # Convert API model to internal dataclass
                internal_signature = signature.to_dataclass()
                
                # Stage 1: Vector similarity search
                self.logger.debug(f"🔍 Stage 1: Vector similarity search for {internal_signature.signature_hash[:8]}...")
                similar_signatures = await self.vector_db.search_similar(
                    query_signature=internal_signature,
                    limit=5,
                    score_threshold=0.7
                )
                
                # Stage 2: Enhanced graph analysis with GDS
                self.logger.debug("🧠 Stage 2: Enhanced graph analysis with GDS...")
                contextual_insights = []

                # Perform consciousness-driven graph analysis
                consciousness_state = {
                    "level": signature.consciousness_level,
                    "coherence": signature.quantum_coherence
                }

                graph_analysis = await self.knowledge_graph.consciousness_driven_analysis(consciousness_state)

                for similar in similar_signatures:
                    causal_context = await self.knowledge_graph.get_causal_context(
                        signature_hash=similar["signature_hash"],
                        depth=3
                    )

                    if causal_context and "error" not in causal_context:
                        insight = {
                            "signature_hash": similar["signature_hash"],
                            "similarity_score": similar["similarity_score"],
                            "consciousness_weighted_score": similar.get("consciousness_weighted_score", 0.0),
                            "causal_context": causal_context,
                            "recommendations": self._generate_recommendations(causal_context),
                            "graph_analysis": graph_analysis  # Enhanced with GDS insights
                        }
                        contextual_insights.append(insight)
                
                # Stage 3: Response enrichment
                search_time = (time.time() - start_time) * 1000
                confidence_score = self._calculate_confidence_score(similar_signatures, contextual_insights)
                recommendations = self._generate_overall_recommendations(contextual_insights)
                
                # Update performance metrics
                self.search_count += 1
                self.total_search_time += search_time
                self.avg_search_time = self.total_search_time / self.search_count
                
                self.logger.info(f"✅ Search completed in {search_time:.2f}ms (confidence: {confidence_score:.3f})")
                
                return SearchResult(
                    query_signature=internal_signature.signature_hash,
                    similar_signatures=[
                        {
                            "signature_hash": s["signature_hash"],
                            "similarity_score": s["similarity_score"],
                            "consciousness_weighted_score": s.get("consciousness_weighted_score", 0.0),
                            "payload": s.get("payload", {})
                        }
                        for s in similar_signatures
                    ],
                    contextual_insights=contextual_insights,
                    confidence_score=confidence_score,
                    search_time_ms=search_time,
                    recommendations=recommendations
                )
                
            except Exception as e:
                self.logger.error(f"❌ Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @app.get("/signatures/{signature_hash}")
        async def get_signature(signature_hash: str):
            """Get a specific signature by hash."""
            try:
                # Get from vector database
                vector_data = await self.vector_db.get_signature_by_hash(signature_hash)
                
                # Get causal context
                causal_context = await self.knowledge_graph.get_causal_context(signature_hash)
                
                if vector_data:
                    return {
                        "signature_hash": signature_hash,
                        "vector_data": vector_data,
                        "causal_context": causal_context,
                        "found": True
                    }
                else:
                    raise HTTPException(status_code=404, detail="Signature not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"❌ Failed to get signature {signature_hash}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/events")
        async def store_event(event: SystemEventAPI, background_tasks: BackgroundTasks):
            """Store a system event."""
            try:
                # Convert to internal format and store
                # This would be implemented based on your specific event handling needs
                background_tasks.add_task(self._process_event, event)
                
                return {"status": "accepted", "event_id": event.event_type}
                
            except Exception as e:
                self.logger.error(f"❌ Failed to store event: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/actions")
        async def store_action(action: AgentActionAPI, background_tasks: BackgroundTasks):
            """Store an agent action."""
            try:
                background_tasks.add_task(self._process_action, action)
                return {"status": "accepted", "action_id": action.action_type}
                
            except Exception as e:
                self.logger.error(f"❌ Failed to store action: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/outcomes")
        async def store_outcome(outcome: OutcomeAPI, background_tasks: BackgroundTasks):
            """Store an action outcome."""
            try:
                background_tasks.add_task(self._process_outcome, outcome)
                return {"status": "accepted", "outcome_id": outcome.action_id}
                
            except Exception as e:
                self.logger.error(f"❌ Failed to store outcome: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            """System health check."""
            try:
                # Check vector database health
                vector_health = await self.vector_db.health_check()
                
                # Check knowledge graph health
                graph_health = await self.knowledge_graph.health_check()
                
                # Overall system health
                overall_status = "healthy"
                if vector_health["status"] != "healthy" or graph_health["status"] != "healthy":
                    overall_status = "degraded"
                
                return {
                    "status": overall_status,
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "vector_database": vector_health,
                        "knowledge_graph": graph_health
                    },
                    "performance": {
                        "avg_search_time_ms": round(self.avg_search_time, 2),
                        "total_searches": self.search_count,
                        "target_performance_met": self.avg_search_time < 50.0  # 50ms target for full search
                    }
                }
                
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        @app.get("/metrics")
        async def get_metrics():
            """Get system performance metrics."""
            try:
                vector_stats = await self.vector_db.get_collection_stats()
                graph_stats = await self.knowledge_graph.get_enhanced_stats()  # Enhanced stats

                return {
                    "search_api": {
                        "total_searches": self.search_count,
                        "avg_search_time_ms": round(self.avg_search_time, 2),
                        "performance_target_met": self.avg_search_time < 50.0
                    },
                    "vector_database": vector_stats,
                    "enhanced_knowledge_graph": graph_stats,  # Enhanced with GDS metrics
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/graph/communities")
        async def detect_communities(consciousness_level: float = 0.5):
            """🧠 Detect signature communities using GDS algorithms."""
            try:
                communities = await self.knowledge_graph.detect_signature_communities(consciousness_level)
                return {
                    "communities": communities,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/graph/centrality")
        async def analyze_centrality(consciousness_level: float = 0.5):
            """📊 Analyze centrality patterns using GDS algorithms."""
            try:
                centrality = await self.knowledge_graph.analyze_centrality_patterns(consciousness_level)
                return {
                    "centrality": centrality,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/graph/predict/{signature_hash}")
        async def predict_patterns(signature_hash: str, consciousness_level: float = 0.5):
            """🔮 Predict future patterns using Graph ML."""
            try:
                predictions = await self.knowledge_graph.predict_future_patterns(
                    signature_hash, consciousness_level
                )
                return {
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/graph/consciousness-analysis")
        async def consciousness_analysis(consciousness_state: dict):
            """🧠 Perform consciousness-driven graph analysis."""
            try:
                analysis = await self.knowledge_graph.consciousness_driven_analysis(consciousness_state)
                return {
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_recommendations(self, causal_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on causal context."""
        recommendations = []
        
        if causal_context.get("outcome"):
            outcome = causal_context["outcome"]
            if outcome.get("success"):
                recommendations.append("✅ Similar pattern previously succeeded - consider similar approach")
            else:
                recommendations.append("⚠️ Similar pattern previously failed - consider alternative approach")
        
        if causal_context.get("action"):
            action = causal_context["action"]
            confidence = action.get("confidence_score", 0.5)
            if confidence > 0.8:
                recommendations.append(f"🎯 High confidence action available (confidence: {confidence:.2f})")
            elif confidence < 0.3:
                recommendations.append(f"⚠️ Low confidence pattern - proceed with caution (confidence: {confidence:.2f})")
        
        if len(causal_context.get("similar_signatures", [])) > 3:
            recommendations.append("📊 Strong pattern match - high confidence in recommendations")
        
        return recommendations
    
    def _generate_overall_recommendations(self, contextual_insights: List[Dict[str, Any]]) -> List[str]:
        """Generate overall recommendations from all insights."""
        if not contextual_insights:
            return ["🔍 No similar patterns found - proceed with standard protocols"]
        
        recommendations = []
        
        # Analyze success patterns
        successful_patterns = [
            insight for insight in contextual_insights
            if insight.get("causal_context", {}).get("outcome", {}).get("success")
        ]
        
        if successful_patterns:
            avg_confidence = sum(
                insight.get("consciousness_weighted_score", 0.0)
                for insight in successful_patterns
            ) / len(successful_patterns)
            
            recommendations.append(f"✅ {len(successful_patterns)} successful similar patterns found (avg confidence: {avg_confidence:.2f})")
        
        # Analyze failure patterns
        failed_patterns = [
            insight for insight in contextual_insights
            if insight.get("causal_context", {}).get("outcome", {}).get("success") is False
        ]
        
        if failed_patterns:
            recommendations.append(f"⚠️ {len(failed_patterns)} failed similar patterns - avoid similar approaches")
        
        return recommendations
    
    def _calculate_confidence_score(self, similar_signatures: List[Dict[str, Any]], 
                                  contextual_insights: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the search results."""
        if not similar_signatures:
            return 0.0
        
        # Base confidence from similarity scores
        similarity_scores = [s["similarity_score"] for s in similar_signatures]
        base_confidence = sum(similarity_scores) / len(similarity_scores)
        
        # Boost confidence if we have causal context
        context_boost = min(0.2, len(contextual_insights) * 0.05)
        
        # Boost confidence for consciousness-weighted scores
        consciousness_scores = [
            s.get("consciousness_weighted_score", s["similarity_score"])
            for s in similar_signatures
        ]
        consciousness_boost = (sum(consciousness_scores) / len(consciousness_scores) - base_confidence) * 0.3
        
        final_confidence = min(1.0, base_confidence + context_boost + consciousness_boost)
        return final_confidence
    
    async def _process_event(self, event: SystemEventAPI):
        """Background task to process system events."""
        # This would implement event processing logic
        self.logger.debug(f"📝 Processing event: {event.event_type}")
    
    async def _process_action(self, action: AgentActionAPI):
        """Background task to process agent actions."""
        self.logger.debug(f"🤖 Processing action: {action.action_type}")
    
    async def _process_outcome(self, outcome: OutcomeAPI):
        """Background task to process outcomes."""
        self.logger.debug(f"📊 Processing outcome for action: {outcome.action_id}")

    def _add_phase2c_routes(self, app: FastAPI):
        """Add Phase 2C Intelligence Flywheel routes to the FastAPI application."""
        try:
            # Import Phase 2C search router
            from aura_intelligence.enterprise.mem0_search import create_search_router

            # Create and mount the Phase 2C search router
            phase2c_router = create_search_router()
            app.include_router(
                phase2c_router,
                prefix="/api/v2c",
                tags=["Phase 2C Intelligence Flywheel"]
            )

            self.logger.info("🧠 Phase 2C Intelligence Flywheel routes added to Search API")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add Phase 2C routes: {e}")
            self.logger.info("🔄 Search API will continue without Phase 2C routes")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server."""
        self.logger.info(f"🚀 Starting Search API Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Factory function for easy initialization
def create_search_api_service(**kwargs) -> SearchAPIService:
    """Create and return a configured SearchAPIService instance."""
    return SearchAPIService(**kwargs)


# For running as a standalone service
if __name__ == "__main__":
    service = create_search_api_service()
    service.run()
