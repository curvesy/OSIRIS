#!/usr/bin/env python3
"""
🧠 Collective Memory Manager - LangMem Integration

Professional memory management for collective intelligence.
Handles context retrieval and continuous learning.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# LangMem integration
try:
    from langmem import Client as LangMemClient
except ImportError:
    # Fallback for development
    class LangMemClient:
        def __init__(self, *args, **kwargs): 
            self.connected = False
        async def search(self, *args, **kwargs): 
            return []
        async def add(self, *args, **kwargs): 
            pass

# Import schemas
schema_dir = Path(__file__).parent.parent / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

try:
    import enums
    import base
    from production_observer_agent import ProductionAgentState
except ImportError:
    # Fallback for testing
    class ProductionAgentState:
        def __init__(self): pass

logger = logging.getLogger(__name__)


class CollectiveMemoryManager:
    """
    Professional LangMem integration for collective intelligence.
    Manages context engineering and continuous learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.namespace = config.get("namespace", "aura_collective_intelligence")
        
        # Initialize LangMem client
        api_key = config.get("langmem_api_key")
        if api_key:
            self.client = LangMemClient(
                api_key=api_key,
                namespace=self.namespace
            )
            self.connected = True
            logger.info(f"✅ LangMem connected: {self.namespace}")
        else:
            self.client = LangMemClient()
            self.connected = False
            logger.warning("⚠️ LangMem API key not provided - using fallback mode")
    
    async def query_relevant_context(self, state: Any) -> Dict[str, Any]:
        """
        Query LangMem for relevant context to inform supervisor decisions.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict containing context insights and recommendations
        """
        
        if not self.connected:
            return self._fallback_context()
        
        try:
            # Create semantic signature for the current situation
            event_signature = self._create_event_signature(state)
            
            # Query similar past workflows
            memories = await self.client.search(
                query=f"Similar incidents to: {event_signature}",
                limit=5,
                filters={"workflow_type": "collective_intelligence"}
            )
            
            if not memories:
                return {
                    "insight": "No similar past incidents found",
                    "confidence": 0.0,
                    "source": "langmem_empty"
                }
            
            # Analyze patterns from past workflows
            insights = self._analyze_memory_patterns(memories)
            insights["source"] = "langmem"
            
            logger.info(f"🧠 Retrieved {len(memories)} relevant memories")
            return insights
            
        except Exception as e:
            logger.error(f"❌ LangMem query failed: {e}")
            return self._fallback_context()
    
    async def learn_from_workflow(self, final_state: Any) -> None:
        """
        Store completed workflow in collective memory for future learning.
        
        Args:
            final_state: Final state of completed workflow
        """
        
        if not self.connected:
            logger.info("📝 Would store workflow in LangMem (fallback mode)")
            return
        
        try:
            workflow_summary = self._create_workflow_summary(final_state)
            
            await self.client.add(
                content=json.dumps(workflow_summary),
                metadata={
                    "workflow_type": "collective_intelligence",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_version": "production_v1.0",
                    "namespace": self.namespace
                }
            )
            
            logger.info(f"✅ Stored workflow in collective memory")
            
        except Exception as e:
            logger.error(f"❌ Failed to store workflow: {e}")
    
    def _create_event_signature(self, state: Any) -> str:
        """Create semantic signature for event matching."""
        
        try:
            if hasattr(state, 'evidence_entries') and state.evidence_entries:
                latest_evidence = state.evidence_entries[-1]
                evidence_type = getattr(latest_evidence, 'evidence_type', 'unknown')
                content = getattr(latest_evidence, 'content', {})
                severity = content.get('severity', 'unknown')
                return f"{evidence_type}_{severity}"
            else:
                return "unknown_event"
        except Exception:
            return "unknown_event"
    
    def _analyze_memory_patterns(self, memories: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns from retrieved memories."""
        
        if not memories:
            return {"confidence": 0.0}
        
        # Analyze success patterns
        success_count = sum(1 for m in memories if m.get("success", False))
        total_count = len(memories)
        success_rate = success_count / total_count if total_count > 0 else 0.5
        
        # Extract common patterns
        patterns = []
        for memory in memories:
            if isinstance(memory, dict) and "patterns" in memory:
                patterns.extend(memory.get("patterns", []))
        
        # Generate recommendations
        if success_rate > 0.8:
            recommended_approach = "standard_analysis"
        elif success_rate > 0.5:
            recommended_approach = "careful_analysis"
        else:
            recommended_approach = "conservative_analysis"
        
        return {
            "similar_incidents_count": total_count,
            "success_rate": success_rate,
            "success_patterns": list(set(patterns)),
            "recommended_approach": recommended_approach,
            "confidence": min(0.9, success_rate + 0.1),
            "context_summary": f"Found {total_count} similar cases with {success_count} successes"
        }
    
    def _create_workflow_summary(self, final_state: Any) -> Dict[str, Any]:
        """Create workflow summary for storage."""
        
        try:
            workflow_id = getattr(final_state, 'workflow_id', 'unknown')
            status = getattr(final_state, 'status', 'unknown')
            evidence_count = len(getattr(final_state, 'evidence_entries', []))
            
            # Calculate processing time
            created_at = getattr(final_state, 'created_at', datetime.utcnow())
            updated_at = getattr(final_state, 'updated_at', datetime.utcnow())
            processing_time = (updated_at - created_at).total_seconds()
            
            return {
                "workflow_id": workflow_id,
                "status": str(status),
                "evidence_count": evidence_count,
                "processing_time_seconds": processing_time,
                "success": str(status) == "TaskStatus.COMPLETED",
                "timestamp": datetime.utcnow().isoformat(),
                "patterns": self._extract_patterns(final_state),
                "lessons_learned": ["workflow_completed_successfully"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create workflow summary: {e}")
            return {
                "workflow_id": "unknown",
                "status": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def _extract_patterns(self, state: Any) -> List[str]:
        """Extract patterns from workflow state."""
        
        patterns = []
        
        try:
            evidence_entries = getattr(state, 'evidence_entries', [])
            
            if len(evidence_entries) > 3:
                patterns.append("high_evidence_volume")
            
            # Check for error patterns
            error_count = 0
            for evidence in evidence_entries:
                content = getattr(evidence, 'content', {})
                if isinstance(content, dict):
                    message = content.get('message', '')
                    if 'error' in str(message).lower():
                        error_count += 1
            
            if error_count > 0:
                patterns.append("error_pattern_detected")
            
        except Exception:
            patterns.append("pattern_extraction_failed")
        
        return patterns
    
    def _fallback_context(self) -> Dict[str, Any]:
        """Fallback context when LangMem is unavailable."""
        
        return {
            "insight": "LangMem unavailable - using default context",
            "confidence": 0.5,
            "recommended_approach": "standard_analysis",
            "source": "fallback",
            "context_summary": "No historical context available"
        }
