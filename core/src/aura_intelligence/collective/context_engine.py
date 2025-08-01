#!/usr/bin/env python3
"""
🧠 Context Engine - Advanced Context Engineering

Professional context engineering for collective intelligence.
Enriches agent state with memory insights and contextual information.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Import schemas
schema_dir = Path(__file__).parent.parent / "agents" / "schemas"
sys.path.insert(0, str(schema_dir))

try:
    import enums
    import base
    from production_observer_agent import ProductionAgentState, ProductionEvidence, AgentConfig
except ImportError:
    # Fallback for testing
    class ProductionAgentState:
        def __init__(self): pass
    class ProductionEvidence:
        def __init__(self, **kwargs): pass
    class AgentConfig:
        def __init__(self): pass

logger = logging.getLogger(__name__)


class ContextEngine:
    """
    Professional context engineering for collective intelligence.
    
    The context engine:
    1. Takes raw agent state and memory insights
    2. Performs sophisticated context enrichment
    3. Creates contextual evidence entries
    4. Returns enriched state ready for supervisor routing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = "context_engine"
        
        # Context engineering parameters
        self.max_context_entries = config.get("max_context_entries", 5)
        self.context_confidence_threshold = config.get("context_confidence_threshold", 0.3)
        
        logger.info("🧠 Context Engine initialized")
    
    async def enrich_state(self, state: Any, memory_context: Dict[str, Any]) -> Any:
        """
        Main context enrichment function.
        
        Args:
            state: Current agent state
            memory_context: Context from collective memory
            
        Returns:
            Enriched state with contextual information
        """
        
        logger.info(f"🧠 Enriching state with context: confidence={memory_context.get('confidence', 0.0)}")
        
        try:
            # Step 1: Analyze current state
            state_analysis = self._analyze_current_state(state)
            
            # Step 2: Merge with memory context
            enriched_context = self._merge_contexts(state_analysis, memory_context)
            
            # Step 3: Create contextual evidence
            context_evidence = self._create_context_evidence(
                state, 
                enriched_context
            )
            
            # Step 4: Add evidence to state (if possible)
            enriched_state = self._add_context_to_state(
                state, 
                context_evidence, 
                enriched_context
            )
            
            logger.info("✅ State enrichment complete")
            return enriched_state
            
        except Exception as e:
            logger.error(f"❌ Context enrichment failed: {e}")
            return state
    
    def _analyze_current_state(self, state: Any) -> Dict[str, Any]:
        """Analyze current state to extract contextual information."""
        
        analysis = {
            "workflow_stage": "unknown",
            "evidence_count": 0,
            "complexity_score": 0.0,
            "urgency_indicators": [],
            "patterns_detected": []
        }
        
        try:
            # Analyze evidence entries
            if hasattr(state, 'evidence_entries'):
                evidence_entries = state.evidence_entries
                analysis["evidence_count"] = len(evidence_entries)
                
                # Determine workflow stage
                if evidence_entries:
                    latest_evidence = evidence_entries[-1]
                    evidence_type = getattr(latest_evidence, 'evidence_type', None)
                    
                    if evidence_type:
                        if str(evidence_type) == "EvidenceType.OBSERVATION":
                            analysis["workflow_stage"] = "observation"
                        elif str(evidence_type) == "EvidenceType.PATTERN":
                            analysis["workflow_stage"] = "analysis"
                        elif str(evidence_type) == "EvidenceType.EXECUTION":
                            analysis["workflow_stage"] = "execution"
                
                # Calculate complexity score
                analysis["complexity_score"] = min(1.0, len(evidence_entries) / 10.0)
                
                # Detect urgency indicators
                for evidence in evidence_entries:
                    content = getattr(evidence, 'content', {})
                    if isinstance(content, dict):
                        message = str(content.get('message', '')).lower()
                        
                        if any(word in message for word in ['critical', 'urgent', 'emergency']):
                            analysis["urgency_indicators"].append("high_priority_keywords")
                        
                        if 'error' in message:
                            analysis["urgency_indicators"].append("error_detected")
                        
                        if 'failure' in message:
                            analysis["urgency_indicators"].append("failure_detected")
                
                # Detect patterns
                if len(evidence_entries) > 3:
                    analysis["patterns_detected"].append("high_volume_evidence")
                
                if len(set(analysis["urgency_indicators"])) > 1:
                    analysis["patterns_detected"].append("multiple_urgency_signals")
            
            # Analyze timing
            if hasattr(state, 'created_at') and hasattr(state, 'updated_at'):
                processing_time = (state.updated_at - state.created_at).total_seconds()
                if processing_time > 300:  # 5 minutes
                    analysis["patterns_detected"].append("long_processing_time")
            
        except Exception as e:
            logger.error(f"State analysis failed: {e}")
        
        return analysis
    
    def _merge_contexts(self, state_analysis: Dict[str, Any], memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge current state analysis with memory context."""
        
        merged = {
            # Current state information
            "current_stage": state_analysis.get("workflow_stage", "unknown"),
            "evidence_count": state_analysis.get("evidence_count", 0),
            "complexity_score": state_analysis.get("complexity_score", 0.0),
            "urgency_indicators": state_analysis.get("urgency_indicators", []),
            "current_patterns": state_analysis.get("patterns_detected", []),
            
            # Memory context information
            "historical_confidence": memory_context.get("confidence", 0.0),
            "similar_incidents": memory_context.get("similar_incidents_count", 0),
            "success_rate": memory_context.get("success_rate", 0.5),
            "recommended_approach": memory_context.get("recommended_approach", "standard_analysis"),
            "historical_patterns": memory_context.get("success_patterns", []),
            "memory_source": memory_context.get("source", "unknown"),
            
            # Derived insights
            "context_quality": self._calculate_context_quality(state_analysis, memory_context),
            "risk_indicators": self._identify_risk_indicators(state_analysis, memory_context),
            "optimization_opportunities": self._identify_optimizations(state_analysis, memory_context)
        }
        
        return merged
    
    def _calculate_context_quality(self, state_analysis: Dict[str, Any], memory_context: Dict[str, Any]) -> float:
        """Calculate overall context quality score."""
        
        quality_factors = []
        
        # Evidence richness
        evidence_count = state_analysis.get("evidence_count", 0)
        evidence_quality = min(1.0, evidence_count / 5.0)
        quality_factors.append(evidence_quality)
        
        # Memory confidence
        memory_confidence = memory_context.get("confidence", 0.0)
        quality_factors.append(memory_confidence)
        
        # Pattern detection
        current_patterns = len(state_analysis.get("patterns_detected", []))
        historical_patterns = len(memory_context.get("success_patterns", []))
        pattern_quality = min(1.0, (current_patterns + historical_patterns) / 4.0)
        quality_factors.append(pattern_quality)
        
        # Calculate weighted average
        if quality_factors:
            return sum(quality_factors) / len(quality_factors)
        else:
            return 0.5
    
    def _identify_risk_indicators(self, state_analysis: Dict[str, Any], memory_context: Dict[str, Any]) -> List[str]:
        """Identify risk indicators from combined context."""
        
        risks = []
        
        # Current state risks
        urgency_indicators = state_analysis.get("urgency_indicators", [])
        if "error_detected" in urgency_indicators:
            risks.append("current_errors_present")
        
        if "failure_detected" in urgency_indicators:
            risks.append("current_failures_present")
        
        # Historical risks
        success_rate = memory_context.get("success_rate", 0.5)
        if success_rate < 0.3:
            risks.append("low_historical_success_rate")
        
        # Combined risks
        complexity_score = state_analysis.get("complexity_score", 0.0)
        memory_confidence = memory_context.get("confidence", 0.0)
        
        if complexity_score > 0.7 and memory_confidence < 0.3:
            risks.append("high_complexity_low_confidence")
        
        return risks
    
    def _identify_optimizations(self, state_analysis: Dict[str, Any], memory_context: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities."""
        
        optimizations = []
        
        # High confidence + simple case = fast track
        complexity_score = state_analysis.get("complexity_score", 0.0)
        memory_confidence = memory_context.get("confidence", 0.0)
        
        if complexity_score < 0.3 and memory_confidence > 0.8:
            optimizations.append("fast_track_eligible")
        
        # Repeated patterns = automation opportunity
        similar_incidents = memory_context.get("similar_incidents_count", 0)
        if similar_incidents > 10:
            optimizations.append("automation_candidate")
        
        # High success rate = standard processing
        success_rate = memory_context.get("success_rate", 0.5)
        if success_rate > 0.9:
            optimizations.append("standard_processing_recommended")
        
        return optimizations
    
    def _create_context_evidence(self, state: Any, enriched_context: Dict[str, Any]) -> Any:
        """Create contextual evidence entry."""
        
        try:
            # Create context evidence using your proven schemas
            context_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.CONTEXT,
                content={
                    "context_type": "collective_intelligence_context",
                    "context_quality": enriched_context.get("context_quality", 0.5),
                    "historical_confidence": enriched_context.get("historical_confidence", 0.0),
                    "similar_incidents": enriched_context.get("similar_incidents", 0),
                    "recommended_approach": enriched_context.get("recommended_approach", "standard_analysis"),
                    "risk_indicators": enriched_context.get("risk_indicators", []),
                    "optimization_opportunities": enriched_context.get("optimization_opportunities", []),
                    "context_timestamp": base.utc_now().isoformat(),
                    "context_engine_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=AgentConfig()  # Use default config
            )
            
            return context_evidence
            
        except Exception as e:
            logger.error(f"Failed to create context evidence: {e}")
            return None
    
    def _add_context_to_state(self, state: Any, context_evidence: Any, enriched_context: Dict[str, Any]) -> Any:
        """Add contextual information to state."""
        
        try:
            # Add context evidence if state supports it
            if hasattr(state, 'add_evidence') and context_evidence:
                enriched_state = state.add_evidence(context_evidence, AgentConfig())
            else:
                enriched_state = state
            
            # Add context metadata
            if hasattr(enriched_state, '__dict__'):
                enriched_state.context_enriched = True
                enriched_state.context_quality = enriched_context.get("context_quality", 0.5)
                enriched_state.context_timestamp = datetime.utcnow().isoformat()
            
            return enriched_state
            
        except Exception as e:
            logger.error(f"Failed to add context to state: {e}")
            return state
