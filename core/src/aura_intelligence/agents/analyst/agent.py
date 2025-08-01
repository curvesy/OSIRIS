#!/usr/bin/env python3
"""
🔍 Analyst Agent - Professional Pattern Analysis

Advanced analyst agent using your proven patterns.
Specialized for deep analysis with TDA integration potential.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Import schemas
schema_dir = Path(__file__).parent.parent / "schemas"
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


class AnalystAgent:
    """
    Professional analyst agent using your proven patterns.
    
    The analyst specializes in:
    1. Deep pattern analysis of evidence
    2. Risk assessment and scoring
    3. Trend detection across workflows
    4. Contextual recommendation generation
    5. Future: TDA integration for topological analysis
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = f"analyst_{config.agent_id}"
        
        # Analysis configuration
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }
        
        self.pattern_weights = {
            "error_patterns": 0.4,
            "volume_patterns": 0.2,
            "timing_patterns": 0.2,
            "context_patterns": 0.2
        }
        
        logger.info(f"🔍 AnalystAgent initialized: {self.agent_id}")
    
    async def analyze_state(self, state: ProductionAgentState) -> ProductionAgentState:
        """
        Main analysis function - the analyst's core capability.
        
        Args:
            state: Current workflow state with evidence
            
        Returns:
            State enriched with analysis evidence
        """
        
        logger.info(f"🔍 AnalystAgent analyzing: {state.workflow_id}")
        
        try:
            # Step 1: Extract and validate evidence
            evidence_entries = getattr(state, 'evidence_entries', [])
            
            if not evidence_entries:
                logger.warning("No evidence to analyze")
                return self._create_no_evidence_analysis(state)
            
            # Step 2: Perform multi-dimensional analysis
            analysis_results = await self._perform_comprehensive_analysis(evidence_entries)
            
            # Step 3: Generate contextual insights
            contextual_insights = self._generate_contextual_insights(state, analysis_results)
            
            # Step 4: Create analysis evidence
            analysis_evidence = self._create_analysis_evidence(
                state, 
                analysis_results, 
                contextual_insights
            )
            
            # Step 5: Update state immutably
            new_state = state.add_evidence(analysis_evidence, self.config)
            
            logger.info(f"✅ Analysis complete: risk_score={analysis_results['risk_score']:.3f}")
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_error_analysis(state, str(e))
    
    async def _perform_comprehensive_analysis(self, evidence_entries: List[Any]) -> Dict[str, Any]:
        """Perform comprehensive multi-dimensional analysis."""
        
        # Initialize analysis dimensions
        error_analysis = self._analyze_error_patterns(evidence_entries)
        volume_analysis = self._analyze_volume_patterns(evidence_entries)
        timing_analysis = self._analyze_timing_patterns(evidence_entries)
        context_analysis = self._analyze_context_patterns(evidence_entries)
        
        # Calculate weighted risk score
        risk_score = (
            error_analysis["risk_contribution"] * self.pattern_weights["error_patterns"] +
            volume_analysis["risk_contribution"] * self.pattern_weights["volume_patterns"] +
            timing_analysis["risk_contribution"] * self.pattern_weights["timing_patterns"] +
            context_analysis["risk_contribution"] * self.pattern_weights["context_patterns"]
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate patterns summary
        patterns_detected = []
        patterns_detected.extend(error_analysis["patterns"])
        patterns_detected.extend(volume_analysis["patterns"])
        patterns_detected.extend(timing_analysis["patterns"])
        patterns_detected.extend(context_analysis["patterns"])
        
        # Calculate confidence
        confidence = self._calculate_analysis_confidence(evidence_entries, patterns_detected)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_score, risk_level, patterns_detected)
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "patterns_detected": patterns_detected,
            "recommendations": recommendations,
            "analysis_dimensions": {
                "error_analysis": error_analysis,
                "volume_analysis": volume_analysis,
                "timing_analysis": timing_analysis,
                "context_analysis": context_analysis
            },
            "evidence_count": len(evidence_entries),
            "analysis_timestamp": base.utc_now().isoformat()
        }
    
    def _analyze_error_patterns(self, evidence_entries: List[Any]) -> Dict[str, Any]:
        """Analyze error-related patterns in evidence."""
        
        error_indicators = ["error", "failure", "exception", "critical", "fatal"]
        warning_indicators = ["warning", "alert", "issue", "problem"]
        
        error_count = 0
        warning_count = 0
        critical_count = 0
        
        for evidence in evidence_entries:
            content = getattr(evidence, 'content', {})
            if isinstance(content, dict):
                message = str(content.get('message', '')).lower()
                
                # Count error types
                if any(indicator in message for indicator in error_indicators):
                    error_count += 1
                    if any(critical in message for critical in ["critical", "fatal"]):
                        critical_count += 1
                
                if any(indicator in message for indicator in warning_indicators):
                    warning_count += 1
        
        total_entries = len(evidence_entries)
        error_rate = error_count / total_entries if total_entries > 0 else 0
        warning_rate = warning_count / total_entries if total_entries > 0 else 0
        
        # Determine patterns
        patterns = []
        if error_rate > 0.5:
            patterns.append("high_error_rate")
        if critical_count > 0:
            patterns.append("critical_errors_present")
        if warning_rate > 0.3:
            patterns.append("high_warning_rate")
        
        # Calculate risk contribution
        risk_contribution = min(1.0, error_rate * 0.7 + (critical_count / total_entries) * 0.3)
        
        return {
            "error_count": error_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "error_rate": error_rate,
            "warning_rate": warning_rate,
            "patterns": patterns,
            "risk_contribution": risk_contribution
        }
    
    def _analyze_volume_patterns(self, evidence_entries: List[Any]) -> Dict[str, Any]:
        """Analyze volume-related patterns."""
        
        evidence_count = len(evidence_entries)
        
        # Volume thresholds
        patterns = []
        if evidence_count > 10:
            patterns.append("high_volume_evidence")
        elif evidence_count > 5:
            patterns.append("moderate_volume_evidence")
        else:
            patterns.append("low_volume_evidence")
        
        # Calculate risk contribution based on volume
        # High volume can indicate either thoroughness or chaos
        if evidence_count > 15:
            risk_contribution = 0.8  # Very high volume = potential chaos
        elif evidence_count > 10:
            risk_contribution = 0.4  # High volume = moderate risk
        elif evidence_count < 2:
            risk_contribution = 0.6  # Too little evidence = uncertainty risk
        else:
            risk_contribution = 0.2  # Normal volume = low risk
        
        return {
            "evidence_count": evidence_count,
            "volume_category": patterns[0] if patterns else "unknown",
            "patterns": patterns,
            "risk_contribution": risk_contribution
        }
    
    def _analyze_timing_patterns(self, evidence_entries: List[Any]) -> Dict[str, Any]:
        """Analyze timing-related patterns."""
        
        if len(evidence_entries) < 2:
            return {
                "patterns": ["insufficient_timing_data"],
                "risk_contribution": 0.3
            }
        
        # Calculate time intervals between evidence
        intervals = []
        for i in range(1, len(evidence_entries)):
            try:
                prev_time = getattr(evidence_entries[i-1], 'timestamp', None)
                curr_time = getattr(evidence_entries[i], 'timestamp', None)
                
                if prev_time and curr_time:
                    interval = (curr_time - prev_time).total_seconds()
                    intervals.append(interval)
            except Exception:
                continue
        
        if not intervals:
            return {
                "patterns": ["no_timing_data"],
                "risk_contribution": 0.3
            }
        
        # Analyze interval patterns
        avg_interval = sum(intervals) / len(intervals)
        max_interval = max(intervals)
        min_interval = min(intervals)
        
        patterns = []
        risk_contribution = 0.2  # Default low risk
        
        if max_interval > 300:  # 5 minutes
            patterns.append("long_processing_gaps")
            risk_contribution = 0.5
        
        if min_interval < 1:  # Less than 1 second
            patterns.append("rapid_fire_evidence")
            risk_contribution = 0.4
        
        if avg_interval > 60:  # Average > 1 minute
            patterns.append("slow_processing")
            risk_contribution = 0.3
        
        return {
            "avg_interval_seconds": avg_interval,
            "max_interval_seconds": max_interval,
            "min_interval_seconds": min_interval,
            "patterns": patterns,
            "risk_contribution": risk_contribution
        }
    
    def _analyze_context_patterns(self, evidence_entries: List[Any]) -> Dict[str, Any]:
        """Analyze contextual patterns in evidence."""
        
        # Analyze evidence types
        evidence_types = {}
        for evidence in evidence_entries:
            evidence_type = getattr(evidence, 'evidence_type', 'unknown')
            evidence_types[str(evidence_type)] = evidence_types.get(str(evidence_type), 0) + 1
        
        # Analyze content diversity
        unique_sources = set()
        for evidence in evidence_entries:
            content = getattr(evidence, 'content', {})
            if isinstance(content, dict):
                source = content.get('source', 'unknown')
                unique_sources.add(source)
        
        patterns = []
        risk_contribution = 0.2
        
        # Pattern detection
        if len(evidence_types) == 1:
            patterns.append("single_evidence_type")
            risk_contribution = 0.4  # Lack of diversity
        
        if len(unique_sources) > len(evidence_entries) * 0.8:
            patterns.append("diverse_sources")
            risk_contribution = 0.1  # Good diversity
        
        return {
            "evidence_type_distribution": evidence_types,
            "unique_sources_count": len(unique_sources),
            "patterns": patterns,
            "risk_contribution": risk_contribution
        }
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        
        if risk_score >= self.risk_thresholds["critical"]:
            return "critical"
        elif risk_score >= self.risk_thresholds["high"]:
            return "high"
        elif risk_score >= self.risk_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _calculate_analysis_confidence(self, evidence_entries: List[Any], patterns: List[str]) -> float:
        """Calculate confidence in the analysis."""
        
        confidence_factors = []
        
        # Evidence quantity factor
        evidence_count = len(evidence_entries)
        quantity_confidence = min(1.0, evidence_count / 5.0)
        confidence_factors.append(quantity_confidence)
        
        # Pattern detection factor
        pattern_confidence = min(1.0, len(patterns) / 3.0)
        confidence_factors.append(pattern_confidence)
        
        # Evidence quality factor (simplified)
        quality_confidence = 0.8  # Assume good quality for now
        confidence_factors.append(quality_confidence)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_recommendations(self, risk_score: float, risk_level: str, patterns: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == "critical":
            recommendations.append("immediate_escalation_required")
            recommendations.append("stop_current_operations")
        elif risk_level == "high":
            recommendations.append("urgent_attention_required")
            recommendations.append("increase_monitoring")
        elif risk_level == "medium":
            recommendations.append("schedule_investigation")
            recommendations.append("continue_with_caution")
        else:
            recommendations.append("continue_normal_operations")
            recommendations.append("maintain_standard_monitoring")
        
        # Pattern-based recommendations
        if "high_error_rate" in patterns:
            recommendations.append("investigate_error_sources")
        
        if "high_volume_evidence" in patterns:
            recommendations.append("review_evidence_collection_efficiency")
        
        if "long_processing_gaps" in patterns:
            recommendations.append("optimize_processing_pipeline")
        
        return recommendations
    
    def _generate_contextual_insights(self, state: Any, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual insights from analysis."""
        
        insights = {
            "workflow_health": "unknown",
            "processing_efficiency": "unknown",
            "attention_priority": "medium"
        }
        
        risk_score = analysis_results.get("risk_score", 0.5)
        patterns = analysis_results.get("patterns_detected", [])
        
        # Determine workflow health
        if risk_score < 0.3:
            insights["workflow_health"] = "healthy"
        elif risk_score < 0.7:
            insights["workflow_health"] = "concerning"
        else:
            insights["workflow_health"] = "critical"
        
        # Determine processing efficiency
        if "long_processing_gaps" in patterns:
            insights["processing_efficiency"] = "poor"
        elif "rapid_fire_evidence" in patterns:
            insights["processing_efficiency"] = "chaotic"
        else:
            insights["processing_efficiency"] = "normal"
        
        # Determine attention priority
        if risk_score > 0.8:
            insights["attention_priority"] = "immediate"
        elif risk_score > 0.5:
            insights["attention_priority"] = "high"
        else:
            insights["attention_priority"] = "normal"
        
        return insights
    
    def _create_analysis_evidence(self, state: Any, analysis_results: Dict[str, Any], insights: Dict[str, Any]) -> Any:
        """Create comprehensive analysis evidence."""
        
        try:
            analysis_evidence = ProductionEvidence(
                evidence_type=enums.EvidenceType.PATTERN,
                content={
                    "analysis_type": "collective_intelligence_analysis",
                    "risk_score": analysis_results["risk_score"],
                    "risk_level": analysis_results["risk_level"],
                    "confidence": analysis_results["confidence"],
                    "patterns_detected": analysis_results["patterns_detected"],
                    "recommendations": analysis_results["recommendations"],
                    "contextual_insights": insights,
                    "analysis_dimensions": analysis_results["analysis_dimensions"],
                    "evidence_analyzed_count": analysis_results["evidence_count"],
                    "analysis_timestamp": analysis_results["analysis_timestamp"],
                    "analyst_id": self.agent_id,
                    "analysis_version": "v1.0"
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return analysis_evidence
            
        except Exception as e:
            logger.error(f"Failed to create analysis evidence: {e}")
            return None
    
    def _create_no_evidence_analysis(self, state: Any) -> Any:
        """Create analysis for state with no evidence."""
        
        try:
            no_evidence_analysis = ProductionEvidence(
                evidence_type=enums.EvidenceType.PATTERN,
                content={
                    "analysis_type": "no_evidence_analysis",
                    "risk_score": 0.5,
                    "risk_level": "medium",
                    "confidence": 0.1,
                    "patterns_detected": ["no_evidence_available"],
                    "recommendations": ["collect_evidence_before_proceeding"],
                    "analysis_timestamp": base.utc_now().isoformat(),
                    "analyst_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return state.add_evidence(no_evidence_analysis, self.config)
            
        except Exception as e:
            logger.error(f"Failed to create no-evidence analysis: {e}")
            return state
    
    def _create_error_analysis(self, state: Any, error_message: str) -> Any:
        """Create analysis evidence for analysis errors."""
        
        try:
            error_analysis = ProductionEvidence(
                evidence_type=enums.EvidenceType.PATTERN,
                content={
                    "analysis_type": "error_analysis",
                    "risk_score": 0.8,
                    "risk_level": "high",
                    "confidence": 0.9,
                    "patterns_detected": ["analysis_failure"],
                    "recommendations": ["retry_analysis", "investigate_analysis_error"],
                    "error_message": error_message,
                    "analysis_timestamp": base.utc_now().isoformat(),
                    "analyst_id": self.agent_id
                },
                workflow_id=getattr(state, 'workflow_id', 'unknown'),
                task_id=getattr(state, 'task_id', 'unknown'),
                config=self.config
            )
            
            return state.add_evidence(error_analysis, self.config)
            
        except Exception as e:
            logger.error(f"Failed to create error analysis: {e}")
            return state
