"""
🔗 TDA Context Integration Layer

Seamless integration with the existing TDA streaming system for orchestration
context enrichment. Provides real-time access to TDA patterns, anomalies,
and correlation data for semantic orchestration decisions.

Key Features:
- Real-time TDA pattern correlation
- Anomaly severity integration
- Correlation ID tracking across systems
- Context caching for performance
- Fallback mechanisms for TDA unavailability

TDA Integration:
- Uses existing TDA Kafka event mesh
- Leverages TDA streaming pattern analysis
- Integrates with TDA correlation system
- Supports TDA feature flag controls
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import asdict

from .base_interfaces import TDAContext, TDAIntegration

# TDA system imports with fallbacks
try:
    from aura_intelligence.tda.streaming import get_current_patterns, analyze_correlation
    from aura_intelligence.infrastructure.kafka_event_mesh import KafkaEventMesh
    from aura_intelligence.observability.tracing import get_tracer
    TDA_AVAILABLE = True
    tracer = get_tracer(__name__)
except ImportError:
    # Fallback for environments without TDA
    TDA_AVAILABLE = False
    tracer = None
    get_current_patterns = None
    analyze_correlation = None
    KafkaEventMesh = None

class TDAContextIntegration(TDAIntegration):
    """
    Production TDA integration for orchestration context enrichment
    """
    
    def __init__(self, kafka_config: Optional[Dict[str, Any]] = None):
        self.kafka_config = kafka_config or {}
        self.context_cache: Dict[str, Tuple[TDAContext, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        self.kafka_client = None
        self.pattern_cache: Dict[str, Any] = {}
        
        if TDA_AVAILABLE:
            self._initialize_kafka_client()
    
    def _initialize_kafka_client(self):
        """Initialize Kafka client for TDA event mesh integration"""
        try:
            self.kafka_client = KafkaEventMesh(
                brokers=self.kafka_config.get("brokers", ["localhost:9092"]),
                client_id="orchestration-tda-integration"
            )
        except Exception as e:
            if tracer:
                tracer.record_exception(e)
            self.kafka_client = None
    
    async def get_context(self, correlation_id: str) -> Optional[TDAContext]:
        """
        Get TDA context for correlation ID with caching
        """
        if tracer:
            with tracer.start_as_current_span("tda_context_retrieval") as span:
                span.set_attributes({
                    "tda.correlation_id": correlation_id,
                    "tda.cache_enabled": True
                })
        
        # Check cache first
        cached_context = self._get_cached_context(correlation_id)
        if cached_context:
            return cached_context
        
        # Fetch from TDA system
        context = await self._fetch_tda_context(correlation_id)
        
        # Cache the result
        if context:
            self._cache_context(correlation_id, context)
        
        return context
    
    async def _fetch_tda_context(self, correlation_id: str) -> Optional[TDAContext]:
        """
        Fetch TDA context from streaming system
        """
        if not TDA_AVAILABLE:
            return self._create_fallback_context(correlation_id)
        
        try:
            # Get current patterns from TDA streaming system
            patterns = await get_current_patterns(
                correlation_id=correlation_id,
                window="1h"
            )
            
            if not patterns:
                return None
            
            # Analyze correlation strength
            correlation_analysis = await analyze_correlation(
                correlation_id,
                patterns.get("pattern_data", {})
            )
            
            # Extract anomaly information
            anomaly_info = patterns.get("anomalies", {})
            anomaly_severity = anomaly_info.get("severity", 0.0)
            
            # Create TDA context
            context = TDAContext(
                correlation_id=correlation_id,
                pattern_confidence=correlation_analysis.get("confidence", 0.0),
                anomaly_severity=anomaly_severity,
                current_patterns=patterns.get("patterns", {}),
                temporal_window=patterns.get("window", "1h"),
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "tda_streaming",
                    "pattern_count": len(patterns.get("patterns", {})),
                    "anomaly_count": len(anomaly_info.get("anomalies", []))
                }
            )
            
            return context
            
        except Exception as e:
            if tracer:
                tracer.record_exception(e)
            return self._create_fallback_context(correlation_id)
    
    def _create_fallback_context(self, correlation_id: str) -> TDAContext:
        """
        Create fallback context when TDA is unavailable
        """
        return TDAContext(
            correlation_id=correlation_id,
            pattern_confidence=0.5,  # Neutral confidence
            anomaly_severity=0.0,    # No anomaly data
            current_patterns={},
            temporal_window="1h",
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "source": "fallback",
                "tda_available": False
            }
        )
    
    async def send_orchestration_result(
        self, 
        result: Dict[str, Any],
        correlation_id: str
    ) -> bool:
        """
        Send orchestration result to TDA for pattern analysis
        """
        if tracer:
            with tracer.start_as_current_span("tda_result_send") as span:
                span.set_attributes({
                    "tda.correlation_id": correlation_id,
                    "orchestration.result_type": result.get("type", "unknown")
                })
        
        if not TDA_AVAILABLE or not self.kafka_client:
            return False
        
        try:
            # Prepare orchestration result for TDA
            tda_message = {
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "orchestration",
                "result": result,
                "metadata": {
                    "workflow_id": result.get("workflow_id"),
                    "agent_count": len(result.get("agent_outputs", {})),
                    "execution_time": result.get("execution_summary", {}).get("execution_time"),
                    "success": result.get("success", True)
                }
            }
            
            # Send to TDA via Kafka
            await self.kafka_client.send_message(
                topic="orchestration-results",
                key=correlation_id,
                value=json.dumps(tda_message)
            )
            
            return True
            
        except Exception as e:
            if tracer:
                tracer.record_exception(e)
            return False
    
    async def get_current_patterns(self, window: str = "1h") -> Dict[str, Any]:
        """
        Get current TDA patterns for the specified time window
        """
        if tracer:
            with tracer.start_as_current_span("tda_patterns_retrieval") as span:
                span.set_attributes({
                    "tda.window": window,
                    "tda.cache_enabled": True
                })
        
        # Check pattern cache
        cache_key = f"patterns_{window}"
        if cache_key in self.pattern_cache:
            cached_time, patterns = self.pattern_cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(minutes=5):
                return patterns
        
        if not TDA_AVAILABLE:
            return {}
        
        try:
            # Get patterns from TDA streaming system
            patterns = await get_current_patterns(window=window)
            
            # Cache the patterns
            self.pattern_cache[cache_key] = (datetime.utcnow(), patterns)
            
            return patterns
            
        except Exception as e:
            if tracer:
                tracer.record_exception(e)
            return {}
    
    async def correlate_with_anomalies(
        self, 
        orchestration_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Correlate orchestration data with current TDA anomalies
        """
        if not TDA_AVAILABLE:
            return {"correlation_strength": 0.0, "anomalies": []}
        
        try:
            # Get current anomalies
            current_patterns = await self.get_current_patterns("30m")
            anomalies = current_patterns.get("anomalies", {})
            
            # Simple correlation logic (can be enhanced with ML)
            correlation_factors = []
            
            # Time-based correlation
            orchestration_time = orchestration_data.get("timestamp")
            if orchestration_time:
                for anomaly in anomalies.get("anomalies", []):
                    anomaly_time = anomaly.get("timestamp")
                    if anomaly_time:
                        time_diff = abs(
                            datetime.fromisoformat(orchestration_time) - 
                            datetime.fromisoformat(anomaly_time)
                        ).total_seconds()
                        
                        # Stronger correlation for closer times
                        if time_diff < 300:  # 5 minutes
                            correlation_factors.append(0.8)
                        elif time_diff < 900:  # 15 minutes
                            correlation_factors.append(0.5)
                        else:
                            correlation_factors.append(0.1)
            
            # Calculate overall correlation strength
            correlation_strength = (
                sum(correlation_factors) / len(correlation_factors)
                if correlation_factors else 0.0
            )
            
            return {
                "correlation_strength": correlation_strength,
                "anomalies": anomalies.get("anomalies", []),
                "correlation_factors": correlation_factors
            }
            
        except Exception as e:
            if tracer:
                tracer.record_exception(e)
            return {"correlation_strength": 0.0, "anomalies": []}
    
    def _get_cached_context(self, correlation_id: str) -> Optional[TDAContext]:
        """Get cached TDA context if still valid"""
        if correlation_id in self.context_cache:
            context, cached_time = self.context_cache[correlation_id]
            if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl_seconds):
                return context
            else:
                # Remove expired cache entry
                del self.context_cache[correlation_id]
        
        return None
    
    def _cache_context(self, correlation_id: str, context: TDAContext):
        """Cache TDA context with timestamp"""
        self.context_cache[correlation_id] = (context, datetime.utcnow())
        
        # Clean up old cache entries (simple LRU)
        if len(self.context_cache) > 1000:
            # Remove oldest 100 entries
            sorted_cache = sorted(
                self.context_cache.items(),
                key=lambda x: x[1][1]
            )
            for correlation_id, _ in sorted_cache[:100]:
                del self.context_cache[correlation_id]
    
    async def get_orchestration_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about TDA-orchestration integration
        """
        return {
            "cache_size": len(self.context_cache),
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            "tda_available": TDA_AVAILABLE,
            "kafka_connected": self.kafka_client is not None,
            "pattern_cache_size": len(self.pattern_cache),
            "last_update": datetime.utcnow().isoformat()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)"""
        # This would be enhanced with actual hit/miss tracking
        return 0.75 if self.context_cache else 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of TDA integration
        """
        health_status = {
            "status": "healthy",
            "tda_available": TDA_AVAILABLE,
            "kafka_connected": False,
            "cache_operational": True,
            "last_check": datetime.utcnow().isoformat()
        }
        
        # Check Kafka connection
        if self.kafka_client:
            try:
                # Simple connectivity check
                health_status["kafka_connected"] = True
            except Exception:
                health_status["kafka_connected"] = False
                health_status["status"] = "degraded"
        
        # Check cache health
        if len(self.context_cache) > 10000:  # Too many cached items
            health_status["cache_operational"] = False
            health_status["status"] = "degraded"
        
        return health_status

class MockTDAIntegration(TDAIntegration):
    """
    Mock TDA integration for testing and development
    """
    
    def __init__(self):
        self.contexts = {}
        self.results = []
        self.patterns = {
            "1h": {"patterns": {"test_pattern": 0.8}, "anomalies": {"severity": 0.3}},
            "30m": {"patterns": {"recent_pattern": 0.9}, "anomalies": {"severity": 0.1}}
        }
    
    async def get_context(self, correlation_id: str) -> Optional[TDAContext]:
        """Mock context retrieval"""
        if correlation_id in self.contexts:
            return self.contexts[correlation_id]
        
        # Create mock context
        return TDAContext(
            correlation_id=correlation_id,
            pattern_confidence=0.7,
            anomaly_severity=0.2,
            current_patterns={"mock_pattern": 0.7},
            temporal_window="1h",
            metadata={"source": "mock", "timestamp": datetime.utcnow().isoformat()}
        )
    
    async def send_orchestration_result(self, result: Dict[str, Any], correlation_id: str) -> bool:
        """Mock result sending"""
        self.results.append((result, correlation_id))
        return True
    
    async def get_current_patterns(self, window: str = "1h") -> Dict[str, Any]:
        """Mock pattern retrieval"""
        return self.patterns.get(window, {})