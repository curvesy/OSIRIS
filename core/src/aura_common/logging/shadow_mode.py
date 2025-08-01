"""
🌙 Shadow Mode Logging Integration
Wrapper for the existing shadow mode logger with enhanced features.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
from pathlib import Path

# Import the existing shadow mode logger
from aura_intelligence.observability.shadow_mode_logger import (
    ShadowModeLogger as _OriginalShadowModeLogger,
    ShadowModeEntry,
)


class ShadowModeLogger(_OriginalShadowModeLogger):
    """
    Enhanced shadow mode logger that preserves existing functionality
    while adding modern patterns.
    """
    
    def __init__(
        self,
        log_dir: Path = Path("shadow_logs"),
        db_path: Optional[Path] = None,
        buffer_size: int = 100,
        flush_interval: float = 60.0
    ):
        """Initialize with same interface as original."""
        super().__init__(log_dir, db_path, buffer_size, flush_interval)
        self._correlation_ids: Dict[str, str] = {}
    
    async def log_prediction_with_context(
        self,
        entry: ShadowModeEntry,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> None:
        """
        Log prediction with additional context.
        
        Args:
            entry: Shadow mode entry
            correlation_id: Request correlation ID
            trace_id: OpenTelemetry trace ID
        """
        # Store correlation for later matching
        if correlation_id:
            self._correlation_ids[entry.workflow_id] = correlation_id
        
        # Add context to entry metadata
        if hasattr(entry, 'metadata'):
            entry.metadata = entry.metadata or {}
            if correlation_id:
                entry.metadata['correlation_id'] = correlation_id
            if trace_id:
                entry.metadata['trace_id'] = trace_id
        
        # Use original logging
        await self.log_prediction(entry)
    
    async def get_analytics_by_correlation(
        self,
        correlation_id: str,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get analytics for all predictions with a correlation ID.
        
        Args:
            correlation_id: Correlation ID to filter by
            time_window: Hours to look back
            
        Returns:
            Analytics for the correlation group
        """
        # Find all workflow IDs with this correlation
        workflow_ids = [
            wid for wid, cid in self._correlation_ids.items()
            if cid == correlation_id
        ]
        
        # Get combined analytics
        all_entries = []
        for workflow_id in workflow_ids:
            entries = await self._query_entries(
                workflow_id=workflow_id,
                time_window=time_window
            )
            all_entries.extend(entries)
        
        # Calculate analytics on combined data
        return self._calculate_analytics(all_entries)


# Global instance for convenience
_shadow_logger: Optional[ShadowModeLogger] = None


async def get_shadow_logger() -> ShadowModeLogger:
    """Get or create global shadow logger instance."""
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowModeLogger()
        await _shadow_logger.initialize()
    return _shadow_logger


async def shadow_log(
    workflow_id: str,
    action: str,
    prediction: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function for shadow logging.
    
    Args:
        workflow_id: Unique workflow identifier
        action: Action being predicted
        prediction: Prediction details
        context: Optional context data
    """
    from .correlation import get_correlation_id
    
    logger = await get_shadow_logger()
    
    entry = ShadowModeEntry(
        workflow_id=workflow_id,
        thread_id=context.get('thread_id', 'default') if context else 'default',
        timestamp=datetime.utcnow(),
        evidence_log=context.get('evidence', []) if context else [],
        memory_context=context.get('memory', {}) if context else {},
        supervisor_decision={'action': action},
        predicted_success_probability=prediction.get('success_probability', 0.5),
        prediction_confidence_score=prediction.get('confidence', 0.5),
        risk_score=prediction.get('risk_score', 0.5),
        predicted_risks=prediction.get('risks', []),
        reasoning_trace=prediction.get('reasoning', []),
        requires_human_approval=prediction.get('requires_approval', False),
        routing_decision=prediction.get('routing', 'unknown'),
        decision_score=prediction.get('decision_score', 0.5)
    )
    
    await logger.log_prediction_with_context(
        entry,
        correlation_id=get_correlation_id()
    )