"""
Shadow Deployment Wrapper
========================

Enables dual-write and comparison between old and new systems
for zero-risk validation in production.
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..tda.models import TDAResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing old vs new system."""
    query_id: str
    old_results: List[str]
    new_results: List[str]
    score_diff: float
    latency_old_ms: float
    latency_new_ms: float
    mismatch: bool


class ShadowMemoryWrapper:
    """
    Wraps old and new memory systems for shadow deployment.
    
    Features:
    - Dual writes to both systems
    - Reads from old system (configurable)
    - Logs comparison results
    - Feature flag support
    """
    
    def __init__(
        self,
        old_system,
        new_system,
        feature_flags: Dict[str, Any],
        comparison_logger=None
    ):
        self.old_system = old_system
        self.new_system = new_system
        self.feature_flags = feature_flags
        self.comparison_logger = comparison_logger or self._default_logger
        
        # Metrics
        self.comparison_count = 0
        self.mismatch_count = 0
    
    def _default_logger(self, result: ComparisonResult):
        """Default comparison logger."""
        logger.info(f"Shadow comparison: {json.dumps({
            'query_id': result.query_id,
            'mismatch': result.mismatch,
            'score_diff': result.score_diff,
            'latency_diff_ms': result.latency_new_ms - result.latency_old_ms
        })}")
    
    def _should_use_new_system(self) -> bool:
        """Check if we should serve from new system."""
        return self.feature_flags.get("SHAPE_MEMORY_V2_SERVE", False)
    
    def _should_shadow_write(self) -> bool:
        """Check if we should write to new system."""
        return self.feature_flags.get("SHAPE_MEMORY_V2_SHADOW", True)
    
    async def store(
        self,
        content: Dict[str, Any],
        tda_result: TDAResult,
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store in both systems (if shadow enabled).
        
        Returns ID from primary system.
        """
        # Always write to old system
        old_id = await self._async_wrapper(
            self.old_system.store,
            content, tda_result, context_type, metadata
        )
        
        # Shadow write to new system
        if self._should_shadow_write():
            try:
                new_id = await self._async_wrapper(
                    self.new_system.store,
                    content, tda_result, context_type, metadata
                )
                logger.debug(f"Shadow stored: old={old_id}, new={new_id}")
            except Exception as e:
                logger.error(f"Shadow store failed: {e}")
                # Don't fail the main operation
        
        return old_id
    
    async def retrieve(
        self,
        query_tda: TDAResult,
        k: int = 10,
        context_filter: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve from primary system and optionally compare.
        """
        query_id = f"q_{int(time.time() * 1000)}"
        
        # Get results from old system
        start_old = time.perf_counter()
        old_results = await self._async_wrapper(
            self.old_system.retrieve,
            query_tda, k, context_filter
        )
        latency_old = (time.perf_counter() - start_old) * 1000
        
        # Shadow query to new system
        if self._should_shadow_write():
            try:
                start_new = time.perf_counter()
                new_results = await self._async_wrapper(
                    self.new_system.retrieve,
                    query_tda, k, context_filter
                )
                latency_new = (time.perf_counter() - start_new) * 1000
                
                # Compare results
                comparison = self._compare_results(
                    query_id, old_results, new_results,
                    latency_old, latency_new
                )
                
                self.comparison_logger(comparison)
                
                # Update metrics
                self.comparison_count += 1
                if comparison.mismatch:
                    self.mismatch_count += 1
                    
            except Exception as e:
                logger.error(f"Shadow retrieve failed: {e}")
        
        # Return results from primary system
        if self._should_use_new_system():
            return new_results if 'new_results' in locals() else old_results
        return old_results
    
    def _compare_results(
        self,
        query_id: str,
        old_results: List[Tuple[Dict[str, Any], float]],
        new_results: List[Tuple[Dict[str, Any], float]],
        latency_old: float,
        latency_new: float
    ) -> ComparisonResult:
        """Compare results from both systems."""
        # Extract top IDs
        old_ids = [r[0]["id"] for r in old_results[:5]]
        new_ids = [r[0]["id"] for r in new_results[:5]]
        
        # Calculate score difference
        score_diffs = []
        for i in range(min(len(old_results), len(new_results))):
            if old_results[i][0]["id"] == new_results[i][0]["id"]:
                score_diffs.append(abs(old_results[i][1] - new_results[i][1]))
        
        avg_score_diff = np.mean(score_diffs) if score_diffs else 1.0
        
        # Check for mismatch
        mismatch = old_ids != new_ids or avg_score_diff > 0.1
        
        return ComparisonResult(
            query_id=query_id,
            old_results=old_ids,
            new_results=new_ids,
            score_diff=avg_score_diff,
            latency_old_ms=latency_old,
            latency_new_ms=latency_new,
            mismatch=mismatch
        )
    
    async def _async_wrapper(self, sync_func, *args, **kwargs):
        """Wrap sync functions for async execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_func, *args, **kwargs)
    
    def get_shadow_stats(self) -> Dict[str, Any]:
        """Get shadow deployment statistics."""
        mismatch_rate = (
            self.mismatch_count / self.comparison_count 
            if self.comparison_count > 0 else 0
        )
        
        return {
            "shadow_enabled": self._should_shadow_write(),
            "serving_from": "new" if self._should_use_new_system() else "old",
            "comparisons": self.comparison_count,
            "mismatches": self.mismatch_count,
            "mismatch_rate": mismatch_rate
        }