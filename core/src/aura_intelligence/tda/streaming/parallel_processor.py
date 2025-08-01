"""
🚀 Multi-Scale Parallel Processing for Streaming TDA
Handles concurrent processing across temporal scales without race conditions
"""

import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import queue

import structlog
from prometheus_client import Counter, Histogram, Gauge

from .windows import StreamingWindow, WindowStats
from .incremental_persistence import VineyardAlgorithm, DiagramUpdate
from ..models import PersistenceDiagram, PersistenceFeature
from ...observability.tracing import get_tracer
from ...common.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

# Metrics
SCALE_UPDATES = Counter('tda_scale_updates_total', 'Updates per scale', ['scale'])
SCALE_LATENCY = Histogram('tda_scale_latency_seconds', 'Processing latency per scale', ['scale'])
RACE_CONDITIONS = Counter('tda_race_conditions_total', 'Race conditions detected')
SYNC_ERRORS = Counter('tda_sync_errors_total', 'Synchronization errors')


@dataclass
class ScaleConfig:
    """Configuration for a temporal scale"""
    name: str
    window_size: int
    slide_interval: int
    max_features: int = 1000
    priority: int = 1  # Higher priority scales get more resources
    

@dataclass
class ScaleState:
    """State for a single scale processor"""
    config: ScaleConfig
    window: StreamingWindow
    algorithm: VineyardAlgorithm
    last_update: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    

class MultiScaleProcessor:
    """
    Processes streaming TDA across multiple temporal scales in parallel
    without race conditions or data corruption
    """
    
    def __init__(
        self,
        scales: List[ScaleConfig],
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.scales = sorted(scales, key=lambda s: s.priority, reverse=True)
        self.scale_states: Dict[str, ScaleState] = {}
        self.max_workers = max_workers or min(len(scales), mp.cpu_count())
        self.use_processes = use_processes
        self.circuit_breaker = circuit_breaker
        
        # Thread-safe queues for each scale
        self.input_queues: Dict[str, queue.Queue] = {}
        self.output_queues: Dict[str, queue.Queue] = {}
        
        # Synchronization primitives
        self.global_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Worker pool
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="tda-scale"
            )
            
        # Initialize scales
        self._initialize_scales()
        
        # Tracer
        self.tracer = get_tracer()
        
        logger.info(
            "multi_scale_processor_initialized",
            scales=len(scales),
            max_workers=self.max_workers,
            use_processes=use_processes
        )
        
    def _initialize_scales(self) -> None:
        """Initialize scale states and queues"""
        for scale in self.scales:
            # Create state
            state = ScaleState(
                config=scale,
                window=StreamingWindow(
                    max_size=scale.window_size,
                    dimension=3  # Assuming 3D point clouds
                ),
                algorithm=VineyardAlgorithm(
                    epsilon=0.1,
                    max_features=scale.max_features
                )
            )
            self.scale_states[scale.name] = state
            
            # Create queues
            self.input_queues[scale.name] = queue.Queue(maxsize=10000)
            self.output_queues[scale.name] = queue.Queue(maxsize=10000)
            
    async def add_points(self, points: np.ndarray) -> Dict[str, DiagramUpdate]:
        """
        Add points to all scales and process in parallel
        
        Returns updates for each scale
        """
        with self.tracer.start_as_current_span("multi_scale_add_points") as span:
            span.set_attribute("num_points", len(points))
            span.set_attribute("num_scales", len(self.scales))
            
            # Distribute points to scale queues
            for scale_name in self.scale_states:
                try:
                    self.input_queues[scale_name].put_nowait(points)
                except queue.Full:
                    logger.warning(
                        "scale_queue_full",
                        scale=scale_name,
                        dropped_points=len(points)
                    )
                    
            # Process scales in parallel
            futures = []
            for scale_name, state in self.scale_states.items():
                future = self.executor.submit(
                    self._process_scale,
                    scale_name,
                    state
                )
                futures.append((scale_name, future))
                
            # Collect results
            updates = {}
            for scale_name, future in futures:
                try:
                    update = await asyncio.get_event_loop().run_in_executor(
                        None, future.result, 5.0  # 5 second timeout
                    )
                    if update:
                        updates[scale_name] = update
                        SCALE_UPDATES.labels(scale=scale_name).inc()
                except Exception as e:
                    logger.error(
                        "scale_processing_error",
                        scale=scale_name,
                        error=str(e)
                    )
                    SYNC_ERRORS.inc()
                    
            return updates
            
    def _process_scale(
        self,
        scale_name: str,
        state: ScaleState
    ) -> Optional[DiagramUpdate]:
        """Process a single scale (runs in worker thread/process)"""
        start_time = datetime.now()
        
        try:
            # Get points from queue
            points_batch = []
            deadline = datetime.now() + timedelta(milliseconds=100)
            
            while datetime.now() < deadline:
                try:
                    points = self.input_queues[scale_name].get_nowait()
                    points_batch.append(points)
                except queue.Empty:
                    break
                    
            if not points_batch:
                return None
                
            # Combine points
            all_points = np.vstack(points_batch)
            
            # Lock for this scale only
            with state.lock:
                # Add to window
                state.window.add_batch(all_points)
                
                # Check if we should process
                if state.window.current_size >= state.config.slide_interval:
                    # Get window data
                    window_data = state.window.get_data()
                    
                    # Compute update
                    update = state.algorithm.incremental_update(
                        window_data,
                        window_id=f"{scale_name}_{state.update_count}"
                    )
                    
                    # Slide window
                    state.window.slide(state.config.slide_interval)
                    
                    # Update state
                    state.last_update = datetime.now()
                    state.update_count += 1
                    
                    # Record metrics
                    latency = (datetime.now() - start_time).total_seconds()
                    SCALE_LATENCY.labels(scale=scale_name).observe(latency)
                    
                    return update
                    
        except Exception as e:
            logger.error(
                "scale_processing_exception",
                scale=scale_name,
                error=str(e)
            )
            raise
            
        return None
        
    def get_scale_diagrams(self) -> Dict[str, PersistenceDiagram]:
        """Get current persistence diagrams for all scales"""
        diagrams = {}
        
        with self.global_lock:
            for scale_name, state in self.scale_states.items():
                with state.lock:
                    diagrams[scale_name] = state.algorithm.get_current_diagram()
                    
        return diagrams
        
    def get_scale_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all scales"""
        stats = {}
        
        for scale_name, state in self.scale_states.items():
            with state.lock:
                window_stats = state.window.get_stats()
                stats[scale_name] = {
                    "window_size": state.config.window_size,
                    "current_size": window_stats.current_size,
                    "total_points_seen": window_stats.total_points_seen,
                    "memory_mb": window_stats.memory_bytes / 1024 / 1024,
                    "update_count": state.update_count,
                    "last_update": state.last_update.isoformat(),
                    "queue_size": self.input_queues[scale_name].qsize()
                }
                
        return stats
        
    async def shutdown(self) -> None:
        """Gracefully shutdown the processor"""
        logger.info("multi_scale_processor_shutdown_started")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop accepting new work
        for q in self.input_queues.values():
            q.put(None)  # Poison pill
            
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("multi_scale_processor_shutdown_complete")


class RaceConditionDetector:
    """
    Detects and prevents race conditions in multi-scale processing
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_log: List[Tuple[datetime, str, str]] = []
        self.lock = threading.Lock()
        
    def log_access(self, scale: str, operation: str) -> None:
        """Log a data access operation"""
        with self.lock:
            self.access_log.append((datetime.now(), scale, operation))
            
            # Keep window size limited
            if len(self.access_log) > self.window_size:
                self.access_log = self.access_log[-self.window_size:]
                
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect potential race conditions"""
        conflicts = []
        
        with self.lock:
            # Group by time windows
            time_groups = defaultdict(list)
            for timestamp, scale, op in self.access_log:
                window = timestamp.replace(microsecond=0)
                time_groups[window].append((scale, op))
                
            # Check for conflicts
            for window, operations in time_groups.items():
                scale_ops = defaultdict(list)
                for scale, op in operations:
                    scale_ops[scale].append(op)
                    
                # Detect write-write conflicts
                for scale, ops in scale_ops.items():
                    writes = [op for op in ops if 'write' in op]
                    if len(writes) > 1:
                        conflicts.append({
                            'type': 'write-write',
                            'scale': scale,
                            'window': window,
                            'operations': writes
                        })
                        RACE_CONDITIONS.inc()
                        
        return conflicts


# Example usage
if __name__ == "__main__":
    # Define scales
    scales = [
        ScaleConfig("1min", window_size=1000, slide_interval=100, priority=3),
        ScaleConfig("5min", window_size=5000, slide_interval=500, priority=2),
        ScaleConfig("15min", window_size=15000, slide_interval=1500, priority=1),
    ]
    
    # Create processor
    processor = MultiScaleProcessor(scales, max_workers=3)
    
    # Simulate streaming data
    async def simulate_stream():
        for i in range(100):
            points = np.random.randn(10, 3)
            updates = await processor.add_points(points)
            
            if updates:
                for scale, update in updates.items():
                    print(f"Scale {scale}: {len(update.added_features)} new features")
                    
            await asyncio.sleep(0.1)
            
        # Get final stats
        stats = processor.get_scale_stats()
        for scale, stat in stats.items():
            print(f"\nScale {scale}:")
            print(f"  Points processed: {stat['total_points_seen']}")
            print(f"  Updates: {stat['update_count']}")
            print(f"  Memory: {stat['memory_mb']:.2f} MB")
            
        await processor.shutdown()
        
    # Run simulation
    asyncio.run(simulate_stream())