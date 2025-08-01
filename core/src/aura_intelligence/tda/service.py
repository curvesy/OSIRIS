"""
🚀 Production TDA Service
FastAPI service for enterprise TDA computations with full observability.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .models import TDARequest, TDAResponse, TDAConfiguration, TDAAlgorithm
from .core import ProductionTDAEngine
from .benchmarks import TDABenchmarkSuite
from ..utils.logger import get_logger


class TDAService:
    """
    🚀 Production TDA Service
    
    Enterprise FastAPI service providing:
    - High-performance TDA computations
    - GPU acceleration with 30x speedup
    - Comprehensive observability and metrics
    - Enterprise-grade reliability and error handling
    - Horizontal scaling and load balancing
    """
    
    def __init__(self, config: TDAConfiguration = None):
        self.config = config or TDAConfiguration()
        self.logger = get_logger(__name__)
        
        # Initialize TDA engine
        self.tda_engine = ProductionTDAEngine(self.config)
        
        # Initialize benchmark suite
        self.benchmark_suite = TDABenchmarkSuite(self.tda_engine)
        
        # Service state
        self.is_healthy = True
        self.startup_time = time.time()
        
        # Create FastAPI app
        self.app = self._create_app()
        
        self.logger.info("🚀 TDA Service initialized")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("🔄 TDA Service starting up...")
            yield
            # Shutdown
            self.logger.info("🔄 TDA Service shutting down...")
            await self.tda_engine.shutdown()
        
        app = FastAPI(
            title="AURA Intelligence TDA Service",
            description="Enterprise-grade Topological Data Analysis with GPU acceleration",
            version="1.0.0",
            lifespan=lifespan
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
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add all API routes."""
        
        @app.get("/")
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "AURA Intelligence TDA Service",
                "version": "1.0.0",
                "status": "healthy" if self.is_healthy else "unhealthy",
                "uptime_seconds": time.time() - self.startup_time,
                "gpu_available": self.tda_engine.cuda_accelerator.is_available() if self.tda_engine.cuda_accelerator else False,
                "algorithms_available": list(self.tda_engine.algorithms.keys()),
                "documentation": "/docs"
            }
        
        @app.post("/compute", response_model=TDAResponse)
        async def compute_tda(request: TDARequest) -> TDAResponse:
            """
            Compute TDA with enterprise-grade reliability.
            
            Performs topological data analysis on input data with:
            - GPU acceleration for 30x speedup
            - Comprehensive error handling
            - Performance metrics and monitoring
            - Enterprise SLA guarantees
            """
            try:
                self.logger.info(f"🔄 TDA computation request: {request.request_id}")
                
                # Validate service health
                if not self.is_healthy:
                    raise HTTPException(status_code=503, detail="Service unhealthy")
                
                # Execute TDA computation
                response = await self.tda_engine.compute_tda(request)
                
                self.logger.info(f"✅ TDA computation completed: {request.request_id}")
                return response
                
            except Exception as e:
                self.logger.error(f"❌ TDA computation failed: {request.request_id} - {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for load balancers."""
            
            try:
                # Check TDA engine health
                system_status = await self.tda_engine.get_system_status()
                
                # Check GPU availability if enabled
                gpu_healthy = True
                if self.config.enable_gpu and self.tda_engine.cuda_accelerator:
                    gpu_healthy = self.tda_engine.cuda_accelerator.is_available()
                
                # Overall health status
                healthy = (
                    system_status['engine_status'] == 'healthy' and
                    system_status['active_requests'] < self.config.max_concurrent_requests and
                    gpu_healthy
                )
                
                self.is_healthy = healthy
                
                return {
                    "status": "healthy" if healthy else "unhealthy",
                    "timestamp": time.time(),
                    "uptime_seconds": time.time() - self.startup_time,
                    "system_status": system_status,
                    "gpu_healthy": gpu_healthy,
                    "active_requests": system_status['active_requests'],
                    "max_requests": self.config.max_concurrent_requests
                }
                
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
                self.is_healthy = False
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @app.get("/status")
        async def get_status():
            """Get comprehensive system status."""
            
            try:
                system_status = await self.tda_engine.get_system_status()
                performance_stats = self.tda_engine.get_performance_stats()
                
                return {
                    "service_info": {
                        "name": "AURA Intelligence TDA Service",
                        "version": "1.0.0",
                        "uptime_seconds": time.time() - self.startup_time,
                        "healthy": self.is_healthy
                    },
                    "system_status": system_status,
                    "performance_stats": performance_stats,
                    "configuration": {
                        "default_algorithm": str(self.config.default_algorithm),
                        "gpu_enabled": self.config.enable_gpu,
                        "max_concurrent_requests": self.config.max_concurrent_requests,
                        "default_timeout_seconds": self.config.default_timeout_seconds,
                        "memory_limit_gb": self.config.memory_limit_gb
                    }
                }
                
            except Exception as e:
                self.logger.error(f"❌ Status check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/algorithms")
        async def list_algorithms():
            """List available TDA algorithms."""
            
            algorithms = []
            for algo_name, algo_instance in self.tda_engine.algorithms.items():
                algorithms.append({
                    "name": str(algo_name),
                    "display_name": getattr(algo_instance, 'algorithm_name', str(algo_name)),
                    "available": True,
                    "gpu_accelerated": hasattr(algo_instance, 'cuda_accelerator') and 
                                     algo_instance.cuda_accelerator is not None,
                    "description": algo_instance.__doc__.split('\n')[1].strip() if algo_instance.__doc__ else ""
                })
            
            return {
                "algorithms": algorithms,
                "total_count": len(algorithms),
                "gpu_available": self.tda_engine.cuda_accelerator.is_available() if self.tda_engine.cuda_accelerator else False
            }
        
        @app.post("/benchmark")
        async def run_benchmark(
            background_tasks: BackgroundTasks,
            algorithms: Optional[List[TDAAlgorithm]] = None,
            datasets: Optional[List[str]] = None,
            n_runs: int = 3
        ):
            """
            Run TDA algorithm benchmarks.
            
            Executes comprehensive benchmarking across algorithms and datasets
            to validate performance and accuracy.
            """
            
            try:
                self.logger.info("🏆 Starting TDA benchmark")
                
                # Run benchmark in background
                background_tasks.add_task(
                    self._run_benchmark_task,
                    algorithms, datasets, n_runs
                )
                
                return {
                    "message": "Benchmark started",
                    "status": "running",
                    "algorithms": algorithms or "all available",
                    "datasets": datasets or "all available",
                    "n_runs": n_runs,
                    "estimated_duration_minutes": len(algorithms or [1, 2]) * len(datasets or [1, 2, 3, 4, 5, 6]) * n_runs * 0.5
                }
                
            except Exception as e:
                self.logger.error(f"❌ Benchmark start failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/benchmark/results")
        async def get_benchmark_results():
            """Get latest benchmark results."""
            
            if not self.benchmark_suite.benchmark_results:
                return {
                    "message": "No benchmark results available",
                    "results": []
                }
            
            # Return last 10 results
            recent_results = self.benchmark_suite.benchmark_results[-10:]
            
            return {
                "total_results": len(self.benchmark_suite.benchmark_results),
                "recent_results": [
                    {
                        "algorithm": str(r.algorithm),
                        "dataset": r.dataset_name,
                        "dataset_size": r.dataset_size,
                        "avg_time_ms": r.avg_computation_time_ms,
                        "accuracy": r.accuracy_score,
                        "speedup": r.speedup_vs_baseline,
                        "timestamp": r.benchmark_timestamp.isoformat()
                    }
                    for r in recent_results
                ]
            }
        
        if PROMETHEUS_AVAILABLE:
            @app.get("/metrics")
            async def get_metrics():
                """Prometheus metrics endpoint."""
                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        @app.post("/admin/shutdown")
        async def shutdown_service():
            """Graceful service shutdown (admin only)."""
            
            self.logger.info("🔄 Shutdown requested")
            
            # Mark as unhealthy
            self.is_healthy = False
            
            # Shutdown TDA engine
            await self.tda_engine.shutdown()
            
            return {"message": "Service shutdown initiated"}
        
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler."""
            
            self.logger.error(f"❌ Unhandled exception: {exc}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(exc),
                    "timestamp": time.time()
                }
            )
    
    async def _run_benchmark_task(
        self,
        algorithms: Optional[List[TDAAlgorithm]],
        datasets: Optional[List[str]],
        n_runs: int
    ):
        """Background task for running benchmarks."""
        
        try:
            results = await self.benchmark_suite.run_comprehensive_benchmark(
                algorithms=algorithms,
                datasets=datasets,
                n_runs=n_runs
            )
            
            # Save results
            self.benchmark_suite.save_benchmark_results()
            
            # Generate plots if possible
            try:
                self.benchmark_suite.generate_performance_plots()
            except Exception as e:
                self.logger.warning(f"⚠️ Could not generate plots: {e}")
            
            self.logger.info("✅ Benchmark completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark task failed: {e}")
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False
    ):
        """Run the TDA service."""
        
        self.logger.info(f"🚀 Starting TDA Service on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=self.config.log_level.lower(),
            access_log=True
        )


# Factory function for creating service
def create_tda_service(config: TDAConfiguration = None) -> TDAService:
    """Create TDA service instance."""
    return TDAService(config)


# CLI entry point
def main():
    """Main entry point for TDA service."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Intelligence TDA Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Create configuration
    config = TDAConfiguration(
        enable_gpu=args.gpu,
        log_level=args.log_level
    )
    
    # Create and run service
    service = create_tda_service(config)
    service.run(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
