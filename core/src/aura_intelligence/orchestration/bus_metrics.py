"""
Event Bus Metrics Exporter
==========================
Exposes Prometheus metrics for the Event Bus.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import time
from aiohttp import web
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

from .bus_redis import RedisBus

logger = logging.getLogger(__name__)

# Define metrics
bus_messages_total = Counter(
    'aura_bus_messages_total',
    'Total messages published to Event Bus',
    ['stream']
)

bus_latency_seconds = Histogram(
    'aura_bus_latency_seconds',
    'Event Bus operation latency',
    ['operation', 'stream'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

bus_pel_size = Gauge(
    'aura_bus_pel_size',
    'Pending Entry List size per consumer group',
    ['stream', 'group']
)

bus_consumer_lag = Gauge(
    'aura_bus_consumer_lag',
    'Consumer lag in messages',
    ['stream', 'group', 'consumer']
)

bus_connections_active = Gauge(
    'aura_bus_connections_active',
    'Active Redis connections'
)

bus_stream_length = Gauge(
    'aura_bus_stream_length',
    'Current stream length',
    ['stream']
)


class EventBusMetrics:
    """Collects and exposes Event Bus metrics."""
    
    def __init__(self, bus: RedisBus, port: int = 9102):
        self.bus = bus
        self.port = port
        self.app = web.Application()
        self.running = False
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/health', self.health_handler)
        
    async def metrics_handler(self, request):
        """Prometheus metrics endpoint."""
        # Collect current metrics
        await self._collect_metrics()
        
        # Generate Prometheus format
        metrics = generate_latest()
        return web.Response(
            body=metrics,
            content_type=CONTENT_TYPE_LATEST
        )
        
    async def health_handler(self, request):
        """Health check endpoint."""
        healthy = await self.bus.health_check()
        status = 200 if healthy else 503
        return web.json_response(
            {"status": "healthy" if healthy else "unhealthy"},
            status=status
        )
        
    async def _collect_metrics(self):
        """Collect current metrics from Redis."""
        try:
            client = await self.bus._get_client()
            
            # Get connection pool stats
            pool_stats = self.bus.pool.get_connection_kwargs()
            bus_connections_active.set(self.bus.pool.connection_kwargs.get('max_connections', 0))
            
            # Collect metrics for known streams
            streams = [
                'topo:failures',
                'evolver:patches', 
                'langgraph:events',
                'aura:dlq'
            ]
            
            for stream in streams:
                try:
                    # Stream length
                    length = await client.xlen(stream)
                    bus_stream_length.labels(stream=stream).set(length)
                    
                    # Consumer group info
                    try:
                        groups = await client.xinfo_groups(stream)
                        for group in groups:
                            group_name = group['name']
                            pending = group['pending']
                            bus_pel_size.labels(
                                stream=stream,
                                group=group_name
                            ).set(pending)
                            
                            # Consumer lag
                            try:
                                consumers = await client.xinfo_consumers(stream, group_name)
                                for consumer in consumers:
                                    consumer_name = consumer['name']
                                    consumer_pending = consumer['pending']
                                    bus_consumer_lag.labels(
                                        stream=stream,
                                        group=group_name,
                                        consumer=consumer_name
                                    ).set(consumer_pending)
                            except Exception:
                                pass
                                
                    except Exception:
                        # No consumer groups yet
                        pass
                        
                except Exception as e:
                    logger.debug(f"Stream {stream} not found: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
    async def start(self):
        """Start metrics server."""
        self.running = True
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"Metrics server started on port {self.port}")
        
    async def stop(self):
        """Stop metrics server."""
        self.running = False
        await self.app.shutdown()
        await self.app.cleanup()


# Instrumented bus wrapper
class InstrumentedRedisBus(RedisBus):
    """Redis Bus with automatic metrics collection."""
    
    async def publish(self, stream: str, data: Dict[str, Any]) -> str:
        """Publish with metrics."""
        start = time.time()
        try:
            result = await super().publish(stream, data)
            bus_messages_total.labels(stream=stream).inc()
            return result
        finally:
            bus_latency_seconds.labels(
                operation='publish',
                stream=stream
            ).observe(time.time() - start)
            
    async def ack(self, stream: str, group: str, event_id: str) -> bool:
        """Acknowledge with metrics."""
        start = time.time()
        try:
            return await super().ack(stream, group, event_id)
        finally:
            bus_latency_seconds.labels(
                operation='ack',
                stream=stream
            ).observe(time.time() - start)


def create_instrumented_bus(url: Optional[str] = None, **kwargs) -> InstrumentedRedisBus:
    """Create an instrumented Redis bus."""
    if url is None:
        url = "redis://localhost:6379"
    return InstrumentedRedisBus(url=url, **kwargs)


async def main():
    """Run standalone metrics server."""
    bus = create_instrumented_bus()
    metrics = EventBusMetrics(bus)
    
    try:
        await metrics.start()
        logger.info("Metrics server running on http://localhost:9102/metrics")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutting down metrics server")
    finally:
        await metrics.stop()
        await bus.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    asyncio.run(main())