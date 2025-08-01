"""
🚀 Redis Streams Transport - High-Performance Agent Communication

Redis Streams-based transport layer for the ACP protocol with:
- High-throughput message delivery
- Consumer groups for load balancing
- Message persistence and replay
- Priority queues and routing
- Integration with existing Redis infrastructure

Based on the proven technologies from kakakagan.md research.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

import redis.asyncio as redis
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..schemas.acp import ACPEnvelope, ACPEndpoint, Priority

tracer = trace.get_tracer(__name__)


@dataclass
class StreamConfig:
    """Configuration for Redis Streams."""
    stream_name: str
    consumer_group: str
    consumer_name: str
    max_length: int = 10000
    block_time_ms: int = 1000
    batch_size: int = 10


class RedisStreamsTransport:
    """
    Redis Streams transport implementation for ACP protocol.
    
    Provides reliable, high-performance message delivery using Redis Streams
    with consumer groups for load balancing and fault tolerance.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        agent_endpoint: Optional[ACPEndpoint] = None,
        stream_prefix: str = "acp",
        consumer_group: str = "agents",
        max_connections: int = 10,
        retry_attempts: int = 3
    ):
        """
        Initialize Redis Streams transport.
        
        Args:
            redis_url: Redis connection URL
            agent_endpoint: This agent's endpoint info
            stream_prefix: Prefix for stream names
            consumer_group: Consumer group name
            max_connections: Maximum Redis connections
            retry_attempts: Number of retry attempts
        """
        self.redis_url = redis_url
        self.agent_endpoint = agent_endpoint
        self.stream_prefix = stream_prefix
        self.consumer_group = consumer_group
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        
        # Redis connections
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Stream configurations
        self.streams: Dict[str, StreamConfig] = {}
        self.priority_streams: Dict[Priority, str] = {}
        
        # Agent discovery
        self.known_agents: Dict[str, ACPEndpoint] = {}
        self.agent_registry_key = f"{stream_prefix}:agents"
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0,
            'last_heartbeat': None
        }
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
    
    async def start(self) -> None:
        """Start the transport and initialize Redis connections."""
        if self._running:
            return
        
        try:
            # Create Redis connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError]
            )
            
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize streams
            await self._initialize_streams()
            
            # Register this agent
            if self.agent_endpoint:
                await self._register_agent(self.agent_endpoint)
            
            self._running = True
            
            # Start background tasks
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._background_tasks.add(heartbeat_task)
            heartbeat_task.add_done_callback(self._background_tasks.discard)
            
            discovery_task = asyncio.create_task(self._agent_discovery_loop())
            self._background_tasks.add(discovery_task)
            discovery_task.add_done_callback(self._background_tasks.discard)
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            raise ConnectionError(f"Failed to start Redis transport: {e}")
    
    async def stop(self) -> None:
        """Stop the transport and cleanup resources."""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Unregister agent
        if self.agent_endpoint:
            await self._unregister_agent(self.agent_endpoint)
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
    
    async def _initialize_streams(self) -> None:
        """Initialize Redis streams and consumer groups."""
        # Create priority-based streams
        priorities = [Priority.CRITICAL, Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW]
        
        for priority in priorities:
            stream_name = f"{self.stream_prefix}:messages:{priority.value}"
            self.priority_streams[priority] = stream_name
            
            # Create stream and consumer group
            try:
                await self.redis_client.xgroup_create(
                    stream_name,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Configure stream
            consumer_name = f"{self.agent_endpoint.agent_id}" if self.agent_endpoint else "unknown"
            self.streams[stream_name] = StreamConfig(
                stream_name=stream_name,
                consumer_group=self.consumer_group,
                consumer_name=consumer_name
            )
    
    @tracer.start_as_current_span("redis_send_message")
    async def send(self, envelope: ACPEnvelope) -> None:
        """
        Send a message via Redis Streams.
        
        Args:
            envelope: Message envelope to send
        """
        span = trace.get_current_span()
        span.set_attributes({
            "message_id": envelope.message_id,
            "recipient": envelope.recipient.agent_id,
            "priority": envelope.priority.value
        })
        
        try:
            # Select stream based on priority
            stream_name = self.priority_streams.get(envelope.priority, self.priority_streams[Priority.NORMAL])
            
            # Serialize envelope
            message_data = {
                'envelope': json.dumps(envelope.to_dict()),
                'recipient': envelope.recipient.agent_id,
                'sender': envelope.sender.agent_id,
                'message_type': envelope.message_type.value,
                'correlation_id': envelope.correlation_id,
                'timestamp': envelope.timestamp_utc
            }
            
            # Send to stream
            message_id = await self.redis_client.xadd(
                stream_name,
                message_data,
                maxlen=10000  # Limit stream size
            )
            
            self.stats['messages_sent'] += 1
            
            span.set_attributes({
                "stream_name": stream_name,
                "redis_message_id": message_id
            })
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    @tracer.start_as_current_span("redis_receive_messages")
    async def receive(self, timeout: float = 1.0) -> List[ACPEnvelope]:
        """
        Receive messages from Redis Streams.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of received message envelopes
        """
        if not self._running or not self.redis_client:
            return []
        
        messages = []
        
        try:
            # Read from all priority streams
            stream_keys = {}
            for stream_name, config in self.streams.items():
                stream_keys[stream_name] = '>'
            
            # Read messages
            result = await self.redis_client.xreadgroup(
                self.consumer_group,
                self.streams[list(self.streams.keys())[0]].consumer_name,
                stream_keys,
                count=10,
                block=int(timeout * 1000)
            )
            
            # Process received messages
            for stream_name, stream_messages in result:
                for message_id, fields in stream_messages:
                    try:
                        # Parse envelope
                        envelope_data = json.loads(fields[b'envelope'])
                        envelope = ACPEnvelope.from_dict(envelope_data)
                        
                        # Check if message is for this agent
                        if (self.agent_endpoint and 
                            envelope.recipient.agent_id == self.agent_endpoint.agent_id):
                            messages.append(envelope)
                            
                            # Acknowledge message
                            await self.redis_client.xack(
                                stream_name,
                                self.consumer_group,
                                message_id
                            )
                            
                            self.stats['messages_received'] += 1
                    
                    except Exception as e:
                        print(f"Error processing message {message_id}: {e}")
                        # Acknowledge to prevent redelivery of bad messages
                        await self.redis_client.xack(
                            stream_name,
                            self.consumer_group,
                            message_id
                        )
        
        except redis.ResponseError as e:
            if "NOGROUP" in str(e):
                # Consumer group doesn't exist, reinitialize
                await self._initialize_streams()
        except Exception as e:
            print(f"Error receiving messages: {e}")
        
        return messages
    
    async def discover_agents(self, roles: Optional[List[str]] = None) -> List[ACPEndpoint]:
        """
        Discover available agents.
        
        Args:
            roles: Filter by specific roles
            
        Returns:
            List of available agent endpoints
        """
        try:
            # Get all registered agents
            agent_data = await self.redis_client.hgetall(self.agent_registry_key)
            
            agents = []
            for agent_id, data in agent_data.items():
                try:
                    agent_info = json.loads(data)
                    endpoint = ACPEndpoint(**agent_info)
                    
                    # Filter by roles if specified
                    if roles is None or endpoint.role in roles:
                        agents.append(endpoint)
                        
                except Exception as e:
                    print(f"Error parsing agent data for {agent_id}: {e}")
            
            return agents
            
        except Exception as e:
            print(f"Error discovering agents: {e}")
            return []
    
    async def _register_agent(self, endpoint: ACPEndpoint) -> None:
        """Register an agent in the discovery registry."""
        try:
            agent_data = {
                'agent_id': endpoint.agent_id,
                'role': endpoint.role,
                'instance_id': endpoint.instance_id,
                'capabilities': endpoint.capabilities,
                'registered_at': datetime.now(timezone.utc).isoformat(),
                'last_heartbeat': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_client.hset(
                self.agent_registry_key,
                endpoint.agent_id,
                json.dumps(agent_data)
            )
            
        except Exception as e:
            print(f"Error registering agent: {e}")
    
    async def _unregister_agent(self, endpoint: ACPEndpoint) -> None:
        """Unregister an agent from the discovery registry."""
        try:
            await self.redis_client.hdel(self.agent_registry_key, endpoint.agent_id)
        except Exception as e:
            print(f"Error unregistering agent: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats."""
        while self._running:
            try:
                if self.agent_endpoint:
                    # Update heartbeat timestamp
                    agent_data = await self.redis_client.hget(
                        self.agent_registry_key,
                        self.agent_endpoint.agent_id
                    )
                    
                    if agent_data:
                        data = json.loads(agent_data)
                        data['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
                        
                        await self.redis_client.hset(
                            self.agent_registry_key,
                            self.agent_endpoint.agent_id,
                            json.dumps(data)
                        )
                        
                        self.stats['last_heartbeat'] = data['last_heartbeat']
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _agent_discovery_loop(self) -> None:
        """Background task to discover and cache agent information."""
        while self._running:
            try:
                # Refresh known agents
                agents = await self.discover_agents()
                self.known_agents = {agent.agent_id: agent for agent in agents}
                
                # Cleanup stale agents (no heartbeat for 5 minutes)
                cutoff_time = datetime.now(timezone.utc).timestamp() - 300
                stale_agents = []
                
                agent_data = await self.redis_client.hgetall(self.agent_registry_key)
                for agent_id, data in agent_data.items():
                    try:
                        agent_info = json.loads(data)
                        last_heartbeat = datetime.fromisoformat(
                            agent_info['last_heartbeat'].replace('Z', '+00:00')
                        ).timestamp()
                        
                        if last_heartbeat < cutoff_time:
                            stale_agents.append(agent_id)
                            
                    except Exception:
                        stale_agents.append(agent_id)
                
                # Remove stale agents
                if stale_agents:
                    await self.redis_client.hdel(self.agent_registry_key, *stale_agents)
                
                await asyncio.sleep(60)  # Discovery every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            **self.stats,
            'known_agents': len(self.known_agents),
            'streams_configured': len(self.streams),
            'redis_connected': self.redis_client is not None and self._running
        }
