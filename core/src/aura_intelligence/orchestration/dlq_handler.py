"""
Dead Letter Queue Handler
=========================
Handles failed messages with retry logic and S3 archival.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import os

from .bus_redis import create_redis_bus
from .bus_protocol import EventBus, Event

logger = logging.getLogger(__name__)


class DLQHandler:
    """
    Handles dead letter queue messages.
    
    Per loothis.md 4.6:
    - Auto-retry up to 3 times
    - Persist to S3 after max retries
    - Expose metrics for monitoring
    """
    
    def __init__(
        self,
        bus: Optional[EventBus] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        s3_bucket: Optional[str] = None
    ):
        self.bus = bus
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.s3_bucket = s3_bucket or os.environ.get('DLQ_S3_BUCKET', 'aura-dlq')
        self.running = False
        self.messages_processed = 0
        self.messages_retried = 0
        self.messages_archived = 0
        
    async def initialize(self):
        """Initialize DLQ handler."""
        if not self.bus:
            self.bus = create_redis_bus()
            
        if not await self.bus.health_check():
            raise RuntimeError("Event Bus not available")
            
        logger.info("DLQ handler initialized")
        logger.info(f"  Max retries: {self.max_retries}")
        logger.info(f"  S3 bucket: {self.s3_bucket}")
        
    async def process_dlq(self):
        """Main DLQ processing loop."""
        self.running = True
        logger.info("DLQ handler started")
        
        try:
            async for event in self.bus.subscribe("aura:dlq", "dlq-handler", "dlq-1"):
                if not self.running:
                    break
                    
                try:
                    await self._process_message(event)
                    self.messages_processed += 1
                except Exception as e:
                    logger.error(f"Error processing DLQ message: {e}")
                finally:
                    # Always acknowledge to prevent redelivery
                    await self.bus.ack("aura:dlq", "dlq-handler", event.metadata.id)
                    
        except Exception as e:
            logger.error(f"Fatal error in DLQ loop: {e}")
        finally:
            self.running = False
            
    async def _process_message(self, event: Event):
        """Process a single DLQ message."""
        logger.info(f"Processing DLQ message: {event.metadata.id}")
        
        # Extract original stream and retry count
        original_stream = event.payload.get("original_stream", "unknown")
        retry_count = event.metadata.retry_count
        error_reason = event.payload.get("error", "Unknown error")
        
        logger.info(f"  Original stream: {original_stream}")
        logger.info(f"  Retry count: {retry_count}/{self.max_retries}")
        logger.info(f"  Error: {error_reason}")
        
        if retry_count < self.max_retries:
            # Retry the message
            await self._retry_message(event, original_stream)
            self.messages_retried += 1
        else:
            # Archive to S3
            await self._archive_message(event)
            self.messages_archived += 1
            
    async def _retry_message(self, event: Event, original_stream: str):
        """Retry a message after delay."""
        logger.info(f"Retrying message after {self.retry_delay}s delay")
        
        # Wait before retry
        await asyncio.sleep(self.retry_delay)
        
        # Prepare retry payload
        retry_payload = event.payload.copy()
        retry_payload["retry_count"] = event.metadata.retry_count + 1
        retry_payload["retry_timestamp"] = datetime.utcnow().isoformat()
        
        try:
            # Publish back to original stream
            retry_id = await self.bus.publish(original_stream, retry_payload)
            logger.info(f"Retried message to {original_stream}: {retry_id}")
            
            # Log retry event
            await self._log_retry(event, original_stream, retry_id)
            
        except Exception as e:
            logger.error(f"Failed to retry message: {e}")
            # If retry fails, archive it
            await self._archive_message(event)
            
    async def _archive_message(self, event: Event):
        """Archive message to S3."""
        logger.info("Archiving message to S3")
        
        # Prepare archive data
        archive_data = {
            "message_id": event.metadata.id,
            "timestamp": event.metadata.timestamp.isoformat(),
            "stream": event.metadata.stream,
            "retry_count": event.metadata.retry_count,
            "payload": event.payload,
            "archived_at": datetime.utcnow().isoformat()
        }
        
        # Generate S3 key
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        message_hash = hashlib.sha256(
            event.metadata.id.encode()
        ).hexdigest()[:8]
        s3_key = f"dlq/{date_prefix}/{event.metadata.stream}/{message_hash}.json"
        
        try:
            # In production, use boto3 to upload to S3
            # For demo, simulate S3 upload
            await self._simulate_s3_upload(s3_key, archive_data)
            
            logger.info(f"Archived to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Publish archive event
            await self._log_archive(event, s3_key)
            
        except Exception as e:
            logger.error(f"Failed to archive message: {e}")
            # Last resort: log to file
            await self._write_to_failsafe_log(archive_data)
            
    async def _simulate_s3_upload(self, key: str, data: Dict[str, Any]):
        """Simulate S3 upload for demo."""
        # In production, replace with:
        # s3_client = boto3.client('s3')
        # s3_client.put_object(
        #     Bucket=self.s3_bucket,
        #     Key=key,
        #     Body=json.dumps(data),
        #     ContentType='application/json'
        # )
        
        # For demo, write to local file
        archive_dir = f"/tmp/aura-dlq/{self.s3_bucket}"
        os.makedirs(archive_dir, exist_ok=True)
        
        file_path = f"{archive_dir}/{key.replace('/', '_')}"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Simulate upload delay
        await asyncio.sleep(0.1)
        
    async def _write_to_failsafe_log(self, data: Dict[str, Any]):
        """Write to local failsafe log as last resort."""
        failsafe_path = "/tmp/aura-dlq-failsafe.jsonl"
        
        with open(failsafe_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
            
        logger.warning(f"Wrote to failsafe log: {failsafe_path}")
        
    async def _log_retry(self, event: Event, stream: str, retry_id: str):
        """Log retry event for monitoring."""
        retry_event = {
            "type": "dlq_retry",
            "original_id": event.metadata.id,
            "retry_id": retry_id,
            "stream": stream,
            "retry_count": event.metadata.retry_count + 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.bus.publish("aura:dlq:events", retry_event)
        
    async def _log_archive(self, event: Event, s3_key: str):
        """Log archive event for monitoring."""
        archive_event = {
            "type": "dlq_archive",
            "message_id": event.metadata.id,
            "s3_bucket": self.s3_bucket,
            "s3_key": s3_key,
            "retry_count": event.metadata.retry_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.bus.publish("aura:dlq:events", archive_event)
        
    async def shutdown(self):
        """Gracefully shut down DLQ handler."""
        logger.info("Shutting down DLQ handler")
        self.running = False
        if self.bus:
            await self.bus.close()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ handler statistics."""
        return {
            "messages_processed": self.messages_processed,
            "messages_retried": self.messages_retried,
            "messages_archived": self.messages_archived,
            "retry_rate": self.messages_retried / max(1, self.messages_processed),
            "archive_rate": self.messages_archived / max(1, self.messages_processed),
            "status": "running" if self.running else "stopped"
        }


async def main():
    """Run DLQ handler standalone."""
    handler = DLQHandler()
    
    try:
        await handler.initialize()
        await handler.process_dlq()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stats = handler.get_stats()
        logger.info(f"DLQ handler stats: {stats}")
        await handler.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    asyncio.run(main())