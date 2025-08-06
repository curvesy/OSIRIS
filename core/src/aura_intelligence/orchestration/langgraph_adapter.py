"""
LangGraph Event Bus Adapter
===========================
Bridges Event Bus with LangGraph orchestration.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
import hashlib

from .bus_redis import create_redis_bus
from .bus_protocol import EventBus, Event
from ..integrations.enhanced_workflow_orchestrator import EnhancedWorkflowOrchestrator

logger = logging.getLogger(__name__)


class LangGraphAdapter:
    """
    Adapter that connects Event Bus to LangGraph.
    
    Responsibilities:
    - Subscribe to evolver:patches stream
    - Apply patches to running graphs
    - Trigger re-execution of affected nodes
    - Publish execution status back to bus
    - Maintain idempotency via patch ledger
    """
    
    def __init__(
        self, 
        bus: Optional[EventBus] = None,
        orchestrator: Optional[EnhancedWorkflowOrchestrator] = None
    ):
        self.bus = bus
        self.orchestrator = orchestrator
        self.running = False
        self.patches_applied = 0
        self.patch_ledger: Set[str] = set()  # Track applied patches
        
    async def initialize(self):
        """Initialize adapter and connections."""
        if not self.bus:
            self.bus = create_redis_bus()
            
        if not await self.bus.health_check():
            raise RuntimeError("Event Bus not available")
            
        if not self.orchestrator:
            self.orchestrator = EnhancedWorkflowOrchestrator()
            await self.orchestrator.initialize()
            
        logger.info("LangGraph adapter initialized")
        
    async def listen_and_apply(self):
        """Main loop: listen for patches and apply them."""
        self.running = True
        logger.info("LangGraph adapter listening for patches...")
        
        try:
            async for event in self.bus.subscribe("evolver:patches", "langgraph", "lg-adapter-1"):
                if not self.running:
                    break
                    
                try:
                    await self._process_patch(event)
                except Exception as e:
                    logger.error(f"Error processing patch: {e}")
                    await self._publish_error(event, str(e))
                    
        except Exception as e:
            logger.error(f"Fatal error in adapter loop: {e}")
        finally:
            self.running = False
            
    async def _process_patch(self, event: Event):
        """Process a single patch event."""
        logger.info(f"Processing patch: {event.metadata.id}")
        
        patch_data = event.payload.get("patch", {})
        target = event.payload.get("target", "unknown")
        failure_id = event.payload.get("failure_id")
        
        # Check idempotency
        patch_hash = self._compute_patch_hash(patch_data)
        if patch_hash in self.patch_ledger:
            logger.info(f"Patch {patch_hash} already applied, skipping")
            await self.bus.ack("evolver:patches", "langgraph", event.metadata.id)
            return
            
        # Validate patch
        if not self._validate_patch(patch_data):
            logger.warning("Invalid patch format")
            await self._publish_error(event, "Invalid patch format")
            await self.bus.ack("evolver:patches", "langgraph", event.metadata.id)
            return
            
        # Apply patch to graph
        result = await self._apply_patch_to_graph(target, patch_data)
        
        if result["success"]:
            # Mark as applied
            self.patch_ledger.add(patch_hash)
            self.patches_applied += 1
            
            # Trigger re-execution
            reexec_result = await self._trigger_reexecution(target, patch_data)
            
            # Publish success event
            await self._publish_success(event, result, reexec_result)
            
            logger.info(f"Successfully applied patch to {target}")
            logger.info(f"  Summary: {patch_data.get('summary', 'N/A')}")
            logger.info(f"  Re-execution: {reexec_result.get('status', 'unknown')}")
        else:
            # Publish failure event
            await self._publish_failure(event, result)
            logger.error(f"Failed to apply patch: {result.get('error', 'Unknown')}")
            
        # Acknowledge the message
        await self.bus.ack("evolver:patches", "langgraph", event.metadata.id)
        
    def _validate_patch(self, patch_data: Dict[str, Any]) -> bool:
        """Validate patch has required fields."""
        required = ["code", "summary", "confidence"]
        return all(field in patch_data for field in required)
        
    def _compute_patch_hash(self, patch_data: Dict[str, Any]) -> str:
        """Compute hash for idempotency check."""
        content = json.dumps(patch_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
        
    async def _apply_patch_to_graph(
        self, 
        target: str, 
        patch_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply patch to the graph component."""
        try:
            # In production, this would:
            # 1. Parse the patch code
            # 2. Identify the target node/component
            # 3. Apply the code changes
            # 4. Validate the changes don't break the graph
            
            # For demo, simulate application
            code = patch_data.get("code", "")
            confidence = patch_data.get("confidence", 0)
            
            if confidence < 0.7:
                return {
                    "success": False,
                    "error": "Confidence too low",
                    "confidence": confidence
                }
                
            # Simulate patch application
            logger.info(f"Applying patch to {target}...")
            await asyncio.sleep(0.5)  # Simulate work
            
            return {
                "success": True,
                "target": target,
                "lines_changed": len(code.split('\n')),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target": target
            }
            
    async def _trigger_reexecution(
        self, 
        target: str, 
        patch_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger re-execution of affected graph nodes."""
        try:
            # In production, this would:
            # 1. Identify affected nodes in the graph
            # 2. Create a new execution context
            # 3. Re-run from the patched node
            # 4. Propagate results downstream
            
            logger.info(f"Triggering re-execution for {target}...")
            
            # For demo, simulate re-execution
            await asyncio.sleep(0.3)
            
            return {
                "status": "completed",
                "nodes_rerun": [target, f"{target}_downstream"],
                "execution_time_ms": 287,
                "result": "improved"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
            
    async def _publish_success(
        self, 
        original_event: Event,
        apply_result: Dict[str, Any],
        reexec_result: Dict[str, Any]
    ):
        """Publish successful patch application."""
        event_data = {
            "type": "patch_applied",
            "source": "langgraph_adapter",
            "patch_id": original_event.metadata.id,
            "target": apply_result.get("target"),
            "apply_result": apply_result,
            "reexecution": reexec_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.bus.publish("langgraph:events", event_data)
        
    async def _publish_failure(
        self, 
        original_event: Event,
        apply_result: Dict[str, Any]
    ):
        """Publish failed patch application."""
        event_data = {
            "type": "patch_failed",
            "source": "langgraph_adapter",
            "patch_id": original_event.metadata.id,
            "error": apply_result.get("error", "Unknown error"),
            "target": apply_result.get("target"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.bus.publish("langgraph:events", event_data)
        
    async def _publish_error(self, original_event: Event, error: str):
        """Publish error event."""
        event_data = {
            "type": "adapter_error",
            "source": "langgraph_adapter",
            "patch_id": original_event.metadata.id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.bus.publish("langgraph:events", event_data)
        
    async def shutdown(self):
        """Gracefully shut down the adapter."""
        logger.info("Shutting down LangGraph adapter")
        self.running = False
        if self.bus:
            await self.bus.close()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "patches_applied": self.patches_applied,
            "patches_in_ledger": len(self.patch_ledger),
            "status": "running" if self.running else "stopped"
        }


async def main():
    """Run the LangGraph adapter standalone."""
    adapter = LangGraphAdapter()
    
    try:
        await adapter.initialize()
        await adapter.listen_and_apply()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stats = adapter.get_stats()
        logger.info(f"Adapter stats: {stats}")
        await adapter.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    asyncio.run(main())