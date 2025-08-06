"""
Evolver Agent - Self-Improving AI Component
===========================================
Listens for failures on the Event Bus and generates patches using AI.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...orchestration.bus_redis import create_redis_bus
from ...orchestration.bus_protocol import Event

logger = logging.getLogger(__name__)


class EvolverAgent:
    """
    The Evolver-Agent that makes AURA self-improving.
    
    Responsibilities:
    - Listen for topology failures from Topo-Fuzzer
    - Analyze the failure context
    - Generate patches using AI (Gemini/GPT-4)
    - Publish patches back to the Event Bus
    - Learn from patch success/failure
    """
    
    def __init__(self, agent_id: str = "evolver-1", ai_provider: str = "mock"):
        self.agent_id = agent_id
        self.ai_provider = ai_provider
        self.bus = None
        self.patches_generated = 0
        self.running = False
        
    async def initialize(self):
        """Initialize the agent and connect to Event Bus."""
        self.bus = create_redis_bus()
        if not await self.bus.health_check():
            raise RuntimeError("Event Bus not available")
        logger.info(f"Evolver-Agent {self.agent_id} initialized")
        
    async def listen_and_evolve(self):
        """Main loop: listen for failures and generate patches."""
        self.running = True
        logger.info(f"Evolver-Agent {self.agent_id} starting to listen...")
        
        try:
            async for event in self.bus.subscribe("topo:failures", "evolvers", self.agent_id):
                if not self.running:
                    break
                    
                try:
                    await self._process_failure(event)
                except Exception as e:
                    logger.error(f"Error processing failure: {e}")
                    # Continue processing other events
                    
        except Exception as e:
            logger.error(f"Fatal error in evolver loop: {e}")
        finally:
            self.running = False
            
    async def _process_failure(self, event: Event):
        """Process a single failure event."""
        logger.info(f"Processing failure: {event.metadata.id}")
        
        failure = event.payload
        
        # Analyze the failure
        analysis = self._analyze_failure(failure)
        logger.info(f"Analysis: {analysis['summary']}")
        
        # Generate patch using AI
        patch = await self._generate_patch_with_ai(failure, analysis)
        
        # Validate patch
        if self._validate_patch(patch):
            # Publish patch to Event Bus
            patch_event = {
                "type": "code_patch",
                "source": self.agent_id,
                "target": failure.get("component", "unknown"),
                "failure_id": event.metadata.id,
                "patch": patch,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            patch_id = await self.bus.publish("evolver:patches", patch_event)
            logger.info(f"Published patch: {patch_id}")
            
            self.patches_generated += 1
            
            # Acknowledge the failure
            await self.bus.ack("topo:failures", "evolvers", event.metadata.id)
        else:
            logger.warning("Generated patch failed validation")
            
    def _analyze_failure(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the failure to understand root cause."""
        error_type = failure.get("error", "").lower()
        
        # Pattern matching for common issues
        if "overflow" in error_type:
            return {
                "type": "numerical_instability",
                "summary": "Numerical overflow detected",
                "recommendation": "Add bounds checking or use stable algorithm"
            }
        elif "dimension" in error_type:
            return {
                "type": "dimension_mismatch", 
                "summary": "Dimension handling issue",
                "recommendation": "Add dimension validation or adaptive handling"
            }
        elif "memory" in error_type:
            return {
                "type": "memory_issue",
                "summary": "Memory constraint violation",
                "recommendation": "Implement chunking or streaming approach"
            }
        else:
            return {
                "type": "generic",
                "summary": "General failure detected",
                "recommendation": "Add error handling and logging"
            }
            
    async def _generate_patch_with_ai(
        self, 
        failure: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a patch using AI (mocked for now)."""
        
        if self.ai_provider == "mock":
            # Mock AI response based on analysis
            return self._generate_mock_patch(failure, analysis)
        else:
            # TODO: Integrate real AI providers (Gemini, GPT-4)
            raise NotImplementedError(f"AI provider {self.ai_provider} not implemented")
            
    def _generate_mock_patch(
        self, 
        failure: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a mock patch for testing."""
        
        component = failure.get("component", "unknown")
        error = failure.get("error", "")
        context = failure.get("context", {})
        
        # Generate contextual patch based on error type
        if analysis["type"] == "numerical_instability":
            code = f"""
# Fix for: {error}
def compute_wasserstein_distance(self, x, y):
    # Add numerical stability check
    max_dim = max(x.shape[0], y.shape[0])
    if max_dim > 512:
        # Use approximate algorithm for high dimensions
        logger.info(f"Using approximate Wasserstein for dim={max_dim}")
        return self._approximate_wasserstein(x, y)
    
    # Add overflow protection
    try:
        distance = ot.wasserstein_distance(x, y)
        if np.isnan(distance) or np.isinf(distance):
            logger.warning("Wasserstein computation resulted in inf/nan")
            return self._fallback_distance(x, y)
        return distance
    except Exception as e:
        logger.error(f"Wasserstein failed: {e}")
        return self._fallback_distance(x, y)
"""
        elif analysis["type"] == "dimension_mismatch":
            code = f"""
# Fix for: {error}
def process_input(self, data):
    # Add dimension validation
    expected_dims = {context.get('expected_dims', 'unknown')}
    actual_dims = data.shape[0] if hasattr(data, 'shape') else len(data)
    
    if actual_dims != expected_dims:
        logger.warning(f"Dimension mismatch: expected {expected_dims}, got {actual_dims}")
        # Adaptive handling
        if actual_dims > expected_dims:
            data = self._reduce_dimensions(data, expected_dims)
        else:
            data = self._pad_dimensions(data, expected_dims)
    
    return self._original_process(data)
"""
        else:
            code = f"""
# Generic fix for: {error}
def {component}_safe(self, *args, **kwargs):
    try:
        return self.{component}(*args, **kwargs)
    except Exception as e:
        logger.error(f"{component} failed: {e}")
        # Fallback behavior
        return self._safe_default_response()
"""
        
        return {
            "summary": f"Fix {analysis['type']} in {component}",
            "description": analysis['recommendation'],
            "code": code.strip(),
            "confidence": 0.85,
            "test_coverage": True,
            "breaking_change": False
        }
        
    def _validate_patch(self, patch: Dict[str, Any]) -> bool:
        """Validate that the patch is safe to apply."""
        # Basic validation
        if not patch.get("code"):
            return False
            
        if patch.get("confidence", 0) < 0.7:
            return False
            
        if patch.get("breaking_change", False):
            logger.warning("Patch contains breaking changes - requires review")
            return False
            
        return True
        
    async def shutdown(self):
        """Gracefully shut down the agent."""
        logger.info(f"Shutting down Evolver-Agent {self.agent_id}")
        self.running = False
        if self.bus:
            await self.bus.close()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "patches_generated": self.patches_generated,
            "status": "running" if self.running else "stopped"
        }


async def main():
    """Run the Evolver-Agent standalone."""
    agent = EvolverAgent()
    
    try:
        await agent.initialize()
        await agent.listen_and_evolve()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    asyncio.run(main())