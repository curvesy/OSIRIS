#!/usr/bin/env python3
"""
🎼 Collective Intelligence Demo - Complete System Integration

Professional demonstration of the complete collective intelligence system.
Shows the full Observer → Supervisor → Analyst → Supervisor → Executor workflow.

This is the culmination of your proven foundation + cutting-edge LangGraph + LangMem.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys

# Add the source directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the complete collective intelligence system
try:
    from aura_intelligence.orchestration import CollectiveWorkflow
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    logger.info("Make sure you're running from the correct directory")
    sys.exit(1)


class CollectiveIntelligenceDemo:
    """
    Professional demonstration of collective intelligence capabilities.
    
    Shows the complete workflow:
    1. Event ingestion
    2. Observer processing (your proven foundation)
    3. Supervisor routing with context engineering
    4. Analyst pattern analysis with risk scoring
    5. Supervisor decision making
    6. Executor action execution
    7. Memory storage and learning
    """
    
    def __init__(self):
        self.demo_config = {
            "langmem_api_key": None,  # Optional - will use fallback mode
            "enable_persistence": True,
            "enable_human_loop": False,  # Disable for demo
            "max_context_entries": 5,
            "context_confidence_threshold": 0.3,
            "db_path": "demo_collective_intelligence.db"
        }
        
        self.workflow = None
        
        logger.info("🎼 Collective Intelligence Demo initialized")
    
    async def run_complete_demo(self):
        """Run the complete collective intelligence demonstration."""
        
        logger.info("🎼 Starting Collective Intelligence Demo")
        logger.info("=" * 80)
        
        try:
            # Step 1: Initialize the collective intelligence system
            await self._initialize_system()
            
            # Step 2: Run different event scenarios
            await self._demo_critical_event()
            await self._demo_normal_event()
            await self._demo_complex_event()
            
            # Step 3: Show system analytics
            await self._show_system_analytics()
            
            # Step 4: Demonstrate memory learning
            await self._demo_memory_learning()
            
            logger.info("=" * 80)
            logger.info("✅ Collective Intelligence Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.workflow:
                await self.workflow.shutdown()
    
    async def _initialize_system(self):
        """Initialize the collective intelligence system."""
        
        logger.info("🎼 Initializing Collective Intelligence System...")
        
        self.workflow = CollectiveWorkflow(self.demo_config)
        await self.workflow.initialize()
        
        # Show system health
        health = self.workflow.get_system_health()
        logger.info(f"🏥 System Health: {json.dumps(health, indent=2)}")
        
        logger.info("✅ System initialized successfully")
    
    async def _demo_critical_event(self):
        """Demonstrate processing of a critical event."""
        
        logger.info("\n" + "🚨" * 20)
        logger.info("🚨 DEMO: Critical Event Processing")
        logger.info("🚨" * 20)
        
        critical_event = {
            "type": "system_error",
            "source": "production_server",
            "severity": "critical",
            "message": "Database connection pool exhausted - 500 errors/minute",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "error_rate": 500,
                "affected_users": 1000,
                "service": "user_authentication",
                "region": "us-east-1"
            }
        }
        
        logger.info(f"📥 Processing critical event: {critical_event['message']}")
        
        # Process through collective intelligence
        result = await self.workflow.process_event(critical_event)
        
        logger.info("📊 Critical Event Results:")
        logger.info(json.dumps(result, indent=2))
        
        # Expected flow:
        # 1. Observer detects critical system error
        # 2. Supervisor routes to Analyst (high priority)
        # 3. Analyst identifies high risk patterns
        # 4. Supervisor routes to Executor (immediate action)
        # 5. Executor sends critical alerts, escalates to human
        
        logger.info("✅ Critical event processing complete")
    
    async def _demo_normal_event(self):
        """Demonstrate processing of a normal event."""
        
        logger.info("\n" + "📊" * 20)
        logger.info("📊 DEMO: Normal Event Processing")
        logger.info("📊" * 20)
        
        normal_event = {
            "type": "performance_metric",
            "source": "monitoring_system",
            "severity": "info",
            "message": "API response time: 150ms (within normal range)",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "response_time_ms": 150,
                "endpoint": "/api/users",
                "status_code": 200,
                "region": "us-west-2"
            }
        }
        
        logger.info(f"📥 Processing normal event: {normal_event['message']}")
        
        # Process through collective intelligence
        result = await self.workflow.process_event(normal_event)
        
        logger.info("📊 Normal Event Results:")
        logger.info(json.dumps(result, indent=2))
        
        # Expected flow:
        # 1. Observer processes performance metric
        # 2. Supervisor routes to Analyst (standard priority)
        # 3. Analyst identifies low risk patterns
        # 4. Supervisor routes to Executor (maintenance actions)
        # 5. Executor logs observation, updates metrics
        
        logger.info("✅ Normal event processing complete")
    
    async def _demo_complex_event(self):
        """Demonstrate processing of a complex multi-faceted event."""
        
        logger.info("\n" + "🔍" * 20)
        logger.info("🔍 DEMO: Complex Event Processing")
        logger.info("🔍" * 20)
        
        complex_event = {
            "type": "security_alert",
            "source": "security_monitoring",
            "severity": "high",
            "message": "Unusual login patterns detected - potential security breach",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "failed_logins": 50,
                "unique_ips": 25,
                "time_window_minutes": 10,
                "affected_accounts": ["user123", "user456", "user789"],
                "geographic_spread": ["US", "RU", "CN"],
                "attack_pattern": "credential_stuffing"
            }
        }
        
        logger.info(f"📥 Processing complex event: {complex_event['message']}")
        
        # Process through collective intelligence
        result = await self.workflow.process_event(complex_event)
        
        logger.info("📊 Complex Event Results:")
        logger.info(json.dumps(result, indent=2))
        
        # Expected flow:
        # 1. Observer detects security alert
        # 2. Supervisor routes to Analyst (high priority + context)
        # 3. Analyst performs deep pattern analysis
        # 4. Supervisor evaluates risk vs. business impact
        # 5. Executor takes security actions, creates incident
        
        logger.info("✅ Complex event processing complete")
    
    async def _show_system_analytics(self):
        """Show system performance analytics."""
        
        logger.info("\n" + "📈" * 20)
        logger.info("📈 DEMO: System Analytics")
        logger.info("📈" * 20)
        
        # Get system health
        health = self.workflow.get_system_health()
        logger.info("🏥 Current System Health:")
        logger.info(json.dumps(health, indent=2))
        
        # Show graph visualization
        logger.info("\n🎼 Workflow Graph Structure:")
        graph_viz = self.workflow.get_graph_visualization()
        logger.info(graph_viz)
        
        logger.info("✅ System analytics complete")
    
    async def _demo_memory_learning(self):
        """Demonstrate collective memory and learning capabilities."""
        
        logger.info("\n" + "🧠" * 20)
        logger.info("🧠 DEMO: Memory & Learning")
        logger.info("🧠" * 20)
        
        # Process a similar event to show learning
        learning_event = {
            "type": "system_error",
            "source": "production_server",
            "severity": "high",
            "message": "Database connection timeout - similar to previous critical event",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "error_rate": 100,
                "affected_users": 200,
                "service": "user_authentication",
                "region": "us-east-1",
                "pattern_similarity": "high"
            }
        }
        
        logger.info(f"📥 Processing learning event: {learning_event['message']}")
        logger.info("🧠 This should show improved context from previous similar events")
        
        # Process through collective intelligence
        result = await self.workflow.process_event(learning_event)
        
        logger.info("📊 Learning Event Results:")
        logger.info(json.dumps(result, indent=2))
        
        # Expected improvements:
        # 1. Memory manager provides context from previous critical event
        # 2. Context engine enriches state with historical patterns
        # 3. Analyst uses improved context for better analysis
        # 4. Supervisor makes more informed routing decisions
        # 5. Executor takes more targeted actions
        
        logger.info("✅ Memory learning demonstration complete")
    
    def _format_demo_results(self, results: dict) -> str:
        """Format demo results for display."""
        
        formatted = []
        formatted.append("🎯 WORKFLOW RESULTS:")
        formatted.append(f"   Status: {results.get('status', 'unknown')}")
        formatted.append(f"   Workflow ID: {results.get('workflow_id', 'unknown')}")
        formatted.append(f"   Evidence Count: {results.get('evidence_count', 0)}")
        
        insights = results.get('insights', {})
        if insights:
            formatted.append("🔍 INSIGHTS:")
            formatted.append(f"   Risk Score: {insights.get('risk_score', 'N/A')}")
            formatted.append(f"   Risk Level: {insights.get('risk_level', 'N/A')}")
            
            patterns = insights.get('patterns_detected', [])
            if patterns:
                formatted.append(f"   Patterns: {', '.join(patterns)}")
            
            recommendations = insights.get('recommendations', [])
            if recommendations:
                formatted.append(f"   Recommendations: {', '.join(recommendations[:3])}")
        
        return "\n".join(formatted)


async def main():
    """Main demo function."""
    
    print("🎼 AURA Intelligence - Collective Intelligence Demo")
    print("=" * 80)
    print("This demo showcases the complete collective intelligence system:")
    print("• Your proven ObserverAgent foundation")
    print("• New AnalystAgent with advanced pattern analysis")
    print("• New ExecutorAgent with enterprise action execution")
    print("• LangGraph StateGraph with Supervisor routing")
    print("• LangMem collective memory integration")
    print("• Context engineering and state enrichment")
    print("• Professional modular architecture")
    print("=" * 80)
    
    demo = CollectiveIntelligenceDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)
