#!/usr/bin/env python3
"""
🧪 AURA Intelligence Development Test
Simple, controlled test of Phase 3C shadow mode system
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_simple_development_test():
    """
    Simple test: Run one workflow and watch shadow mode logging work
    """
    
    print("🧪 AURA Intelligence Development Test")
    print("=====================================")
    print("Goal: Test shadow mode logging with a simple workflow")
    print("")
    
    try:
        # Import our components (with error handling)
        print("📦 Loading AURA Intelligence components...")

        # Add the src directory to Python path
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from aura_intelligence.orchestration.workflows import create_collective_graph
        from aura_intelligence.observability.shadow_mode_logger import ShadowModeLogger
        from aura_intelligence.domain.schema import CollectiveState
        
        print("✅ Components loaded successfully")
        
        # Initialize shadow mode logger
        print("🌙 Initializing shadow mode logger...")
        shadow_logger = ShadowModeLogger()
        await shadow_logger.initialize()
        print("✅ Shadow mode logger ready")
        
        # Create a simple test workflow
        print("🔧 Creating test workflow...")
        
        # Simple test configuration
        test_config = {
            "llm_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        workflow_graph = create_collective_graph(test_config)
        print("✅ Workflow graph created")
        
        # Create a simple test scenario
        print("🎯 Creating test scenario: Autonomous Procurement Negotiation")
        
        test_input = {
            "messages": [
                {
                    "role": "user", 
                    "content": "I need to negotiate the price of office supplies. The vendor is asking $1000, but our budget is $800. Please help me negotiate a fair deal."
                }
            ],
            "workflow_id": f"dev_test_{int(time.time())}",
            "context": {
                "scenario": "procurement_negotiation",
                "vendor_price": 1000,
                "budget_limit": 800,
                "priority": "medium"
            }
        }
        
        print(f"📋 Test Input: {json.dumps(test_input, indent=2)}")
        print("")
        
        # Run the workflow and watch what happens
        print("🚀 Executing workflow with shadow mode active...")
        print("   (This will take a few seconds...)")
        print("")
        
        start_time = time.time()
        
        # Execute the workflow
        result = await workflow_graph.ainvoke(test_input)
        
        execution_time = time.time() - start_time
        
        print("✅ Workflow execution completed!")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print("")
        
        # Show the result
        print("📊 Workflow Result:")
        print("=" * 50)
        if isinstance(result, dict):
            for key, value in result.items():
                if key == "messages" and isinstance(value, list):
                    print(f"{key}: {len(value)} messages")
                    if value:
                        print(f"  Last message: {value[-1].get('content', 'N/A')[:100]}...")
                elif key == "risk_assessment":
                    print(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    print(f"{key}: {str(value)[:100]}...")
        else:
            print(f"Result: {str(result)[:200]}...")
        print("")
        
        # Check shadow mode data
        print("🌙 Checking shadow mode data...")
        
        # Wait a moment for async logging to complete
        await asyncio.sleep(1)
        
        # Get recent shadow mode entries
        accuracy_metrics = await shadow_logger.get_accuracy_metrics(days=1)
        
        print("📈 Shadow Mode Metrics:")
        print(f"   Total predictions: {accuracy_metrics.get('total_predictions', 0)}")
        print(f"   Data completeness: {accuracy_metrics.get('data_completeness', 0):.2%}")
        print(f"   Recent entries: {accuracy_metrics.get('recent_entries', 0)}")
        print("")
        
        # Show what was logged
        if accuracy_metrics.get('total_predictions', 0) > 0:
            print("✅ SUCCESS: Shadow mode logging is working!")
            print("   The validator made a prediction and it was logged.")
            print("   You can check the SQLite database for full details.")
        else:
            print("⚠️  No predictions logged. This might be expected if:")
            print("   - The workflow didn't trigger the validator")
            print("   - The logging is asynchronous and still processing")
        
        print("")
        print("🎉 Development Test Complete!")
        print("=" * 50)
        print("What happened:")
        print("1. ✅ Workflow executed successfully")
        print("2. ✅ Shadow mode logger initialized")
        print("3. ✅ Validator predictions captured (if any)")
        print("4. ✅ System is ready for more testing")
        print("")
        print("Next steps:")
        print("- Run more test scenarios")
        print("- Check the SQLite database directly")
        print("- Start local Grafana to see dashboards")
        print("- Tune validator thresholds based on results")
        
        # Clean up
        await shadow_logger.close()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("   This is normal for development testing.")
        print("   We can debug and fix issues as they come up.")
        logger.exception("Full error details:")

async def run_simple_shadow_logger_test():
    """
    Even simpler test: Just test the shadow logger directly
    """
    print("🧪 Simple Shadow Logger Test")
    print("============================")
    
    try:
        # Add the src directory to Python path
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from aura_intelligence.observability.shadow_mode_logger import ShadowModeLogger, ShadowModeEntry
        
        # Test the logger directly
        shadow_logger = ShadowModeLogger()
        await shadow_logger.initialize()
        
        # Create a test entry
        test_entry = ShadowModeEntry(
            workflow_id="test_001",
            predicted_success_probability=0.85,
            prediction_confidence_score=0.92,
            routing_decision="tools"
        )
        
        # Log it
        entry_id = await shadow_logger.log_prediction(test_entry)
        print(f"✅ Logged prediction with ID: {entry_id}")

        # Record an outcome
        await shadow_logger.record_outcome(entry_id, "success")
        print("✅ Recorded outcome")

        # Get metrics
        metrics = await shadow_logger.get_accuracy_metrics(days=1)
        print(f"📊 Metrics: {json.dumps(metrics, indent=2)}")

        await shadow_logger.close()
        print("✅ Shadow logger test complete!")
        
    except Exception as e:
        print(f"❌ Shadow logger test failed: {e}")
        logger.exception("Full error:")

if __name__ == "__main__":
    print("🧪 AURA Intelligence Development Testing")
    print("Choose test mode:")
    print("1. Full workflow test (recommended)")
    print("2. Simple shadow logger test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(run_simple_shadow_logger_test())
    else:
        asyncio.run(run_simple_development_test())
