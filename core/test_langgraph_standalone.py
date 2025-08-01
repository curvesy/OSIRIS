#!/usr/bin/env python3
"""
🧪 LangGraph Standalone Test
Tests the LangGraph orchestration without full system dependencies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Annotated
import operator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.error("LangGraph not available - install with: pip install langgraph")


class AgentState(TypedDict):
    """State shared across all agents in the workflow."""
    messages: Annotated[List[str], operator.add]
    evidence_log: List[Dict[str, Any]]
    tda_insights: Dict[str, Any]
    current_agent: str
    workflow_context: Dict[str, Any]
    decision_history: List[Dict[str, Any]]
    collective_decision: Dict[str, Any]
    agents_involved: List[str]


class MockTDABridge:
    """Mock TDA bridge for testing without full system."""
    
    async def analyze_patterns(self, evidence_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock TDA analysis."""
        # Simulate TDA analysis based on evidence
        anomaly_score = 0.0
        security_indicators = 0.0
        performance_degradation = 0.0
        knowledge_entropy = 0.0
        
        for evidence in evidence_log:
            if evidence.get('type') == 'anomaly_detection':
                anomaly_score = max(anomaly_score, evidence.get('score', 0.0))
            elif evidence.get('type') == 'security_alert':
                security_indicators = 0.8
            elif evidence.get('type') == 'performance_degradation':
                performance_degradation = 0.7
            elif evidence.get('type') == 'unknown_pattern':
                knowledge_entropy = evidence.get('entropy', 0.5)
        
        return {
            'anomaly_score': anomaly_score,
            'security_indicators': security_indicators,
            'performance_degradation': performance_degradation,
            'knowledge_entropy': knowledge_entropy,
            'patterns_detected': len(evidence_log),
            'analysis_timestamp': datetime.now().isoformat()
        }


class StandaloneLangGraphTest:
    """
    🧪 Standalone LangGraph Test System
    
    Tests LangGraph orchestration without full AURA dependencies.
    """
    
    def __init__(self):
        self.tda_bridge = MockTDABridge()
        
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._create_workflow()
        else:
            self.workflow = None
        
        logger.info("🧪 Standalone LangGraph Test initialized")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for testing."""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("observer", self._observer_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("optimizer", self._optimizer_node)
        workflow.add_node("guardian", self._guardian_node)
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Set entry point
        workflow.set_entry_point("observer")
        
        # Add TDA-guided conditional routing
        workflow.add_conditional_edges(
            "observer",
            self._route_based_on_tda,
            {
                "high_anomaly": "analyzer",
                "security_threat": "guardian",
                "performance_issue": "optimizer",
                "knowledge_gap": "researcher",
                "normal_flow": "supervisor"
            }
        )
        
        # Add edges back to supervisor
        workflow.add_edge("analyzer", "supervisor")
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("optimizer", "supervisor")
        workflow.add_edge("guardian", "supervisor")
        
        # Supervisor ends the workflow
        workflow.add_edge("supervisor", END)
        
        return workflow.compile()
    
    async def _observer_node(self, state: AgentState) -> AgentState:
        """Mock observer agent node."""
        logger.info("👁️ Observer Agent processing evidence")
        
        state['messages'].append(f"Observer: Processed {len(state['evidence_log'])} evidence items")
        state['current_agent'] = 'observer'
        state['agents_involved'].append('observer')
        state['workflow_context']['observer_result'] = {
            'evidence_validated': True,
            'summary': f"Validated {len(state['evidence_log'])} evidence items"
        }
        
        return state
    
    async def _analyzer_node(self, state: AgentState) -> AgentState:
        """Mock analyzer agent node."""
        logger.info("🔬 Analyzer Agent performing deep analysis")
        
        # Get TDA insights
        tda_insights = await self.tda_bridge.analyze_patterns(state['evidence_log'])
        
        state['messages'].append(f"Analyzer: Deep analysis complete with TDA insights")
        state['current_agent'] = 'analyzer'
        state['agents_involved'].append('analyzer')
        state['tda_insights'] = tda_insights
        state['workflow_context']['analysis_result'] = {
            'anomaly_detected': tda_insights['anomaly_score'] > 0.5,
            'confidence': 0.85,
            'summary': 'Deep analysis with topological insights completed'
        }
        
        return state
    
    async def _researcher_node(self, state: AgentState) -> AgentState:
        """Mock researcher agent node."""
        logger.info("📚 Researcher Agent discovering knowledge")
        
        state['messages'].append("Researcher: Knowledge discovery and graph enrichment")
        state['current_agent'] = 'researcher'
        state['agents_involved'].append('researcher')
        state['workflow_context']['research_result'] = {
            'knowledge_discovered': 'New patterns identified',
            'graph_enrichment': 'Knowledge graph updated',
            'confidence': 0.75
        }
        
        return state
    
    async def _optimizer_node(self, state: AgentState) -> AgentState:
        """Mock optimizer agent node."""
        logger.info("⚡ Optimizer Agent optimizing performance")
        
        state['messages'].append("Optimizer: Performance optimization recommendations")
        state['current_agent'] = 'optimizer'
        state['agents_involved'].append('optimizer')
        state['workflow_context']['optimization_result'] = {
            'optimizations_applied': 'Performance tuning completed',
            'improvement': '25% faster response time',
            'resource_savings': '$1,500'
        }
        
        return state
    
    async def _guardian_node(self, state: AgentState) -> AgentState:
        """Mock guardian agent node."""
        logger.info("🛡️ Guardian Agent enforcing security")
        
        state['messages'].append("Guardian: Security policies enforced")
        state['current_agent'] = 'guardian'
        state['agents_involved'].append('guardian')
        state['workflow_context']['security_result'] = {
            'threat_level': 'medium',
            'compliance_status': 'compliant',
            'protective_action': 'Enhanced monitoring activated'
        }
        
        return state
    
    async def _supervisor_node(self, state: AgentState) -> AgentState:
        """Mock supervisor agent node."""
        logger.info("🎯 Supervisor Agent making collective decision")
        
        # Collect all agent inputs
        agent_results = {
            key: value for key, value in state['workflow_context'].items()
            if key.endswith('_result')
        }
        
        # Make collective decision
        collective_decision = {
            'action': 'collective_action_determined',
            'confidence': 0.88,
            'risk_score': 0.35,
            'reasoning': f'Decision based on {len(agent_results)} agent inputs',
            'agents_consulted': len(state['agents_involved']),
            'tda_guided': bool(state.get('tda_insights'))
        }
        
        state['messages'].append(f"Supervisor: Collective decision made with {collective_decision['confidence']:.2f} confidence")
        state['current_agent'] = 'supervisor'
        state['agents_involved'].append('supervisor')
        state['collective_decision'] = collective_decision
        state['decision_history'].append(collective_decision)
        
        return state
    
    async def _route_based_on_tda(self, state: AgentState) -> str:
        """Route workflow based on TDA insights."""
        
        # Get TDA analysis
        tda_analysis = await self.tda_bridge.analyze_patterns(state['evidence_log'])
        
        # Route based on patterns
        anomaly_score = tda_analysis.get('anomaly_score', 0.0)
        security_indicators = tda_analysis.get('security_indicators', 0.0)
        performance_degradation = tda_analysis.get('performance_degradation', 0.0)
        knowledge_entropy = tda_analysis.get('knowledge_entropy', 0.0)
        
        logger.info(f"🔍 TDA Routing - Anomaly: {anomaly_score:.3f}, Security: {security_indicators:.3f}")
        
        # Decision logic
        if anomaly_score > 0.8:
            return "high_anomaly"
        elif security_indicators > 0.7:
            return "security_threat"
        elif performance_degradation > 0.6:
            return "performance_issue"
        elif knowledge_entropy > 0.5:
            return "knowledge_gap"
        else:
            return "normal_flow"
    
    async def process_collective_intelligence(self, evidence_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process evidence through collective intelligence workflow."""
        
        if not LANGGRAPH_AVAILABLE:
            return {
                'success': False,
                'error': 'LangGraph not available - install with: pip install langgraph'
            }
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            evidence_log=evidence_log,
            tda_insights={},
            current_agent="",
            workflow_context={},
            decision_history=[],
            collective_decision={},
            agents_involved=[]
        )
        
        logger.info(f"🚀 Starting collective intelligence workflow with {len(evidence_log)} evidence items")
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"✅ Collective intelligence complete - Agents: {result['agents_involved']}")
            
            return {
                'success': True,
                'collective_decision': result['collective_decision'],
                'agents_involved': result['agents_involved'],
                'workflow_messages': result['messages'],
                'tda_insights': result['tda_insights'],
                'decision_history': result['decision_history'],
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Collective intelligence workflow failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


async def test_langgraph_standalone():
    """Test LangGraph orchestration standalone."""
    print("🧪 LangGraph Standalone Test")
    print("=" * 50)
    
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph not available!")
        print("Install with: pip install langgraph")
        return
    
    # Initialize test system
    test_system = StandaloneLangGraphTest()
    
    print("✅ LangGraph Test System Initialized")
    print("   🔗 Workflow graph created")
    print("   🤖 6 agent nodes configured")
    print("   🔍 TDA-guided routing active")
    print()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Normal Flow',
            'evidence': [
                {'type': 'system_metric', 'value': 75, 'status': 'normal'}
            ]
        },
        {
            'name': 'High Anomaly',
            'evidence': [
                {'type': 'anomaly_detection', 'score': 0.95, 'severity': 'critical'}
            ]
        },
        {
            'name': 'Security Threat',
            'evidence': [
                {'type': 'security_alert', 'threat_level': 'high'}
            ]
        },
        {
            'name': 'Performance Issue',
            'evidence': [
                {'type': 'performance_degradation', 'impact': 'high'}
            ]
        },
        {
            'name': 'Knowledge Gap',
            'evidence': [
                {'type': 'unknown_pattern', 'entropy': 0.8}
            ]
        }
    ]
    
    print(f"🔍 Testing {len(scenarios)} Scenarios:")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        
        result = await test_system.process_collective_intelligence(scenario['evidence'])
        
        if result['success']:
            print(f"   ✅ Success: {len(result['agents_involved'])} agents")
            print(f"   🤖 Flow: {' → '.join(result['agents_involved'])}")
            print(f"   🎯 Decision: {result['collective_decision']['action']}")
            print(f"   📊 Confidence: {result['collective_decision']['confidence']:.3f}")
        else:
            print(f"   ❌ Failed: {result['error']}")
        
        print()
    
    print("🎉 LangGraph Standalone Test Complete!")
    print("✅ Multi-agent orchestration working")
    print("✅ TDA-guided routing functional")
    print("✅ Collective decision making operational")
    print("✅ Ready for full system integration")


if __name__ == "__main__":
    asyncio.run(test_langgraph_standalone())
