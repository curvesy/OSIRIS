"""
🧠 TDA-Guided Coordinator Agent
LangGraph orchestration with topological routing intelligence.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
try:
    from langgraph.prebuilt import ToolNode as ToolExecutor
except ImportError:
    # Fallback for older LangGraph versions
    ToolExecutor = None

from ..agents.tda_analyzer import TDAAnalyzerAgent
from ..agents.real_agents.guardian_agent import RealGuardianAgent
from ..agents.real_agents.optimizer_agent import RealOptimizerAgent
from ..agents.real_agents.researcher_agent import RealResearcherAgent
from ..memory.causal_pattern_store import CausalPatternStore
from ..utils.logger import get_logger


class TDACoordinatorState(TypedDict):
    """State for TDA-guided coordination workflow."""
    
    # Input data
    events: List[Dict[str, Any]]
    request_id: str
    priority: str
    
    # TDA analysis results
    tda_analysis: Optional[Dict[str, Any]]
    topological_pattern: Optional[str]
    anomaly_score: Optional[float]
    routing_decision: Optional[str]
    
    # Agent responses
    agent_responses: Annotated[List[Dict[str, Any]], operator.add]
    
    # Workflow state
    current_step: str
    workflow_status: str
    error_messages: Annotated[List[str], operator.add]
    
    # Results
    final_response: Optional[Dict[str, Any]]
    causal_patterns: Annotated[List[Dict[str, Any]], operator.add]
    
    # Metadata
    processing_start_time: float
    step_timestamps: Dict[str, float]


class TDACoordinator:
    """
    🧠 TDA-Guided Coordinator
    
    Orchestrates the AURA Intelligence system using topological insights:
    - Analyzes events with TDA service
    - Routes tasks based on persistence entropy and Betti numbers
    - Coordinates specialist agents based on topological patterns
    - Manages learning loop with causal pattern storage
    """
    
    def __init__(self, tda_service_url: str = "http://localhost:8080"):
        self.logger = get_logger(__name__)
        
        # Initialize TDA analyzer
        self.tda_analyzer = TDAAnalyzerAgent(tda_service_url)
        
        # Initialize specialist agents
        self.guardian_agent = RealGuardianAgent()
        self.optimizer_agent = RealOptimizerAgent()
        self.researcher_agent = RealResearcherAgent()

        # Initialize causal pattern store
        self.pattern_store = CausalPatternStore()
        
        # Routing logic based on topological patterns
        self.routing_map = {
            'Pattern_7_Failure': self._route_to_guardian,
            'Pattern_9_Cascade_Failure': self._route_to_guardian,
            'Pattern_6_Emerging_Threat': self._route_to_guardian,
            'Pattern_3_Isolation': self._route_to_optimizer,
            'Pattern_4_Gradual_Drift': self._route_to_optimizer,
            'Pattern_5_Feedback_Loop': self._route_to_analyzer_deep_dive,
            'Pattern_8_System_Change': self._route_to_researcher,
            'Pattern_1_Normal': self._route_to_monitor,
            'Pattern_2_Expected_Behavior': self._route_to_monitor
        }
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        self.logger.info("🧠 TDA Coordinator initialized with topological routing")
    
    def _build_workflow(self) -> StateGraph:
        """Build the TDA-guided LangGraph workflow."""
        
        workflow = StateGraph(TDACoordinatorState)
        
        # Add nodes
        workflow.add_node("tda_analysis", self._tda_analysis_node)
        workflow.add_node("routing_decision", self._routing_decision_node)
        workflow.add_node("guardian_response", self._guardian_node)
        workflow.add_node("optimizer_response", self._optimizer_node)
        workflow.add_node("researcher_response", self._researcher_node)
        workflow.add_node("analyzer_deep_dive", self._analyzer_deep_dive_node)
        workflow.add_node("monitor_response", self._monitor_node)
        workflow.add_node("pattern_storage", self._pattern_storage_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        
        # Set entry point
        workflow.set_entry_point("tda_analysis")
        
        # Add edges
        workflow.add_edge("tda_analysis", "routing_decision")
        
        # Conditional routing based on TDA analysis
        workflow.add_conditional_edges(
            "routing_decision",
            self._route_based_on_topology,
            {
                "guardian": "guardian_response",
                "optimizer": "optimizer_response", 
                "researcher": "researcher_response",
                "analyzer": "analyzer_deep_dive",
                "monitor": "monitor_response"
            }
        )
        
        # All specialist responses go to pattern storage
        workflow.add_edge("guardian_response", "pattern_storage")
        workflow.add_edge("optimizer_response", "pattern_storage")
        workflow.add_edge("researcher_response", "pattern_storage")
        workflow.add_edge("analyzer_deep_dive", "pattern_storage")
        workflow.add_edge("monitor_response", "pattern_storage")
        
        # Pattern storage leads to final synthesis
        workflow.add_edge("pattern_storage", "final_synthesis")
        workflow.add_edge("final_synthesis", END)
        
        return workflow.compile()
    
    async def coordinate_response(self, events: List[Dict[str, Any]], 
                                request_id: str = None,
                                priority: str = "medium") -> Dict[str, Any]:
        """
        Coordinate system response using TDA-guided routing.
        
        Args:
            events: List of events to analyze and respond to
            request_id: Optional request identifier
            priority: Request priority level
            
        Returns:
            Coordinated response with topological insights
        """
        start_time = asyncio.get_event_loop().time()
        
        if not request_id:
            request_id = f"tda_coord_{int(start_time)}"
        
        # Initialize state
        initial_state = TDACoordinatorState(
            events=events,
            request_id=request_id,
            priority=priority,
            tda_analysis=None,
            topological_pattern=None,
            anomaly_score=None,
            routing_decision=None,
            agent_responses=[],
            current_step="initialization",
            workflow_status="running",
            error_messages=[],
            final_response=None,
            causal_patterns=[],
            processing_start_time=start_time,
            step_timestamps={}
        )
        
        try:
            self.logger.info(f"🧠 Starting TDA-guided coordination: {request_id}")
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            self.logger.info(
                f"✅ TDA coordination completed: {request_id} "
                f"({processing_time:.1f}ms, pattern: {final_state.get('topological_pattern')})"
            )
            
            return final_state.get('final_response', {})
            
        except Exception as e:
            self.logger.error(f"❌ TDA coordination failed: {request_id} - {e}")
            return self._create_error_response(request_id, str(e))
    
    async def _tda_analysis_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Perform TDA analysis of events."""
        
        state['current_step'] = "tda_analysis"
        state['step_timestamps']['tda_analysis'] = asyncio.get_event_loop().time()
        
        try:
            # Analyze events with TDA
            tda_analysis = await self.tda_analyzer.analyze_events(
                state['events'],
                analysis_type="anomaly_detection"
            )
            
            state['tda_analysis'] = tda_analysis
            state['topological_pattern'] = tda_analysis['tda_results']['pattern_classification']
            state['anomaly_score'] = tda_analysis['tda_results']['anomaly_score']
            
            self.logger.info(
                f"🔍 TDA analysis complete: {state['topological_pattern']} "
                f"(anomaly: {state['anomaly_score']:.3f})"
            )
            
        except Exception as e:
            error_msg = f"TDA analysis failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            
            # Set fallback values
            state['topological_pattern'] = 'Pattern_Unknown'
            state['anomaly_score'] = 0.5
        
        return state
    
    async def _routing_decision_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Make routing decision based on TDA analysis."""
        
        state['current_step'] = "routing_decision"
        state['step_timestamps']['routing_decision'] = asyncio.get_event_loop().time()
        
        pattern = state['topological_pattern']
        anomaly_score = state['anomaly_score']
        
        # Determine routing based on pattern and anomaly score
        if pattern in self.routing_map:
            routing_decision = self._get_agent_name_from_pattern(pattern)
        else:
            # Fallback routing based on anomaly score
            if anomaly_score > 0.7:
                routing_decision = "guardian"
            elif anomaly_score > 0.5:
                routing_decision = "optimizer"
            else:
                routing_decision = "monitor"
        
        state['routing_decision'] = routing_decision
        
        self.logger.info(
            f"🎯 Routing decision: {pattern} → {routing_decision} "
            f"(anomaly: {anomaly_score:.3f})"
        )
        
        return state
    
    def _route_based_on_topology(self, state: TDACoordinatorState) -> str:
        """Route based on topological analysis results."""
        return state['routing_decision']
    
    def _get_agent_name_from_pattern(self, pattern: str) -> str:
        """Get agent name from topological pattern."""
        pattern_to_agent = {
            'Pattern_7_Failure': 'guardian',
            'Pattern_9_Cascade_Failure': 'guardian',
            'Pattern_6_Emerging_Threat': 'guardian',
            'Pattern_3_Isolation': 'optimizer',
            'Pattern_4_Gradual_Drift': 'optimizer',
            'Pattern_5_Feedback_Loop': 'analyzer',
            'Pattern_8_System_Change': 'researcher',
            'Pattern_1_Normal': 'monitor',
            'Pattern_2_Expected_Behavior': 'monitor'
        }
        return pattern_to_agent.get(pattern, 'monitor')
    
    async def _guardian_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Execute Guardian agent response."""
        
        state['current_step'] = "guardian_response"
        state['step_timestamps']['guardian_response'] = asyncio.get_event_loop().time()
        
        try:
            # Convert events for Guardian agent
            evidence_log = self._convert_events_to_evidence(state['events'])
            
            # Execute Guardian response
            guardian_response = await self.guardian_agent.enforce_security(evidence_log)
            
            agent_response = {
                'agent': 'guardian',
                'response': guardian_response,
                'topological_context': {
                    'pattern': state['topological_pattern'],
                    'anomaly_score': state['anomaly_score']
                }
            }
            
            state['agent_responses'].append(agent_response)
            
            self.logger.info(f"🛡️ Guardian response: {guardian_response.threat_level}")
            
        except Exception as e:
            error_msg = f"Guardian agent failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _optimizer_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Execute Optimizer agent response."""
        
        state['current_step'] = "optimizer_response"
        state['step_timestamps']['optimizer_response'] = asyncio.get_event_loop().time()
        
        try:
            evidence_log = self._convert_events_to_evidence(state['events'])
            optimizer_response = await self.optimizer_agent.optimize_performance(evidence_log)
            
            agent_response = {
                'agent': 'optimizer',
                'response': optimizer_response,
                'topological_context': {
                    'pattern': state['topological_pattern'],
                    'anomaly_score': state['anomaly_score']
                }
            }
            
            state['agent_responses'].append(agent_response)
            
            self.logger.info(f"⚡ Optimizer response: {len(optimizer_response.optimizations_applied)} optimizations")
            
        except Exception as e:
            error_msg = f"Optimizer agent failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _researcher_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Execute Researcher agent response."""
        
        state['current_step'] = "researcher_response"
        state['step_timestamps']['researcher_response'] = asyncio.get_event_loop().time()
        
        try:
            evidence_log = self._convert_events_to_evidence(state['events'])
            researcher_response = await self.researcher_agent.research_knowledge_gap(evidence_log)
            
            agent_response = {
                'agent': 'researcher',
                'response': researcher_response,
                'topological_context': {
                    'pattern': state['topological_pattern'],
                    'anomaly_score': state['anomaly_score']
                }
            }
            
            state['agent_responses'].append(agent_response)
            
            self.logger.info(f"📚 Researcher response: {len(researcher_response.knowledge_discovered)} discoveries")
            
        except Exception as e:
            error_msg = f"Researcher agent failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _analyzer_deep_dive_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Execute deep analysis for complex patterns."""
        
        state['current_step'] = "analyzer_deep_dive"
        state['step_timestamps']['analyzer_deep_dive'] = asyncio.get_event_loop().time()
        
        try:
            # Perform deeper TDA analysis
            deep_analysis = await self.tda_analyzer.analyze_events(
                state['events'],
                analysis_type="topology_analysis"
            )
            
            agent_response = {
                'agent': 'analyzer_deep_dive',
                'response': deep_analysis,
                'topological_context': {
                    'pattern': state['topological_pattern'],
                    'anomaly_score': state['anomaly_score']
                }
            }
            
            state['agent_responses'].append(agent_response)
            
            self.logger.info("🔍 Deep analysis completed")
            
        except Exception as e:
            error_msg = f"Deep analysis failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _monitor_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Execute monitoring response for normal patterns."""
        
        state['current_step'] = "monitor_response"
        state['step_timestamps']['monitor_response'] = asyncio.get_event_loop().time()
        
        # Simple monitoring response
        monitor_response = {
            'status': 'monitoring',
            'pattern': state['topological_pattern'],
            'anomaly_score': state['anomaly_score'],
            'action': 'continue_monitoring',
            'next_check': datetime.now().isoformat()
        }
        
        agent_response = {
            'agent': 'monitor',
            'response': monitor_response,
            'topological_context': {
                'pattern': state['topological_pattern'],
                'anomaly_score': state['anomaly_score']
            }
        }
        
        state['agent_responses'].append(agent_response)
        
        self.logger.info("👁️ Monitor response: continue monitoring")
        
        return state
    
    async def _pattern_storage_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Store causal patterns for learning."""
        
        state['current_step'] = "pattern_storage"
        state['step_timestamps']['pattern_storage'] = asyncio.get_event_loop().time()
        
        try:
            # Create causal pattern record
            causal_pattern = {
                'pattern_id': f"{state['topological_pattern']}_{int(asyncio.get_event_loop().time())}",
                'topological_pattern': state['topological_pattern'],
                'anomaly_score': state['anomaly_score'],
                'betti_numbers': state['tda_analysis']['tda_results']['betti_numbers'] if state['tda_analysis'] else [],
                'persistence_entropy': state['tda_analysis']['tda_results']['persistence_entropy'] if state['tda_analysis'] else 0.0,
                'topological_signature': state['tda_analysis']['tda_results']['topological_signature'] if state['tda_analysis'] else '',
                'confidence': state['tda_analysis']['tda_results']['confidence'] if state['tda_analysis'] else 0.0,
                'events_processed': len(state['events']),
                'agent_responses': len(state['agent_responses']),
                'timestamp': datetime.now().isoformat(),
                'request_id': state['request_id']
            }

            # Store in Neo4j knowledge graph
            await self.pattern_store.store_pattern(causal_pattern)

            state['causal_patterns'].append(causal_pattern)

            self.logger.info(f"💾 Pattern stored in knowledge graph: {state['topological_pattern']}")

        except Exception as e:
            error_msg = f"Pattern storage failed: {e}"
            state['error_messages'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _final_synthesis_node(self, state: TDACoordinatorState) -> TDACoordinatorState:
        """Synthesize final response."""
        
        state['current_step'] = "final_synthesis"
        state['step_timestamps']['final_synthesis'] = asyncio.get_event_loop().time()
        state['workflow_status'] = "completed"
        
        processing_time = (asyncio.get_event_loop().time() - state['processing_start_time']) * 1000
        
        final_response = {
            'request_id': state['request_id'],
            'status': 'success',
            'topological_analysis': {
                'pattern': state['topological_pattern'],
                'anomaly_score': state['anomaly_score'],
                'betti_numbers': state['tda_analysis']['tda_results']['betti_numbers'] if state['tda_analysis'] else [],
                'routing_decision': state['routing_decision']
            },
            'agent_responses': state['agent_responses'],
            'causal_patterns': state['causal_patterns'],
            'performance': {
                'total_processing_time_ms': processing_time,
                'step_timestamps': state['step_timestamps'],
                'events_processed': len(state['events'])
            },
            'errors': state['error_messages'],
            'timestamp': datetime.now().isoformat()
        }
        
        state['final_response'] = final_response
        
        self.logger.info(f"✅ Final synthesis complete: {processing_time:.1f}ms")
        
        return state
    
    def _convert_events_to_evidence(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert events to evidence format for agents."""
        return events  # For now, pass through directly
    
    def _create_error_response(self, request_id: str, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'request_id': request_id,
            'status': 'error',
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
