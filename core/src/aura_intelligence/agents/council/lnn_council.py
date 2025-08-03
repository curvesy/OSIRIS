"""
LNN Council Agent for Multi-Agent Systems.

This module provides an LNN-based agent that can participate in
multi-agent councils, leveraging context-aware inference and
Byzantine consensus when needed.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode as ToolExecutor

from ...agents.base import AgentBase, AgentConfig, AgentState
from ...neural.context_integration import ContextAwareLNN
from ...neural.lnn import LNNConfig
from ...neural.memory_hooks import LNNMemoryHooks
from ...resilience import resilient, ResilienceLevel
from ...observability import create_tracer

logger = logging.getLogger(__name__)
tracer = create_tracer("lnn_council_agent")


class VoteType(str, Enum):
    """Types of votes in council decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"


@dataclass
class CouncilTask:
    """Task submitted to the council."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    context: Dict[str, Any]
    priority: int = 5
    deadline: Optional[datetime] = None


@dataclass
class CouncilVote:
    """Vote cast by an agent."""
    agent_id: str
    vote: VoteType
    confidence: float
    reasoning: str
    supporting_evidence: List[Dict[str, Any]]
    timestamp: datetime


class LNNCouncilAgent(AgentBase):
    """LNN agent participating in multi-agent councils."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize LNN with context awareness
        lnn_config = config.get("lnn_config", self._default_lnn_config())
        self.lnn = ContextAwareLNN(
            lnn_config=lnn_config,
            memory_manager=getattr(self, "memory", None),
            knowledge_graph=getattr(self, "graph", None),
            event_producer=getattr(self, "events", None),
            feature_flags=config.get("feature_flags", {})
        )
        
        # Memory hooks for background indexing
        self.memory_hooks = LNNMemoryHooks(
            memory_manager=getattr(self, "memory", None)
        )
        
        # Council-specific configuration
        self.vote_threshold = config.get("vote_threshold", 0.7)
        self.delegation_threshold = config.get("delegation_threshold", 0.3)
        self.expertise_domains = config.get("expertise_domains", ["general"])
        
        # Performance tracking
        self._votes_cast = 0
        self._votes_agreed_with_consensus = 0
        self._average_confidence = 0.0
        
    def _default_lnn_config(self) -> LNNConfig:
        """Default LNN configuration for council agent."""
        return LNNConfig(
            input_size=256,  # Suitable for encoded council tasks
            hidden_sizes=[128, 64],
            output_size=4,  # One per vote type
            time_constant=1.0,
            solver_type="rk4",
            adapt_time_constant=True,
            sparsity=0.7
        )
        
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for council participation."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("lnn_inference", self._lnn_inference)
        workflow.add_node("formulate_vote", self._formulate_vote)
        workflow.add_node("submit_vote", self._submit_vote)
        workflow.add_node("update_memory", self._update_memory)
        
        # Define edges
        workflow.add_edge("analyze_task", "retrieve_context")
        workflow.add_edge("retrieve_context", "lnn_inference")
        workflow.add_edge("lnn_inference", "formulate_vote")
        workflow.add_edge("formulate_vote", "submit_vote")
        workflow.add_edge("submit_vote", "update_memory")
        workflow.add_edge("update_memory", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_task")
        
        return workflow
        
    async def process(self, task: CouncilTask) -> CouncilVote:
        """Process a council task and return a vote."""
        with tracer.start_as_current_span("process_council_task") as span:
            span.set_attribute("task.id", task.task_id)
            span.set_attribute("task.type", task.task_type)
            span.set_attribute("task.priority", task.priority)
            
            # Feature flag for A/B testing
            if self.config.get("use_lnn_inference", True):
                return await self._process_with_lnn(task)
            else:
                # Fall back to legacy processing
                return await self._legacy_process(task)
                
    @resilient(criticality=ResilienceLevel.CRITICAL)
    async def _process_with_lnn(self, task: CouncilTask) -> CouncilVote:
        """Process task using LNN inference."""
        # Initialize state
        initial_state = AgentState(
            messages=[],
            task=task.payload,
            context=task.context,
            metadata={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "priority": task.priority,
                "start_time": datetime.utcnow()
            }
        )
        
        # Run workflow
        result = await self.graph.ainvoke(initial_state)
        
        # Extract vote from result
        vote_data = result.get("vote", {})
        vote = CouncilVote(
            agent_id=self.agent_id,
            vote=vote_data.get("vote_type", VoteType.ABSTAIN),
            confidence=vote_data.get("confidence", 0.0),
            reasoning=vote_data.get("reasoning", ""),
            supporting_evidence=vote_data.get("evidence", []),
            timestamp=datetime.utcnow()
        )
        
        # Update metrics
        self._votes_cast += 1
        self._average_confidence = (
            (self._average_confidence * (self._votes_cast - 1) + vote.confidence) 
            / self._votes_cast
        )
        
        return vote
        
    async def _analyze_task(self, state: AgentState) -> AgentState:
        """Analyze the council task."""
        task = state["task"]
        
        # Extract key features
        analysis = {
            "complexity": self._assess_complexity(task),
            "domain_match": self._check_domain_match(task),
            "urgency": self._assess_urgency(state["metadata"]),
            "risk_level": self._assess_risk(task)
        }
        
        state["analysis"] = analysis
        state["messages"].append({
            "role": "system",
            "content": f"Task analysis complete: {analysis}"
        })
        
        return state
        
    async def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context for decision making."""
        # Query context based on task
        query_context = {
            "entities": self._extract_entities(state["task"]),
            "temporal": {
                "deadline": state["metadata"].get("deadline"),
                "created": state["metadata"].get("start_time")
            }
        }
        
        # Search for similar past decisions
        if self.memory_hooks:
            similar_decisions = await self.memory_hooks.search_similar_decisions(
                query_features=self._encode_task(state["task"]),
                limit=5,
                time_window_hours=168  # Last week
            )
            state["similar_decisions"] = similar_decisions
            
        state["query_context"] = query_context
        state["messages"].append({
            "role": "system",
            "content": f"Retrieved {len(state.get('similar_decisions', []))} similar decisions"
        })
        
        return state
        
    async def _lnn_inference(self, state: AgentState) -> AgentState:
        """Run LNN inference on the task."""
        # Encode task for LNN
        import torch
        task_tensor = torch.tensor(
            self._encode_task(state["task"]),
            dtype=torch.float32
        )
        
        # Run context-aware inference
        result = await self.lnn.context_aware_inference(
            input_data=task_tensor,
            query_context=state.get("query_context")
        )
        
        # Store inference result
        state["lnn_result"] = result
        state["messages"].append({
            "role": "assistant",
            "content": f"LNN inference complete. Confidence: {result['confidence']:.2f}"
        })
        
        # Trigger memory hook
        if self.memory_hooks:
            await self.memory_hooks.on_inference_complete(result)
            
        return state
        
    async def _formulate_vote(self, state: AgentState) -> AgentState:
        """Formulate vote based on LNN inference."""
        lnn_result = state["lnn_result"]
        analysis = state["analysis"]
        
        # Interpret LNN output as vote probabilities
        predictions = lnn_result["predictions"]
        if isinstance(predictions, list) and len(predictions) >= 4:
            vote_probs = {
                VoteType.APPROVE: predictions[0],
                VoteType.REJECT: predictions[1],
                VoteType.ABSTAIN: predictions[2],
                VoteType.DELEGATE: predictions[3]
            }
        else:
            # Fallback interpretation
            confidence = lnn_result["confidence"]
            if confidence > self.vote_threshold:
                vote_probs = {VoteType.APPROVE: confidence, VoteType.REJECT: 1-confidence}
            else:
                vote_probs = {VoteType.ABSTAIN: 1.0}
                
        # Select vote with highest probability
        vote_type = max(vote_probs, key=vote_probs.get)
        confidence = vote_probs[vote_type]
        
        # Check delegation threshold
        if confidence < self.delegation_threshold and analysis["domain_match"] < 0.5:
            vote_type = VoteType.DELEGATE
            
        # Generate reasoning
        reasoning = self._generate_reasoning(
            vote_type=vote_type,
            confidence=confidence,
            analysis=analysis,
            lnn_result=lnn_result,
            similar_decisions=state.get("similar_decisions", [])
        )
        
        # Collect supporting evidence
        evidence = []
        if "similar_patterns" in lnn_result:
            evidence.extend(lnn_result["similar_patterns"])
        if state.get("similar_decisions"):
            evidence.extend([
                {
                    "type": "historical_decision",
                    "decision": d["content"],
                    "relevance": d.get("similarity", 0.0)
                }
                for d in state["similar_decisions"][:3]
            ])
            
        state["vote"] = {
            "vote_type": vote_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence
        }
        
        return state
        
    async def _submit_vote(self, state: AgentState) -> AgentState:
        """Submit vote to the council."""
        vote_data = state["vote"]
        
        # Create formal vote
        vote = CouncilVote(
            agent_id=self.agent_id,
            vote=vote_data["vote_type"],
            confidence=vote_data["confidence"],
            reasoning=vote_data["reasoning"],
            supporting_evidence=vote_data["evidence"],
            timestamp=datetime.utcnow()
        )
        
        # Emit vote event
        if hasattr(self, "events") and self.events:
            await self.events.publish("council.vote", {
                "task_id": state["metadata"]["task_id"],
                "agent_id": self.agent_id,
                "vote": vote.vote.value,
                "confidence": vote.confidence,
                "timestamp": vote.timestamp.isoformat()
            })
            
        state["submitted_vote"] = vote
        state["messages"].append({
            "role": "assistant",
            "content": f"Vote submitted: {vote.vote.value} (confidence: {vote.confidence:.2f})"
        })
        
        return state
        
    async def _update_memory(self, state: AgentState) -> AgentState:
        """Update memory with decision outcome."""
        # Record pattern if high confidence
        if state["vote"]["confidence"] > 0.8:
            pattern = {
                "type": "council_decision",
                "features": self._encode_task(state["task"]),
                "confidence": state["vote"]["confidence"],
                "context": {
                    "task_type": state["metadata"]["task_type"],
                    "vote": state["vote"]["vote_type"].value,
                    "domain_match": state["analysis"]["domain_match"]
                }
            }
            
            if self.memory_hooks:
                await self.memory_hooks.on_pattern_detected(pattern)
                
        return state
        
    async def _legacy_process(self, task: CouncilTask) -> CouncilVote:
        """Legacy processing without LNN."""
        # Simple rule-based voting
        if task.priority > 8:
            vote_type = VoteType.APPROVE
            confidence = 0.9
        elif task.priority < 3:
            vote_type = VoteType.REJECT
            confidence = 0.8
        else:
            vote_type = VoteType.ABSTAIN
            confidence = 0.5
            
        return CouncilVote(
            agent_id=self.agent_id,
            vote=vote_type,
            confidence=confidence,
            reasoning="Legacy rule-based decision",
            supporting_evidence=[],
            timestamp=datetime.utcnow()
        )
        
    def _assess_complexity(self, task: Dict[str, Any]) -> float:
        """Assess task complexity (0-1)."""
        # Simple heuristic based on task size and structure
        complexity = 0.0
        
        # Factor in data size
        task_str = str(task)
        complexity += min(1.0, len(task_str) / 1000)
        
        # Factor in nested structure
        def count_depth(obj, depth=0):
            if isinstance(obj, dict):
                return max([count_depth(v, depth+1) for v in obj.values()] + [depth])
            elif isinstance(obj, list):
                return max([count_depth(v, depth+1) for v in obj] + [depth])
            return depth
            
        depth = count_depth(task)
        complexity += min(1.0, depth / 5)
        
        return min(1.0, complexity / 2)
        
    def _check_domain_match(self, task: Dict[str, Any]) -> float:
        """Check how well task matches agent expertise."""
        task_str = str(task).lower()
        matches = sum(1 for domain in self.expertise_domains if domain in task_str)
        return min(1.0, matches / len(self.expertise_domains))
        
    def _assess_urgency(self, metadata: Dict[str, Any]) -> float:
        """Assess task urgency based on deadline."""
        deadline = metadata.get("deadline")
        if not deadline:
            return 0.5
            
        time_remaining = (deadline - datetime.utcnow()).total_seconds()
        if time_remaining < 300:  # 5 minutes
            return 1.0
        elif time_remaining < 3600:  # 1 hour
            return 0.8
        elif time_remaining < 86400:  # 1 day
            return 0.5
        else:
            return 0.2
            
    def _assess_risk(self, task: Dict[str, Any]) -> float:
        """Assess risk level of the task."""
        # Look for risk indicators
        risk_keywords = ["critical", "urgent", "failure", "error", "security", "breach"]
        task_str = str(task).lower()
        
        risk_score = sum(0.2 for keyword in risk_keywords if keyword in task_str)
        return min(1.0, risk_score)
        
    def _extract_entities(self, task: Dict[str, Any]) -> List[str]:
        """Extract entity IDs from task."""
        entities = []
        
        # Simple extraction - look for ID patterns
        task_str = str(task)
        import re
        
        # UUID pattern
        uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
        entities.extend(re.findall(uuid_pattern, task_str, re.I))
        
        # ID fields
        if isinstance(task, dict):
            for key, value in task.items():
                if "id" in key.lower() and isinstance(value, str):
                    entities.append(value)
                    
        return list(set(entities))[:10]  # Limit to 10
        
    def _encode_task(self, task: Dict[str, Any]) -> List[float]:
        """Encode task as feature vector."""
        # Simplified encoding - in production use proper encoder
        features = []
        
        # Task size
        features.append(min(1.0, len(str(task)) / 1000))
        
        # Key presence
        important_keys = ["priority", "deadline", "type", "action", "target"]
        for key in important_keys:
            features.append(1.0 if key in task else 0.0)
            
        # Numeric values
        if isinstance(task, dict):
            numeric_values = [v for v in task.values() if isinstance(v, (int, float))]
            if numeric_values:
                features.append(sum(numeric_values) / len(numeric_values) / 100)
            else:
                features.append(0.0)
                
        # Pad to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return features
        
    def _generate_reasoning(
        self,
        vote_type: VoteType,
        confidence: float,
        analysis: Dict[str, Any],
        lnn_result: Dict[str, Any],
        similar_decisions: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable reasoning for vote."""
        reasoning_parts = []
        
        # Vote explanation
        if vote_type == VoteType.APPROVE:
            reasoning_parts.append(f"Recommending approval with {confidence:.1%} confidence.")
        elif vote_type == VoteType.REJECT:
            reasoning_parts.append(f"Recommending rejection with {confidence:.1%} confidence.")
        elif vote_type == VoteType.ABSTAIN:
            reasoning_parts.append("Abstaining due to insufficient confidence or information.")
        else:  # DELEGATE
            reasoning_parts.append("Delegating to more suitable agent due to low domain match.")
            
        # Analysis factors
        if analysis["complexity"] > 0.7:
            reasoning_parts.append("Task exhibits high complexity.")
        if analysis["urgency"] > 0.8:
            reasoning_parts.append("High urgency detected.")
        if analysis["risk_level"] > 0.6:
            reasoning_parts.append("Elevated risk factors present.")
            
        # Historical context
        if similar_decisions:
            outcomes = [d["content"].get("vote", {}).get("vote_type") for d in similar_decisions]
            if outcomes:
                most_common = max(set(outcomes), key=outcomes.count)
                reasoning_parts.append(
                    f"Historical precedent suggests {most_common} "
                    f"({outcomes.count(most_common)}/{len(outcomes)} similar cases)."
                )
                
        # Context influence
        if lnn_result.get("context_influence", 0) > 0.5:
            reasoning_parts.append("Decision strongly influenced by historical context.")
            
        return " ".join(reasoning_parts)
        
    async def participate_in_consensus(
        self,
        task: CouncilTask,
        other_votes: List[CouncilVote]
    ) -> Optional[CouncilVote]:
        """Participate in Byzantine consensus if needed."""
        # Check if consensus is required
        if len(other_votes) < 2:
            return None
            
        # Cast our vote
        our_vote = await self.process(task)
        
        # Check agreement with majority
        vote_counts = {}
        for vote in other_votes + [our_vote]:
            vote_counts[vote.vote] = vote_counts.get(vote.vote, 0) + 1
            
        majority_vote = max(vote_counts, key=vote_counts.get)
        if our_vote.vote == majority_vote:
            self._votes_agreed_with_consensus += 1
            
        # If we strongly disagree with emerging consensus, may revise
        if our_vote.vote != majority_vote and our_vote.confidence < 0.6:
            # Consider changing vote
            logger.info(
                f"LNN agent {self.agent_id} considering vote revision. "
                f"Original: {our_vote.vote}, Majority: {majority_vote}"
            )
            
        return our_vote
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        consensus_rate = (
            self._votes_agreed_with_consensus / self._votes_cast 
            if self._votes_cast > 0 else 0.0
        )
        
        return {
            "votes_cast": self._votes_cast,
            "average_confidence": self._average_confidence,
            "consensus_agreement_rate": consensus_rate,
            "expertise_domains": self.expertise_domains,
            "lnn_metrics": self.lnn.lnn.get_metrics().to_dict() if hasattr(self.lnn.lnn, 'get_metrics') else {}
        }