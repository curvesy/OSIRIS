"""
🧠 Professional Predictive Validator - July 2025 Production Implementation

The "prefrontal cortex" of our digital organism. Implements prospection, risk assessment,
and human-in-the-loop governance for agent actions.

Based on the definitive architectural directive from what.md. Incorporates:
- Hybrid rule-based/LLM approach for reliability
- Caching for performance optimization  
- Accuracy tracking for continuous improvement
- Human-in-the-loop governance for high-risk actions
- Structured JSON output for auditability
"""

import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    # Mock for testing
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
    
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content


@dataclass
class ValidationResult:
    """Structured validation output with all necessary decision factors."""
    predicted_success_probability: float
    prediction_confidence_score: float
    risk_score: float
    predicted_risks: List[Dict[str, str]]
    reasoning_trace: str
    requires_human_approval: bool = False
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class AccuracyTracker:
    """Tracks prediction accuracy for continuous improvement and meta-learning."""
    
    def __init__(self, max_size: int = 1000):
        self.predictions = deque(maxlen=max_size)
        self.accuracy_window = deque(maxlen=100)  # Recent accuracy tracking
    
    def log_prediction(self, prediction_result: ValidationResult, actual_outcome: bool):
        """Logs a prediction and its eventual outcome for accuracy tracking."""
        predicted_success = prediction_result.predicted_success_probability > 0.5
        
        self.predictions.append({
            "predicted_success": predicted_success,
            "actual_success": actual_outcome,
            "confidence": prediction_result.prediction_confidence_score,
            "risk_score": prediction_result.risk_score,
            "timestamp": datetime.now()
        })
        
        # Update accuracy window
        self.accuracy_window.append(predicted_success == actual_outcome)
    
    def get_accuracy(self) -> float:
        """Returns current accuracy over the tracked window."""
        if not self.predictions:
            return 0.5  # Assume 50% for cold start
        
        correct_predictions = sum(1 for p in self.predictions 
                                if p['predicted_success'] == p['actual_success'])
        return correct_predictions / len(self.predictions)
    
    def get_recent_accuracy(self) -> float:
        """Returns accuracy over the most recent predictions."""
        if not self.accuracy_window:
            return 0.5
        return sum(self.accuracy_window) / len(self.accuracy_window)
    
    def get_confidence_calibration(self) -> Dict[str, float]:
        """Returns calibration metrics for confidence scores."""
        if len(self.predictions) < 10:
            return {"calibration_error": 0.0, "sample_size": len(self.predictions)}
        
        # Simple calibration: how well does confidence correlate with accuracy
        high_conf_predictions = [p for p in self.predictions if p['confidence'] > 0.8]
        if not high_conf_predictions:
            return {"calibration_error": 0.0, "sample_size": 0}
        
        high_conf_accuracy = sum(1 for p in high_conf_predictions 
                               if p['predicted_success'] == p['actual_success']) / len(high_conf_predictions)
        
        expected_accuracy = sum(p['confidence'] for p in high_conf_predictions) / len(high_conf_predictions)
        calibration_error = abs(high_conf_accuracy - expected_accuracy)
        
        return {
            "calibration_error": calibration_error,
            "high_confidence_accuracy": high_conf_accuracy,
            "expected_accuracy": expected_accuracy,
            "sample_size": len(high_conf_predictions)
        }


class ProfessionalPredictiveValidator:
    """
    Production-ready predictive validator implementing the hybrid rule-based/LLM approach.
    
    This is the "prefrontal cortex" of our digital organism - providing prospection,
    risk assessment, and governance for all agent actions.
    """
    
    def __init__(self, llm, risk_threshold: float = 0.75, cache_ttl_seconds: int = 3600):
        """
        Initialize the professional predictive validator.
        
        Args:
            llm: Language model for structured validation
            risk_threshold: Threshold above which human approval is required
            cache_ttl_seconds: Time-to-live for cached validation results
        """
        self.llm = llm
        self.risk_threshold = risk_threshold
        self.decision_cache = {}
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.accuracy_tracker = AccuracyTracker()
        self.rulebook = self._load_rulebook()
        self.system_prompt = self._create_validation_prompt()
        
        # Performance metrics
        self.validation_count = 0
        self.cache_hits = 0
        self.rule_hits = 0
        self.llm_calls = 0
    
    def _load_rulebook(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads known high-risk patterns for the hybrid rule-based approach.
        
        In production, this would be loaded from a configuration file or database.
        These rules provide immediate, deterministic risk assessment for known patterns.
        """
        return {
            "critical_service_restart": {
                "risk_score": 0.9,
                "success_probability": 0.2,
                "reason": "Restarting a critical service carries inherent high risk and can impact availability.",
                "mitigation": "Ensure backup systems are active and schedule during maintenance window."
            },
            "database_schema_change": {
                "risk_score": 0.95,
                "success_probability": 0.1,
                "reason": "Database schema changes are high-risk, potentially irreversible operations.",
                "mitigation": "Require database backup, rollback plan, and manual DBA approval."
            },
            "delete_production_data": {
                "risk_score": 1.0,
                "success_probability": 0.05,
                "reason": "Direct deletion of production data is a critical operation with potential for data loss.",
                "mitigation": "Require explicit backup verification and multi-person approval."
            },
            "production_deployment": {
                "risk_score": 0.8,
                "success_probability": 0.3,
                "reason": "Production deployments can introduce bugs or service disruptions.",
                "mitigation": "Ensure staging validation, rollback plan, and monitoring alerts are configured."
            },
            "user_data_access": {
                "risk_score": 0.85,
                "success_probability": 0.25,
                "reason": "Accessing user data requires privacy compliance and audit trails.",
                "mitigation": "Verify authorization, log access, and ensure GDPR/privacy compliance."
            }
        }
    
    def _create_validation_prompt(self) -> str:
        """
        Creates a structured prompt that forces the LLM into rigorous validation mode.
        
        This prompt is designed to extract maximum analytical value from the LLM
        while ensuring consistent, parseable JSON output.
        """
        return """You are a Predictive Validation Engine operating in July 2025. Your sole purpose is to rigorously assess a proposed action and predict its outcome with justifiable confidence.

## ANALYSIS FRAMEWORK ##
You will analyze:
1. Current situation from evidence log
2. Historical precedents from memory context  
3. The specific proposed action
4. Potential failure modes and risks
5. Success probability with confidence intervals

## CRITICAL REQUIREMENTS ##
- Be pessimistic and conservative in risk assessment
- Consider cascading effects and edge cases
- Provide specific, actionable risk mitigations
- Base predictions on evidence, not optimism

## OUTPUT FORMAT ##
Generate ONLY a valid JSON object with this exact schema:

{
  "predicted_success_probability": 0.0-1.0,
  "prediction_confidence_score": 0.0-1.0,
  "risk_score": 0.0-1.0,
  "predicted_risks": [
    {
      "risk": "Specific description of potential failure mode",
      "mitigation": "Concrete action to reduce this risk"
    }
  ],
  "reasoning_trace": "Step-by-step explanation of how you derived these assessments from the provided context"
}

Do not include any text outside this JSON object."""
    
    def _generate_cache_key(self, proposed_action: str, context: List[Dict]) -> str:
        """
        Creates a stable cache key based on action and situational context.
        
        Uses SHA-256 for consistency and collision resistance.
        """
        # Sort context by timestamp for consistent ordering
        sorted_context = sorted(context, key=lambda x: x.get('timestamp', ''))
        context_str = json.dumps(sorted_context, sort_keys=True)
        content = f"{proposed_action}:{context_str}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[ValidationResult]:
        """
        Checks validation cache with TTL expiration.
        
        Returns cached result if valid, None if expired or missing.
        """
        if cache_key not in self.decision_cache:
            return None
        
        cached_entry = self.decision_cache[cache_key]
        if (datetime.now() - cached_entry['timestamp']) > self.cache_ttl:
            # Remove expired entry
            del self.decision_cache[cache_key]
            return None
        
        self.cache_hits += 1
        cached_result = cached_entry['result']
        cached_result.cache_hit = True
        return cached_result
    
    def _check_rulebook(self, proposed_action: str) -> Optional[ValidationResult]:
        """
        Checks action against known high-risk patterns in the rulebook.
        
        This provides immediate, deterministic risk assessment for critical operations.
        """
        action_lower = proposed_action.lower().replace("_", " ").replace("-", " ")
        
        for rule_keyword, details in self.rulebook.items():
            # Check if any rule keyword appears in the proposed action
            rule_words = rule_keyword.replace("_", " ").split()
            if any(word in action_lower for word in rule_words):
                self.rule_hits += 1
                
                return ValidationResult(
                    predicted_success_probability=details['success_probability'],
                    prediction_confidence_score=0.99,  # High confidence in rule-based assessment
                    risk_score=details['risk_score'],
                    predicted_risks=[{
                        "risk": details['reason'],
                        "mitigation": details['mitigation']
                    }],
                    reasoning_trace=f"Action flagged by internal high-risk rulebook: '{rule_keyword}'. "
                                  f"Rule-based assessment indicates high risk requiring manual oversight.",
                    requires_human_approval=True
                )
        
        return None

    async def validate_action(self, proposed_action: str, memory_context: List[Any],
                            current_evidence: List[Dict[str, Any]]) -> ValidationResult:
        """
        Runs the complete professional validation process.

        This is the main entry point that orchestrates the hybrid validation approach:
        1. Rule-based check for known critical risks
        2. Cache lookup for efficiency
        3. LLM-based structured validation for novel situations
        4. Human-in-the-loop governance layer

        Args:
            proposed_action: The action to validate
            memory_context: Historical context from knowledge graph
            current_evidence: Current evidence log

        Returns:
            ValidationResult with complete risk assessment
        """
        self.validation_count += 1
        start_time = datetime.now()

        try:
            # Step 1: Rule-based check for known critical risks
            rule_result = self._check_rulebook(proposed_action)
            if rule_result:
                return rule_result

            # Step 2: Cache lookup for efficiency
            cache_key = self._generate_cache_key(proposed_action, current_evidence)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result

            # Step 3: LLM-based structured validation for novel situations
            validation_result = await self._llm_validate(
                proposed_action, memory_context, current_evidence
            )

            # Step 4: Apply human-in-the-loop governance
            validation_result.requires_human_approval = (
                validation_result.risk_score > self.risk_threshold
            )

            # Step 5: Cache the result for future use
            self.decision_cache[cache_key] = {
                'timestamp': datetime.now(),
                'result': validation_result
            }

            return validation_result

        except Exception as e:
            # Fallback to conservative decision on any error
            return self._fallback_decision(proposed_action, str(e))

    async def _llm_validate(self, proposed_action: str, memory_context: List[Any],
                          current_evidence: List[Dict[str, Any]]) -> ValidationResult:
        """
        Performs LLM-based structured validation for novel situations.

        This handles cases not covered by the rulebook, using the LLM's reasoning
        capabilities to assess risk and predict outcomes.
        """
        self.llm_calls += 1

        # Format the validation prompt with all available context
        prompt_input = self._format_validation_input(
            proposed_action, memory_context, current_evidence
        )

        # Invoke LLM with structured prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt_input)
        ]

        response = await self.llm.ainvoke(messages)

        # Parse and validate the response
        return self._parse_validation_response(response.content)

    def _format_validation_input(self, proposed_action: str, memory_context: List[Any],
                               current_evidence: List[Dict[str, Any]]) -> str:
        """
        Formats all available context into a structured input for the LLM.

        This ensures the LLM has complete situational awareness for accurate validation.
        """
        # Format current evidence
        evidence_str = json.dumps(current_evidence, indent=2, default=str)

        # Format memory context (handle various types)
        memory_str = json.dumps([
            ctx for ctx in memory_context if ctx is not None
        ], indent=2, default=str)

        # Get current system metrics for context
        accuracy_metrics = self.accuracy_tracker.get_confidence_calibration()

        return f"""## CURRENT SITUATION (Evidence Log)
{evidence_str}

## HISTORICAL PRECEDENT (Memory Context)
{memory_str}

## SYSTEM PERFORMANCE CONTEXT
- Current Validator Accuracy: {self.accuracy_tracker.get_recent_accuracy():.2f}
- Calibration Error: {accuracy_metrics.get('calibration_error', 0.0):.3f}
- Total Validations: {self.validation_count}
- Cache Hit Rate: {(self.cache_hits / max(1, self.validation_count)):.2f}

## PROPOSED ACTION FOR VALIDATION
'{proposed_action}'

## VALIDATION TASK
Assess this proposed action considering all context above. Provide conservative risk assessment with specific failure modes and mitigations."""

    def _parse_validation_response(self, response_content: str) -> ValidationResult:
        """
        Safely parses the LLM's JSON output with robust error handling.

        Falls back to conservative assessment if parsing fails.
        """
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")

            json_str = response_content[json_start:json_end]
            parsed = json.loads(json_str)

            # Validate required fields and ranges
            success_prob = max(0.0, min(1.0, parsed.get('predicted_success_probability', 0.3)))
            confidence = max(0.0, min(1.0, parsed.get('prediction_confidence_score', 0.1)))
            risk_score = max(0.0, min(1.0, parsed.get('risk_score', 0.9)))

            # Ensure risks is a list of dicts with required keys
            risks = parsed.get('predicted_risks', [])
            if not isinstance(risks, list):
                risks = []

            validated_risks = []
            for risk in risks:
                if isinstance(risk, dict) and 'risk' in risk and 'mitigation' in risk:
                    validated_risks.append({
                        'risk': str(risk['risk']),
                        'mitigation': str(risk['mitigation'])
                    })

            if not validated_risks:
                validated_risks = [{
                    'risk': 'Unknown risks may exist',
                    'mitigation': 'Proceed with caution and monitoring'
                }]

            return ValidationResult(
                predicted_success_probability=success_prob,
                prediction_confidence_score=confidence,
                risk_score=risk_score,
                predicted_risks=validated_risks,
                reasoning_trace=str(parsed.get('reasoning_trace', 'LLM validation completed'))
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Return conservative fallback on parsing failure
            return ValidationResult(
                predicted_success_probability=0.3,
                prediction_confidence_score=0.1,
                risk_score=0.9,  # High risk on parsing failure
                predicted_risks=[{
                    'risk': f'Failed to parse validation response: {str(e)}',
                    'mitigation': 'Escalate for manual review and validation'
                }],
                reasoning_trace=f'Validation parsing failed: {str(e)}. Using conservative fallback.'
            )

    def _fallback_decision(self, proposed_action: str, error_msg: str) -> ValidationResult:
        """
        Conservative fallback decision when validation fails completely.

        Always errs on the side of caution with high risk assessment.
        """
        return ValidationResult(
            predicted_success_probability=0.2,
            prediction_confidence_score=0.05,
            risk_score=0.95,
            predicted_risks=[{
                'risk': f'Validation system error: {error_msg}',
                'mitigation': 'Manual review required due to validation system failure'
            }],
            reasoning_trace=f'Fallback decision due to validation error: {error_msg}',
            requires_human_approval=True
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Returns comprehensive performance metrics for monitoring and optimization.

        Useful for dashboards, alerting, and continuous improvement.
        """
        return {
            'total_validations': self.validation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.validation_count),
            'rule_hits': self.rule_hits,
            'rule_hit_rate': self.rule_hits / max(1, self.validation_count),
            'llm_calls': self.llm_calls,
            'llm_call_rate': self.llm_calls / max(1, self.validation_count),
            'current_accuracy': self.accuracy_tracker.get_accuracy(),
            'recent_accuracy': self.accuracy_tracker.get_recent_accuracy(),
            'confidence_calibration': self.accuracy_tracker.get_confidence_calibration(),
            'cache_size': len(self.decision_cache),
            'prediction_history_size': len(self.accuracy_tracker.predictions)
        }

    def log_actual_outcome(self, validation_result: ValidationResult, actual_success: bool):
        """
        Logs the actual outcome of a validated action for accuracy tracking.

        This is crucial for continuous improvement and meta-learning.
        """
        self.accuracy_tracker.log_prediction(validation_result, actual_success)

    def clear_cache(self):
        """Clears the validation cache (useful for testing or cache invalidation)."""
        self.decision_cache.clear()
        self.cache_hits = 0


# Factory function for easy instantiation
def create_professional_validator(llm, risk_threshold: float = 0.75,
                                cache_ttl_seconds: int = 3600) -> ProfessionalPredictiveValidator:
    """
    Factory function to create a professional predictive validator.

    Args:
        llm: Language model instance for validation
        risk_threshold: Risk threshold for human approval (0.0-1.0)
        cache_ttl_seconds: Cache time-to-live in seconds

    Returns:
        Configured ProfessionalPredictiveValidator instance
    """
    return ProfessionalPredictiveValidator(
        llm=llm,
        risk_threshold=risk_threshold,
        cache_ttl_seconds=cache_ttl_seconds
    )
