"""
🏷️ Schema Enums - Unified Domain Taxonomy

Centralized enums for all schema types with:
- Consistent naming and values
- Lifecycle and version tracking
- Validation support
- Documentation and examples

Single source of truth for all enumerated values.
"""

from enum import Enum
from typing import Dict, List


# ============================================================================
# CORE WORKFLOW ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"
    
    @classmethod
    def get_active_statuses(cls) -> List['TaskStatus']:
        """Get statuses that indicate active work."""
        return [cls.PENDING, cls.IN_PROGRESS, cls.WAITING_FOR_INPUT, cls.WAITING_FOR_APPROVAL]
    
    @classmethod
    def get_terminal_statuses(cls) -> List['TaskStatus']:
        """Get statuses that indicate workflow completion."""
        return [cls.COMPLETED, cls.FAILED, cls.CANCELLED, cls.TIMEOUT]
    
    def is_active(self) -> bool:
        """Check if this status indicates active work."""
        return self in self.get_active_statuses()
    
    def is_terminal(self) -> bool:
        """Check if this status indicates completion."""
        return self in self.get_terminal_statuses()


class Priority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    
    def get_numeric_value(self) -> int:
        """Get numeric priority value for sorting."""
        priority_values = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.NORMAL: 2,
            Priority.LOW: 1
        }
        return priority_values[self]


class Urgency(str, Enum):
    """Task urgency levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0
    
    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """Convert numeric confidence score to level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH
    
    def get_min_score(self) -> float:
        """Get minimum score for this confidence level."""
        score_ranges = {
            ConfidenceLevel.VERY_LOW: 0.0,
            ConfidenceLevel.LOW: 0.2,
            ConfidenceLevel.MEDIUM: 0.4,
            ConfidenceLevel.HIGH: 0.6,
            ConfidenceLevel.VERY_HIGH: 0.8
        }
        return score_ranges[self]


# ============================================================================
# EVIDENCE ENUMS
# ============================================================================

class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""
    OBSERVATION = "observation"
    METRIC = "metric"
    LOG_ENTRY = "log_entry"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    HISTORICAL = "historical"
    EXTERNAL = "external"
    SYNTHETIC = "synthetic"
    HUMAN_INPUT = "human_input"
    
    def get_category(self) -> str:
        """Get the high-level category for this evidence type."""
        categories = {
            EvidenceType.OBSERVATION: "direct",
            EvidenceType.METRIC: "quantitative",
            EvidenceType.LOG_ENTRY: "direct",
            EvidenceType.PATTERN: "analytical",
            EvidenceType.CORRELATION: "analytical",
            EvidenceType.PREDICTION: "analytical",
            EvidenceType.HISTORICAL: "contextual",
            EvidenceType.EXTERNAL: "contextual",
            EvidenceType.SYNTHETIC: "generated",
            EvidenceType.HUMAN_INPUT: "expert"
        }
        return categories.get(self, "unknown")


class EvidenceQuality(str, Enum):
    """Quality assessment for evidence."""
    EXCELLENT = "excellent"    # 0.9 - 1.0
    GOOD = "good"             # 0.7 - 0.9
    FAIR = "fair"             # 0.5 - 0.7
    POOR = "poor"             # 0.3 - 0.5
    UNRELIABLE = "unreliable" # 0.0 - 0.3


# ============================================================================
# ACTION ENUMS
# ============================================================================

class ActionCategory(str, Enum):
    """High-level action categories."""
    INFRASTRUCTURE = "infrastructure"
    COMMUNICATION = "communication"
    INVESTIGATION = "investigation"
    REMEDIATION = "remediation"
    ESCALATION = "escalation"
    LEARNING = "learning"
    MONITORING = "monitoring"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    TESTING = "testing"


class ActionType(str, Enum):
    """Structured taxonomy of agent actions."""
    # Infrastructure Actions
    RESTART_K8S_POD = "restart_k8s_pod"
    SCALE_DEPLOYMENT = "scale_deployment"
    UPDATE_CONFIG = "update_config"
    RESTART_SERVICE = "restart_service"
    DEPLOY_HOTFIX = "deploy_hotfix"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    
    # Communication Actions
    NOTIFY_STAKEHOLDER = "notify_stakeholder"
    SEND_ALERT = "send_alert"
    CREATE_TICKET = "create_ticket"
    UPDATE_STATUS = "update_status"
    SEND_EMAIL = "send_email"
    POST_SLACK_MESSAGE = "post_slack_message"
    
    # Investigation Actions
    QUERY_LOGS = "query_logs"
    ANALYZE_METRICS = "analyze_metrics"
    RUN_DIAGNOSTIC = "run_diagnostic"
    COLLECT_EVIDENCE = "collect_evidence"
    TRACE_REQUEST = "trace_request"
    PROFILE_PERFORMANCE = "profile_performance"
    
    # Remediation Actions
    APPLY_FIX = "apply_fix"
    ROLLBACK_CHANGE = "rollback_change"
    ISOLATE_COMPONENT = "isolate_component"
    DRAIN_TRAFFIC = "drain_traffic"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION = "reset_connection"
    
    # Escalation Actions
    ESCALATE_INCIDENT = "escalate_incident"
    REQUEST_APPROVAL = "request_approval"
    HANDOFF_TO_HUMAN = "handoff_to_human"
    TRIGGER_RUNBOOK = "trigger_runbook"
    CALL_ONCALL = "call_oncall"
    CREATE_WAR_ROOM = "create_war_room"
    
    # Learning Actions
    UPDATE_MODEL = "update_model"
    RECORD_PATTERN = "record_pattern"
    CREATE_RULE = "create_rule"
    VALIDATE_HYPOTHESIS = "validate_hypothesis"
    UPDATE_KNOWLEDGE_BASE = "update_knowledge_base"
    TRAIN_CLASSIFIER = "train_classifier"
    
    # Security Actions
    BLOCK_IP = "block_ip"
    REVOKE_ACCESS = "revoke_access"
    ROTATE_CREDENTIALS = "rotate_credentials"
    ENABLE_MFA = "enable_mfa"
    QUARANTINE_SYSTEM = "quarantine_system"
    AUDIT_PERMISSIONS = "audit_permissions"
    
    # Testing Actions
    RUN_HEALTH_CHECK = "run_health_check"
    EXECUTE_TEST_SUITE = "execute_test_suite"
    VALIDATE_CONFIGURATION = "validate_configuration"
    BENCHMARK_PERFORMANCE = "benchmark_performance"
    CHAOS_ENGINEERING = "chaos_engineering"
    LOAD_TEST = "load_test"


class ActionResult(str, Enum):
    """Possible outcomes of agent actions."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRY_NEEDED = "retry_needed"
    PENDING_APPROVAL = "pending_approval"
    BLOCKED = "blocked"
    
    def is_successful(self) -> bool:
        """Check if this result indicates success."""
        return self in [ActionResult.SUCCESS, ActionResult.PARTIAL_SUCCESS]
    
    def requires_retry(self) -> bool:
        """Check if this result indicates retry is needed."""
        return self in [ActionResult.TIMEOUT, ActionResult.RETRY_NEEDED]


class RiskLevel(str, Enum):
    """Risk levels for actions and decisions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    
    def get_numeric_value(self) -> int:
        """Get numeric risk value for comparison."""
        risk_values = {
            RiskLevel.CRITICAL: 5,
            RiskLevel.HIGH: 4,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 2,
            RiskLevel.MINIMAL: 1
        }
        return risk_values[self]


# ============================================================================
# DECISION ENUMS
# ============================================================================

class DecisionType(str, Enum):
    """Types of decisions that can be made."""
    BINARY = "binary"              # Yes/No decisions
    MULTIPLE_CHOICE = "multiple_choice"  # Select one from many
    RANKING = "ranking"            # Order options by preference
    SCORING = "scoring"            # Assign scores to options
    APPROVAL = "approval"          # Approve/reject/defer
    ROUTING = "routing"            # Route to different paths


class DecisionMethod(str, Enum):
    """Methods used for decision making."""
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    HUMAN_EXPERT = "human_expert"
    CONSENSUS = "consensus"
    WEIGHTED_SCORING = "weighted_scoring"
    COST_BENEFIT = "cost_benefit"
    RISK_ANALYSIS = "risk_analysis"
    MULTI_CRITERIA = "multi_criteria"


# ============================================================================
# CRYPTOGRAPHIC ENUMS
# ============================================================================

class SignatureAlgorithm(str, Enum):
    """Supported cryptographic signature algorithms."""
    HMAC_SHA256 = "hmac_sha256"
    RSA_PSS_SHA256 = "rsa_pss_sha256"
    ECDSA_P256_SHA256 = "ecdsa_p256_sha256"
    ED25519 = "ed25519"
    # Post-quantum algorithms
    DILITHIUM2 = "dilithium2"
    FALCON512 = "falcon512"
    
    def is_post_quantum(self) -> bool:
        """Check if this is a post-quantum algorithm."""
        return self in [SignatureAlgorithm.DILITHIUM2, SignatureAlgorithm.FALCON512]
    
    def get_key_size_bits(self) -> int:
        """Get typical key size in bits."""
        key_sizes = {
            SignatureAlgorithm.HMAC_SHA256: 256,
            SignatureAlgorithm.RSA_PSS_SHA256: 2048,
            SignatureAlgorithm.ECDSA_P256_SHA256: 256,
            SignatureAlgorithm.ED25519: 256,
            SignatureAlgorithm.DILITHIUM2: 1312,
            SignatureAlgorithm.FALCON512: 897
        }
        return key_sizes.get(self, 256)


class HashAlgorithm(str, Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE3 = "blake3"


# ============================================================================
# CLASSIFICATION ENUMS
# ============================================================================

class SecurityClassification(str, Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"
    
    def get_numeric_level(self) -> int:
        """Get numeric classification level."""
        levels = {
            SecurityClassification.PUBLIC: 0,
            SecurityClassification.INTERNAL: 1,
            SecurityClassification.CONFIDENTIAL: 2,
            SecurityClassification.RESTRICTED: 3,
            SecurityClassification.TOP_SECRET: 4
        }
        return levels[self]


class RetentionPolicy(str, Enum):
    """Data retention policies."""
    IMMEDIATE = "immediate"        # Delete immediately after use
    SHORT_TERM = "short_term"      # 30 days
    MEDIUM_TERM = "medium_term"    # 1 year
    LONG_TERM = "long_term"        # 7 years
    PERMANENT = "permanent"        # Keep forever
    LEGAL_HOLD = "legal_hold"      # Keep until legal hold lifted
    
    def get_retention_days(self) -> int:
        """Get retention period in days."""
        retention_days = {
            RetentionPolicy.IMMEDIATE: 0,
            RetentionPolicy.SHORT_TERM: 30,
            RetentionPolicy.MEDIUM_TERM: 365,
            RetentionPolicy.LONG_TERM: 2555,  # 7 years
            RetentionPolicy.PERMANENT: -1,    # Forever
            RetentionPolicy.LEGAL_HOLD: -1    # Until lifted
        }
        return retention_days[self]


# ============================================================================
# AGENT ROLE ENUMS
# ============================================================================

class AgentRole(str, Enum):
    """Standard agent roles in The Collective."""
    OBSERVER = "observer"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    ROUTER = "router"
    CONSENSUS = "consensus"
    SUPERVISOR = "supervisor"
    HUMAN = "human"
    SYSTEM = "system"


class AgentCapability(str, Enum):
    """Agent capabilities for routing and discovery."""
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    SECURITY = "security"


# ============================================================================
# TAXONOMY MAPPINGS
# ============================================================================

# Action type to category mapping
ACTION_CATEGORIES: Dict[ActionType, ActionCategory] = {
    # Infrastructure
    ActionType.RESTART_K8S_POD: ActionCategory.INFRASTRUCTURE,
    ActionType.SCALE_DEPLOYMENT: ActionCategory.INFRASTRUCTURE,
    ActionType.UPDATE_CONFIG: ActionCategory.INFRASTRUCTURE,
    ActionType.RESTART_SERVICE: ActionCategory.INFRASTRUCTURE,
    ActionType.DEPLOY_HOTFIX: ActionCategory.INFRASTRUCTURE,
    ActionType.ROLLBACK_DEPLOYMENT: ActionCategory.INFRASTRUCTURE,
    
    # Communication
    ActionType.NOTIFY_STAKEHOLDER: ActionCategory.COMMUNICATION,
    ActionType.SEND_ALERT: ActionCategory.COMMUNICATION,
    ActionType.CREATE_TICKET: ActionCategory.COMMUNICATION,
    ActionType.UPDATE_STATUS: ActionCategory.COMMUNICATION,
    ActionType.SEND_EMAIL: ActionCategory.COMMUNICATION,
    ActionType.POST_SLACK_MESSAGE: ActionCategory.COMMUNICATION,
    
    # Investigation
    ActionType.QUERY_LOGS: ActionCategory.INVESTIGATION,
    ActionType.ANALYZE_METRICS: ActionCategory.INVESTIGATION,
    ActionType.RUN_DIAGNOSTIC: ActionCategory.INVESTIGATION,
    ActionType.COLLECT_EVIDENCE: ActionCategory.INVESTIGATION,
    ActionType.TRACE_REQUEST: ActionCategory.INVESTIGATION,
    ActionType.PROFILE_PERFORMANCE: ActionCategory.INVESTIGATION,
    
    # Remediation
    ActionType.APPLY_FIX: ActionCategory.REMEDIATION,
    ActionType.ROLLBACK_CHANGE: ActionCategory.REMEDIATION,
    ActionType.ISOLATE_COMPONENT: ActionCategory.REMEDIATION,
    ActionType.DRAIN_TRAFFIC: ActionCategory.REMEDIATION,
    ActionType.CLEAR_CACHE: ActionCategory.REMEDIATION,
    ActionType.RESET_CONNECTION: ActionCategory.REMEDIATION,
    
    # Escalation
    ActionType.ESCALATE_INCIDENT: ActionCategory.ESCALATION,
    ActionType.REQUEST_APPROVAL: ActionCategory.ESCALATION,
    ActionType.HANDOFF_TO_HUMAN: ActionCategory.ESCALATION,
    ActionType.TRIGGER_RUNBOOK: ActionCategory.ESCALATION,
    ActionType.CALL_ONCALL: ActionCategory.ESCALATION,
    ActionType.CREATE_WAR_ROOM: ActionCategory.ESCALATION,
    
    # Learning
    ActionType.UPDATE_MODEL: ActionCategory.LEARNING,
    ActionType.RECORD_PATTERN: ActionCategory.LEARNING,
    ActionType.CREATE_RULE: ActionCategory.LEARNING,
    ActionType.VALIDATE_HYPOTHESIS: ActionCategory.LEARNING,
    ActionType.UPDATE_KNOWLEDGE_BASE: ActionCategory.LEARNING,
    ActionType.TRAIN_CLASSIFIER: ActionCategory.LEARNING,
    
    # Security
    ActionType.BLOCK_IP: ActionCategory.SECURITY,
    ActionType.REVOKE_ACCESS: ActionCategory.SECURITY,
    ActionType.ROTATE_CREDENTIALS: ActionCategory.SECURITY,
    ActionType.ENABLE_MFA: ActionCategory.SECURITY,
    ActionType.QUARANTINE_SYSTEM: ActionCategory.SECURITY,
    ActionType.AUDIT_PERMISSIONS: ActionCategory.SECURITY,
    
    # Testing
    ActionType.RUN_HEALTH_CHECK: ActionCategory.TESTING,
    ActionType.EXECUTE_TEST_SUITE: ActionCategory.TESTING,
    ActionType.VALIDATE_CONFIGURATION: ActionCategory.TESTING,
    ActionType.BENCHMARK_PERFORMANCE: ActionCategory.TESTING,
    ActionType.CHAOS_ENGINEERING: ActionCategory.TESTING,
    ActionType.LOAD_TEST: ActionCategory.TESTING,
}


def get_action_category(action_type: ActionType) -> ActionCategory:
    """Get the category for an action type."""
    return ACTION_CATEGORIES.get(action_type, ActionCategory.INFRASTRUCTURE)


# Export all enums
__all__ = [
    # Core workflow
    'TaskStatus', 'Priority', 'Urgency', 'ConfidenceLevel',
    
    # Evidence
    'EvidenceType', 'EvidenceQuality',
    
    # Actions
    'ActionCategory', 'ActionType', 'ActionResult', 'RiskLevel',
    
    # Decisions
    'DecisionType', 'DecisionMethod',
    
    # Cryptography
    'SignatureAlgorithm', 'HashAlgorithm',
    
    # Classification
    'SecurityClassification', 'RetentionPolicy',
    
    # Agents
    'AgentRole', 'AgentCapability',
    
    # Utilities
    'ACTION_CATEGORIES', 'get_action_category'
]
