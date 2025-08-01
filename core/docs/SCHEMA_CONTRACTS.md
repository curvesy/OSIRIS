# 📋 AURA Intelligence Schema Contracts - Canonical Documentation

## **IMMUTABLE CONTRACTS FOR MULTI-AGENT SYSTEMS**

This document defines the **canonical contracts** between agents, infrastructure, and humans in The Collective. These schemas are **immutable interfaces** that ensure system integrity, security, and explainability.

---

## 🔐 **CORE PRINCIPLES**

### **1. Immutability Contract**
- **ALL state modifications MUST return new instances**
- **NO in-place mutations allowed in distributed workflows**
- **Thread-safe by design, race-condition free**

### **2. Cryptographic Integrity Contract**
- **ALL evidence, actions, and decisions MUST be cryptographically signed**
- **Signature algorithms MUST be agile and configurable**
- **Tamper detection MUST be verifiable at any time**

### **3. Temporal Consistency Contract**
- **ALL timestamps MUST use DateTimeField objects**
- **ALL temporal operations MUST be timezone-aware UTC**
- **NO string-based datetime operations allowed**

### **4. Trace Context Contract**
- **ALL schemas MUST include W3C trace context**
- **End-to-end correlation MUST be maintained**
- **OpenTelemetry integration MUST be complete**

### **5. Explainability Contract**
- **ALL decisions MUST include structured rationale**
- **Evidence relationships MUST be first-class**
- **Decision options MUST include scoring and rejection analysis**

---

## 📊 **SCHEMA HIERARCHY**

```
AgentState (Root State Container)
├── DossierEntry[] (Cryptographically Signed Evidence)
│   ├── EvidenceContent (Union of Typed Evidence)
│   │   ├── LogEvidence
│   │   ├── MetricEvidence
│   │   ├── PatternEvidence
│   │   ├── PredictionEvidence
│   │   ├── CorrelationEvidence
│   │   ├── ObservationEvidence
│   │   └── ExternalEvidence
│   └── EvidenceReference (Global Lineage)
├── ActionRecord[] (Cryptographically Signed Actions)
│   ├── ActionIntent (Structured Rationale)
│   └── ActionReference (Global Lineage)
├── DecisionPoint[] (Cryptographically Signed Decisions)
│   ├── DecisionOption[] (Scored Options with Evidence)
│   ├── DecisionCriterion[] (Weighted Criteria)
│   └── Enhanced Explainability
└── Metadata (Versioning, Classification, Retention)
```

---

## 🛡️ **SECURITY CONTRACTS**

### **Cryptographic Signature Requirements**
```python
# REQUIRED on ALL critical entities
class CryptographicContract:
    signing_agent_id: str                    # WHO signed
    agent_public_key: str                    # HOW to verify
    signature_algorithm: SignatureAlgorithm  # WHICH algorithm
    content_signature: str                   # THE signature
    signature_timestamp: DateTimeField       # WHEN signed
    
    # REQUIRED methods
    def sign_content(self, private_key: str) -> str
    def verify_signature(self, private_key: str) -> bool
```

### **Supported Signature Algorithms**
- **HMAC_SHA256** - Default for development/testing
- **RSA_PSS_SHA256** - Enterprise PKI integration
- **ECDSA_P256_SHA256** - Mobile/IoT efficiency
- **ED25519** - Modern high-performance
- **DILITHIUM2** - Post-quantum ready
- **FALCON512** - Post-quantum alternative

---

## 🕐 **TEMPORAL CONTRACTS**

### **DateTime Field Requirements**
```python
# REQUIRED: Use DateTimeField, NOT strings
collection_timestamp: DateTimeField = Field(default_factory=utc_now)
execution_timestamp: DateTimeField = Field(default_factory=utc_now)
decision_timestamp: DateTimeField = Field(default_factory=utc_now)

# FORBIDDEN: String timestamps
# timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())  # ❌
```

### **Temporal Utility Functions**
```python
def utc_now() -> datetime:
    """Get current UTC datetime - USE THIS"""
    return datetime.now(timezone.utc)

def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string for serialization"""
    return dt.isoformat()

def iso_to_datetime(iso_str: str) -> datetime:
    """Convert ISO string to datetime for deserialization"""
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1] + '+00:00'
    return datetime.fromisoformat(iso_str)
```

---

## 🔍 **TRACE CONTEXT CONTRACTS**

### **Required OpenTelemetry Fields**
```python
# REQUIRED on ALL schemas
traceparent: str = Field(..., description="W3C trace context")
tracestate: Optional[str] = Field(None, description="Vendor trace state")
span_id: Optional[str] = Field(None, description="OpenTelemetry span ID")
```

### **Trace Propagation Rules**
1. **MUST** propagate `traceparent` across all agent interactions
2. **MUST** create child spans for all major operations
3. **MUST** include correlation_id in all trace attributes
4. **MUST** record exceptions and errors in spans

---

## 🧠 **EXPLAINABILITY CONTRACTS**

### **Decision Point Requirements**
```python
class DecisionContract:
    # REQUIRED: Structured criteria with weights
    criteria: List[DecisionCriterion]
    
    # REQUIRED: Scored options with evidence
    options: List[DecisionOption]
    
    # REQUIRED: Detailed rationale
    decision_rationale: str
    
    # REQUIRED: Evidence relationships
    supporting_evidence: List[EvidenceReference]
    
    # REQUIRED: Option comparison
    option_comparison: Dict[str, Any]
```

### **Action Intent Requirements**
```python
class ActionIntentContract:
    # REQUIRED: Structured intent
    primary_goal: str
    expected_outcome: str
    success_criteria: List[str]
    
    # REQUIRED: Risk assessment
    risk_level: str
    potential_failures: List[str]
    mitigation_strategies: List[str]
    
    # REQUIRED: Business context
    business_justification: str
    compliance_notes: Optional[str]
```

---

## 🔄 **IMMUTABILITY CONTRACTS**

### **State Update Rules**
```python
# ✅ CORRECT: Pure functional updates
new_state = current_state.with_evidence(evidence, agent_id, private_key)
new_state = current_state.with_action(action, agent_id, private_key)
new_state = current_state.with_decision(decision, agent_id, private_key)

# ❌ FORBIDDEN: In-place mutations
current_state.add_evidence(evidence)  # NEVER DO THIS
current_state.action_log.append(action)  # NEVER DO THIS
```

### **Pydantic Configuration**
```python
class Config:
    # REQUIRED for all state classes
    allow_mutation = False      # Enforce immutability
    validate_assignment = True  # Catch mutation attempts
    use_enum_values = True     # Consistent serialization
```

---

## 🆔 **GLOBAL ID CONTRACTS**

### **Composite ID Format**
```python
# REQUIRED format for global uniqueness
def to_global_id(self) -> str:
    return f"{self.workflow_id}:{self.task_id}:{self.entry_id}"

# Example: "wf_incident_001:task_investigate_002:evidence_log_12345"
```

### **Reference Requirements**
```python
class ReferenceContract:
    workflow_id: str     # Workflow context
    task_id: str         # Task context  
    entry_id: str        # Entity identifier
    agent_id: str        # Creating agent
    event_timestamp: str # Creation time
    schema_version: str  # Schema version
```

---

## 📋 **VALIDATION CONTRACTS**

### **Required Validators**
```python
@validator('confidence', 'source_reliability', 'freshness')
def validate_scores(cls, v):
    if not 0.0 <= v <= 1.0:
        raise ValueError("Score must be between 0.0 and 1.0")
    return v

@validator('content_signature', 'action_signature', 'decision_signature')
def validate_signature_format(cls, v):
    if not v or len(v) < 32:
        raise ValueError("Signature must be at least 32 characters")
    return v
```

---

## 🔧 **INTEGRATION CONTRACTS**

### **LangGraph Integration**
- **MUST** pass immutable state copies between nodes
- **MUST** use pure functional updates only
- **MUST** maintain trace context across graph execution

### **Temporal Integration**
- **MUST** support schema versioning for long-lived workflows
- **MUST** provide migration paths between schema versions
- **MUST** handle backwards compatibility gracefully

---

## ⚠️ **BREAKING CHANGE POLICY**

### **NEVER ALLOWED:**
- Removing required fields
- Changing field types
- Modifying enum values
- Breaking signature formats

### **ALLOWED WITH VERSION BUMP:**
- Adding optional fields
- Adding new enum values
- Adding new evidence types
- Extending metadata

### **MIGRATION REQUIREMENTS:**
- Auto-migration scripts for all version changes
- Backwards compatibility for at least 2 major versions
- Clear deprecation warnings before removal

---

## 🎯 **COMPLIANCE CHECKLIST**

Before deploying any schema changes:

- [ ] All required fields present
- [ ] Cryptographic signatures implemented
- [ ] DateTime fields used (no strings)
- [ ] Trace context included
- [ ] Immutability enforced
- [ ] Validation rules complete
- [ ] Global IDs properly formatted
- [ ] Documentation updated
- [ ] Migration scripts provided
- [ ] Tests passing

---

**These contracts are IMMUTABLE and form the foundation of The Collective's integrity, security, and explainability. Adherence is mandatory for all agents, infrastructure, and human interfaces.**
