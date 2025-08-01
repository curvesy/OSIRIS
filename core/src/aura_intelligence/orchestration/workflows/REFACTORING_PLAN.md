# 📋 Workflows Modularization Plan

## Current State
- **File**: `workflows.py`
- **Lines**: 1,216
- **Responsibilities**: Mixed (state, nodes, routing, tools, builder, config)

## Target Structure
```
orchestration/workflows/
├── __init__.py              # Public API exports
├── state.py                 # State definitions (CollectiveState)
├── nodes/                   # Individual workflow nodes
│   ├── __init__.py
│   ├── observer.py          # Observer node logic
│   ├── supervisor.py        # Supervisor node logic
│   ├── analyst.py           # Analyst node logic
│   ├── executor.py          # Executor node logic
│   └── error_handler.py     # Error recovery node
├── tools/                   # LangGraph tools
│   ├── __init__.py
│   ├── evidence.py          # Evidence collection tools
│   ├── memory.py            # Memory access tools
│   └── execution.py         # Execution tools
├── routing/                 # Routing logic
│   ├── __init__.py
│   ├── conditions.py        # Routing conditions
│   └── strategies.py        # Routing strategies
├── shadow_mode/             # Shadow mode integration
│   ├── __init__.py
│   ├── logger.py            # Shadow mode logger wrapper
│   └── analytics.py         # Shadow mode analytics
├── builder.py               # Workflow graph builder
├── config.py                # Workflow configuration
└── utils.py                 # Shared utilities
```

## Refactoring Steps

### Step 1: Extract State (50 lines)
- Move `CollectiveState` to `state.py`
- Use our shared `aura_common` for type definitions

### Step 2: Extract Nodes (400 lines)
- Each node becomes a separate module
- Use dependency injection for services
- Apply single responsibility principle

### Step 3: Extract Tools (200 lines)
- Group tools by functionality
- Use `@tool` decorator consistently
- Add proper error handling from `aura_common`

### Step 4: Extract Routing (150 lines)
- Separate routing conditions
- Make strategies configurable
- Use feature flags for A/B testing

### Step 5: Shadow Mode Integration (100 lines)
- Wrap existing shadow logger
- Add correlation ID support
- Enable feature flag control

### Step 6: Builder Pattern (200 lines)
- Clean workflow construction
- Configuration-driven
- Testable components

## Testing Strategy

### Unit Tests
- Test each node in isolation
- Mock dependencies
- Cover error cases

### Integration Tests
- Test node interactions
- Verify routing logic
- Check state transitions

### Scenario Tests
- Multi-agent consensus
- Error recovery flows
- Shadow mode validation

## Migration Plan
1. Create new structure
2. Move code incrementally
3. Update imports
4. Run tests at each step
5. Delete old file when complete