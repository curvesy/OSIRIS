# 🔧 Configuration Naming Convention & Migration Strategy

## Overview

The AURA Intelligence configuration system maintains backward compatibility between two naming conventions:
- **Modern Convention**: `*Settings` classes (e.g., `AURASettings`, `AgentSettings`)
- **Legacy Convention**: `*Config` classes (e.g., `UltimateAURAConfig`, `AgentConfig`)

This document explains the rationale, implementation, and migration path.

## Current State

### Modern Implementation (Primary)
The configuration system uses Pydantic v2 models with the `*Settings` naming convention:
- `AURASettings` - Main configuration class
- `AgentSettings` - Agent-specific configuration
- `MemorySettings` - Memory system configuration
- `APISettings` - API keys and endpoints
- `ObservabilitySettings` - Monitoring and logging
- `SecuritySettings` - Security and authentication
- `DeploymentSettings` - Deployment modes and features

### Legacy Aliases (Backward Compatibility)
To support existing code that expects `*Config` names, we provide type aliases:
```python
UltimateAURAConfig = AURASettings
AgentConfig = AgentSettings
MemoryConfig = MemorySettings
KnowledgeConfig = AURASettings  # Knowledge is part of main settings
TopologyConfig = AURASettings   # Topology is part of main settings
```

## Implementation Details

### Location: `core/src/aura_intelligence/config/__init__.py`

The configuration module exports both naming conventions:
1. **Settings Classes**: The actual implementation
2. **Config Aliases**: Type aliases for backward compatibility
3. **Factory Functions**: Convenience functions for common configurations

### Factory Functions
```python
get_ultimate_config() -> AURASettings  # Ultimate features enabled
get_production_config() -> AURASettings  # Production-ready settings
get_enterprise_config() -> AURASettings  # Enterprise features
get_development_config() -> AURASettings  # Development/testing
```

## Usage Examples

### Using Modern Settings (Recommended)
```python
from aura_intelligence.config import AURASettings, AgentSettings

# Direct instantiation
config = AURASettings()

# Access sub-configurations
agent_config = config.agent  # Returns AgentSettings instance
```

### Using Legacy Config Names (Backward Compatible)
```python
from aura_intelligence.config import UltimateAURAConfig, AgentConfig

# Works exactly the same as AURASettings
config = UltimateAURAConfig()

# AgentConfig is an alias for AgentSettings
agent_config: AgentConfig = config.agent
```

### Using Factory Functions
```python
from aura_intelligence.config import get_ultimate_config

# Get pre-configured ultimate settings
config = get_ultimate_config()
# Sets: enhancement_level=ULTIMATE, agent_count=10, consciousness=True
```

## Migration Strategy

### Phase 1: Current State ✅
- Maintain dual naming support through aliases
- All existing code continues to work
- New code can use either convention

### Phase 2: Gradual Migration (Recommended)
1. **New Code**: Use `*Settings` naming convention
2. **Refactoring**: When touching existing code, update imports
3. **Documentation**: Update examples to use modern naming
4. **Tests**: Write new tests with `*Settings` names

### Phase 3: Deprecation (Future)
1. Add deprecation warnings to `*Config` aliases
2. Provide automated migration script
3. Set timeline for removal (e.g., v2.0)

### Phase 4: Removal (Future)
1. Remove `*Config` aliases
2. Update all remaining code
3. Simplify configuration module

## Special Cases

### TopologicalSignature
The `TopologicalSignature` class was missing and has been added to:
- **Location**: `core/src/aura_intelligence/core/topology.py`
- **Purpose**: Represents topological features from TDA analysis
- **Usage**: Returned by TDA engine, used in search/comparison

### Knowledge and Topology Config
These don't have separate settings classes because:
- Knowledge configuration is part of the main `AURASettings`
- Topology configuration is part of the main `AURASettings`
- The aliases point to `AURASettings` for compatibility

## Best Practices

1. **Import Once**: Import configuration at module level
   ```python
   from aura_intelligence.config import AURASettings
   ```

2. **Type Hints**: Use the actual class for type hints
   ```python
   def process(config: AURASettings) -> None:
       ...
   ```

3. **Environment Variables**: All settings support env vars
   ```bash
   AURA_ENVIRONMENT=production
   AURA_AGENT__AGENT_COUNT=10
   ```

4. **Validation**: Use built-in validation
   ```python
   config = get_production_config()
   warnings = config.validate_configuration()
   ```

## Troubleshooting

### Import Errors
If you get `ImportError: cannot import name 'XConfig'`:
1. Check you're importing from `aura_intelligence.config`
2. Ensure the alias exists in `__all__` export
3. Verify the Settings class exists

### Type Checking
Type checkers (mypy, pyright) should recognize aliases:
```python
config: UltimateAURAConfig = get_ultimate_config()  # Valid
config: AURASettings = get_ultimate_config()        # Also valid
```

### Configuration Not Found
If configuration seems missing:
1. Check environment variable naming: `AURA_*`
2. Verify nested config access: `config.agent.enhancement_level`
3. Use `config.model_dump()` to inspect full configuration

## Future Considerations

1. **Unified Naming**: Eventually standardize on `*Settings`
2. **Config Versioning**: Add version field for migrations
3. **Schema Export**: Generate JSON schema for validation
4. **Config Templates**: Provide more specialized factories

## Summary

The dual naming convention provides a smooth transition path while maintaining backward compatibility. New code should prefer the `*Settings` convention, but `*Config` aliases ensure existing code continues to work without modification.