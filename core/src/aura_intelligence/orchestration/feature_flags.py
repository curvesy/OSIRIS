"""Feature flag management for orchestration capabilities"""
from enum import Enum
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """Available feature flags for orchestration"""
    # Existing feature flags
    ASYNC_ORCHESTRATION = "async_orchestration"
    MULTI_AGENT_COUNCIL = "multi_agent_council"
    DISTRIBUTED_MEMORY = "distributed_memory"
    ADVANCED_MONITORING = "advanced_monitoring"
    RAY_SERVE_ORCHESTRATION = "ray_serve_orchestration"
    TEMPORAL_WORKFLOWS = "temporal_workflows"
    CREWAI_INTEGRATION = "crewai_integration"
    LANGGRAPH_ENHANCEMENTS = "langgraph_enhancements"
    
    # Power Sprint Week 1 feature flags
    KV_MIRROR_ENABLED = "kv_mirror_enabled"
    ENTROPY_COMPACTION_ENABLED = "entropy_compaction_enabled"
    RESOURCEFLAME_ENABLED = "resourceflame_enabled"
    
    # Power Sprint Week 2 feature flags
    LAZY_WITNESS_ENABLED = "lazy_witness_enabled"
    MATRIX_PH_GPU_ENABLED = "matrix_ph_gpu_enabled"
    PHFORMER_TINY_ENABLED = "phformer_tiny_enabled"
    CHUNKED_MICROBATCH_ENABLED = "chunked_microbatch_enabled"
    
    # Power Sprint Week 3 feature flags
    ADAPTIVE_CHECKPOINT_ENABLED = "adaptive_checkpoint_enabled"
    TEMPORAL_SIGNALFIRST_ENABLED = "temporal_signalfirst_enabled"
    NEO4J_MOTIFCOST_ENABLED = "neo4j_motifcost_enabled"
    
    # Power Sprint Week 4 feature flags
    DIFF_COMM_V2B_ENABLED = "diff_comm_v2b_enabled"
    WEBSUB_ALERTS_ENABLED = "websub_alerts_enabled"
    HASH_CARRY_SEEDS_ENABLED = "hash_carry_seeds_enabled"
    KMUX_EBPF_ENABLED = "kmux_ebpf_enabled"
    
    # Migration support flags
    DUAL_WRITE_MODE = "dual_write_mode"
    REDIS_READS_ENABLED = "redis_reads_enabled"


# Default feature flag configurations
FEATURE_CONFIGS: Dict[FeatureFlag, Dict[str, Any]] = {
    # Existing features
    FeatureFlag.ASYNC_ORCHESTRATION: {
        "enabled": True,
        "description": "Enable asynchronous orchestration patterns",
        "rollout_percentage": 100
    },
    FeatureFlag.MULTI_AGENT_COUNCIL: {
        "enabled": True,
        "description": "Enable multi-agent council consensus",
        "rollout_percentage": 100
    },
    FeatureFlag.DISTRIBUTED_MEMORY: {
        "enabled": True,
        "description": "Enable distributed memory across agents",
        "rollout_percentage": 100
    },
    FeatureFlag.ADVANCED_MONITORING: {
        "enabled": True,
        "description": "Enable advanced monitoring with OpenTelemetry",
        "rollout_percentage": 100
    },
    FeatureFlag.RAY_SERVE_ORCHESTRATION: {
        "enabled": False,
        "description": "Use Ray Serve for distributed orchestration",
        "rollout_percentage": 0
    },
    FeatureFlag.TEMPORAL_WORKFLOWS: {
        "enabled": True,
        "description": "Enable Temporal workflow orchestration",
        "rollout_percentage": 100
    },
    FeatureFlag.CREWAI_INTEGRATION: {
        "enabled": False,
        "description": "Enable CrewAI integration for hierarchical agents",
        "rollout_percentage": 0
    },
    FeatureFlag.LANGGRAPH_ENHANCEMENTS: {
        "enabled": True,
        "description": "Enable LangGraph enhancements",
        "rollout_percentage": 100
    },
    
    # Power Sprint Week 1 - Quick Wins
    FeatureFlag.KV_MIRROR_ENABLED: {
        "enabled": False,
        "description": "Use NATS JetStream KV-Mirror instead of Redis",
        "rollout_percentage": 0,
        "power_sprint_week": 1
    },
    FeatureFlag.ENTROPY_COMPACTION_ENABLED: {
        "enabled": True,  # CRITICAL: Must be true from start
        "description": "HyperOak entropy-aware compaction (40% storage reduction)",
        "rollout_percentage": 100,
        "power_sprint_week": 1,
        "irreversible": True
    },
    FeatureFlag.RESOURCEFLAME_ENABLED: {
        "enabled": False,
        "description": "Grafana ResourceFlame panel for GPU/NUMA visualization",
        "rollout_percentage": 0,
        "power_sprint_week": 1
    },
    
    # Power Sprint Week 2 - Compute Optimizations
    FeatureFlag.LAZY_WITNESS_ENABLED: {
        "enabled": False,
        "description": "Lazy Witness + Witness-Z-Rips for 3x TDA speedup",
        "rollout_percentage": 0,
        "power_sprint_week": 2
    },
    FeatureFlag.MATRIX_PH_GPU_ENABLED: {
        "enabled": False,
        "description": "Matrix-PH GPU fusion kernel (1.6x speedup)",
        "rollout_percentage": 0,
        "power_sprint_week": 2
    },
    FeatureFlag.PHFORMER_TINY_ENABLED: {
        "enabled": False,
        "description": "PHFormer-Tiny-8B model (same accuracy, 1/3 size)",
        "rollout_percentage": 0,
        "power_sprint_week": 2
    },
    FeatureFlag.CHUNKED_MICROBATCH_ENABLED: {
        "enabled": False,
        "description": "MAX Serve chunked micro-batching (70%+ GPU util)",
        "rollout_percentage": 0,
        "power_sprint_week": 2
    },
    
    # Power Sprint Week 3 - Infrastructure & Durability
    FeatureFlag.ADAPTIVE_CHECKPOINT_ENABLED: {
        "enabled": False,
        "description": "Adaptive checkpoint coalescing (45% fewer DB writes)",
        "rollout_percentage": 0,
        "power_sprint_week": 3
    },
    FeatureFlag.TEMPORAL_SIGNALFIRST_ENABLED: {
        "enabled": False,
        "description": "Temporal SignalFirst (20ms latency reduction)",
        "rollout_percentage": 0,
        "power_sprint_week": 3
    },
    FeatureFlag.NEO4J_MOTIFCOST_ENABLED: {
        "enabled": False,
        "description": "Neo4j MotifCost index (4-6x query speedup)",
        "rollout_percentage": 0,
        "power_sprint_week": 3
    },
    
    # Power Sprint Week 4 - Network & Security
    FeatureFlag.DIFF_COMM_V2B_ENABLED: {
        "enabled": False,
        "description": "Diff-Comm-v2b with dynamic header pruning (3x bandwidth reduction)",
        "rollout_percentage": 0,
        "power_sprint_week": 4
    },
    FeatureFlag.WEBSUB_ALERTS_ENABLED: {
        "enabled": False,
        "description": "Web-sub protocol for alerts (bank-friendly)",
        "rollout_percentage": 0,
        "power_sprint_week": 4
    },
    FeatureFlag.HASH_CARRY_SEEDS_ENABLED: {
        "enabled": False,
        "description": "Hash-with-Carry seeds for deterministic replay",
        "rollout_percentage": 0,
        "power_sprint_week": 4
    },
    FeatureFlag.KMUX_EBPF_ENABLED: {
        "enabled": False,
        "description": "KMUX multiprobe eBPF (reduced kernel modules)",
        "rollout_percentage": 0,
        "power_sprint_week": 4
    },
    
    # Migration support
    FeatureFlag.DUAL_WRITE_MODE: {
        "enabled": False,
        "description": "Write to both Redis and NATS during migration",
        "rollout_percentage": 0,
        "migration_flag": True
    },
    FeatureFlag.REDIS_READS_ENABLED: {
        "enabled": True,
        "description": "Read from Redis (disable after KV-Mirror migration)",
        "rollout_percentage": 100,
        "migration_flag": True
    }
}


class FeatureFlagManager:
    """Manages feature flags with support for runtime updates and per-tenant overrides"""
    
    def __init__(self):
        self.flags = FEATURE_CONFIGS.copy()
        self.tenant_overrides: Dict[str, Dict[FeatureFlag, bool]] = {}
        self._load_overrides()
    
    def _load_overrides(self):
        """Load feature flag overrides from environment or config file"""
        # Check for environment variable overrides
        for flag in FeatureFlag:
            env_key = f"AURA_FF_{flag.value.upper()}"
            if env_key in os.environ:
                try:
                    self.flags[flag]["enabled"] = os.environ[env_key].lower() == "true"
                    logger.info(f"Feature flag {flag.value} overridden by env: {self.flags[flag]['enabled']}")
                except Exception as e:
                    logger.error(f"Error parsing feature flag env var {env_key}: {e}")
        
        # Load from config file if exists
        config_path = os.environ.get("AURA_FF_CONFIG_PATH", "/etc/aura/feature_flags.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    for flag_name, settings in config.items():
                        try:
                            flag = FeatureFlag(flag_name)
                            self.flags[flag].update(settings)
                            logger.info(f"Feature flag {flag_name} loaded from config")
                        except ValueError:
                            logger.warning(f"Unknown feature flag in config: {flag_name}")
            except Exception as e:
                logger.error(f"Error loading feature flag config: {e}")
    
    def is_enabled(self, flag: FeatureFlag, tenant_id: Optional[str] = None) -> bool:
        """Check if a feature flag is enabled, with optional tenant override"""
        # Check tenant-specific override first
        if tenant_id and tenant_id in self.tenant_overrides:
            if flag in self.tenant_overrides[tenant_id]:
                return self.tenant_overrides[tenant_id][flag]
        
        # Check global flag
        return self.flags.get(flag, {}).get("enabled", False)
    
    def set_override(self, flag: FeatureFlag, enabled: bool, tenant_id: Optional[str] = None):
        """Set a feature flag override"""
        if tenant_id:
            if tenant_id not in self.tenant_overrides:
                self.tenant_overrides[tenant_id] = {}
            self.tenant_overrides[tenant_id][flag] = enabled
            logger.info(f"Set tenant override for {tenant_id}: {flag.value}={enabled}")
        else:
            self.flags[flag]["enabled"] = enabled
            logger.info(f"Set global flag: {flag.value}={enabled}")
    
    def clear_override(self, flag: FeatureFlag, tenant_id: str):
        """Clear a tenant-specific override"""
        if tenant_id in self.tenant_overrides and flag in self.tenant_overrides[tenant_id]:
            del self.tenant_overrides[tenant_id][flag]
            logger.info(f"Cleared tenant override for {tenant_id}: {flag.value}")
    
    def get_all_flags(self, tenant_id: Optional[str] = None) -> Dict[str, bool]:
        """Get all feature flag states"""
        result = {}
        for flag in FeatureFlag:
            result[flag.value] = self.is_enabled(flag, tenant_id)
        return result
    
    def get_config(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Get full configuration for a feature flag"""
        return self.flags.get(flag, {})
    
    def get_power_sprint_status(self) -> Dict[int, Dict[str, Any]]:
        """Get Power Sprint optimization status by week"""
        status = {1: {}, 2: {}, 3: {}, 4: {}}
        
        for flag, config in self.flags.items():
            if "power_sprint_week" in config:
                week = config["power_sprint_week"]
                status[week][flag.value] = {
                    "enabled": config["enabled"],
                    "description": config["description"],
                    "irreversible": config.get("irreversible", False)
                }
        
        return status
    
    def enable_power_sprint_week(self, week: int):
        """Enable all optimizations for a specific Power Sprint week"""
        if week not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid Power Sprint week: {week}")
        
        enabled_flags = []
        for flag, config in self.flags.items():
            if config.get("power_sprint_week") == week and not config.get("irreversible"):
                self.flags[flag]["enabled"] = True
                enabled_flags.append(flag.value)
        
        logger.info(f"Enabled Power Sprint Week {week} flags: {enabled_flags}")
        return enabled_flags


# Global feature flag manager instance
_feature_flag_manager = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get or create the global feature flag manager"""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def is_feature_enabled(flag: FeatureFlag, tenant_id: Optional[str] = None) -> bool:
    """Convenience function to check if a feature is enabled"""
    return get_feature_flag_manager().is_enabled(flag, tenant_id)