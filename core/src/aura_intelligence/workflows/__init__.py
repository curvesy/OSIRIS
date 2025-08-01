"""
Workflows Package for AURA Intelligence.

Contains Temporal workflow definitions for various use cases.
"""

# GPU Allocation workflow - graceful import
try:
    from .gpu_allocation import (
        GPUAllocationWorkflow,
        GPUAllocationRequest,
        GPUAllocationResult,
        GPUType,
        AllocationPriority,
        GPUAllocationActivities
    )
    _gpu_allocation_available = True
except ImportError:
    _gpu_allocation_available = False
    GPUAllocationWorkflow = None
    GPUAllocationRequest = None
    GPUAllocationResult = None
    GPUType = None
    AllocationPriority = None
    GPUAllocationActivities = None

__all__ = []

# Add available workflows
if _gpu_allocation_available:
    __all__.extend([
        "GPUAllocationWorkflow",
        "GPUAllocationRequest", 
        "GPUAllocationResult",
        "GPUType",
        "AllocationPriority",
        "GPUAllocationActivities"
    ])