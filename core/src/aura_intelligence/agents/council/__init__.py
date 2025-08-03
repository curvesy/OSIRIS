"""
Multi-Agent Council Package.

This package provides agents that can participate in multi-agent
councils for collaborative decision making.
"""

# LNN Council Agent - graceful import
try:
    from .lnn_council import (
        LNNCouncilAgent,
        CouncilTask,
        CouncilVote,
        VoteType
    )
    _lnn_council_available = True
except ImportError:
    _lnn_council_available = False
    LNNCouncilAgent = None
    CouncilTask = None
    CouncilVote = None
    VoteType = None

# Production LNN Council Agent - graceful import
try:
    from .production_lnn_council import ProductionLNNCouncilAgent
    _production_lnn_available = True
except ImportError:
    _production_lnn_available = False
    ProductionLNNCouncilAgent = None

__all__ = []

# Add LNN council components if available
if _lnn_council_available:
    __all__.extend([
        "LNNCouncilAgent",
        "CouncilTask",
        "CouncilVote",
        "VoteType"
    ])

# Add production LNN council if available
if _production_lnn_available:
    __all__.append("ProductionLNNCouncilAgent")