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

__all__ = []

# Add LNN council components if available
if _lnn_council_available:
    __all__.extend([
        "LNNCouncilAgent",
        "CouncilTask",
        "CouncilVote",
        "VoteType"
    ])