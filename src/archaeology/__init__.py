"""
Archaeology package pro DeepResearchTool
Moduly pro archeologické prohledávání webu a legacy protokolů.
"""

from .historical_excavator import HistoricalWebExcavator
from .legacy_protocols import LegacyProtocolDetector

__all__ = [
    "HistoricalWebExcavator",
    "LegacyProtocolDetector"
]
