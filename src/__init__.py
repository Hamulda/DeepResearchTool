"""
DeepResearchTool - Plně automatický, lokální, evidence-based research agent
Version 3.0 - BEZ human-in-the-loop

Core modules:
- retrieval: HyDE, hybrid search (Qdrant + BM25), RRF
- rank: Re-ranking, MMR diversification, contextual compression
- compress: Adaptive chunking, discourse-aware processing
- synthesis: Template-driven s per-claim evidence binding
- verify: Adversarial verification, claim graph, contradiction detection
- connectors: Specialized sources (Common Crawl, Memento, Ahmia, Legal APIs, OpenAlex)
- metrics: Evaluation suite (recall@k, nDCG, evidence coverage, groundedness)
- utils: Gates, compliance, token budgeting

Author: Senior Python/MLOps Agent
"""

__version__ = "3.0.0"
__author__ = "Senior Python/MLOps Agent"

# Core pipeline components
from .core import *
from .retrieval import *
from .rank import *
from .compress import *
from .synthesis import *
from .verify import *

# Specialized connectors
from .connectors import *

# Utilities and gates
from .utils.gates import GateKeeper, EvidenceGateError, ComplianceGateError, MetricsGateError
from .metrics import *

__all__ = [
    # Core
    "ResearchPipeline",
    "Config",

    # Gates
    "GateKeeper",
    "EvidenceGateError",
    "ComplianceGateError",
    "MetricsGateError",

    # Pipeline stages
    "HybridRetrieval",
    "ReRanker",
    "ContextualCompressor",
    "TemplateSynthesizer",
    "AdversarialVerifier",

    # Connectors
    "CommonCrawlConnector",
    "MementoConnector",
    "AhmiaConnector",
    "LegalAPIsConnector",
    "OpenAlexConnector",

    # Metrics
    "EvaluationSuite",
    "MetricsCalculator"
]
