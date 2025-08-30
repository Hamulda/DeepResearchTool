# Synthesis module for Phase 3 - Intelligence Synthesis

from .correlation_engine import (
    CorrelationEngine,
    Entity,
    EntityCluster,
    NetworkAnalysisResult,
    Relationship,
)
from .credibility_assessor import (
    ContentQualityMetrics,
    CredibilityAssessment,
    CredibilityAssessor,
    DomainReputationMetrics,
    SourceMetadata,
    TemporalRelevanceMetrics,
)
from .deep_pattern_detector import ArtefactExtraction, DeepPatternDetector, PatternMatch

# Enhanced synthesis engine (existing)
from .enhanced_synthesis_engine import EnhancedSynthesisEngine
from .intelligence_synthesis_engine import (
    IntelligenceSource,
    IntelligenceSynthesisEngine,
    IntelligenceSynthesisResult,
)
from .steganography_analyzer import (
    FrequencyAnalysisResult,
    LSBAnalysisResult,
    SteganographyAnalyzer,
    SteganographyResult,
)

__all__ = [
    # Pattern Detection
    "DeepPatternDetector",
    "PatternMatch",
    "ArtefactExtraction",
    # Steganography Analysis
    "SteganographyAnalyzer",
    "SteganographyResult",
    "LSBAnalysisResult",
    "FrequencyAnalysisResult",
    # Correlation Engine
    "CorrelationEngine",
    "Entity",
    "Relationship",
    "EntityCluster",
    "NetworkAnalysisResult",
    # Credibility Assessment
    "CredibilityAssessor",
    "SourceMetadata",
    "CredibilityAssessment",
    "ContentQualityMetrics",
    "DomainReputationMetrics",
    "TemporalRelevanceMetrics",
    # Main Intelligence Synthesis
    "IntelligenceSynthesisEngine",
    "IntelligenceSynthesisResult",
    "IntelligenceSource",
    # Legacy
    "EnhancedSynthesisEngine",
]
