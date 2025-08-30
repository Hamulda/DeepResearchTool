"""Reporting Engine for Evidence-Bound Synthesis
Generate reports with citations to source artifacts, confidence metrics
Export to CSV/Markdown with contradiction detection and provenance tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class EvidenceType(Enum):
    """Types of evidence sources"""

    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    API_RESPONSE = "api_response"
    OSINT_FINDING = "osint_finding"
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    STEGANOGRAPHY = "steganography"


class ConfidenceLevel(Enum):
    """Confidence levels for claims"""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


@dataclass
class Citation:
    """Citation to source evidence"""

    citation_id: str
    source_type: EvidenceType
    source_url: str | None
    source_title: str
    excerpt: str
    page_number: int | None = None
    timestamp: datetime | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.citation_id:
            content = f"{self.source_type.value}:{self.source_title}:{self.excerpt[:50]}"
            self.citation_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class Claim:
    """Individual claim with supporting evidence"""

    claim_id: str
    statement: str
    confidence: float
    supporting_citations: list[Citation]
    contradicting_citations: list[Citation] = field(default_factory=list)
    reasoning: str | None = None
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.md5(self.statement.encode()).hexdigest()[:12]

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum"""
        if self.confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        if self.confidence < 0.6:
            return ConfidenceLevel.MEDIUM
        if self.confidence < 0.8:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.VERY_HIGH

    def has_contradictions(self) -> bool:
        """Check if claim has contradicting evidence"""
        return len(self.contradicting_citations) > 0

    def get_citation_count(self) -> int:
        """Get total citation count"""
        return len(self.supporting_citations) + len(self.contradicting_citations)


@dataclass
class ReportSection:
    """Section of a report"""

    section_id: str
    title: str
    content: str
    claims: list[Claim]
    subsections: list["ReportSection"] = field(default_factory=list)
    order: int = 0


@dataclass
class EvidenceReport:
    """Complete evidence-bound report"""

    report_id: str
    title: str
    summary: str
    created_at: datetime
    query: str

    sections: list[ReportSection]
    all_citations: list[Citation]

    # Metadata
    total_claims: int = 0
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    contradiction_count: int = 0
    source_types_used: set[str] = field(default_factory=set)

    # Generation info
    generation_time_seconds: float = 0.0
    model_used: str | None = None

    def __post_init__(self):
        if not self.report_id:
            content = f"{self.title}:{self.created_at.isoformat()}"
            self.report_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Calculate metadata
        self._calculate_metadata()

    def _calculate_metadata(self):
        """Calculate report metadata"""
        all_claims = []
        for section in self.sections:
            all_claims.extend(section.claims)
            for subsection in section.subsections:
                all_claims.extend(subsection.claims)

        self.total_claims = len(all_claims)

        # Confidence distribution
        confidence_counts = {level.value: 0 for level in ConfidenceLevel}
        for claim in all_claims:
            level = claim.get_confidence_level()
            confidence_counts[level.value] += 1
        self.confidence_distribution = confidence_counts

        # Contradiction count
        self.contradiction_count = sum(1 for claim in all_claims if claim.has_contradictions())

        # Source types
        for citation in self.all_citations:
            self.source_types_used.add(citation.source_type.value)


@dataclass
class ReportConfig:
    """Report generation configuration"""

    # Citation requirements
    min_citations_per_claim: int = 2
    require_diverse_sources: bool = True
    include_contradictions: bool = True

    # Output formats
    export_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    include_raw_data: bool = False

    # Content settings
    max_excerpt_length: int = 200
    include_confidence_scores: bool = True
    include_source_metadata: bool = True

    # Quality thresholds
    min_claim_confidence: float = 0.3
    flag_contradictions: bool = True
    require_evidence_for_claims: bool = True


class ContradictionDetector:
    """Detect contradictions between claims and evidence"""

    def __init__(self):
        self.contradiction_patterns = [
            ("not", "does not", "cannot", "never"),
            ("increase", "decrease", "rise", "fall"),
            ("true", "false", "correct", "incorrect"),
            ("support", "oppose", "agree", "disagree"),
        ]

    async def detect_contradictions(self, claims: list[Claim]) -> list[tuple[str, str, float]]:
        """Detect contradictions between claims"""
        contradictions = []

        for i, claim1 in enumerate(claims):
            for claim2 in claims[i + 1 :]:
                contradiction_score = await self._compare_claims(claim1, claim2)
                if contradiction_score > 0.5:
                    contradictions.append((claim1.claim_id, claim2.claim_id, contradiction_score))

        return contradictions

    async def _compare_claims(self, claim1: Claim, claim2: Claim) -> float:
        """Compare two claims for contradictions"""
        # Simple text-based contradiction detection
        text1 = claim1.statement.lower()
        text2 = claim2.statement.lower()

        # Look for opposing keywords
        contradiction_score = 0.0

        for pattern in self.contradiction_patterns:
            for word1 in pattern[:2]:
                for word2 in pattern[2:]:
                    if (word1 in text1 and word2 in text2) or (word2 in text1 and word1 in text2):
                        contradiction_score += 0.3

        # Check for negation patterns
        if ("not" in text1 and "not" not in text2) or ("not" in text2 and "not" not in text1):
            # Same base claim but one is negated
            base1 = text1.replace("not ", "").replace(" not", "")
            base2 = text2.replace("not ", "").replace(" not", "")
            if self._text_similarity(base1, base2) > 0.7:
                contradiction_score += 0.5

        return min(contradiction_score, 1.0)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union


class ReportingEngine:
    """Main reporting engine for evidence-bound synthesis"""

    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self.contradiction_detector = ContradictionDetector()

        # Report storage
        self.reports: dict[str, EvidenceReport] = {}

    async def generate_report(
        self,
        title: str,
        query: str,
        sections_data: list[dict[str, Any]],
        citations: list[Citation],
        model_used: str | None = None,
    ) -> EvidenceReport:
        """Generate evidence-bound report"""
        start_time = datetime.now()

        logger.info(f"📊 Generating evidence-bound report: {title}")

        # Create report sections
        sections = []
        for i, section_data in enumerate(sections_data):
            section = await self._create_section(section_data, citations, i)
            sections.append(section)

        # Detect contradictions across all claims
        all_claims = []
        for section in sections:
            all_claims.extend(section.claims)

        if self.config.include_contradictions:
            contradictions = await self.contradiction_detector.detect_contradictions(all_claims)
            await self._mark_contradictions(all_claims, contradictions)

        # Generate summary
        summary = await self._generate_summary(sections, all_claims)

        # Create report
        generation_time = (datetime.now() - start_time).total_seconds()

        report = EvidenceReport(
            report_id="",
            title=title,
            summary=summary,
            created_at=start_time,
            query=query,
            sections=sections,
            all_citations=citations,
            generation_time_seconds=generation_time,
            model_used=model_used,
        )

        # Store report
        self.reports[report.report_id] = report

        logger.info(
            f"✅ Report generated: {report.total_claims} claims, {len(citations)} citations"
        )
        return report

    async def _create_section(
        self, section_data: dict[str, Any], all_citations: list[Citation], order: int
    ) -> ReportSection:
        """Create report section with claims and citations"""
        # Extract claims from section content
        claims = await self._extract_claims_from_content(
            section_data.get("content", ""), all_citations
        )

        # Create subsections if present
        subsections = []
        if "subsections" in section_data:
            for j, subsection_data in enumerate(section_data["subsections"]):
                subsection = await self._create_section(subsection_data, all_citations, j)
                subsections.append(subsection)

        return ReportSection(
            section_id=f"section_{order}",
            title=section_data.get("title", f"Section {order + 1}"),
            content=section_data.get("content", ""),
            claims=claims,
            subsections=subsections,
            order=order,
        )

    async def _extract_claims_from_content(
        self, content: str, citations: list[Citation]
    ) -> list[Claim]:
        """Extract verifiable claims from content"""
        # This is a simplified implementation
        # In practice, would use NLP to identify factual claims

        claims = []
        sentences = content.split(". ")

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip short sentences
                continue

            # Simple heuristic: sentences with certain patterns are likely claims
            claim_indicators = [
                "research shows",
                "study found",
                "data indicates",
                "analysis reveals",
                "evidence suggests",
                "according to",
            ]

            if any(indicator in sentence.lower() for indicator in claim_indicators):
                # Find relevant citations for this claim
                relevant_citations = self._find_relevant_citations(sentence, citations)

                if len(relevant_citations) >= self.config.min_citations_per_claim:
                    confidence = self._calculate_claim_confidence(sentence, relevant_citations)

                    if confidence >= self.config.min_claim_confidence:
                        claim = Claim(
                            claim_id="",
                            statement=sentence.strip(),
                            confidence=confidence,
                            supporting_citations=relevant_citations,
                            reasoning=f"Extracted from content analysis, supported by {len(relevant_citations)} citations",
                        )
                        claims.append(claim)

        return claims

    def _find_relevant_citations(
        self, claim_text: str, citations: list[Citation]
    ) -> list[Citation]:
        """Find citations relevant to a claim"""
        relevant = []
        claim_words = set(claim_text.lower().split())

        for citation in citations:
            # Check overlap between claim and citation excerpt
            citation_words = set(citation.excerpt.lower().split())
            overlap = len(claim_words.intersection(citation_words))

            if overlap >= 3:  # Minimum word overlap
                relevant.append(citation)

        return relevant[:5]  # Limit to top 5 most relevant

    def _calculate_claim_confidence(self, claim_text: str, citations: list[Citation]) -> float:
        """Calculate confidence score for a claim"""
        if not citations:
            return 0.0

        # Base confidence from citation count
        citation_confidence = min(len(citations) / 3, 1.0)  # 3+ citations = full confidence

        # Average citation confidence
        avg_citation_confidence = sum(c.confidence for c in citations) / len(citations)

        # Source diversity bonus
        source_types = set(c.source_type for c in citations)
        diversity_bonus = len(source_types) * 0.1

        # Combine factors
        total_confidence = (
            citation_confidence * 0.4 + avg_citation_confidence * 0.5 + diversity_bonus
        )

        return min(total_confidence, 1.0)

    async def _mark_contradictions(
        self, claims: list[Claim], contradictions: list[tuple[str, str, float]]
    ):
        """Mark contradictory claims"""
        claim_map = {claim.claim_id: claim for claim in claims}

        for claim1_id, claim2_id, score in contradictions:
            if claim1_id in claim_map and claim2_id in claim_map:
                claim1 = claim_map[claim1_id]
                claim2 = claim_map[claim2_id]

                # Add contradiction citations
                contradiction_citation = Citation(
                    citation_id="",
                    source_type=EvidenceType.DOCUMENT,
                    source_url=None,
                    source_title="Internal Contradiction",
                    excerpt=f"Contradicts: {claim2.statement[:100]}...",
                    confidence=score,
                    metadata={"contradiction_score": score, "contradicted_claim": claim2_id},
                )

                claim1.contradicting_citations.append(contradiction_citation)

    async def _generate_summary(self, sections: list[ReportSection], claims: list[Claim]) -> str:
        """Generate report summary"""
        total_claims = len(claims)
        high_confidence_claims = sum(1 for c in claims if c.confidence > 0.7)
        contradictions = sum(1 for c in claims if c.has_contradictions())

        summary_parts = [
            f"This report analyzes {total_claims} key claims based on available evidence.",
            f"{high_confidence_claims} claims have high confidence (>70%).",
        ]

        if contradictions > 0:
            summary_parts.append(
                f"{contradictions} claims have contradictory evidence requiring further investigation."
            )

        summary_parts.append(f"The analysis covers {len(sections)} main areas of investigation.")

        return " ".join(summary_parts)

    async def export_report(
        self, report: EvidenceReport, output_path: Path, formats: list[str] | None = None
    ) -> dict[str, Path]:
        """Export report in specified formats"""
        formats = formats or self.config.export_formats
        exported_files = {}

        output_path.mkdir(parents=True, exist_ok=True)

        for format_type in formats:
            if format_type == "markdown":
                file_path = await self._export_markdown(report, output_path)
                exported_files["markdown"] = file_path
            elif format_type == "json":
                file_path = await self._export_json(report, output_path)
                exported_files["json"] = file_path
            elif format_type == "csv":
                file_path = await self._export_csv(report, output_path)
                exported_files["csv"] = file_path

        return exported_files

    async def _export_markdown(self, report: EvidenceReport, output_path: Path) -> Path:
        """Export report as Markdown"""
        markdown_path = output_path / f"report_{report.report_id}.md"

        markdown_content = []

        # Header
        markdown_content.extend(
            [
                f"# {report.title}",
                "",
                f"**Generated:** {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Query:** {report.query}",
                f"**Total Claims:** {report.total_claims}",
                f"**Contradictions:** {report.contradiction_count}",
                "",
                "## Summary",
                "",
                report.summary,
                "",
            ]
        )

        # Sections
        for section in report.sections:
            markdown_content.extend([f"## {section.title}", "", section.content, ""])

            # Claims in section
            if section.claims:
                markdown_content.append("### Key Claims")
                markdown_content.append("")

                for i, claim in enumerate(section.claims, 1):
                    confidence_level = claim.get_confidence_level().value.replace("_", " ").title()

                    markdown_content.extend(
                        [
                            f"**Claim {i}:** {claim.statement}",
                            f"- **Confidence:** {claim.confidence:.2f} ({confidence_level})",
                            f"- **Citations:** {len(claim.supporting_citations)}",
                        ]
                    )

                    if claim.has_contradictions():
                        markdown_content.append(
                            f"- **⚠️ Contradictions:** {len(claim.contradicting_citations)}"
                        )

                    # Citations
                    for j, citation in enumerate(claim.supporting_citations[:3], 1):
                        markdown_content.append(
                            f"  {j}. {citation.source_title} - {citation.excerpt[:100]}..."
                        )

                    markdown_content.append("")

        # Citations section
        markdown_content.extend(["## All Citations", ""])

        for i, citation in enumerate(report.all_citations, 1):
            markdown_content.extend(
                [
                    f"**[{i}]** {citation.source_title}",
                    f"- Type: {citation.source_type.value}",
                    f"- Excerpt: {citation.excerpt}",
                    f"- Confidence: {citation.confidence:.2f}",
                ]
            )

            if citation.source_url:
                markdown_content.append(f"- URL: {citation.source_url}")

            markdown_content.append("")

        # Write file
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        return markdown_path

    async def _export_json(self, report: EvidenceReport, output_path: Path) -> Path:
        """Export report as JSON"""
        json_path = output_path / f"report_{report.report_id}.json"

        # Convert to serializable format
        report_data = {
            "report_id": report.report_id,
            "title": report.title,
            "summary": report.summary,
            "created_at": report.created_at.isoformat(),
            "query": report.query,
            "metadata": {
                "total_claims": report.total_claims,
                "confidence_distribution": report.confidence_distribution,
                "contradiction_count": report.contradiction_count,
                "source_types_used": list(report.source_types_used),
                "generation_time_seconds": report.generation_time_seconds,
                "model_used": report.model_used,
            },
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "content": section.content,
                    "order": section.order,
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "statement": claim.statement,
                            "confidence": claim.confidence,
                            "confidence_level": claim.get_confidence_level().value,
                            "supporting_citations": [
                                c.citation_id for c in claim.supporting_citations
                            ],
                            "contradicting_citations": [
                                c.citation_id for c in claim.contradicting_citations
                            ],
                            "reasoning": claim.reasoning,
                        }
                        for claim in section.claims
                    ],
                }
                for section in report.sections
            ],
            "citations": [
                {
                    "citation_id": citation.citation_id,
                    "source_type": citation.source_type.value,
                    "source_url": citation.source_url,
                    "source_title": citation.source_title,
                    "excerpt": citation.excerpt,
                    "confidence": citation.confidence,
                    "timestamp": citation.timestamp.isoformat() if citation.timestamp else None,
                    "metadata": citation.metadata,
                }
                for citation in report.all_citations
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return json_path

    async def _export_csv(self, report: EvidenceReport, output_path: Path) -> Path:
        """Export report data as CSV"""
        if not POLARS_AVAILABLE:
            logger.warning("Polars not available - skipping CSV export")
            return None

        csv_path = output_path / f"report_{report.report_id}_claims.csv"

        # Prepare claims data
        claims_data = []
        for section in report.sections:
            for claim in section.claims:
                claims_data.append(
                    {
                        "claim_id": claim.claim_id,
                        "section": section.title,
                        "statement": claim.statement,
                        "confidence": claim.confidence,
                        "confidence_level": claim.get_confidence_level().value,
                        "supporting_citations_count": len(claim.supporting_citations),
                        "contradicting_citations_count": len(claim.contradicting_citations),
                        "has_contradictions": claim.has_contradictions(),
                        "reasoning": claim.reasoning or "",
                    }
                )

        if claims_data:
            df = pl.DataFrame(claims_data)
            df.write_csv(csv_path)

        return csv_path

    def get_reporting_stats(self) -> dict[str, Any]:
        """Get reporting statistics"""
        if not self.reports:
            return {"total_reports": 0}

        reports = list(self.reports.values())

        return {
            "total_reports": len(reports),
            "average_claims_per_report": sum(r.total_claims for r in reports) / len(reports),
            "average_citations_per_report": sum(len(r.all_citations) for r in reports)
            / len(reports),
            "reports_with_contradictions": sum(1 for r in reports if r.contradiction_count > 0),
            "average_generation_time": sum(r.generation_time_seconds for r in reports)
            / len(reports),
        }


# Utility functions
async def create_simple_report(
    title: str, content: str, citations: list[Citation], output_path: Path
) -> EvidenceReport:
    """Create simple report from content and citations"""
    config = ReportConfig(min_citations_per_claim=1, export_formats=["markdown", "json"])

    engine = ReportingEngine(config)

    sections_data = [{"title": "Analysis", "content": content}]

    report = await engine.generate_report(
        title=title,
        query="Simple report generation",
        sections_data=sections_data,
        citations=citations,
    )

    await engine.export_report(report, output_path)
    return report


__all__ = [
    "Citation",
    "Claim",
    "ConfidenceLevel",
    "ContradictionDetector",
    "EvidenceReport",
    "EvidenceType",
    "ReportConfig",
    "ReportSection",
    "ReportingEngine",
    "create_simple_report",
]
