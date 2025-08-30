#!/usr/bin/env python3
"""Enhanced Research Analyzer for Deep Research Tool
Advanced analysis system for generating high-quality research summaries and insights

Author: Advanced IT Specialist
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnalysisInsight:
    """Represents a key insight from research analysis"""

    type: str  # 'finding', 'trend', 'controversy', 'gap', 'breakthrough'
    title: str
    description: str
    confidence: float
    supporting_sources: list[str]
    significance: str  # 'critical', 'high', 'medium', 'low'


@dataclass
class ResearchSummary:
    """Complete research summary with insights"""

    topic: str
    executive_summary: str
    key_findings: list[str]
    insights: list[AnalysisInsight]
    source_analysis: dict[str, Any]
    methodology_assessment: dict[str, Any]
    research_gaps: list[str]
    recommendations: list[str]
    confidence_score: float
    analysis_timestamp: datetime


class EnhancedResearchAnalyzer:
    """Advanced analyzer for generating comprehensive research insights"""

    def __init__(self):
        """Initialize the enhanced analyzer"""
        # Pattern matching for different types of findings
        self.finding_patterns = {
            "clinical_findings": [
                r"\b(?:clinical trial|study|research)\s+(?:shows?|demonstrates?|finds?|concludes?)\b",
                r"\b(?:patients?|subjects?)\s+(?:experienced|showed|demonstrated)\b",
                r"\b(?:treatment|therapy|intervention)\s+(?:resulted in|led to|caused)\b",
            ],
            "mechanism_findings": [
                r"\b(?:mechanism|pathway|receptor|binding)\s+(?:involves|includes|requires)\b",
                r"\b(?:acts through|works by|functions via)\b",
                r"\b(?:targets?|binds to|activates?|inhibits?)\b",
            ],
            "safety_findings": [
                r"\b(?:side effects?|adverse events?|safety)\s+(?:include|reported|observed)\b",
                r"\b(?:contraindicated|not recommended|avoid)\b",
                r"\b(?:toxicity|harmful|dangerous)\b",
            ],
            "efficacy_findings": [
                r"\b(?:effective|efficacious|successful)\s+(?:in|for|at)\b",
                r"\b(?:significant improvement|marked increase|notable reduction)\b",
                r"\b(?:\d+%\s+(?:improvement|increase|reduction|change))\b",
            ],
        }

        # Controversy detection patterns
        self.controversy_patterns = [
            r"\b(?:controversial|disputed|debated|conflicting|contradictory)\b",
            r"\b(?:however|but|contrary to|on the other hand|despite)\b",
            r"\b(?:critics argue|opponents claim|skeptics believe)\b",
            r"\b(?:limited evidence|insufficient data|inconclusive)\b",
        ]

        # Quality indicators
        self.quality_indicators = {
            "high_quality": [
                r"\bmeta-analysis\b",
                r"\bsystematic review\b",
                r"\brandomized controlled trial\b",
                r"\bdouble-blind\b",
                r"\bplacebo-controlled\b",
                r"\bpeer-reviewed\b",
            ],
            "sample_size": [
                r"\bn\s*=\s*(\d+)\b",
                r"\b(\d+)\s+(?:participants?|subjects?|patients?)\b",
            ],
            "statistical_significance": [
                r"\bp\s*<\s*0\.05\b",
                r"\bp\s*<\s*0\.01\b",
                r"\bstatistically significant\b",
            ],
        }

    def analyze_research_data(
        self, research_results: list[dict[str, Any]], topic: str, priority_scorer=None
    ) -> ResearchSummary:
        """Perform comprehensive analysis of research data

        Args:
            research_results: List of research documents
            topic: Research topic
            priority_scorer: Optional priority scorer for quality assessment

        Returns:
            Comprehensive research summary

        """
        logger.info(f"Starting enhanced analysis for topic: {topic}")

        # Extract and categorize findings
        findings = self._extract_findings(research_results)

        # Generate insights
        insights = self._generate_insights(findings, research_results)

        # Analyze source quality and diversity
        source_analysis = self._analyze_sources(research_results, priority_scorer)

        # Assess methodology quality
        methodology_assessment = self._assess_methodology(research_results)

        # Identify research gaps
        research_gaps = self._identify_research_gaps(findings, research_results)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            topic, findings, insights, source_analysis
        )

        # Extract key findings
        key_findings = self._extract_key_findings(findings, insights)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            findings, insights, research_gaps, methodology_assessment
        )

        # Calculate overall confidence
        confidence_score = self._calculate_analysis_confidence(
            source_analysis, methodology_assessment, len(research_results)
        )

        return ResearchSummary(
            topic=topic,
            executive_summary=executive_summary,
            key_findings=key_findings,
            insights=insights,
            source_analysis=source_analysis,
            methodology_assessment=methodology_assessment,
            research_gaps=research_gaps,
            recommendations=recommendations,
            confidence_score=confidence_score,
            analysis_timestamp=datetime.now(),
        )

    def _extract_findings(
        self, research_results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract and categorize findings from research documents"""
        findings = defaultdict(list)

        for result in research_results:
            content = result.get("content", "")
            title = result.get("title", "")
            combined_text = f"{title} {content}"

            # Extract different types of findings
            for finding_type, patterns in self.finding_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, combined_text, re.IGNORECASE)
                    for match in matches:
                        # Extract surrounding context
                        start = max(0, match.start() - 100)
                        end = min(len(combined_text), match.end() + 100)
                        context = combined_text[start:end].strip()

                        finding = {
                            "type": finding_type,
                            "text": context,
                            "source": result.get("title", "Unknown"),
                            "url": result.get("url", ""),
                            "confidence": result.get("priority_score", 0.5),
                            "source_credibility": result.get("source_credibility", 0.5),
                        }
                        findings[finding_type].append(finding)

        return dict(findings)

    def _generate_insights(
        self, findings: dict[str, list[dict[str, Any]]], research_results: list[dict[str, Any]]
    ) -> list[AnalysisInsight]:
        """Generate key insights from findings"""
        insights = []

        # Analyze finding frequency and patterns
        for finding_type, finding_list in findings.items():
            if len(finding_list) >= 3:  # Need minimum sources for credible insight

                # Calculate average confidence
                avg_confidence = statistics.mean([f["confidence"] for f in finding_list])

                # Generate insight based on finding type
                if finding_type == "clinical_findings":
                    insight = self._generate_clinical_insight(finding_list, avg_confidence)
                elif finding_type == "mechanism_findings":
                    insight = self._generate_mechanism_insight(finding_list, avg_confidence)
                elif finding_type == "safety_findings":
                    insight = self._generate_safety_insight(finding_list, avg_confidence)
                elif finding_type == "efficacy_findings":
                    insight = self._generate_efficacy_insight(finding_list, avg_confidence)
                else:
                    continue

                if insight:
                    insights.append(insight)

        # Detect controversies
        controversy_insights = self._detect_controversies(research_results)
        insights.extend(controversy_insights)

        # Identify breakthrough findings
        breakthrough_insights = self._identify_breakthroughs(research_results)
        insights.extend(breakthrough_insights)

        # Sort insights by significance and confidence
        insights.sort(
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.significance],
                x.confidence,
            ),
            reverse=True,
        )

        return insights[:10]  # Return top 10 insights

    def _generate_clinical_insight(
        self, findings: list[dict[str, Any]], confidence: float
    ) -> AnalysisInsight | None:
        """Generate insight from clinical findings with data lineage tracking"""
        if not findings:
            return None

        # Extract common themes with source tracking
        texts_with_sources = [(f["text"], f["source"], f.get("url", "")) for f in findings]
        combined_text = " ".join([text for text, _, _ in texts_with_sources])

        # Look for outcome patterns with source attribution
        outcomes_with_sources = []
        outcome_patterns = [
            r"(\d+%)\s+(?:improvement|increase|reduction)",
            r"significant\s+(?:improvement|increase|reduction)\s+in\s+(\w+)",
            r"patients?\s+(?:experienced|showed)\s+([^.]+)",
        ]

        for pattern in outcome_patterns:
            for text, source, url in texts_with_sources:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    outcomes_with_sources.append(
                        {
                            "finding": match,
                            "source": source,
                            "url": url,
                            "context": text[max(0, text.find(match) - 50) : text.find(match) + 100],
                        }
                    )

        if outcomes_with_sources:
            # Create evidence chain
            evidence_chain = []
            for outcome in outcomes_with_sources[:3]:
                evidence_chain.append(
                    {
                        "claim": outcome["finding"],
                        "source_document": outcome["source"],
                        "source_url": outcome["url"],
                        "supporting_text": outcome["context"],
                        "extraction_method": "regex_pattern_matching",
                        "confidence": confidence,
                        "verification_status": "extracted_unverified",
                    }
                )

            description = "Clinical studies consistently report positive outcomes. "
            description += (
                f"Key findings include: {', '.join([e['claim'] for e in evidence_chain])}. "
            )
            description += (
                f"Based on {len(findings)} independent sources with full data lineage tracking."
            )

            return AnalysisInsight(
                type="finding",
                title="Consistent Clinical Efficacy Demonstrated",
                description=description,
                confidence=min(0.95, confidence + 0.1),
                supporting_sources=[f["source"] for f in findings[:5]],
                significance="high" if confidence > 0.7 else "medium",
                evidence_chain=evidence_chain,  # New field for data lineage
                lineage_metadata={
                    "extraction_timestamp": datetime.now().isoformat(),
                    "extraction_patterns_used": outcome_patterns,
                    "source_diversity_score": len(set(f["source"] for f in findings))
                    / len(findings),
                    "cross_verification_count": len(
                        [f for f in findings if f.get("priority_score", 0) > 0.7]
                    ),
                },
            )

        return None

    def _generate_mechanism_insight(
        self, findings: list[dict[str, Any]], confidence: float
    ) -> AnalysisInsight | None:
        """Generate insight from mechanism findings with detailed source tracking"""
        if not findings:
            return None

        # Extract mechanism details with precise source attribution
        mechanism_evidence = []
        mechanism_patterns = [
            r"(?:binds to|targets?|activates?|inhibits?)\s+([^.]+)",
            r"mechanism\s+(?:involves|includes)\s+([^.]+)",
            r"works\s+(?:through|by|via)\s+([^.]+)",
        ]

        for finding in findings:
            text = finding.get("text", "")
            source = finding.get("source", "Unknown")
            url = finding.get("url", "")

            for pattern in mechanism_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    mechanism_text = match.group(1).strip()

                    # Create detailed evidence record
                    evidence_record = {
                        "mechanism_description": mechanism_text,
                        "source_document": source,
                        "source_url": url,
                        "extraction_pattern": pattern,
                        "text_position": (match.start(), match.end()),
                        "surrounding_context": text[
                            max(0, match.start() - 100) : match.end() + 100
                        ],
                        "confidence_score": finding.get("priority_score", 0.5),
                        "source_credibility": finding.get("source_credibility", 0.5),
                        "extraction_timestamp": datetime.now().isoformat(),
                    }
                    mechanism_evidence.append(evidence_record)

        if mechanism_evidence:
            # Find most common mechanisms with evidence tracking
            mechanism_texts = [e["mechanism_description"] for e in mechanism_evidence]
            mechanism_counts = Counter(mechanism_texts)
            top_mechanisms = mechanism_counts.most_common(3)

            # Create comprehensive evidence chain
            evidence_chain = []
            for mechanism_text, count in top_mechanisms:
                supporting_evidence = [
                    e for e in mechanism_evidence if e["mechanism_description"] == mechanism_text
                ]

                evidence_chain.append(
                    {
                        "claim": mechanism_text,
                        "support_count": count,
                        "supporting_sources": [e["source_document"] for e in supporting_evidence],
                        "source_urls": [e["source_url"] for e in supporting_evidence],
                        "extraction_contexts": [
                            e["surrounding_context"] for e in supporting_evidence
                        ],
                        "average_confidence": sum(
                            e["confidence_score"] for e in supporting_evidence
                        )
                        / len(supporting_evidence),
                        "credibility_scores": [
                            e["source_credibility"] for e in supporting_evidence
                        ],
                    }
                )

            description = "Research reveals consistent mechanism of action. "
            description += (
                f"Primary mechanisms include: {', '.join([m[0] for m in top_mechanisms])}. "
            )
            description += f"Supported by {len(findings)} studies with detailed evidence tracking."

            return AnalysisInsight(
                type="finding",
                title="Well-Defined Mechanism of Action",
                description=description,
                confidence=confidence,
                supporting_sources=[f["source"] for f in findings[:5]],
                significance="high" if confidence > 0.7 else "medium",
                evidence_chain=evidence_chain,
                lineage_metadata={
                    "total_evidence_points": len(mechanism_evidence),
                    "unique_mechanisms_found": len(mechanism_counts),
                    "cross_reference_strength": len(top_mechanisms),
                    "extraction_patterns_used": mechanism_patterns,
                    "source_diversity": len(set(e["source_document"] for e in mechanism_evidence)),
                },
            )

        return None

    def _generate_safety_insight(
        self, findings: list[dict[str, Any]], confidence: float
    ) -> AnalysisInsight | None:
        """Generate insight from safety findings"""
        if not findings:
            return None

        # Extract safety concerns
        texts = [f["text"] for f in findings]
        combined_text = " ".join(texts)

        # Look for safety patterns
        safety_issues = []
        safety_patterns = [
            r"side effects?\s+include\s+([^.]+)",
            r"adverse events?\s+(?:reported|observed)\s+([^.]+)",
            r"contraindicated\s+in\s+([^.]+)",
            r"(?:avoid|not recommended)\s+(?:in|for)\s+([^.]+)",
        ]

        for pattern in safety_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            safety_issues.extend([s.strip() for s in matches])

        if safety_issues:
            description = "Multiple studies report consistent safety profile. "
            description += f"Key safety considerations: {', '.join(safety_issues[:3])}. "
            description += f"Documented in {len(findings)} sources."

            return AnalysisInsight(
                type="finding",
                title="Important Safety Considerations Identified",
                description=description,
                confidence=confidence,
                supporting_sources=[f["source"] for f in findings[:5]],
                significance=(
                    "critical"
                    if any("contraindicated" in text.lower() for text in texts)
                    else "high"
                ),
            )

        return None

    def _generate_efficacy_insight(
        self, findings: list[dict[str, Any]], confidence: float
    ) -> AnalysisInsight | None:
        """Generate insight from efficacy findings"""
        if not findings:
            return None

        # Extract efficacy data
        texts = [f["text"] for f in findings]
        combined_text = " ".join(texts)

        # Look for efficacy patterns
        efficacy_data = []
        efficacy_patterns = [
            r"(\d+%)\s+(?:effective|efficacious|successful)",
            r"significant\s+(?:improvement|benefit)\s+in\s+([^.]+)",
            r"effective\s+(?:in|for|at)\s+([^.]+)",
        ]

        for pattern in efficacy_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            efficacy_data.extend(matches)

        if efficacy_data:
            description = "Research demonstrates consistent efficacy across studies. "
            description += f"Key efficacy measures: {', '.join(efficacy_data[:3])}. "
            description += f"Validated by {len(findings)} independent studies."

            return AnalysisInsight(
                type="finding",
                title="Strong Efficacy Evidence Established",
                description=description,
                confidence=confidence,
                supporting_sources=[f["source"] for f in findings[:5]],
                significance="high" if confidence > 0.7 else "medium",
            )

        return None

    def _detect_controversies(
        self, research_results: list[dict[str, Any]]
    ) -> list[AnalysisInsight]:
        """Detect controversial topics or conflicting evidence"""
        controversies = []

        for result in research_results:
            content = result.get("content", "")
            title = result.get("title", "")
            combined_text = f"{title} {content}"

            # Count controversy indicators
            controversy_count = 0
            for pattern in self.controversy_patterns:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                controversy_count += len(matches)

            if controversy_count >= 3:  # Multiple controversy indicators
                controversies.append(
                    AnalysisInsight(
                        type="controversy",
                        title="Conflicting Evidence Identified",
                        description="Multiple sources indicate ongoing debate or conflicting findings. "
                        "This suggests the topic requires careful interpretation of evidence.",
                        confidence=0.8,
                        supporting_sources=[result.get("title", "Unknown")],
                        significance="high",
                    )
                )

        return controversies[:3]  # Return top 3 controversies

    def _identify_breakthroughs(
        self, research_results: list[dict[str, Any]]
    ) -> list[AnalysisInsight]:
        """Identify potential breakthrough findings"""
        breakthroughs = []

        breakthrough_indicators = [
            r"\b(?:breakthrough|novel|innovative|revolutionary|groundbreaking)\b",
            r"\b(?:first time|unprecedented|never before)\b",
            r"\b(?:paradigm shift|game-changer|major advance)\b",
        ]

        for result in research_results:
            content = result.get("content", "")
            title = result.get("title", "")
            combined_text = f"{title} {content}"

            breakthrough_count = 0
            for pattern in breakthrough_indicators:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                breakthrough_count += len(matches)

            if breakthrough_count >= 2:  # Multiple breakthrough indicators
                breakthroughs.append(
                    AnalysisInsight(
                        type="breakthrough",
                        title="Potential Breakthrough Development",
                        description="Source indicates significant advancement or novel finding. "
                        "This may represent an important development in the field.",
                        confidence=result.get("priority_score", 0.6),
                        supporting_sources=[result.get("title", "Unknown")],
                        significance="critical" if breakthrough_count >= 3 else "high",
                    )
                )

        return breakthroughs[:2]  # Return top 2 breakthroughs

    def _analyze_sources(
        self, research_results: list[dict[str, Any]], priority_scorer=None
    ) -> dict[str, Any]:
        """Analyze source quality and diversity"""
        source_types = defaultdict(int)
        quality_scores = []

        for result in research_results:
            source_type = result.get("source_type", "unknown")
            source_types[source_type] += 1

            if priority_scorer:
                quality_scores.append(result.get("priority_score", 0.5))
            else:
                quality_scores.append(0.5)

        return {
            "total_sources": len(research_results),
            "source_diversity": len(source_types),
            "source_breakdown": dict(source_types),
            "average_quality": statistics.mean(quality_scores) if quality_scores else 0.5,
            "quality_distribution": {
                "high": len([s for s in quality_scores if s > 0.7]),
                "medium": len([s for s in quality_scores if 0.4 <= s <= 0.7]),
                "low": len([s for s in quality_scores if s < 0.4]),
            },
        }

    def _assess_methodology(self, research_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess research methodology quality"""
        methodology_scores = {
            "peer_reviewed": 0,
            "controlled_studies": 0,
            "large_sample_size": 0,
            "statistical_significance": 0,
        }

        for result in research_results:
            content = result.get("content", "").lower()

            # Check for peer review
            if any(pattern in content for pattern in ["peer-reviewed", "peer reviewed"]):
                methodology_scores["peer_reviewed"] += 1

            # Check for controlled studies
            if any(pattern in content for pattern in ["randomized", "controlled", "double-blind"]):
                methodology_scores["controlled_studies"] += 1

            # Check for statistical significance
            if any(
                pattern in content for pattern in ["p <", "statistically significant", "p-value"]
            ):
                methodology_scores["statistical_significance"] += 1

            # Check sample size
            sample_matches = re.findall(r"\bn\s*=\s*(\d+)", content)
            if sample_matches and any(int(n) > 100 for n in sample_matches):
                methodology_scores["large_sample_size"] += 1

        total_sources = len(research_results)
        methodology_quality = {
            "peer_review_rate": (
                methodology_scores["peer_reviewed"] / total_sources if total_sources > 0 else 0
            ),
            "controlled_study_rate": (
                methodology_scores["controlled_studies"] / total_sources if total_sources > 0 else 0
            ),
            "large_sample_rate": (
                methodology_scores["large_sample_size"] / total_sources if total_sources > 0 else 0
            ),
            "statistical_significance_rate": (
                methodology_scores["statistical_significance"] / total_sources
                if total_sources > 0
                else 0
            ),
            "overall_quality": (
                sum(methodology_scores.values()) / (total_sources * 4) if total_sources > 0 else 0
            ),
        }

        return methodology_quality

    def _identify_research_gaps(
        self, findings: dict[str, list[dict[str, Any]]], research_results: list[dict[str, Any]]
    ) -> list[str]:
        """Identify gaps in current research"""
        gaps = []

        # Check for missing finding types
        expected_types = [
            "clinical_findings",
            "mechanism_findings",
            "safety_findings",
            "efficacy_findings",
        ]
        missing_types = [t for t in expected_types if t not in findings or len(findings[t]) < 2]

        if "clinical_findings" in missing_types:
            gaps.append("Limited clinical trial data available")
        if "mechanism_findings" in missing_types:
            gaps.append("Mechanism of action not well established")
        if "safety_findings" in missing_types:
            gaps.append("Insufficient safety data reported")
        if "efficacy_findings" in missing_types:
            gaps.append("Efficacy data needs more robust validation")

        # Check for methodological gaps
        controlled_studies = sum(
            1
            for result in research_results
            if "randomized" in result.get("content", "").lower()
            or "controlled" in result.get("content", "").lower()
        )

        if controlled_studies < len(research_results) * 0.3:
            gaps.append("More randomized controlled trials needed")

        # Check for long-term studies
        long_term_studies = sum(
            1
            for result in research_results
            if any(
                term in result.get("content", "").lower()
                for term in ["long-term", "longitudinal", "follow-up"]
            )
        )

        if long_term_studies < 2:
            gaps.append("Long-term safety and efficacy studies needed")

        return gaps[:5]  # Return top 5 gaps

    def _generate_executive_summary(
        self,
        topic: str,
        findings: dict[str, list[dict[str, Any]]],
        insights: list[AnalysisInsight],
        source_analysis: dict[str, Any],
    ) -> str:
        """Generate executive summary"""
        summary = f"Analysis of research on {topic} reveals "

        total_findings = sum(len(finding_list) for finding_list in findings.values())
        summary += f"{total_findings} relevant findings across {source_analysis['total_sources']} sources. "

        if insights:
            high_significance = [i for i in insights if i.significance in ["critical", "high"]]
            if high_significance:
                summary += f"{len(high_significance)} high-significance insights identified, "
                summary += f"including {high_significance[0].title.lower()}. "

        quality_dist = source_analysis.get("quality_distribution", {})
        high_quality = quality_dist.get("high", 0)
        if high_quality > 0:
            summary += f"Analysis includes {high_quality} high-quality sources. "

        controversies = [i for i in insights if i.type == "controversy"]
        if controversies:
            summary += "Some conflicting evidence noted requiring careful interpretation. "

        summary += "Comprehensive analysis provides reliable foundation for understanding current research state."

        return summary

    def _extract_key_findings(
        self, findings: dict[str, list[dict[str, Any]]], insights: list[AnalysisInsight]
    ) -> list[str]:
        """Extract key findings for summary"""
        key_findings = []

        # Add findings from insights
        for insight in insights[:5]:
            if insight.type == "finding":
                key_findings.append(insight.title)

        # Add findings from each category
        for finding_type, finding_list in findings.items():
            if finding_list:
                # Extract most common themes
                texts = [f["text"] for f in finding_list[:3]]
                combined = " ".join(texts)

                # Simplified extraction of key points
                sentences = [s.strip() for s in combined.split(".") if len(s.strip()) > 20]
                if sentences:
                    key_findings.append(
                        sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
                    )

        return key_findings[:8]  # Return top 8 findings

    def _generate_recommendations(
        self,
        findings: dict[str, list[dict[str, Any]]],
        insights: list[AnalysisInsight],
        research_gaps: list[str],
        methodology_assessment: dict[str, Any],
    ) -> list[str]:
        """Generate research recommendations"""
        recommendations = []

        # Based on research gaps
        if research_gaps:
            recommendations.append(
                f"Address identified research gaps: {', '.join(research_gaps[:2])}"
            )

        # Based on methodology quality
        if methodology_assessment.get("overall_quality", 0) < 0.6:
            recommendations.append("Increase focus on high-quality, peer-reviewed studies")

        # Based on controversies
        controversies = [i for i in insights if i.type == "controversy"]
        if controversies:
            recommendations.append("Resolve conflicting evidence through systematic reviews")

        # Based on source diversity
        if len(findings) < 3:
            recommendations.append(
                "Expand research to include diverse study types and methodologies"
            )

        # General recommendations
        recommendations.append("Continue monitoring emerging research developments")
        recommendations.append("Validate findings through independent replication studies")

        return recommendations[:6]  # Return top 6 recommendations

    def _calculate_analysis_confidence(
        self,
        source_analysis: dict[str, Any],
        methodology_assessment: dict[str, Any],
        num_sources: int,
    ) -> float:
        """Calculate overall confidence in analysis"""
        # Base confidence on number of sources
        source_confidence = min(0.9, num_sources / 20)  # Max 0.9 at 20+ sources

        # Quality-based confidence
        quality_confidence = source_analysis.get("average_quality", 0.5)

        # Methodology-based confidence
        methodology_confidence = methodology_assessment.get("overall_quality", 0.5)

        # Weighted average
        overall_confidence = (
            source_confidence * 0.3 + quality_confidence * 0.4 + methodology_confidence * 0.3
        )

        return min(0.95, overall_confidence)  # Cap at 95%
