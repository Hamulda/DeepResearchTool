#!/usr/bin/env python3
"""Intelligence Synthesis Engine - Hlavní integrační modul pro Fázi 3
Koordinace všech komponent syntézy zpravodajských informací

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any

from .correlation_engine import CorrelationEngine, Entity, NetworkAnalysisResult, Relationship
from .credibility_assessor import CredibilityAssessment, CredibilityAssessor, SourceMetadata
from .deep_pattern_detector import ArtefactExtraction, DeepPatternDetector, PatternMatch
from .steganography_analyzer import SteganographyAnalyzer, SteganographyResult

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceSynthesisResult:
    """Kompletní výsledek syntézy zpravodajských informací"""

    analysis_id: str
    timestamp: str

    # Pattern detection results
    pattern_matches: list[PatternMatch]
    extracted_artefacts: list[ArtefactExtraction]
    pattern_statistics: dict[str, Any]

    # Steganography analysis results
    steganography_results: list[SteganographyResult]
    media_analysis_summary: dict[str, Any]

    # Correlation analysis results
    entities: list[Entity]
    relationships: list[Relationship]
    network_analysis: NetworkAnalysisResult
    correlation_statistics: dict[str, Any]

    # Credibility assessment results
    credibility_assessments: list[CredibilityAssessment]
    credibility_report: dict[str, Any]

    # Overall synthesis metrics
    synthesis_confidence: float
    intelligence_score: float
    risk_level: str
    key_findings: list[dict[str, Any]]
    recommendations: list[str]


@dataclass
class IntelligenceSource:
    """Zdroj informací pro analýzu"""

    source_id: str
    content_type: str  # text, image, audio, document
    content: Any  # text, bytes, file path
    metadata: SourceMetadata
    processing_priority: int = 1  # 1-5, where 5 is highest priority


class IntelligenceSynthesisEngine:
    """Hlavní engine pro syntézu zpravodajských informací"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.synthesis_config = config.get("intelligence_synthesis", {})

        # Inicializace komponent
        self.pattern_detector = DeepPatternDetector(config)
        self.steganography_analyzer = SteganographyAnalyzer(config)
        self.correlation_engine = CorrelationEngine(config)
        self.credibility_assessor = CredibilityAssessor(config)

        # Konfigurace zpracování
        self.enable_pattern_detection = self.synthesis_config.get("enable_pattern_detection", True)
        self.enable_steganography = self.synthesis_config.get("enable_steganography_analysis", True)
        self.enable_correlation = self.synthesis_config.get("enable_correlation_analysis", True)
        self.enable_credibility = self.synthesis_config.get("enable_credibility_assessment", True)

        # Prahy a limity
        self.max_concurrent_analyses = self.synthesis_config.get("max_concurrent_analyses", 5)
        self.intelligence_threshold = self.synthesis_config.get("intelligence_threshold", 0.7)
        self.risk_threshold = self.synthesis_config.get("risk_threshold", 0.6)

        # Cache a statistiky
        self.analysis_cache: dict[str, IntelligenceSynthesisResult] = {}
        self.processing_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
        }

    async def synthesize_intelligence(
        self, sources: list[IntelligenceSource], analysis_name: str = "intelligence_analysis"
    ) -> IntelligenceSynthesisResult:
        """Hlavní metoda pro syntézu zpravodajských informací"""
        analysis_id = f"{analysis_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info(f"Starting intelligence synthesis: {analysis_id} with {len(sources)} sources")

        try:
            # Seřazení zdrojů podle priority
            sources.sort(key=lambda s: s.processing_priority, reverse=True)

            # Rozdělení zdrojů podle typu obsahu
            text_sources = [s for s in sources if s.content_type == "text"]
            media_sources = [s for s in sources if s.content_type in ["image", "audio", "document"]]

            # Paralelní analýza různých komponent
            tasks = []

            # 1. Pattern Detection na textových zdrojích
            if self.enable_pattern_detection and text_sources:
                tasks.append(self._analyze_patterns(text_sources))

            # 2. Steganography Analysis na mediálních souborech
            if self.enable_steganography and media_sources:
                tasks.append(self._analyze_steganography(media_sources))

            # 3. Correlation Analysis na všech zdrojích
            if self.enable_correlation:
                tasks.append(self._analyze_correlations(sources))

            # 4. Credibility Assessment na všech zdrojích
            if self.enable_credibility:
                tasks.append(self._assess_credibility(sources))

            # Spuštění paralelních analýz
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Zpracování výsledků
            pattern_results = None
            steganography_results = None
            correlation_results = None
            credibility_results = None

            result_index = 0
            if self.enable_pattern_detection and text_sources:
                pattern_results = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            if self.enable_steganography and media_sources:
                steganography_results = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            if self.enable_correlation:
                correlation_results = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            if self.enable_credibility:
                credibility_results = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            # Syntéza výsledků
            synthesis_result = await self._synthesize_results(
                analysis_id=analysis_id,
                pattern_results=pattern_results,
                steganography_results=steganography_results,
                correlation_results=correlation_results,
                credibility_results=credibility_results,
                sources=sources,
            )

            # Cache výsledek
            self.analysis_cache[analysis_id] = synthesis_result

            # Statistiky
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats["total_analyses"] += 1
            self.processing_stats["successful_analyses"] += 1
            self.processing_stats["average_processing_time"] = (
                self.processing_stats["average_processing_time"]
                * (self.processing_stats["total_analyses"] - 1)
                + processing_time
            ) / self.processing_stats["total_analyses"]

            logger.info(
                f"Intelligence synthesis completed: {analysis_id} in {processing_time:.2f}s"
            )

            return synthesis_result

        except Exception as e:
            logger.error(f"Error in intelligence synthesis {analysis_id}: {e}")
            self.processing_stats["total_analyses"] += 1
            self.processing_stats["failed_analyses"] += 1

            # Fallback výsledek
            return await self._create_fallback_result(analysis_id, str(e))

    async def _analyze_patterns(
        self, text_sources: list[IntelligenceSource]
    ) -> tuple[list[PatternMatch], list[ArtefactExtraction], dict[str, Any]]:
        """Analýza vzorů v textových zdrojích"""
        logger.info("Starting pattern detection analysis")

        all_matches = []
        all_artefacts = []

        for source in text_sources:
            try:
                if isinstance(source.content, str):
                    matches = await self.pattern_detector.detect_patterns(
                        source.content, source.source_id
                    )
                    all_matches.extend(matches)

                    # Extrakce artefaktů z nalezených vzorů
                    artefacts = await self.pattern_detector.extract_artefacts(matches)
                    all_artefacts.extend(artefacts)

            except Exception as e:
                logger.warning(
                    f"Error processing source {source.source_id} for pattern detection: {e}"
                )

        # Generování statistik
        pattern_statistics = self.pattern_detector.get_detection_statistics()
        pattern_report = await self.pattern_detector.generate_pattern_report(all_matches)
        pattern_statistics.update(pattern_report)

        logger.info(
            f"Pattern detection completed: {len(all_matches)} patterns, {len(all_artefacts)} artefacts"
        )

        return all_matches, all_artefacts, pattern_statistics

    async def _analyze_steganography(
        self, media_sources: list[IntelligenceSource]
    ) -> tuple[list[SteganographyResult], dict[str, Any]]:
        """Analýza steganografie v mediálních souborech"""
        logger.info("Starting steganography analysis")

        steganography_results = []

        # Omezení souběžných analýz (steganografie je náročná na výpočty)
        semaphore = asyncio.Semaphore(self.max_concurrent_analyses)

        async def analyze_single_media(source: IntelligenceSource):
            async with semaphore:
                try:
                    if isinstance(source.content, bytes):
                        result = await self.steganography_analyzer.analyze_media_file(
                            source.source_id, source.content
                        )
                        return result
                    if isinstance(source.content, str):  # File path
                        with open(source.content, "rb") as f:
                            content = f.read()
                        result = await self.steganography_analyzer.analyze_media_file(
                            source.content, content
                        )
                        return result
                except Exception as e:
                    logger.warning(f"Error analyzing media {source.source_id}: {e}")
                    return None

        # Paralelní analýza mediálních souborů
        tasks = [analyze_single_media(source) for source in media_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrování úspěšných výsledků
        steganography_results = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]

        # Generování souhrnné zprávy
        media_analysis_summary = await self.steganography_analyzer.generate_steganography_report(
            steganography_results
        )

        logger.info(
            f"Steganography analysis completed: {len(steganography_results)} files analyzed"
        )

        return steganography_results, media_analysis_summary

    async def _analyze_correlations(
        self, sources: list[IntelligenceSource]
    ) -> tuple[list[Entity], list[Relationship], NetworkAnalysisResult, dict[str, Any]]:
        """Analýza korelací a vztahů mezi entitami"""
        logger.info("Starting correlation analysis")

        # Příprava dokumentů pro analýzu
        documents = {}
        for source in sources:
            if source.content_type == "text" and isinstance(source.content, str):
                documents[source.source_id] = source.content

        if not documents:
            logger.warning("No text documents available for correlation analysis")
            return (
                [],
                [],
                NetworkAnalysisResult(
                    total_entities=0,
                    total_relationships=0,
                    network_density=0.0,
                    connected_components=0,
                    largest_component_size=0,
                    key_entities=[],
                    important_relationships=[],
                    entity_clusters=[],
                    network_metrics={},
                    anomalous_patterns=[],
                ),
                {},
            )

        # Budování grafu vztahů
        relationship_graph = await self.correlation_engine.build_relationship_graph(documents)

        # Analýza síťových vlastností
        network_analysis = await self.correlation_engine.analyze_network()

        # Extrakce entit a vztahů
        all_entities = list(self.correlation_engine.entities.values())
        all_relationships = network_analysis.important_relationships

        # Statistiky
        correlation_statistics = self.correlation_engine.get_correlation_statistics()

        logger.info(
            f"Correlation analysis completed: {len(all_entities)} entities, {len(all_relationships)} relationships"
        )

        return all_entities, all_relationships, network_analysis, correlation_statistics

    async def _assess_credibility(
        self, sources: list[IntelligenceSource]
    ) -> tuple[list[CredibilityAssessment], dict[str, Any]]:
        """Hodnocení důvěryhodnosti zdrojů"""
        logger.info("Starting credibility assessment")

        credibility_assessments = []

        # Příprava zdrojů pro hodnocení
        assessment_tasks = []
        for source in sources:
            if source.content_type == "text" and isinstance(source.content, str):
                assessment_tasks.append(
                    self.credibility_assessor.assess_source_credibility(
                        source.content, source.metadata
                    )
                )

        # Paralelní hodnocení
        if assessment_tasks:
            credibility_assessments = await asyncio.gather(
                *assessment_tasks, return_exceptions=True
            )
            credibility_assessments = [
                a for a in credibility_assessments if not isinstance(a, Exception)
            ]

        # Generování zprávy o důvěryhodnosti
        credibility_report = await self.credibility_assessor.generate_credibility_report(
            credibility_assessments
        )

        logger.info(
            f"Credibility assessment completed: {len(credibility_assessments)} sources assessed"
        )

        return credibility_assessments, credibility_report

    async def _synthesize_results(
        self,
        analysis_id: str,
        pattern_results: tuple | None,
        steganography_results: tuple | None,
        correlation_results: tuple | None,
        credibility_results: tuple | None,
        sources: list[IntelligenceSource],
    ) -> IntelligenceSynthesisResult:
        """Syntéza všech výsledků analýzy"""
        # Extrakce výsledků jednotlivých komponent
        pattern_matches = pattern_results[0] if pattern_results else []
        extracted_artefacts = pattern_results[1] if pattern_results else []
        pattern_statistics = pattern_results[2] if pattern_results else {}

        steganography_analysis = steganography_results[0] if steganography_results else []
        media_analysis_summary = steganography_results[1] if steganography_results else {}

        entities = correlation_results[0] if correlation_results else []
        relationships = correlation_results[1] if correlation_results else []
        network_analysis = (
            correlation_results[2]
            if correlation_results
            else NetworkAnalysisResult(
                total_entities=0,
                total_relationships=0,
                network_density=0.0,
                connected_components=0,
                largest_component_size=0,
                key_entities=[],
                important_relationships=[],
                entity_clusters=[],
                network_metrics={},
                anomalous_patterns=[],
            )
        )
        correlation_statistics = correlation_results[3] if correlation_results else {}

        credibility_assessments = credibility_results[0] if credibility_results else []
        credibility_report = credibility_results[1] if credibility_results else {}

        # Výpočet celkových metrik
        synthesis_confidence = await self._calculate_synthesis_confidence(
            pattern_results, steganography_results, correlation_results, credibility_results
        )

        intelligence_score = await self._calculate_intelligence_score(
            pattern_matches, extracted_artefacts, entities, relationships, credibility_assessments
        )

        risk_level = await self._determine_risk_level(
            extracted_artefacts, steganography_analysis, credibility_assessments, intelligence_score
        )

        # Generování klíčových poznatků
        key_findings = await self._generate_key_findings(
            pattern_matches,
            extracted_artefacts,
            entities,
            relationships,
            steganography_analysis,
            credibility_assessments,
        )

        # Generování doporučení
        recommendations = await self._generate_recommendations(
            key_findings, risk_level, intelligence_score, credibility_assessments
        )

        return IntelligenceSynthesisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            pattern_matches=pattern_matches,
            extracted_artefacts=extracted_artefacts,
            pattern_statistics=pattern_statistics,
            steganography_results=steganography_analysis,
            media_analysis_summary=media_analysis_summary,
            entities=entities,
            relationships=relationships,
            network_analysis=network_analysis,
            correlation_statistics=correlation_statistics,
            credibility_assessments=credibility_assessments,
            credibility_report=credibility_report,
            synthesis_confidence=synthesis_confidence,
            intelligence_score=intelligence_score,
            risk_level=risk_level,
            key_findings=key_findings,
            recommendations=recommendations,
        )

    async def _calculate_synthesis_confidence(
        self,
        pattern_results: tuple | None,
        steganography_results: tuple | None,
        correlation_results: tuple | None,
        credibility_results: tuple | None,
    ) -> float:
        """Výpočet celkové konfidence syntézy"""
        confidence_components = []

        # Konfidence z jednotlivých komponent
        if pattern_results:
            confidence_components.append(0.8)  # Pattern detection je spolehlivá

        if steganography_results:
            confidence_components.append(0.7)  # Steganografie má středně vysokou spolehlivost

        if correlation_results:
            confidence_components.append(0.9)  # Korelační analýza je velmi spolehlivá

        if credibility_results:
            confidence_components.append(0.85)  # Credibility assessment je spolehlivá

        if not confidence_components:
            return 0.3  # Minimální konfidence pokud žádná analýza neproběhla

        # Vážený průměr s bonusem za více komponent
        base_confidence = sum(confidence_components) / len(confidence_components)
        component_bonus = min(0.1, len(confidence_components) * 0.02)

        return min(1.0, base_confidence + component_bonus)

    async def _calculate_intelligence_score(
        self,
        pattern_matches: list[PatternMatch],
        extracted_artefacts: list[ArtefactExtraction],
        entities: list[Entity],
        relationships: list[Relationship],
        credibility_assessments: list[CredibilityAssessment],
    ) -> float:
        """Výpočet celkového intelligence skóre"""
        score_components = []

        # Skóre z pattern detection
        if pattern_matches:
            high_confidence_patterns = [p for p in pattern_matches if p.confidence > 0.8]
            pattern_score = min(1.0, len(high_confidence_patterns) / 10.0)
            score_components.append(("patterns", pattern_score, 0.25))

        # Skóre z artefaktů
        if extracted_artefacts:
            valuable_artefacts = [a for a in extracted_artefacts if a.confidence > 0.7]
            artefact_score = min(1.0, len(valuable_artefacts) / 5.0)
            score_components.append(("artefacts", artefact_score, 0.3))

        # Skóre z entit a vztahů
        if entities and relationships:
            entity_density = min(1.0, len(entities) / 50.0)
            relationship_strength = (
                sum(r.strength for r in relationships[:10]) / 10.0 if relationships else 0
            )
            correlation_score = (entity_density + relationship_strength) / 2.0
            score_components.append(("correlations", correlation_score, 0.25))

        # Skóre z důvěryhodnosti
        if credibility_assessments:
            avg_credibility = sum(
                a.overall_credibility_score for a in credibility_assessments
            ) / len(credibility_assessments)
            score_components.append(("credibility", avg_credibility, 0.2))

        # Vážený průměr
        if not score_components:
            return 0.3

        weighted_score = sum(score * weight for _, score, weight in score_components)
        total_weight = sum(weight for _, _, weight in score_components)

        return weighted_score / total_weight if total_weight > 0 else 0.3

    async def _determine_risk_level(
        self,
        extracted_artefacts: list[ArtefactExtraction],
        steganography_analysis: list[SteganographyResult],
        credibility_assessments: list[CredibilityAssessment],
        intelligence_score: float,
    ) -> str:
        """Určení úrovně rizika"""
        risk_factors = []

        # Rizikové artefakty
        high_risk_artefacts = [
            a
            for a in extracted_artefacts
            if a.artefact_type in ["cryptocurrency", "darknet", "communication"]
            and a.confidence > 0.7
        ]
        if high_risk_artefacts:
            risk_factors.append(len(high_risk_artefacts) * 0.2)

        # Steganografická rizika
        suspicious_media = [s for s in steganography_analysis if s.suspicion_score > 0.7]
        if suspicious_media:
            risk_factors.append(len(suspicious_media) * 0.3)

        # Nedůvěryhodné zdroje
        low_credibility_sources = [
            a for a in credibility_assessments if a.overall_credibility_score < 0.4
        ]
        if low_credibility_sources:
            risk_factors.append(len(low_credibility_sources) * 0.1)

        # Vysoké intelligence skóre může znamenat vyšší riziko
        if intelligence_score > 0.8:
            risk_factors.append(0.2)

        total_risk_score = sum(risk_factors)

        if total_risk_score > self.risk_threshold:
            return "HIGH"
        if total_risk_score > 0.3:
            return "MEDIUM"
        return "LOW"

    async def _generate_key_findings(
        self,
        pattern_matches: list[PatternMatch],
        extracted_artefacts: list[ArtefactExtraction],
        entities: list[Entity],
        relationships: list[Relationship],
        steganography_analysis: list[SteganographyResult],
        credibility_assessments: list[CredibilityAssessment],
    ) -> list[dict[str, Any]]:
        """Generování klíčových poznatků"""
        findings = []

        # Top pattern matches
        high_confidence_patterns = sorted(
            [p for p in pattern_matches if p.confidence > 0.8],
            key=lambda x: x.confidence,
            reverse=True,
        )[:5]

        for pattern in high_confidence_patterns:
            findings.append(
                {
                    "type": "pattern_detection",
                    "title": f"High-confidence {pattern.pattern_type} detected",
                    "description": f"Found {pattern.pattern_name}: {pattern.matched_text[:50]}...",
                    "confidence": pattern.confidence,
                    "importance": "high",
                }
            )

        # Valuable artefacts
        valuable_artefacts = sorted(
            [a for a in extracted_artefacts if a.confidence > 0.7],
            key=lambda x: x.confidence,
            reverse=True,
        )[:3]

        for artefact in valuable_artefacts:
            findings.append(
                {
                    "type": "artefact_extraction",
                    "title": f"Valuable {artefact.artefact_type} artefact",
                    "description": f"Extracted {artefact.value[:50]}...",
                    "confidence": artefact.confidence,
                    "importance": "high",
                }
            )

        # Key entities
        key_entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:3]
        for entity in key_entities:
            findings.append(
                {
                    "type": "entity_detection",
                    "title": f"Key {entity.label} entity identified",
                    "description": f"Entity: {entity.text}",
                    "confidence": entity.confidence,
                    "importance": "medium",
                }
            )

        # Suspicious media
        suspicious_media = [s for s in steganography_analysis if s.suspicion_score > 0.7]
        for media in suspicious_media[:2]:
            findings.append(
                {
                    "type": "steganography_detection",
                    "title": "Suspicious media file detected",
                    "description": f"File {media.file_path} shows steganography indicators",
                    "confidence": media.confidence,
                    "importance": "high",
                }
            )

        # Low credibility sources
        low_cred_sources = sorted(
            [a for a in credibility_assessments if a.overall_credibility_score < 0.4],
            key=lambda x: x.overall_credibility_score,
        )[:2]

        for source in low_cred_sources:
            findings.append(
                {
                    "type": "credibility_warning",
                    "title": "Low credibility source detected",
                    "description": f"Source {source.source_metadata.domain} has low credibility score",
                    "confidence": source.confidence,
                    "importance": "medium",
                }
            )

        return findings

    async def _generate_recommendations(
        self,
        key_findings: list[dict[str, Any]],
        risk_level: str,
        intelligence_score: float,
        credibility_assessments: list[CredibilityAssessment],
    ) -> list[str]:
        """Generování doporučení na základě analýzy"""
        recommendations = []

        # Doporučení podle úrovně rizika
        if risk_level == "HIGH":
            recommendations.append(
                "Immediate investigation recommended due to high-risk indicators"
            )
            recommendations.append("Verify all findings through independent sources")
            recommendations.append("Consider implementing enhanced security monitoring")
        elif risk_level == "MEDIUM":
            recommendations.append("Further analysis recommended for identified patterns")
            recommendations.append("Cross-reference findings with threat intelligence databases")
        else:
            recommendations.append("Continue routine monitoring of identified patterns")

        # Doporučení podle intelligence skóre
        if intelligence_score > self.intelligence_threshold:
            recommendations.append("High intelligence value detected - prioritize this analysis")
            recommendations.append("Share findings with relevant intelligence teams")

        # Doporučení podle credibility
        low_cred_count = len(
            [a for a in credibility_assessments if a.overall_credibility_score < 0.4]
        )
        if low_cred_count > len(credibility_assessments) * 0.3:
            recommendations.append(
                "High proportion of low-credibility sources - verify findings independently"
            )

        # Doporučení podle typu nálezů
        finding_types = [f["type"] for f in key_findings]
        if "steganography_detection" in finding_types:
            recommendations.append("Forensic analysis recommended for suspicious media files")

        if "artefact_extraction" in finding_types:
            recommendations.append("Investigate extracted artefacts for operational intelligence")

        return recommendations

    async def _create_fallback_result(
        self, analysis_id: str, error: str
    ) -> IntelligenceSynthesisResult:
        """Vytvoření fallback výsledku při chybě"""
        return IntelligenceSynthesisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            pattern_matches=[],
            extracted_artefacts=[],
            pattern_statistics={"error": error},
            steganography_results=[],
            media_analysis_summary={"error": error},
            entities=[],
            relationships=[],
            network_analysis=NetworkAnalysisResult(
                total_entities=0,
                total_relationships=0,
                network_density=0.0,
                connected_components=0,
                largest_component_size=0,
                key_entities=[],
                important_relationships=[],
                entity_clusters=[],
                network_metrics={},
                anomalous_patterns=[],
            ),
            correlation_statistics={"error": error},
            credibility_assessments=[],
            credibility_report={"error": error},
            synthesis_confidence=0.1,
            intelligence_score=0.0,
            risk_level="UNKNOWN",
            key_findings=[
                {
                    "type": "error",
                    "title": "Analysis failed",
                    "description": error,
                    "confidence": 0.0,
                    "importance": "high",
                }
            ],
            recommendations=[
                "Re-run analysis after addressing the error",
                "Check input data quality",
            ],
        )

    async def export_analysis_results(
        self, result: IntelligenceSynthesisResult, export_path: str, format: str = "json"
    ) -> str:
        """Export výsledků analýzy"""
        output_path = f"{export_path}.{format}"

        try:
            if format.lower() == "json":
                # Převod dataclasses na dict pro JSON serialization
                result_dict = self._dataclass_to_dict(result)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

            elif format.lower() == "html":
                html_content = await self._generate_html_report(result)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Analysis results exported to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            raise

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Rekurzivní převod dataclasses na dictionary"""
        if hasattr(obj, "__dataclass_fields__"):
            return {
                field: self._dataclass_to_dict(getattr(obj, field))
                for field in obj.__dataclass_fields__
            }
        if isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        return obj

    async def _generate_html_report(self, result: IntelligenceSynthesisResult) -> str:
        """Generování HTML zprávy"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Intelligence Synthesis Report - {result.analysis_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }}
                .high-risk {{ background-color: #ffebee; }}
                .medium-risk {{ background-color: #fff3e0; }}
                .low-risk {{ background-color: #e8f5e8; }}
                .finding {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Intelligence Synthesis Report</h1>
                <p><strong>Analysis ID:</strong> {result.analysis_id}</p>
                <p><strong>Timestamp:</strong> {result.timestamp}</p>
                <p><strong>Intelligence Score:</strong> {result.intelligence_score:.2f}</p>
                <p><strong>Risk Level:</strong> <span class="{result.risk_level.lower()}-risk">{result.risk_level}</span></p>
                <p><strong>Synthesis Confidence:</strong> {result.synthesis_confidence:.2f}</p>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                {"".join(f'<div class="finding"><strong>{f["title"]}</strong><br>{f["description"]}<br><em>Confidence: {f["confidence"]:.2f}</em></div>' for f in result.key_findings)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in result.recommendations)}
                </ul>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <table>
                    <tr><th>Component</th><th>Results</th></tr>
                    <tr><td>Pattern Matches</td><td>{len(result.pattern_matches)}</td></tr>
                    <tr><td>Extracted Artefacts</td><td>{len(result.extracted_artefacts)}</td></tr>
                    <tr><td>Entities</td><td>{len(result.entities)}</td></tr>
                    <tr><td>Relationships</td><td>{len(result.relationships)}</td></tr>
                    <tr><td>Steganography Results</td><td>{len(result.steganography_results)}</td></tr>
                    <tr><td>Credibility Assessments</td><td>{len(result.credibility_assessments)}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        return html_template

    def get_synthesis_statistics(self) -> dict[str, Any]:
        """Získání statistik syntézy"""
        return {
            "processing_stats": self.processing_stats,
            "cache_size": len(self.analysis_cache),
            "component_stats": {
                "pattern_detector": self.pattern_detector.get_detection_statistics(),
                "steganography_analyzer": self.steganography_analyzer.get_analysis_statistics(),
                "correlation_engine": self.correlation_engine.get_correlation_statistics(),
                "credibility_assessor": self.credibility_assessor.get_assessment_statistics(),
            },
            "configuration": {
                "enable_pattern_detection": self.enable_pattern_detection,
                "enable_steganography": self.enable_steganography,
                "enable_correlation": self.enable_correlation,
                "enable_credibility": self.enable_credibility,
                "intelligence_threshold": self.intelligence_threshold,
                "risk_threshold": self.risk_threshold,
            },
        }
