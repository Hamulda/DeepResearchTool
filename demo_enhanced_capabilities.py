#!/usr/bin/env python3
"""
Demo script pro testování rozšířených funkcionalit Deep Research Tool
Demonstrace specializovaných scraperů, AI autentifikace a cross-source korelace

Author: Advanced IT Specialist
"""

import asyncio
import logging
import yaml
from datetime import datetime
from typing import Dict, Any, List
import json

# Import implementovaných modulů
from src.scrapers.declassified_scraper import DeclassifiedScraper
from src.scrapers.deep_web_crawler import DeepWebCrawler
from src.scrapers.osint_collector import OSINTCollector
from src.scrapers.historical_archives_scraper import HistoricalArchivesScraper
from src.analysis.specialized_analyzer import SpecializedAnalyzer
from src.analysis.ai_content_authentication import AIContentAuthenticator
from src.analysis.timeline_reconstruction import TimelineReconstructor

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedResearchDemo:
    """Demo třída pro testování rozšířených funkcionalit"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.initialize_components()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Načtení konfigurace"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Výchozí konfigurace pro demo"""
        return {
            "research_config": {
                "specialized_sources": {
                    "declassified_archives": {
                        "cia_crest": {"enabled": True, "rate_limit": 6},
                        "national_archives": {"enabled": True, "rate_limit": 20},
                    },
                    "deep_web": {
                        "tor_integration": {"enabled": False},  # Vypnuto pro demo
                        "safety_checks": True,
                    },
                    "historical_archives": {
                        "qatar_digital_library": {"enabled": True, "rate_limit": 15},
                        "chinese_text_project": {"enabled": True, "rate_limit": 10},
                    },
                },
                "osint": {"professional_tools": {"social_intelligence": {"enabled": True}}},
                "security": {"content_validation": {"enabled": True}},
            }
        }

    def initialize_components(self):
        """Inicializace všech komponent"""
        research_config = self.config.get("research_config", {})

        # Specializované scrapery
        self.declassified_scraper = DeclassifiedScraper(
            research_config.get("specialized_sources", {}).get("declassified_archives", {})
        )

        self.deep_web_crawler = DeepWebCrawler(research_config)

        self.osint_collector = OSINTCollector(research_config)

        self.historical_scraper = HistoricalArchivesScraper(research_config)

        # Analytické komponenty
        self.specialized_analyzer = SpecializedAnalyzer(research_config)

        self.ai_authenticator = AIContentAuthenticator(research_config)

        self.timeline_reconstructor = TimelineReconstructor(research_config)

    async def demo_declassified_documents_research(self, query: str) -> Dict[str, Any]:
        """Demo výzkumu declassified dokumentů"""
        logger.info(f"🔓 Demo: Searching declassified documents for '{query}'")

        results = {
            "query": query,
            "source_type": "declassified_documents",
            "documents": [],
            "analysis": {},
            "timeline": {},
            "authentication": {},
        }

        try:
            # Vyhledání declassified dokumentů
            documents = []
            async for doc in self.declassified_scraper.search(query, max_results=5):
                doc_dict = {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "classification_level": doc.classification_level,
                    "agency": doc.agency,
                    "topics": doc.topics,
                    "entities": doc.entities,
                    "redacted_sections": len(doc.redacted_sections),
                    "confidence_score": doc.confidence_score,
                }
                documents.append(doc_dict)
                results["documents"].append(doc_dict)

            logger.info(f"Found {len(documents)} declassified documents")

            # Analýza dokumentů
            if documents:
                analysis = await self.specialized_analyzer.analyze_declassified_documents(documents)
                results["analysis"] = {
                    "classification_patterns": len(analysis.get("classification_analysis", [])),
                    "entity_networks": analysis.get("entity_networks", {}),
                    "redaction_analysis": analysis.get("redaction_analysis", {}),
                    "historical_correlation": analysis.get("historical_correlation", {}),
                }

                # Timeline rekonstrukce
                timeline = await self.timeline_reconstructor.reconstruct_timeline(documents)
                results["timeline"] = {
                    "total_events": timeline.get("total_events", 0),
                    "confidence_metrics": timeline.get("confidence_metrics", {}),
                    "temporal_patterns": len(timeline.get("temporal_patterns", [])),
                }

                # AI autentifikace prvního dokumentu
                if documents:
                    auth_result = await self.ai_authenticator.authenticate_content(documents[0])
                    results["authentication"] = {
                        "authenticity_score": auth_result.authenticity_score,
                        "verification_status": auth_result.verification_status,
                        "confidence_level": auth_result.confidence_level,
                        "anomaly_flags": auth_result.anomaly_flags,
                    }

        except Exception as e:
            logger.error(f"Error in declassified documents demo: {e}")
            results["error"] = str(e)

        return results

    async def demo_historical_archives_research(self, query: str) -> Dict[str, Any]:
        """Demo výzkumu historických archivů"""
        logger.info(f"📜 Demo: Searching historical archives for '{query}'")

        results = {
            "query": query,
            "source_type": "historical_archives",
            "documents": [],
            "analysis": {},
            "supported_archives": [],
        }

        try:
            # Získání podporovaných archivů
            results["supported_archives"] = self.historical_scraper.get_supported_archives()

            # Vyhledání historických dokumentů
            documents = []
            async for doc in self.historical_scraper.search(
                query, archive_types=["qatar", "chinese"], max_results=3
            ):
                doc_dict = {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "time_period": doc.time_period,
                    "geographic_origin": doc.geographic_origin,
                    "original_language": doc.original_language,
                    "source_archive": doc.source_archive,
                    "historical_significance": doc.historical_significance,
                    "content_preview": (
                        doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                    ),
                }
                documents.append(doc_dict)
                results["documents"].append(doc_dict)

            logger.info(f"Found {len(documents)} historical documents")

            # Základní analýza
            if documents:
                language_distribution = {}
                time_periods = {}

                for doc in documents:
                    lang = doc["original_language"]
                    period = doc["time_period"]

                    language_distribution[lang] = language_distribution.get(lang, 0) + 1
                    time_periods[period] = time_periods.get(period, 0) + 1

                results["analysis"] = {
                    "language_distribution": language_distribution,
                    "time_period_distribution": time_periods,
                    "geographic_coverage": len(set(doc["geographic_origin"] for doc in documents)),
                }

        except Exception as e:
            logger.error(f"Error in historical archives demo: {e}")
            results["error"] = str(e)

        return results

    async def demo_osint_collection(self, target: str) -> Dict[str, Any]:
        """Demo OSINT sběru informací"""
        logger.info(f"🕵️ Demo: OSINT collection for '{target}'")

        results = {
            "target": target,
            "source_type": "osint_intelligence",
            "intelligence_results": [],
            "correlation_analysis": {},
            "statistics": {},
        }

        try:
            # OSINT sběr
            osint_results = []
            async for result in self.osint_collector.search(
                target,
                target_type="general",
                platforms=["reddit"],  # Omezeno pro demo
                max_results=3,
            ):
                result_dict = {
                    "source": result.source,
                    "result_type": result.result_type,
                    "confidence_score": result.confidence_score,
                    "source_reliability": result.source_reliability,
                    "information_credibility": result.information_credibility,
                    "collection_date": result.collection_date.isoformat(),
                    "data_summary": (
                        str(result.data)[:200] + "..."
                        if len(str(result.data)) > 200
                        else str(result.data)
                    ),
                }
                osint_results.append(result_dict)
                results["intelligence_results"].append(result_dict)

            logger.info(f"Collected {len(osint_results)} OSINT results")

            # Korelační analýza
            if osint_results:
                correlation = self.osint_collector.correlate_intelligence(
                    [self._convert_dict_to_osint_result(r) for r in osint_results]
                )
                results["correlation_analysis"] = {
                    "confidence_aggregate": correlation.get("confidence_aggregate", 0),
                    "source_diversity": correlation.get("source_diversity", 0),
                    "reliability_assessment": correlation.get("reliability_assessment", "unknown"),
                    "key_findings_count": len(correlation.get("key_findings", [])),
                }

            # Statistiky
            stats = self.osint_collector.get_collection_stats()
            results["statistics"] = stats

        except Exception as e:
            logger.error(f"Error in OSINT demo: {e}")
            results["error"] = str(e)

        return results

    async def demo_cross_source_correlation(self, query: str) -> Dict[str, Any]:
        """Demo cross-source korelace a verifikace"""
        logger.info(f"🔗 Demo: Cross-source correlation for '{query}'")

        results = {
            "query": query,
            "analysis_type": "cross_source_correlation",
            "sources_analyzed": 0,
            "correlations": [],
            "credibility_assessments": [],
            "timeline_analysis": {},
            "ai_authentication": {},
        }

        try:
            # Simulace více zdrojů pro korelaci
            mock_sources = [
                {
                    "id": "source_1",
                    "content": f"Information about {query} from government source. Official statement released.",
                    "source_type": "government_official",
                    "reliability_score": 0.9,
                    "creation_date": datetime(2023, 1, 15),
                },
                {
                    "id": "source_2",
                    "content": f"News report about {query}. Sources confirm the details.",
                    "source_type": "verified_news",
                    "reliability_score": 0.7,
                    "creation_date": datetime(2023, 1, 16),
                },
                {
                    "id": "source_3",
                    "content": f"Social media discussion about {query}. Mixed reactions from users.",
                    "source_type": "social_media_unverified",
                    "reliability_score": 0.3,
                    "creation_date": datetime(2023, 1, 17),
                },
            ]

            results["sources_analyzed"] = len(mock_sources)

            # Cross-source analýza
            correlation_analysis = await self.specialized_analyzer.perform_cross_source_correlation(
                mock_sources
            )

            results["correlations"] = [
                {
                    "correlation_id": corr.correlation_id,
                    "correlation_strength": corr.correlation_strength,
                    "semantic_similarity": corr.semantic_similarity,
                    "temporal_alignment": corr.temporal_alignment,
                    "factual_consistency": corr.factual_consistency,
                    "matching_entities_count": len(corr.matching_entities),
                }
                for corr in correlation_analysis.get("correlations", [])
            ]

            results["credibility_assessments"] = [
                {
                    "source_id": assess.source_id,
                    "reliability_score": assess.reliability_score,
                    "consistency_score": assess.consistency_score,
                    "bias_indicators_count": len(assess.bias_indicators),
                    "temporal_consistency": assess.temporal_consistency,
                }
                for assess in correlation_analysis.get("credibility_assessments", [])
            ]

            # Timeline analýza
            timeline_results = correlation_analysis.get("timeline_reconstruction", {})
            results["timeline_analysis"] = {
                "events_identified": len(timeline_results.get("timeline", [])),
                "confidence_metrics": timeline_results.get("confidence_metrics", {}),
                "methodology": timeline_results.get("methodology", {}),
            }

            # AI autentifikace
            if mock_sources:
                auth_result = await self.ai_authenticator.authenticate_content(mock_sources[0])
                bias_result = await self.ai_authenticator.detect_bias_and_propaganda(
                    mock_sources[0]
                )

                results["ai_authentication"] = {
                    "authenticity_score": auth_result.authenticity_score,
                    "verification_status": auth_result.verification_status,
                    "bias_strength": bias_result.bias_strength,
                    "bias_direction": bias_result.bias_direction,
                    "propaganda_indicators_count": len(bias_result.propaganda_indicators),
                }

        except Exception as e:
            logger.error(f"Error in cross-source correlation demo: {e}")
            results["error"] = str(e)

        return results

    async def demo_comprehensive_research(self, query: str) -> Dict[str, Any]:
        """Komprehensivní demo kombinující všechny funkcionality"""
        logger.info(f"🚀 Demo: Comprehensive research for '{query}'")

        comprehensive_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "research_modules": {},
            "summary": {},
            "recommendations": [],
        }

        # Spuštění všech demo modulů
        logger.info("Running declassified documents research...")
        comprehensive_results["research_modules"]["declassified"] = (
            await self.demo_declassified_documents_research(query)
        )

        logger.info("Running historical archives research...")
        comprehensive_results["research_modules"]["historical"] = (
            await self.demo_historical_archives_research(query)
        )

        logger.info("Running OSINT collection...")
        comprehensive_results["research_modules"]["osint"] = await self.demo_osint_collection(query)

        logger.info("Running cross-source correlation...")
        comprehensive_results["research_modules"]["correlation"] = (
            await self.demo_cross_source_correlation(query)
        )

        # Souhrnná analýza
        total_sources = 0
        total_documents = 0
        avg_confidence = 0
        confidence_scores = []

        for module_name, module_results in comprehensive_results["research_modules"].items():
            if "documents" in module_results:
                docs = module_results["documents"]
                total_documents += len(docs)

                for doc in docs:
                    if "confidence_score" in doc:
                        confidence_scores.append(doc["confidence_score"])

            if "intelligence_results" in module_results:
                intel = module_results["intelligence_results"]
                total_sources += len(intel)

                for result in intel:
                    if "confidence_score" in result:
                        confidence_scores.append(result["confidence_score"])

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

        comprehensive_results["summary"] = {
            "total_sources_searched": total_sources + total_documents,
            "total_documents_found": total_documents,
            "average_confidence": round(avg_confidence, 3),
            "modules_executed": len(comprehensive_results["research_modules"]),
            "research_depth": "comprehensive",
        }

        # Doporučení
        recommendations = []

        if avg_confidence > 0.7:
            recommendations.append(
                "High confidence in findings - results are reliable for analysis"
            )
        elif avg_confidence > 0.5:
            recommendations.append("Moderate confidence - consider additional source verification")
        else:
            recommendations.append("Low confidence - requires extensive cross-verification")

        if total_documents > 5:
            recommendations.append("Sufficient document base for trend analysis")

        if any("error" in module for module in comprehensive_results["research_modules"].values()):
            recommendations.append(
                "Some modules encountered errors - check configuration and API access"
            )

        comprehensive_results["recommendations"] = recommendations

        return comprehensive_results

    def _convert_dict_to_osint_result(self, result_dict: Dict[str, Any]):
        """Pomocná metoda pro konverzi slovníku na OSINT result objekt"""

        # Mock implementace pro demo
        class MockOSINTResult:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)

        return MockOSINTResult(result_dict)

    async def run_demo(self, query: str = "Cold War intelligence operations"):
        """Spuštění kompletního demo"""
        print("🔬 Deep Research Tool - Enhanced Capabilities Demo")
        print("=" * 60)
        print(f"Research Query: {query}")
        print("=" * 60)

        try:
            results = await self.demo_comprehensive_research(query)

            # Výstup výsledků
            print("\n📊 RESEARCH SUMMARY")
            print("-" * 30)
            summary = results["summary"]
            for key, value in summary.items():
                print(f"{key.replace('_', ' ').title()}: {value}")

            print("\n🎯 RECOMMENDATIONS")
            print("-" * 30)
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"{i}. {rec}")

            print("\n📋 MODULE RESULTS")
            print("-" * 30)
            for module_name, module_results in results["research_modules"].items():
                print(f"\n{module_name.upper()} MODULE:")
                if "error" in module_results:
                    print(f"  ❌ Error: {module_results['error']}")
                else:
                    if "documents" in module_results:
                        print(f"  📄 Documents found: {len(module_results['documents'])}")
                    if "intelligence_results" in module_results:
                        print(
                            f"  🔍 Intelligence results: {len(module_results['intelligence_results'])}"
                        )
                    if "analysis" in module_results:
                        print(f"  📈 Analysis completed: ✅")

            # Uložení výsledků
            output_file = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f"\n💾 Detailed results saved to: {output_file}")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)


async def main():
    """Hlavní funkce pro spuštění demo"""
    demo = EnhancedResearchDemo()

    # Testovací queries
    test_queries = [
        "Cold War intelligence operations",
        "CIA declassified documents",
        "Ming dynasty historical records",
        "Middle East diplomatic relations",
    ]

    print("🚀 Starting Enhanced Deep Research Tool Demo")
    print("Available test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")

    try:
        choice = input("\nSelect query number (1-4) or enter custom query: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(test_queries):
            selected_query = test_queries[int(choice) - 1]
        else:
            selected_query = choice if choice else test_queries[0]

        await demo.run_demo(selected_query)

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
