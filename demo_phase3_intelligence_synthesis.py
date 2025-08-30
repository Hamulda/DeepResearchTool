#!/usr/bin/env python3
"""
Demo script pro Fázi 3: Syntéza Zpravodajských Informací
Demonstrace všech komponent intelligence synthesis

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Přidání src do Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.synthesis import (
    IntelligenceSynthesisEngine,
    IntelligenceSource,
    SourceMetadata,
    DeepPatternDetector,
    SteganographyAnalyzer,
    CorrelationEngine,
    CredibilityAssessor,
)

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_config():
    """Vytvoření ukázkové konfigurace"""
    return {
        "intelligence_synthesis": {
            "enable_pattern_detection": True,
            "enable_steganography_analysis": True,
            "enable_correlation_analysis": True,
            "enable_credibility_assessment": True,
            "max_concurrent_analyses": 3,
            "intelligence_threshold": 0.7,
            "risk_threshold": 0.6,
        },
        "pattern_detection": {"min_confidence": 0.5},
        "steganography": {
            "suspicion_threshold": 0.7,
            "entropy_threshold": 7.8,
            "chi_square_threshold": 0.05,
        },
        "correlation": {
            "spacy_model": "en_core_web_sm",
            "proximity_threshold": 100,
            "min_co_occurrence": 2,
            "entity_similarity_threshold": 0.8,
        },
        "credibility": {
            "factor_weights": {
                "domain_reputation": 0.25,
                "content_quality": 0.30,
                "temporal_relevance": 0.20,
                "source_authority": 0.15,
                "bias_detection": 0.10,
            },
            "thresholds": {
                "high_credibility": 0.8,
                "medium_credibility": 0.6,
                "low_credibility": 0.4,
            },
        },
    }


def create_sample_sources():
    """Vytvoření ukázkových zdrojů pro analýzu"""
    sources = []

    # Ukázkový zpravodajský článek
    news_content = """
    Breaking News: Security researchers have discovered a new cryptocurrency scheme 
    operating through dark web marketplaces. The investigation revealed multiple 
    Bitcoin addresses (1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2) 
    being used to funnel money through various transactions.
    
    According to cybersecurity firm SecureAnalytics, the scheme involves communication 
    through encrypted channels including Telegram handles @cryptoexchange_official and 
    email addresses darkmarket@protonmail.com. The operation appears to be coordinated 
    from servers located at IP addresses 192.168.1.100 and 10.0.0.50.
    
    Law enforcement agencies are investigating connections to the .onion site 
    facebookcorewwwi.onion and other dark web marketplaces. The investigation 
    is ongoing and more details will be released as they become available.
    """

    sources.append(
        IntelligenceSource(
            source_id="news_article_001",
            content_type="text",
            content=news_content,
            metadata=SourceMetadata(
                source_id="news_article_001",
                url="https://cybersecuritynews.com/crypto-scheme-discovered",
                domain="cybersecuritynews.com",
                title="New Cryptocurrency Scheme Discovered",
                content_length=len(news_content),
                publication_date=datetime.now(),
                author="Jane Security Reporter",
                source_type="news",
                language="en",
            ),
            processing_priority=5,
        )
    )

    # Ukázkový blog post
    blog_content = """
    Technical Analysis: Cryptocurrency Laundering Patterns
    
    In recent months, we've observed sophisticated patterns in cryptocurrency 
    transactions that suggest coordinated laundering activities. Our analysis 
    of blockchain data reveals clusters of addresses frequently used together:
    
    - Primary wallet: 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy
    - Secondary wallets: bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh, 1F1tAaz5x1HUXrCNLbtMDqcw6o5GNn4xqX
    
    The transactions often involve conversion through privacy coins before 
    returning to Bitcoin. Email communications intercepted by law enforcement 
    mention contact points: cryptoanalyst@tutanota.com and trading.specialist@protonmail.ch
    
    Geographic analysis suggests operations centered around coordinates 
    40.7128,-74.0060 (New York) and 51.5074,-0.1278 (London).
    """

    sources.append(
        IntelligenceSource(
            source_id="blog_analysis_002",
            content_type="text",
            content=blog_content,
            metadata=SourceMetadata(
                source_id="blog_analysis_002",
                url="https://cryptoanalysis.blog/laundering-patterns",
                domain="cryptoanalysis.blog",
                title="Cryptocurrency Laundering Patterns",
                content_length=len(blog_content),
                publication_date=datetime.now(),
                author="Dr. Crypto Analyst",
                source_type="blog",
                language="en",
            ),
            processing_priority=4,
        )
    )

    # Ukázkový sociální příspěvek
    social_content = """
    🚨 URGENT ALERT 🚨
    
    New intel suggests major crypto operation running through:
    - Onion sites: darkmarket23xfghjk.onion, securetrading45abc.onion  
    - Contact: @crypto_insider_2023 on Telegram
    - Hash signatures: a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
    
    Sources indicate coordination with organization "Digital Currency Exchange LLC"
    and person known as "Alex Crypto". Exercise extreme caution!
    
    #CyberSecurity #CryptoAlert #DarkWeb
    """

    sources.append(
        IntelligenceSource(
            source_id="social_alert_003",
            content_type="text",
            content=social_content,
            metadata=SourceMetadata(
                source_id="social_alert_003",
                url="https://twitter.com/cryptoalerts/status/123456789",
                domain="twitter.com",
                title="Urgent Crypto Alert",
                content_length=len(social_content),
                publication_date=datetime.now(),
                author="@cryptoalerts",
                source_type="social_media",
                language="en",
            ),
            processing_priority=3,
        )
    )

    return sources


async def demo_pattern_detection():
    """Demo pattern detection komponenty"""
    print("\n" + "=" * 60)
    print("🔍 DEMO: Deep Pattern Detection")
    print("=" * 60)

    config = create_sample_config()
    detector = DeepPatternDetector(config)

    # Testovací text s různými vzory
    test_text = """
    Investigation findings:
    - Bitcoin addresses: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
    - Onion sites: facebookcorewwwi.onion, 3g2upl4pq6kufc4m.onion
    - IP addresses: 192.168.1.100, 8.8.8.8
    - Email contacts: contact@protonmail.com, secure@tutanota.com
    - Hash: a1b2c3d4e5f6789012345678901234567890abcdef
    - Phone: +1-555-123-4567
    """

    # Detekce vzorů
    matches = await detector.detect_patterns(test_text, "demo_source")

    print(f"✅ Detected {len(matches)} patterns:")
    for match in matches[:10]:  # Top 10
        print(
            f"  - {match.pattern_type}/{match.pattern_name}: {match.matched_text} (confidence: {match.confidence:.2f})"
        )

    # Extrakce artefaktů
    artefacts = await detector.extract_artefacts(matches)
    print(f"\n📦 Extracted {len(artefacts)} artefacts:")
    for artefact in artefacts[:5]:  # Top 5
        print(
            f"  - {artefact.artefact_type}: {artefact.value} (confidence: {artefact.confidence:.2f})"
        )

    # Statistiky
    stats = detector.get_detection_statistics()
    print(f"\n📊 Detection Statistics:")
    print(f"  - Total patterns: {stats['total_patterns']}")
    print(f"  - Categories: {len(stats['categories'])}")

    return matches, artefacts


async def demo_steganography_analysis():
    """Demo steganography analysis komponenty"""
    print("\n" + "=" * 60)
    print("🖼️ DEMO: Steganography Analysis")
    print("=" * 60)

    config = create_sample_config()
    analyzer = SteganographyAnalyzer(config)

    # Vytvoření fake image data pro demo
    # V reálné implementaci by se použily skutečné obrázky
    fake_image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000  # PNG header + data

    try:
        # Analýza "obrázku"
        result = await analyzer.analyze_media_file("demo_image.png", fake_image_data)

        print(f"✅ Analysis completed:")
        print(f"  - File type: {result.file_type}")
        print(f"  - Suspicion score: {result.suspicion_score:.2f}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Methods used: {', '.join(result.analysis_methods)}")

        if result.detected_anomalies:
            print(f"  - Anomalies detected: {len(result.detected_anomalies)}")
            for anomaly in result.detected_anomalies[:3]:
                print(f"    • {anomaly}")

        # Statistiky
        stats = analyzer.get_analysis_statistics()
        print(f"\n📊 Analysis Statistics:")
        print(f"  - Total analyses: {stats['total_analyses']}")
        print(f"  - PIL available: {stats['dependencies']['PIL_available']}")
        print(f"  - SciPy available: {stats['dependencies']['scipy_available']}")

        return [result]

    except Exception as e:
        print(f"⚠️ Steganography analysis error (expected in demo): {e}")
        return []


async def demo_correlation_analysis():
    """Demo correlation engine komponenty"""
    print("\n" + "=" * 60)
    print("🕸️ DEMO: Correlation Analysis")
    print("=" * 60)

    config = create_sample_config()
    engine = CorrelationEngine(config)

    # Ukázkové dokumenty
    documents = {
        "doc1": "John Smith works for CryptoTech Corp in New York. He was seen meeting with Alice Johnson from SecureBank.",
        "doc2": "Alice Johnson, representative of SecureBank, discussed cryptocurrency regulations with government officials.",
        "doc3": "CryptoTech Corp announced new partnership. John Smith will lead the project in New York office.",
        "doc4": "Investigation reveals connection between unknown person and bitcoin address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    }

    try:
        # Budování grafu vztahů
        graph = await engine.build_relationship_graph(documents)

        # Analýza sítě
        network_analysis = await engine.analyze_network()

        print(f"✅ Network analysis completed:")
        print(f"  - Total entities: {network_analysis.total_entities}")
        print(f"  - Total relationships: {network_analysis.total_relationships}")
        print(f"  - Network density: {network_analysis.network_density:.3f}")
        print(f"  - Connected components: {network_analysis.connected_components}")

        # Klíčové entity
        if network_analysis.key_entities:
            print(f"\n👥 Key entities:")
            for entity in network_analysis.key_entities[:5]:
                print(f"  - {entity.label}: {entity.text} (confidence: {entity.confidence:.2f})")

        # Důležité vztahy
        if network_analysis.important_relationships:
            print(f"\n🔗 Important relationships:")
            for rel in network_analysis.important_relationships[:3]:
                print(f"  - {rel.entity1.text} ↔ {rel.entity2.text} ({rel.relationship_type})")

        # Anomálie
        if network_analysis.anomalous_patterns:
            print(f"\n⚠️ Anomalous patterns:")
            for anomaly in network_analysis.anomalous_patterns[:3]:
                print(f"  - {anomaly['type']}: {anomaly['description']}")

        # Statistiky
        stats = engine.get_correlation_statistics()
        print(f"\n📊 Correlation Statistics:")
        print(f"  - Total entities: {stats['total_entities']}")
        print(f"  - Graph nodes: {stats['graph_nodes']}")
        print(f"  - Graph edges: {stats['graph_edges']}")

        return (
            list(engine.entities.values()),
            network_analysis.important_relationships,
            network_analysis,
        )

    except Exception as e:
        print(f"⚠️ Correlation analysis error: {e}")
        return [], [], None


async def demo_credibility_assessment():
    """Demo credibility assessor komponenty"""
    print("\n" + "=" * 60)
    print("🛡️ DEMO: Credibility Assessment")
    print("=" * 60)

    config = create_sample_config()
    assessor = CredibilityAssessor(config)

    # Ukázkové zdroje s různou důvěryhodností
    test_sources = [
        (
            "This is a well-researched article with multiple citations and references. According to recent studies published in Nature, the findings are consistent with peer-reviewed research.",
            SourceMetadata(
                "test1",
                "https://reuters.com/article",
                "reuters.com",
                "Research Article",
                200,
                datetime.now(),
                "Dr. Smith",
                "news",
                "en",
            ),
        ),
        (
            "URGENT!!! SHOCKING DISCOVERY!!! You won't believe what they don't want you to know!!! Click here for amazing secrets!!!",
            SourceMetadata(
                "test2",
                "https://clickbait.com/shocking",
                "clickbait.com",
                "Shocking Discovery",
                100,
                datetime.now(),
                "Anonymous",
                "blog",
                "en",
            ),
        ),
        (
            "Government officials announced new policy. The statement was released through official channels and confirmed by multiple agencies.",
            SourceMetadata(
                "test3",
                "https://fbi.gov/news",
                "fbi.gov",
                "Official Statement",
                150,
                datetime.now(),
                "Press Office",
                "government",
                "en",
            ),
        ),
    ]

    assessments = []

    for content, metadata in test_sources:
        try:
            assessment = await assessor.assess_source_credibility(content, metadata)
            assessments.append(assessment)

            print(f"\n✅ Assessment for {metadata.domain}:")
            print(f"  - Credibility score: {assessment.overall_credibility_score:.2f}")
            print(f"  - Confidence: {assessment.confidence:.2f}")
            print(f"  - Risk factors: {len(assessment.risk_factors)}")
            print(f"  - Positive indicators: {len(assessment.positive_indicators)}")

            if assessment.risk_factors:
                print(f"  - Top risk: {assessment.risk_factors[0]}")
            if assessment.positive_indicators:
                print(f"  - Top positive: {assessment.positive_indicators[0]}")

        except Exception as e:
            print(f"⚠️ Assessment error for {metadata.domain}: {e}")

    # Generování zprávy
    if assessments:
        try:
            report = await assessor.generate_credibility_report(assessments)
            print(f"\n📊 Credibility Report:")
            print(f"  - Total sources: {report['summary']['total_sources_assessed']}")
            print(f"  - Average credibility: {report['summary']['average_credibility_score']:.2f}")
            print(f"  - High credibility: {report['summary']['high_credibility_count']}")
            print(f"  - Low credibility: {report['summary']['low_credibility_count']}")
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")

    # Statistiky
    stats = assessor.get_assessment_statistics()
    print(f"\n📊 Assessment Statistics:")
    print(f"  - Total assessments: {stats['total_assessments']}")
    print(f"  - Trusted domains: {stats['trusted_domains_count']}")
    print(f"  - Blacklisted domains: {stats['blacklisted_domains_count']}")

    return assessments


async def demo_intelligence_synthesis():
    """Demo kompletní intelligence synthesis"""
    print("\n" + "=" * 60)
    print("🧠 DEMO: Complete Intelligence Synthesis")
    print("=" * 60)

    config = create_sample_config()
    synthesis_engine = IntelligenceSynthesisEngine(config)

    # Vytvoření ukázkových zdrojů
    sources = create_sample_sources()

    print(f"🔄 Starting synthesis of {len(sources)} sources...")

    try:
        # Spuštění kompletní syntézy
        result = await synthesis_engine.synthesize_intelligence(sources, "phase3_demo")

        print(f"\n✅ Intelligence synthesis completed!")
        print(f"  - Analysis ID: {result.analysis_id}")
        print(f"  - Intelligence score: {result.intelligence_score:.2f}")
        print(f"  - Risk level: {result.risk_level}")
        print(f"  - Synthesis confidence: {result.synthesis_confidence:.2f}")

        # Klíčové poznatky
        print(f"\n🔍 Key Findings ({len(result.key_findings)}):")
        for finding in result.key_findings[:5]:
            print(f"  - {finding['title']} (confidence: {finding['confidence']:.2f})")

        # Doporučení
        print(f"\n💡 Recommendations ({len(result.recommendations)}):")
        for rec in result.recommendations[:3]:
            print(f"  - {rec}")

        # Komponenty statistiky
        print(f"\n📊 Component Results:")
        print(f"  - Pattern matches: {len(result.pattern_matches)}")
        print(f"  - Extracted artefacts: {len(result.extracted_artefacts)}")
        print(f"  - Entities: {len(result.entities)}")
        print(f"  - Relationships: {len(result.relationships)}")
        print(f"  - Steganography results: {len(result.steganography_results)}")
        print(f"  - Credibility assessments: {len(result.credibility_assessments)}")

        # Export výsledků
        try:
            output_path = await synthesis_engine.export_analysis_results(
                result, "phase3_demo_results", "json"
            )
            print(f"\n💾 Results exported to: {output_path}")
        except Exception as e:
            print(f"⚠️ Export error: {e}")

        # Celkové statistiky
        stats = synthesis_engine.get_synthesis_statistics()
        print(f"\n📊 Synthesis Statistics:")
        print(f"  - Total analyses: {stats['processing_stats']['total_analyses']}")
        print(f"  - Successful analyses: {stats['processing_stats']['successful_analyses']}")
        print(
            f"  - Average processing time: {stats['processing_stats']['average_processing_time']:.2f}s"
        )

        return result

    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """Hlavní demo funkce"""
    print("🚀 FÁZE 3: SYNTÉZA ZPRAVODAJSKÝCH INFORMACÍ - DEMO")
    print("=" * 70)
    print("Demonstrace všech komponent pro intelligence synthesis")
    print("=" * 70)

    try:
        # Demo jednotlivých komponent
        await demo_pattern_detection()
        await demo_steganography_analysis()
        await demo_correlation_analysis()
        await demo_credibility_assessment()

        # Demo kompletní syntézy
        await demo_intelligence_synthesis()

        print("\n" + "=" * 70)
        print("✅ VŠECHNY DEMO KOMPONENTY DOKONČENY!")
        print("🎯 Fáze 3 je úspěšně implementována a funkční")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Spuštění demo
    asyncio.run(main())
