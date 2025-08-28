#!/usr/bin/env python3
"""
Demonstration script for Enhanced Deep Research Tool
Shows capabilities of all new specialized components

Author: Advanced IT Specialist
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.enhanced_orchestrator import EnhancedResearchOrchestrator, ResearchRequest
from analysis.specialized_analyzer import SpecializedSourceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchDemonstrator:
    """Demonstrates capabilities of enhanced research tool"""

    def __init__(self):
        self.orchestrator = EnhancedResearchOrchestrator()

    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all capabilities"""
        print("🔍 Enhanced Deep Research Tool - Comprehensive Demonstration")
        print("=" * 70)

        # Initialize orchestrator
        print("\n📡 Initializing Enhanced Research Orchestrator...")
        if not await self.orchestrator.initialize():
            print("❌ Failed to initialize orchestrator")
            return

        print("✅ Orchestrator initialized successfully")

        # Demonstrate different research scenarios
        scenarios = [
            {
                'name': 'CIA Declassified Intelligence Analysis',
                'topic': 'MKUltra mind control experiments',
                'sources': ['declassified', 'academic'],
                'depth': 'comprehensive'
            },
            {
                'name': 'Academic Research Synthesis',
                'topic': 'quantum computing algorithms',
                'sources': ['academic'],
                'depth': 'standard'
            },
            {
                'name': 'OSINT Investigation',
                'topic': 'cybersecurity threat intelligence',
                'sources': ['osint', 'academic'],
                'depth': 'comprehensive'
            }
        ]

        results_summary = []

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🎯 Scenario {i}: {scenario['name']}")
            print("-" * 50)

            # Create research request
            request = ResearchRequest(
                topic=scenario['topic'],
                research_depth=scenario['depth'],
                source_types=scenario['sources'],
                security_level='high',
                correlation_analysis=True,
                generate_report=True,
                max_results_per_source=20  # Reduced for demo
            )

            # Conduct research
            print(f"🔎 Researching: {scenario['topic']}")
            results = await self.orchestrator.conduct_research(request)

            # Display results
            self._display_results_summary(results)
            results_summary.append(results)

        # Generate overall performance report
        print("\n📊 Overall Performance Report")
        print("-" * 40)
        performance = await self.orchestrator.get_performance_report()
        self._display_performance_report(performance)

        # Demonstrate export capabilities
        if results_summary:
            print("\n💾 Export Demonstration")
            print("-" * 30)
            await self._demonstrate_exports(results_summary[0])

        # Cleanup
        await self.orchestrator.close()
        print("\n✅ Demonstration completed successfully")

    def _display_results_summary(self, results):
        """Display summary of research results"""
        stats = results.source_statistics

        print(f"⏱️  Execution time: {results.execution_time:.2f} seconds")
        print(f"📄 Total documents found: {stats.get('total_documents', 0)}")

        breakdown = stats.get('source_breakdown', {})
        for source, count in breakdown.items():
            if count > 0:
                print(f"   • {source.title()}: {count} documents")

        print(f"🔗 Correlations found: {stats.get('correlations_found', 0)}")

        # Display quality assessment
        quality = stats.get('data_quality', {})
        if quality:
            print("📈 Data quality scores:")
            for source, score in quality.items():
                print(f"   • {source.title()}: {score:.2f}")

        # Display intelligence report summary if available
        if results.intelligence_report:
            report = results.intelligence_report
            print(f"🎯 Intelligence confidence: {report.confidence_level}")
            print(f"📋 Key findings: {len(report.key_findings)}")
            if report.key_findings:
                print(f"   Top finding: {report.key_findings[0][:100]}...")

    def _display_performance_report(self, performance):
        """Display performance metrics"""
        metrics = performance['performance_metrics']

        print(f"🚀 Total requests processed: {metrics['total_requests']}")
        print(f"✅ Successful searches: {metrics['successful_searches']}")
        print(f"⚡ Average response time: {metrics['average_response_time']:.2f}s")
        print(f"🔗 Total correlations found: {metrics['correlations_found']}")

        sources = performance['sources_available']
        print("\n📡 Available sources:")
        for source, available in sources.items():
            status = "✅" if available else "❌"
            print(f"   {status} {source.replace('_', ' ').title()}")

    async def _demonstrate_exports(self, results):
        """Demonstrate export capabilities"""
        try:
            # JSON export
            json_export = await self.orchestrator.export_results(results, "json")
            print(f"📝 JSON export: {len(json_export)} characters")

            # HTML export
            html_export = await self.orchestrator.export_results(results, "html")
            print(f"🌐 HTML export: {len(html_export)} characters")

            # Save to files for demonstration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            json_file = f"demo_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_export)
            print(f"💾 Saved JSON to: {json_file}")

            html_file = f"demo_results_{timestamp}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_export)
            print(f"💾 Saved HTML to: {html_file}")

        except Exception as e:
            print(f"❌ Export demonstration failed: {str(e)}")

async def run_individual_component_tests():
    """Test individual components separately"""
    print("\n🧪 Individual Component Testing")
    print("=" * 40)

    # Test Specialized Analyzer
    print("\n🔬 Testing Specialized Source Analyzer...")
    try:
        analyzer = SpecializedSourceAnalyzer()
        await analyzer.initialize()
        print("✅ Specialized Analyzer initialized")

        # Mock correlation analysis
        mock_data = {
            'test_source': [{'title': 'Test Document', 'content': 'Sample analysis content'}]
        }

        correlations = await analyzer.correlation_analysis(mock_data)
        print(f"🔗 Mock correlation analysis: {len(correlations)} results")

    except Exception as e:
        print(f"❌ Specialized Analyzer test failed: {str(e)}")

    print("\n✅ Component testing completed")

def display_implementation_summary():
    """Display summary of what was implemented"""
    print("\n🎉 IMPLEMENTATION SUMMARY")
    print("=" * 50)

    implemented_features = [
        "🏛️  CIA CREST Declassified Document Scraper",
        "   • 10+ million pages of declassified documents",
        "   • Advanced classification level detection",
        "   • Redaction pattern analysis",
        "   • Historical context mapping",
        "",
        "🌐 Deep Web/Dark Web Crawler",
        "   • Tor network integration with safety protocols",
        "   • .onion site discovery and analysis",
        "   • Security validation and risk assessment",
        "   • Circuit rotation for anonymity",
        "",
        "📚 BASE Bielefeld Academic Integration",
        "   • 150+ million documents from 7000+ sources",
        "   • Advanced quality scoring",
        "   • Multi-language support",
        "   • Repository credibility assessment",
        "",
        "🕵️  Advanced OSINT Collector",
        "   • Multi-source intelligence gathering",
        "   • Entity relationship analysis",
        "   • Correlation scoring",
        "   • Cross-reference verification",
        "",
        "🧠 Specialized Source Analyzer",
        "   • Cross-source correlation analysis",
        "   • Entity extraction with spaCy",
        "   • Timeline reconstruction",
        "   • Credibility assessment",
        "   • Network topology analysis",
        "",
        "🎛️  Enhanced Research Orchestrator",
        "   • Parallel multi-source data collection",
        "   • Intelligent result correlation",
        "   • Performance monitoring",
        "   • Multiple export formats",
        "",
        "🔐 Security & Compliance Framework",
        "   • VPN rotation and anonymity protocols",
        "   • Content validation pipeline",
        "   • GDPR compliance features",
        "   • Audit trail logging",
        "",
        "⚙️  Advanced Configuration System",
        "   • Specialized source configuration",
        "   • Security protocol settings",
        "   • Rate limiting per source",
        "   • Quality thresholds"
    ]

    for feature in implemented_features:
        print(feature)

    print("\n🚀 KEY CAPABILITIES ACHIEVED:")
    capabilities = [
        "✅ Access to 10+ million declassified CIA documents",
        "✅ 150+ million academic documents from deep web sources",
        "✅ Secure Tor-based dark web research capabilities",
        "✅ Advanced cross-source correlation analysis",
        "✅ Entity relationship mapping and timeline reconstruction",
        "✅ Multi-layer security and anonymity protocols",
        "✅ Comprehensive intelligence report generation",
        "✅ Real-time performance monitoring and optimization"
    ]

    for capability in capabilities:
        print(capability)

async def main():
    """Main demonstration function"""
    print("🌟 Enhanced Deep Research Tool - Advanced Capabilities Demo")
    print("Building on existing foundation with specialized intelligence sources")
    print("=" * 80)

    # Display implementation summary
    display_implementation_summary()

    # Ask user what to demonstrate
    print("\n🎮 What would you like to demonstrate?")
    print("1. Full comprehensive research scenarios")
    print("2. Individual component testing only")
    print("3. Configuration overview only")

    try:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            demonstrator = ResearchDemonstrator()
            await demonstrator.run_comprehensive_demo()
        elif choice == "2":
            await run_individual_component_tests()
        elif choice == "3":
            print("\n📋 Configuration has been enhanced with specialized sources.")
            print("Check config.yaml for detailed settings.")
        else:
            print("ℹ️  Demo skipped. All components are ready for use.")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("💡 Note: Some features require additional setup (Tor, API keys, etc.)")

if __name__ == "__main__":
    asyncio.run(main())
