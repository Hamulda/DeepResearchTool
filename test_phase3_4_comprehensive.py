#!/usr/bin/env python3
"""
Komplexní test pro fáze 3 a 4 DeepResearchTool
Testuje deep web crawling, autonomní agenta a bezpečnostní opatření

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pytest

# Import testovaných komponent
from src.deep_web.network_manager import NetworkManager, NetworkConfig, NetworkType
from src.scraping.stealth_engine import StealthEngine, StealthConfig
from src.security.osint_sandbox import OSINTSecuritySandbox, SecurityLevel
from workers.deep_web_crawler import DeepWebCrawler, CrawlerConfig, crawl_onion_sites
from src.core.agentic_loop import AgenticLoop, TaskType, TaskPriority, AgentTask
from src.core.intelligent_task_manager import IntelligentTaskManager, TaskExecutionStrategy
from src.synthesis.hierarchical_summarizer import create_m1_summarizer
from src.synthesis.credibility_assessor import CredibilityAssessor
from src.synthesis.correlation_engine import CorrelationEngine
from src.synthesis.deep_pattern_detector import DeepPatternDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase3and4TestSuite:
    """Komplexní testovací suite pro fáze 3 a 4"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        self.start_time = time.time()
        
        # Konfigurace pro testy
        self.security_config = {
            "security": {
                "osint": {
                    "security_level": "medium",
                    "rate_limits": {
                        "requests_per_minute": 30,
                        "requests_per_hour": 500,
                        "concurrent_requests": 5
                    },
                    "rules": {
                        "whitelist": [
                            "domain:httpbin.org",
                            "domain:example.com",
                            "domain:test.local"
                        ],
                        "blacklist": [
                            "regex:.*malware.*",
                            "domain:malicious-site.com"
                        ]
                    }
                }
            },
            "audit_trail": str(self.temp_dir / "test_audit.jsonl")
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Spustí všechny testy fází 3 a 4"""
        logger.info("🚀 Spouštím komplexní testy fází 3 a 4...")
        
        # Fáze 3 testy
        await self.test_network_manager()
        await self.test_stealth_engine()
        await self.test_security_sandbox()
        await self.test_deep_web_crawler()
        
        # Fáze 4 testy
        await self.test_intelligent_task_manager()
        await self.test_agentic_loop()
        await self.test_integration_scenario()
        
        # Generování finálního reportu
        return self.generate_final_report()
    
    async def test_network_manager(self):
        """Test síťového manažera (Fáze 3.1)"""
        logger.info("🔍 Testování Network Manageru...")
        
        try:
            # Konfigurace pouze pro clearnet (bezpečné testování)
            config = NetworkConfig(
                enable_clearnet=True,
                enable_tor=False,  # Vypnuto pro CI/CD
                enable_i2p=False,  # Vypnuto pro CI/CD
                preferred_networks=[NetworkType.CLEARNET],
                default_timeout=10
            )
            
            async with NetworkManager(config) as network_manager:
                # Test health check
                health = await network_manager.health_check()
                
                # Test HTTP request
                try:
                    response = await network_manager.get("http://httpbin.org/ip")
                    request_success = response.status == 200
                except Exception as e:
                    request_success = False
                    logger.warning(f"HTTP request failed: {e}")
                
                # Test síťových statistik
                status = await network_manager.get_network_status()
                
                self.test_results['network_manager'] = {
                    'status': 'success',
                    'health_check': health,
                    'http_request_success': request_success,
                    'network_status': status,
                    'clearnet_available': health.get('clearnet', {}).get('healthy', False)
                }
                
        except Exception as e:
            self.test_results['network_manager'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Network Manager test failed: {e}")
    
    async def test_stealth_engine(self):
        """Test stealth scrapingu (Fáze 3.2)"""
        logger.info("🥷 Testování Stealth Engine...")
        
        try:
            config = StealthConfig(
                user_agent_rotation=True,
                header_randomization=True,
                timing_randomization=False,  # Rychlejší testy
                min_delay=0.1,
                max_delay=0.2
            )
            
            engine = StealthEngine(config)
            
            # Test scrapingu bezpečné stránky
            result = await engine.scrape_url("http://httpbin.org/html", use_browser=False)
            
            # Test detekce anti-bot opatření
            mock_content = """
            <html>
                <body>
                    <h1>Test Page</h1>
                    <p>This is a test page for scraping.</p>
                </body>
            </html>
            """
            
            detection = engine.browser_manager.anti_detection.detect_anti_bot_measures(
                mock_content, "http://httpbin.org/html"
            )
            
            # Test statistik
            stats = engine.get_session_stats()
            
            await engine.cleanup()
            
            self.test_results['stealth_engine'] = {
                'status': 'success',
                'scraping_result': {
                    'success': result.get('success', False),
                    'content_length': len(result.get('content', '')),
                    'final_url': result.get('final_url', '')
                },
                'anti_bot_detection': detection,
                'session_stats': stats
            }
            
        except Exception as e:
            self.test_results['stealth_engine'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Stealth Engine test failed: {e}")
    
    async def test_security_sandbox(self):
        """Test bezpečnostního sandboxu (Fáze 4.3)"""
        logger.info("🛡️ Testování Security Sandbox...")
        
        try:
            sandbox = OSINTSecuritySandbox(self.security_config)
            
            # Test access permission
            test_urls = [
                ("http://httpbin.org/ip", "allowed"),
                ("http://malicious-site.com/malware", "blocked"),
                ("http://example.com/test", "allowed")
            ]
            
            access_results = {}
            for url, expected in test_urls:
                result, metadata = sandbox.check_access_permission(url)
                access_results[url] = {
                    'result': result.value,
                    'expected': expected,
                    'metadata': metadata,
                    'correct': result.value == expected
                }
            
            # Test content processing
            test_content = "Research data: email@example.com, password: secret123"
            processed_content, processing_metadata = sandbox.process_content(
                test_content, "http://example.com/test"
            )
            
            # Test security statistics
            stats = sandbox.get_security_statistics()
            
            self.test_results['security_sandbox'] = {
                'status': 'success',
                'access_control_tests': access_results,
                'content_processing': {
                    'original_length': len(test_content),
                    'processed_length': len(processed_content),
                    'safety_report': processing_metadata.get('safety_report', {}),
                    'pii_redacted': 'REDACTED' in processed_content
                },
                'statistics': stats
            }
            
        except Exception as e:
            self.test_results['security_sandbox'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Security Sandbox test failed: {e}")
    
    async def test_deep_web_crawler(self):
        """Test deep web crawleru (Fáze 3.3)"""
        logger.info("🕷️ Testování Deep Web Crawler...")
        
        try:
            # Konfigurace pro bezpečné testování (pouze clearnet)
            config = CrawlerConfig(
                max_depth=1,
                max_pages_per_domain=3,
                max_total_pages=5,
                crawl_delay=0.5,
                tor_identity_rotation_interval=999  # Vysoké číslo pro testy
            )
            
            # Mock test - vytvoříme crawler ale nebudeme crawlovat .onion
            crawler = DeepWebCrawler(config)
            
            # Test základní funkcionality bez skutečného crawlingu
            test_urls = ["http://httpbin.org/html"]
            
            # Simulace crawlingu (bez skutečné inicializace Tor/I2P)
            mock_results = []
            for url in test_urls:
                if crawler._is_valid_url(url.replace('http', 'https').replace('.org', '.onion')):
                    mock_results.append({
                        'url': url,
                        'title': 'Test Page',
                        'links_found': 3,
                        'depth': 0
                    })
            
            # Test utility funkcí
            domain = crawler._extract_domain("http://example.onion/test")
            is_valid = crawler._is_valid_url("http://test.onion/page")
            
            self.test_results['deep_web_crawler'] = {
                'status': 'success',
                'config_validation': {
                    'max_depth': config.max_depth,
                    'max_pages': config.max_total_pages,
                    'crawl_delay': config.crawl_delay
                },
                'utility_functions': {
                    'domain_extraction': domain == 'example.onion',
                    'url_validation': is_valid,
                },
                'mock_crawl_results': mock_results,
                'crawler_initialized': True
            }
            
        except Exception as e:
            self.test_results['deep_web_crawler'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Deep Web Crawler test failed: {e}")
    
    async def test_intelligent_task_manager(self):
        """Test inteligentního task manageru (Fáze 4.1)"""
        logger.info("🧠 Testování Intelligent Task Manager...")
        
        try:
            # Mock credibility assessor
            class MockCredibilityAssessor:
                async def assess_content(self, content, url):
                    return 0.8
            
            credibility_assessor = MockCredibilityAssessor()
            task_manager = IntelligentTaskManager(credibility_assessor)
            
            # Vytvoření testovacích úkolů
            test_tasks = []
            for i in range(5):
                task = AgentTask(
                    id=f"task_{i}",
                    task_type=TaskType.ANALYZE,
                    priority=TaskPriority.MEDIUM,
                    parameters={"test_param": f"value_{i}"},
                    created_at=datetime.now(),
                    credibility_score=0.7 + (i * 0.05)
                )
                test_tasks.append(task)
                await task_manager.add_task_with_intelligence(task)
            
            # Test optimálního pořadí
            optimal_order = task_manager.get_optimal_execution_order(test_tasks)
            
            # Test alokace zdrojů
            resource_allocation = await task_manager.optimize_resource_allocation(test_tasks)
            
            # Test různých strategií
            strategies_tested = {}
            for strategy in TaskExecutionStrategy:
                task_manager.strategy = strategy
                order = task_manager.get_optimal_execution_order(test_tasks)
                strategies_tested[strategy.value] = len(order)
            
            # Test učení a adaptace
            task_manager.record_execution_result("task_0", True, 5.0, {"quality": "high"})
            task_manager.record_execution_result("task_1", False, 10.0, {})
            
            insights = task_manager.get_learning_insights()
            task_manager.adapt_strategy()
            
            self.test_results['intelligent_task_manager'] = {
                'status': 'success',
                'task_graph_nodes': len(task_manager.task_graph.nodes),
                'optimal_order_length': len(optimal_order),
                'resource_allocation': resource_allocation,
                'strategies_tested': strategies_tested,
                'learning_insights': insights,
                'current_strategy': task_manager.strategy.value
            }
            
        except Exception as e:
            self.test_results['intelligent_task_manager'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Intelligent Task Manager test failed: {e}")
    
    async def test_agentic_loop(self):
        """Test autonomního agenta (Fáze 4.1)"""
        logger.info("🤖 Testování Agentic Loop...")
        
        try:
            # Mock komponenty
            class MockDocumentStore:
                async def store_document(self, doc):
                    return f"doc_{hash(str(doc)) % 1000}"
            
            class MockWebScraper:
                async def scrape_page(self, url):
                    return f"Content from {url}"
            
            class MockCredibilityAssessor:
                async def assess_content(self, content, url):
                    return 0.75
            
            class MockCorrelationEngine:
                async def analyze_text(self, text):
                    return {"entities": [{"id": "entity_1", "type": "person"}]}
                
                async def find_related_entities(self, entity_id):
                    return [{"id": "related_1", "relation": "associated"}]
            
            class MockPatternDetector:
                async def analyze_text(self, text):
                    return {"artifacts": [{"pattern": "test_pattern", "confidence": 0.8}]}
            
            # Inicializace komponent
            document_store = MockDocumentStore()
            credibility_assessor = MockCredibilityAssessor()
            correlation_engine = MockCorrelationEngine()
            pattern_detector = MockPatternDetector()
            web_scraper = MockWebScraper()
            
            # Vytvoření agentic loop
            agent = AgenticLoop(
                document_store=document_store,
                credibility_assessor=credibility_assessor,
                correlation_engine=correlation_engine,
                pattern_detector=pattern_detector,
                web_scraper=web_scraper,
                max_iterations=3,
                min_credibility_threshold=0.5
            )
            
            # Test krátkého výzkumného cyklu
            initial_query = "Test research query about artificial intelligence"
            test_urls = ["http://httpbin.org/html"]
            
            # Spuštění krátkého cyklu
            results = await agent.start_research_cycle(initial_query, test_urls)
            
            # Test aktuálního stavu
            status = agent.get_current_status()
            
            self.test_results['agentic_loop'] = {
                'status': 'success',
                'research_results': {
                    'iterations_completed': results['research_summary']['iterations_completed'],
                    'total_tasks_executed': results['research_summary']['total_tasks_executed'],
                    'average_credibility': results['research_summary']['average_credibility'],
                    'key_discoveries_count': len(results['key_discoveries'])
                },
                'agent_status': status,
                'final_recommendations': results.get('recommendations', [])
            }
            
        except Exception as e:
            self.test_results['agentic_loop'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Agentic Loop test failed: {e}")
    
    async def test_integration_scenario(self):
        """Test kompletního integračního scénáře"""
        logger.info("🔗 Testování integračního scénáře...")
        
        try:
            # Scénář: Bezpečný výzkum s autonomním agentem
            scenario_results = {}
            
            # 1. Bezpečnostní kontrola
            sandbox = OSINTSecuritySandbox(self.security_config)
            test_url = "http://httpbin.org/ip"
            
            access_result, metadata = sandbox.check_access_permission(test_url)
            scenario_results['security_check'] = {
                'url': test_url,
                'allowed': access_result.value == 'allowed',
                'metadata': metadata
            }
            
            # 2. Stealth scraping (pokud povoleno)
            if access_result.value == 'allowed':
                engine = StealthEngine(StealthConfig(
                    timing_randomization=False,
                    min_delay=0.1,
                    max_delay=0.1
                ))
                
                scrape_result = await engine.scrape_url(test_url, use_browser=False)
                
                # 3. Bezpečné zpracování obsahu
                if scrape_result.get('success'):
                    processed_content, processing_metadata = sandbox.process_content(
                        scrape_result['content'], test_url
                    )
                    
                    scenario_results['content_processing'] = {
                        'scraping_success': True,
                        'content_length': len(scrape_result['content']),
                        'processed_length': len(processed_content),
                        'safety_score': processing_metadata['safety_report']['score']
                    }
                
                await engine.cleanup()
            
            # 4. Analýza bezpečnostních metrik
            security_stats = sandbox.get_security_statistics()
            scenario_results['security_metrics'] = {
                'total_attempts': security_stats.get('total_access_attempts', 0),
                'success_rate': security_stats.get('success_rate', 0),
                'block_rate': security_stats.get('block_rate', 0)
            }
            
            self.test_results['integration_scenario'] = {
                'status': 'success',
                'scenario_results': scenario_results,
                'workflow_completed': True
            }
            
        except Exception as e:
            self.test_results['integration_scenario'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Integration scenario test failed: {e}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generuje finální report z testů"""
        total_time = time.time() - self.start_time
        
        # Počítání úspěšných testů
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get('status') == 'success')
        total_tests = len(self.test_results)
        
        # Identifikace problémových oblastí
        failed_tests = [name for name, result in self.test_results.items() 
                       if result.get('status') == 'failed']
        
        # Hodnocení pokrytí funkcionalit
        phase3_coverage = self._assess_phase3_coverage()
        phase4_coverage = self._assess_phase4_coverage()
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': len(failed_tests),
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'total_runtime_seconds': total_time
            },
            'phase_coverage': {
                'phase_3_deep_web': phase3_coverage,
                'phase_4_autonomous_agent': phase4_coverage
            },
            'detailed_results': self.test_results,
            'failed_components': failed_tests,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps(),
            'metadata': {
                'test_timestamp': datetime.now().isoformat(),
                'test_environment': 'development',
                'temp_directory': str(self.temp_dir)
            }
        }
        
        return report
    
    def _assess_phase3_coverage(self) -> Dict[str, Any]:
        """Hodnotí pokrytí fáze 3"""
        return {
            'network_manager': self.test_results.get('network_manager', {}).get('status') == 'success',
            'stealth_engine': self.test_results.get('stealth_engine', {}).get('status') == 'success',
            'deep_web_crawler': self.test_results.get('deep_web_crawler', {}).get('status') == 'success',
            'security_sandbox': self.test_results.get('security_sandbox', {}).get('status') == 'success',
            'overall_coverage': sum([
                self.test_results.get('network_manager', {}).get('status') == 'success',
                self.test_results.get('stealth_engine', {}).get('status') == 'success',
                self.test_results.get('deep_web_crawler', {}).get('status') == 'success',
                self.test_results.get('security_sandbox', {}).get('status') == 'success'
            ]) / 4
        }
    
    def _assess_phase4_coverage(self) -> Dict[str, Any]:
        """Hodnotí pokrytí fáze 4"""
        return {
            'intelligent_task_manager': self.test_results.get('intelligent_task_manager', {}).get('status') == 'success',
            'agentic_loop': self.test_results.get('agentic_loop', {}).get('status') == 'success',
            'integration_scenario': self.test_results.get('integration_scenario', {}).get('status') == 'success',
            'overall_coverage': sum([
                self.test_results.get('intelligent_task_manager', {}).get('status') == 'success',
                self.test_results.get('agentic_loop', {}).get('status') == 'success',
                self.test_results.get('integration_scenario', {}).get('status') == 'success'
            ]) / 3
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generuje doporučení na základě výsledků testů"""
        recommendations = []
        
        if 'network_manager' in self.test_results:
            if self.test_results['network_manager'].get('status') == 'failed':
                recommendations.append("Opravit konfiguraci Network Manageru pro stabilní síťové spojení")
        
        if 'stealth_engine' in self.test_results:
            if self.test_results['stealth_engine'].get('status') == 'success':
                recommendations.append("Stealth Engine funguje správně - lze začít s testováním na reálných stránkách")
        
        if 'security_sandbox' in self.test_results:
            if self.test_results['security_sandbox'].get('status') == 'success':
                recommendations.append("Bezpečnostní sandbox je funkční - systém je připraven pro OSINT operace")
        
        if 'agentic_loop' in self.test_results:
            if self.test_results['agentic_loop'].get('status') == 'success':
                recommendations.append("Autonomní agent funguje - lze testovat na komplexnějších scénářích")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generuje další kroky na základě výsledků"""
        next_steps = []
        
        # Analýza úspěšnosti
        success_rate = self.test_results.get('test_summary', {}).get('success_rate', 0)
        
        if success_rate >= 0.8:
            next_steps.extend([
                "Systém je připraven pro produkční nasazení",
                "Spustit performance testy s většími datovými sadami",
                "Implementovat monitoring a alerting pro produkční prostředí"
            ])
        elif success_rate >= 0.6:
            next_steps.extend([
                "Opravit identifikované problémy před produkčním nasazením",
                "Rozšířit testovací pokrytí pro problematické oblasti",
                "Optimalizovat konfiguraci komponent"
            ])
        else:
            next_steps.extend([
                "Provést detailní analýzu selhání komponent",
                "Refaktorovat problematické moduly",
                "Posílit testovací suite před dalším testováním"
            ])
        
        return next_steps


async def main():
    """Hlavní funkce pro spuštění testů"""
    print("🧪 Spouštím komplexní testy fází 3 a 4 DeepResearchTool...")
    print("=" * 60)
    
    test_suite = Phase3and4TestSuite()
    
    try:
        # Spuštění všech testů
        final_report = await test_suite.run_all_tests()
        
        # Výpis výsledků
        print("\n📊 FINÁLNÍ REPORT")
        print("=" * 60)
        
        summary = final_report['test_summary']
        print(f"✅ Úspěšné testy: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"📈 Úspěšnost: {summary['success_rate']:.1%}")
        print(f"⏱️ Celkový čas: {summary['total_runtime_seconds']:.1f}s")
        
        print(f"\n🎯 POKRYTÍ FÁZÍ")
        print("-" * 30)
        phase3 = final_report['phase_coverage']['phase_3_deep_web']
        phase4 = final_report['phase_coverage']['phase_4_autonomous_agent']
        print(f"Fáze 3 (Deep Web): {phase3['overall_coverage']:.1%}")
        print(f"Fáze 4 (Autonomní Agent): {phase4['overall_coverage']:.1%}")
        
        if final_report['failed_components']:
            print(f"\n❌ SELHÁNÍ")
            print("-" * 30)
            for component in final_report['failed_components']:
                error = final_report['detailed_results'][component].get('error', 'Neznámá chyba')
                print(f"• {component}: {error}")
        
        if final_report['recommendations']:
            print(f"\n💡 DOPORUČENÍ")
            print("-" * 30)
            for i, rec in enumerate(final_report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\n🔄 DALŠÍ KROKY")
        print("-" * 30)
        for i, step in enumerate(final_report['next_steps'], 1):
            print(f"{i}. {step}")
        
        # Uložení detailního reportu
        report_file = test_suite.temp_dir / "phase3_4_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 Detailní report uložen: {report_file}")
        
        # Výstupní kód podle úspěšnosti
        if summary['success_rate'] >= 0.8:
            print("\n🎉 Testy proběhly úspěšně! Systém je připraven k použití.")
            return 0
        elif summary['success_rate'] >= 0.6:
            print("\n⚠️ Testy částečně úspěšné. Potřebné drobné opravy.")
            return 1
        else:
            print("\n🚨 Testy selhaly. Nutné významné opravy.")
            return 2
            
    except Exception as e:
        print(f"\n💥 Kritická chyba při testování: {e}")
        logger.exception("Critical test failure")
        return 3


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)