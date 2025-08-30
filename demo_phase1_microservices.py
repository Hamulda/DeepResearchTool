                    payload = {"url": url}
                    async with session.post(f"{self.api_url}/scrape", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            task_id = data["task_id"]
                            logger.info(f"   ✅ Úloha vytvořena: {task_id}")

                            # Počkej a zkontroluj status
                            await asyncio.sleep(2)

                            async with session.get(f"{self.api_url}/task/{task_id}") as status_response:
                                if status_response.status == 200:
                                    status_data = await status_response.json()
                                    logger.info(f"   📊 Status: {status_data['status']}")

                            results.append({
                                "url": url,
                                "task_id": task_id,
                                "success": True
                            })
                        else:
                            logger.error(f"   ❌ Chyba při vytváření úlohy: {response.status}")
                            results.append({
                                "url": url,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            })

                except Exception as e:
                    logger.error(f"   ❌ Chyba při scraping {url}: {e}")
                    results.append({
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })

        return results

    async def test_system_stats(self):
        """Test systémových statistik"""
        logger.info("📊 Získávání systémových statistik...")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_url}/stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        logger.info("✅ Systémové statistiky:")
                        logger.info(f"   Redis připojení: {stats['redis']['connected_clients']}")
                        logger.info(f"   Paměť Redis: {stats['redis']['used_memory_human']}")
                        logger.info(f"   Fronta akvizice: {stats['queues']['acquisition']}")
                        logger.info(f"   Fronta zpracování: {stats['queues']['processing']}")
                        return stats
                    else:
                        logger.error(f"❌ Chyba při získávání statistik: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"❌ Chyba při získávání statistik: {e}")
                return None

    async def run_demo(self):
        """Spusť kompletní demo"""
        logger.info("🚀 Spouštění Demo Fáze 1 - Mikroslužbová architektura")
        logger.info("=" * 60)

        # Test 1: Zdraví API
        health_ok = await self.test_api_health()
        if not health_ok:
            logger.error("❌ API není dostupné. Ukončuji demo.")
            return False

        # Test 2: Scraping workflow
        scraping_results = await self.test_scraping_workflow()
        successful_scrapes = sum(1 for r in scraping_results if r["success"])
        logger.info(f"📈 Úspěšně zpracováno: {successful_scrapes}/{len(scraping_results)} URL")

        # Test 3: Systémové statistiky
        stats = await self.test_system_stats()

        # Shrnutí
        logger.info("=" * 60)
        logger.info("📋 SHRNUTÍ DEMO FÁZE 1:")
        logger.info(f"   ✅ API Gateway: {'OK' if health_ok else 'FAIL'}")
        logger.info(f"   ✅ Scraping: {successful_scrapes}/{len(scraping_results)} úspěšných")
        logger.info(f"   ✅ Statistiky: {'OK' if stats else 'FAIL'}")

        if health_ok and successful_scrapes > 0 and stats:
            logger.info("🎉 Fáze 1 je ÚSPĚŠNĚ implementována!")
            return True
        else:
            logger.info("❌ Fáze 1 má problémy - zkontrolujte logy.")
            return False

async def main():
    """Main funkce"""
    demo = Phase1Demo()
    success = await demo.run_demo()

    if success:
        print("\n🎯 DALŠÍ KROKY:")
        print("   1. Pokračujte s Fází 2 - Pokročilá akvizice dat a anonymizační vrstvy")
        print("   2. Přidejte Tor proxy a advanced scraping")
        print("   3. Implementujte persona management")
    else:
        print("\n🔧 ŘEŠENÍ PROBLÉMŮ:")
        print("   1. Zkontrolujte že všechny služby běží: docker-compose -f docker-compose.microservices.yml ps")
        print("   2. Zkontrolujte logy: docker-compose -f docker-compose.microservices.yml logs")
        print("   3. Restartujte služby: ./scripts/start_microservices.sh")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Demo skript pro testování mikroslužbové architektury Fáze 1
Ověřuje základní funkcionalitu: akvizice -> zpracování -> uložení
"""

import asyncio
import aiohttp
import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase1Demo:
    """Demo pro Fázi 1 - Základní architektura a klíčová infrastruktura"""

    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://example.com"
        ]

    async def test_api_health(self):
        """Test zdraví API"""
        logger.info("🔍 Testování zdraví API...")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ API je zdravé: {data}")
                        return True
                    else:
                        logger.error(f"❌ API není zdravé: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Chyba při testování API: {e}")
                return False

    async def test_scraping_workflow(self):
        """Test kompletního scraping workflow"""
        logger.info("🕷️ Testování scraping workflow...")

        results = []

        async with aiohttp.ClientSession() as session:
            for url in self.test_urls:
                try:
                    logger.info(f"   Scraping: {url}")

                    # Vytvoř scraping úlohu
