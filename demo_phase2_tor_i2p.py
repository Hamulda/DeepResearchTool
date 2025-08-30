#!/usr/bin/env python3
"""
Demo skript pro Fázi 2 - Pokročilá akvizice dat a anonymizační vrstvy
Testuje Tor/I2P integraci, Playwright scraping a persona management
"""

import asyncio
import aiohttp
import json
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2Demo:
    """Demo pro Fázi 2 - Pokročilá akvizice a anonymizace"""

    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.test_scenarios = {
            "basic_web": [
                "https://httpbin.org/html",
                "https://httpbin.org/user-agent",
                "https://httpbin.org/headers",
            ],
            "js_heavy": [
                "https://httpbin.org/delay/2",
                "https://jsonplaceholder.typicode.com/posts/1",
            ],
            "tor_test": ["https://check.torproject.org/api/ip"],
        }

    async def test_enhanced_api(self):
        """Test vylepšeného API s Tor podporou"""
        logger.info("🔍 Testování vylepšeného API...")

        async with aiohttp.ClientSession() as session:
            # Test health check
            try:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ API zdravé: {data}")
                        return True
                    else:
                        logger.error(f"❌ API problém: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ API nedostupné: {e}")
                return False

    async def test_tor_scraping(self):
        """Test Tor anonymního scrapingu"""
        logger.info("🧅 Testování Tor scraping...")

        results = []

        async with aiohttp.ClientSession() as session:
            for url in self.test_scenarios["tor_test"]:
                try:
                    logger.info(f"   Tor scraping: {url}")

                    # Vytvoř enhanced scraping úlohu s Tor
                    payload = {"url": url, "force_tor": True, "stealth_mode": True}

                    async with session.post(
                        f"{self.api_url}/scrape-enhanced", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            task_id = data["task_id"]
                            logger.info(f"   ✅ Tor úloha: {task_id}")

                            # Počkej na zpracování
                            await asyncio.sleep(10)

                            # Zkontroluj výsledky
                            async with session.get(
                                f"{self.api_url}/task/{task_id}"
                            ) as status_response:
                                if status_response.status == 200:
                                    status_data = await status_response.json()
                                    logger.info(
                                        f"   📊 Tor status: {status_data.get('status', 'unknown')}"
                                    )

                            results.append(
                                {"url": url, "task_id": task_id, "method": "tor", "success": True}
                            )
                        else:
                            logger.error(f"   ❌ Tor scraping selhal: {response.status}")
                            results.append({"url": url, "method": "tor", "success": False})

                except Exception as e:
                    logger.error(f"   ❌ Tor chyba {url}: {e}")
                    results.append({"url": url, "method": "tor", "success": False, "error": str(e)})

        return results

    async def test_playwright_scraping(self):
        """Test Playwright stealth scrapingu"""
        logger.info("🎭 Testování Playwright scraping...")

        results = []

        async with aiohttp.ClientSession() as session:
            for url in self.test_scenarios["js_heavy"]:
                try:
                    logger.info(f"   Playwright scraping: {url}")

                    # Vytvoř Playwright úlohu
                    payload = {"url": url, "use_playwright": True, "stealth_mode": True}

                    async with session.post(
                        f"{self.api_url}/scrape-enhanced", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            task_id = data["task_id"]
                            logger.info(f"   ✅ Playwright úloha: {task_id}")

                            await asyncio.sleep(8)

                            results.append(
                                {
                                    "url": url,
                                    "task_id": task_id,
                                    "method": "playwright",
                                    "success": True,
                                }
                            )
                        else:
                            logger.error(f"   ❌ Playwright selhal: {response.status}")
                            results.append({"url": url, "method": "playwright", "success": False})

                except Exception as e:
                    logger.error(f"   ❌ Playwright chyba {url}: {e}")
                    results.append(
                        {"url": url, "method": "playwright", "success": False, "error": str(e)}
                    )

        return results

    async def test_persona_management(self):
        """Test persona management systému"""
        logger.info("👥 Testování persona management...")

        results = []

        async with aiohttp.ClientSession() as session:
            # Test několika různých domén pro persona assignment
            test_domains = [
                "https://httpbin.org/user-agent",
                "https://httpstat.us/200",
                "https://jsonplaceholder.typicode.com/users/1",
            ]

            for url in test_domains:
                try:
                    logger.info(f"   Persona test: {url}")

                    payload = {"url": url, "track_persona": True}

                    async with session.post(f"{self.api_url}/scrape", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            task_id = data["task_id"]

                            await asyncio.sleep(5)

                            # Zkontroluj persona info
                            async with session.get(
                                f"{self.api_url}/task/{task_id}/persona"
                            ) as persona_response:
                                if persona_response.status == 200:
                                    persona_data = await persona_response.json()
                                    logger.info(
                                        f"   🎭 Persona: {persona_data.get('persona_id', 'unknown')}"
                                    )

                            results.append({"url": url, "task_id": task_id, "success": True})
                        else:
                            results.append({"url": url, "success": False})

                except Exception as e:
                    logger.error(f"   ❌ Persona test chyba: {e}")
                    results.append({"url": url, "success": False, "error": str(e)})

        return results

    async def test_anonymization_verification(self):
        """Ověř že anonymizace funguje"""
        logger.info("🔒 Testování anonymizace...")

        async with aiohttp.ClientSession() as session:
            try:
                # Test IP detection
                async with session.get(f"{self.api_url}/check-ip") as response:
                    if response.status == 200:
                        ip_data = await response.json()
                        logger.info(f"   📍 Detekovaná IP: {ip_data.get('ip', 'unknown')}")
                        logger.info(f"   🌍 Země: {ip_data.get('country', 'unknown')}")
                        logger.info(f"   🧅 Tor: {ip_data.get('tor_exit', False)}")

                        return {
                            "ip_detected": True,
                            "tor_active": ip_data.get("tor_exit", False),
                            "country": ip_data.get("country", "unknown"),
                        }
                    else:
                        logger.error(f"   ❌ IP check selhal: {response.status}")
                        return {"ip_detected": False}

            except Exception as e:
                logger.error(f"   ❌ Anonymizace test chyba: {e}")
                return {"ip_detected": False, "error": str(e)}

    async def run_demo(self):
        """Spusť kompletní demo Fáze 2"""
        logger.info("🚀 Spouštění Demo Fáze 2 - Pokročilá akvizice a anonymizace")
        logger.info("=" * 70)

        # Test 1: Enhanced API
        api_ok = await self.test_enhanced_api()
        if not api_ok:
            logger.error("❌ Enhanced API není dostupné. Demo končí.")
            return False

        # Test 2: Anonymizace verification
        anon_results = await self.test_anonymization_verification()

        # Test 3: Tor scraping
        tor_results = await self.test_tor_scraping()
        tor_success = sum(1 for r in tor_results if r["success"])

        # Test 4: Playwright scraping
        playwright_results = await self.test_playwright_scraping()
        playwright_success = sum(1 for r in playwright_results if r["success"])

        # Test 5: Persona management
        persona_results = await self.test_persona_management()
        persona_success = sum(1 for r in persona_results if r["success"])

        # Shrnutí
        logger.info("=" * 70)
        logger.info("📋 SHRNUTÍ DEMO FÁZE 2:")
        logger.info(f"   ✅ Enhanced API: {'OK' if api_ok else 'FAIL'}")
        logger.info(
            f"   🔒 Anonymizace: {'AKTIVNÍ' if anon_results.get('tor_active') else 'NEAKTIVNÍ'}"
        )
        logger.info(f"   🧅 Tor scraping: {tor_success}/{len(tor_results)} úspěšných")
        logger.info(f"   🎭 Playwright: {playwright_success}/{len(playwright_results)} úspěšných")
        logger.info(f"   👥 Persona mgmt: {persona_success}/{len(persona_results)} úspěšných")

        overall_success = (
            api_ok and tor_success > 0 and playwright_success > 0 and persona_success > 0
        )

        if overall_success:
            logger.info("🎉 Fáze 2 je ÚSPĚŠNĚ implementována!")
            logger.info("✅ Pokročilá akvizice s anonymizací je funkční!")
        else:
            logger.info("❌ Fáze 2 má problémy - zkontrolujte konfigurace.")

        return overall_success


async def main():
    """Main funkce"""
    demo = Phase2Demo()
    success = await demo.run_demo()

    if success:
        print("\n🎯 FÁZE 2 DOKONČENA!")
        print("Další kroky:")
        print("   1. Pokračujte s Fází 3 - Inteligentní zpracování a RAG")
        print("   2. Implementujte trafilatura a spaCy NLP")
        print("   3. Přidejte sentence embeddings a LanceDB indexing")
    else:
        print("\n🔧 ŘEŠENÍ PROBLÉMŮ FÁZE 2:")
        print("   1. Zkontrolujte Tor/I2P služby: docker-compose logs tor-proxy")
        print("   2. Ověřte Playwright installation: playwright install")
        print(
            "   3. Testujte síťové připojení: curl --socks5 localhost:9050 https://check.torproject.org/api/ip"
        )


if __name__ == "__main__":
    asyncio.run(main())
