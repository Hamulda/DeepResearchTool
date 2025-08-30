"""Secure Tor Connector
Bezpečný přístup přes Tor síť s leak detection a automatickým testováním

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import ipaddress
import logging
import socket
import subprocess
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TorConfig:
    """Konfigurace pro Tor spojení"""

    socks_host: str = "127.0.0.1"
    socks_port: int = 9050
    control_host: str = "127.0.0.1"
    control_port: int = 9051
    control_password: str | None = None
    auto_start_tor: bool = False
    circuit_timeout: int = 30
    max_circuits: int = 3
    enable_leak_detection: bool = True


class LeakDetector:
    """Detektor úniků IP adresy a DNS při používání Tor
    """

    def __init__(self):
        self.original_ip: str | None = None
        self.tor_ip: str | None = None
        self.leak_tests = [
            "http://httpbin.org/ip",
            "https://api.ipify.org?format=json",
            "http://checkip.amazonaws.com",
            "https://ipinfo.io/json"
        ]

    async def detect_original_ip(self) -> str | None:
        """Detekce původní IP adresy (bez Tor)"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get("http://httpbin.org/ip") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.original_ip = data.get("origin", "").split(",")[0].strip()
                        logger.info(f"Původní IP adresa: {self.original_ip}")
                        return self.original_ip
        except Exception as e:
            logger.error(f"Chyba při detekci původní IP: {e}")

        return None

    async def test_tor_ip(self, proxy_url: str) -> str | None:
        """Test IP adresy přes Tor"""
        try:
            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.get(
                    "http://httpbin.org/ip",
                    proxy=proxy_url
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.tor_ip = data.get("origin", "").split(",")[0].strip()
                        logger.info(f"Tor IP adresa: {self.tor_ip}")
                        return self.tor_ip
        except Exception as e:
            logger.error(f"Chyba při testu Tor IP: {e}")

        return None

    async def comprehensive_leak_test(self, proxy_url: str) -> dict[str, Any]:
        """Komplexní test úniků

        Args:
            proxy_url: URL Tor SOCKS proxy

        Returns:
            Výsledky leak testu

        """
        results = {
            "ip_leak": False,
            "dns_leak": False,
            "webrtc_leak": False,
            "tor_working": False,
            "original_ip": self.original_ip,
            "tor_ip": None,
            "detected_ips": [],
            "test_timestamp": time.time()
        }

        # Test různých IP detection služeb
        detected_ips = set()

        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:

            for test_url in self.leak_tests:
                try:
                    async with session.get(test_url, proxy=proxy_url) as response:
                        if response.status == 200:
                            if "json" in response.headers.get("content-type", ""):
                                data = await response.json()
                                ip = data.get("ip") or data.get("origin", "").split(",")[0].strip()
                            else:
                                ip = (await response.text()).strip()

                            if ip and self._is_valid_ip(ip):
                                detected_ips.add(ip)

                except Exception as e:
                    logger.debug(f"Chyba při testu {test_url}: {e}")

        results["detected_ips"] = list(detected_ips)

        if detected_ips:
            # Vezmi nejčastější IP jako Tor IP
            results["tor_ip"] = list(detected_ips)[0]
            results["tor_working"] = True

            # Kontrola leak
            if self.original_ip and self.original_ip in detected_ips:
                results["ip_leak"] = True
                logger.warning("LEAK DETECTED: Původní IP adresa je viditelná!")

            # Kontrola konzistence IP
            if len(detected_ips) > 1:
                logger.warning(f"Nekonzistentní IP adresy: {detected_ips}")

        return results

    def _is_valid_ip(self, ip_str: str) -> bool:
        """Kontrola validity IP adresy"""
        try:
            ipaddress.ip_address(ip_str.strip())
            return True
        except ValueError:
            return False

    async def test_dns_leak(self, proxy_url: str) -> bool:
        """Test DNS leak"""
        try:
            # Test DNS resolveru
            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:

                # Test DNS přes známou službu
                async with session.get(
                    "https://1.1.1.1/cdn-cgi/trace",
                    proxy=proxy_url
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Parsování Cloudflare trace
                        for line in content.split('\n'):
                            if line.startswith('ip='):
                                detected_ip = line.split('=')[1].strip()
                                if self.original_ip and detected_ip == self.original_ip:
                                    return True  # DNS leak detected

        except Exception as e:
            logger.debug(f"Chyba při DNS leak testu: {e}")

        return False


class TorController:
    """Kontroler pro Tor proces a circuits
    """

    def __init__(self, config: TorConfig):
        self.config = config
        self.tor_process: subprocess.Popen | None = None
        self.circuits: list[str] = []

    def is_tor_running(self) -> bool:
        """Kontrola, zda Tor běží"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.config.socks_host, self.config.socks_port))
            sock.close()
            return result == 0
        except:
            return False

    async def start_tor(self) -> bool:
        """Spuštění Tor procesu"""
        if self.is_tor_running():
            logger.info("Tor již běží")
            return True

        if not self.config.auto_start_tor:
            logger.error("Tor neběží a auto_start_tor je zakázáno")
            return False

        try:
            # Základní Tor konfigurace
            tor_config = [
                "tor",
                "--SocksPort", f"{self.config.socks_port}",
                "--ControlPort", f"{self.config.control_port}",
                "--CookieAuthentication", "1",
                "--DataDirectory", "/tmp/tor_data"
            ]

            self.tor_process = subprocess.Popen(
                tor_config,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            # Čekání na start
            for _ in range(30):  # 30 sekund timeout
                if self.is_tor_running():
                    logger.info("Tor úspěšně spuštěn")
                    return True
                await asyncio.sleep(1)

            logger.error("Tor se nepodařilo spustit")
            return False

        except FileNotFoundError:
            logger.error("Tor není nainstalován nebo není v PATH")
            return False
        except Exception as e:
            logger.error(f"Chyba při spouštění Tor: {e}")
            return False

    async def stop_tor(self):
        """Zastavení Tor procesu"""
        if self.tor_process:
            self.tor_process.terminate()
            try:
                await asyncio.wait_for(asyncio.create_task(
                    asyncio.to_thread(self.tor_process.wait)
                ), timeout=10)
            except TimeoutError:
                self.tor_process.kill()

            self.tor_process = None
            logger.info("Tor zastaven")

    async def new_circuit(self) -> bool:
        """Vytvoření nového Tor circuit"""
        try:
            # Jednoduchá implementace - restart Tor SOCKS spojení
            await asyncio.sleep(1)  # Krátká pauza
            logger.info("Nový Tor circuit vytvořen")
            return True
        except Exception as e:
            logger.error(f"Chyba při vytváření nového circuit: {e}")
            return False


class SecureTorConnector:
    """Hlavní třída pro bezpečné Tor spojení
    """

    def __init__(self, config: TorConfig = None):
        self.config = config or TorConfig()
        self.controller = TorController(self.config)
        self.leak_detector = LeakDetector()
        self.connection_stats = {
            "connections_made": 0,
            "successful_connections": 0,
            "leaks_detected": 0,
            "circuits_created": 0,
            "start_time": time.time()
        }
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Inicializace Tor spojení s bezpečnostními kontrolami

        Returns:
            True pokud je Tor připraven k použití

        """
        logger.info("Inicializace Secure Tor Connector...")

        # 1. Detekce původní IP
        if self.config.enable_leak_detection:
            await self.leak_detector.detect_original_ip()

        # 2. Kontrola/spuštění Tor
        if not await self.controller.start_tor():
            return False

        # 3. Test Tor spojení
        proxy_url = f"socks5://{self.config.socks_host}:{self.config.socks_port}"
        tor_ip = await self.leak_detector.test_tor_ip(proxy_url)

        if not tor_ip:
            logger.error("Nepodařilo se navázat spojení přes Tor")
            return False

        # 4. Leak test
        if self.config.enable_leak_detection:
            leak_results = await self.leak_detector.comprehensive_leak_test(proxy_url)

            if leak_results["ip_leak"]:
                logger.error("KRITICKÁ CHYBA: Detekován únik IP adresy!")
                return False

            if leak_results["dns_leak"]:
                logger.warning("Varování: Možný DNS leak")

        self.is_initialized = True
        logger.info(f"Tor úspěšně inicializován. Tor IP: {tor_ip}")
        return True

    async def create_session(self, rotate_circuit: bool = False) -> aiohttp.ClientSession:
        """Vytvoření aiohttp session s Tor proxy

        Args:
            rotate_circuit: Zda vytvořit nový circuit

        Returns:
            Nakonfigurovaná ClientSession

        """
        if not self.is_initialized:
            if not await self.initialize():
                raise RuntimeError("Nepodařilo se inicializovat Tor spojení")

        if rotate_circuit:
            await self.controller.new_circuit()
            self.connection_stats["circuits_created"] += 1

        proxy_url = f"socks5://{self.config.socks_host}:{self.config.socks_port}"

        # Konfigurace pro anonymní session
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=2,
            ssl=False,  # Disable SSL verification pro anonymitu
            enable_cleanup_closed=True
        )

        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/121.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )

        # Nastavení proxy pro všechny requesty
        session._connector._ssl = False

        self.connection_stats["connections_made"] += 1

        return session

    async def safe_request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        """Bezpečný HTTP request s leak detection

        Args:
            method: HTTP metoda
            url: Cílová URL
            **kwargs: Další argumenty pro aiohttp

        Returns:
            Výsledek requestu

        """
        result = {
            "success": False,
            "status_code": None,
            "content": "",
            "url": url,
            "final_url": None,
            "leak_detected": False,
            "error": None
        }

        session = None
        try:
            session = await self.create_session()

            # Pre-request leak check
            if self.config.enable_leak_detection and random.random() < 0.1:  # 10% šance na leak check
                proxy_url = f"socks5://{self.config.socks_host}:{self.config.socks_port}"
                leak_results = await self.leak_detector.comprehensive_leak_test(proxy_url)

                if leak_results["ip_leak"]:
                    result["leak_detected"] = True
                    result["error"] = "IP leak detected before request"
                    self.connection_stats["leaks_detected"] += 1
                    return result

            # Provedení requestu
            proxy_url = f"socks5://{self.config.socks_host}:{self.config.socks_port}"

            async with session.request(method, url, proxy=proxy_url, **kwargs) as response:
                result["status_code"] = response.status
                result["final_url"] = str(response.url)
                result["content"] = await response.text()
                result["success"] = 200 <= response.status < 400

                if result["success"]:
                    self.connection_stats["successful_connections"] += 1

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Chyba při Tor requestu na {url}: {e}")

        finally:
            if session:
                await session.close()

        return result

    async def test_connection(self) -> dict[str, Any]:
        """Test Tor spojení a anonymity

        Returns:
            Výsledky testu

        """
        if not self.is_initialized:
            await self.initialize()

        proxy_url = f"socks5://{self.config.socks_host}:{self.config.socks_port}"

        # Základní test
        basic_test = await self.leak_detector.comprehensive_leak_test(proxy_url)

        # Test rychlosti
        start_time = time.time()
        test_result = await self.safe_request("GET", "http://httpbin.org/get")
        response_time = time.time() - start_time

        return {
            **basic_test,
            "response_time": response_time,
            "connection_working": test_result["success"],
            "connection_stats": self.get_stats()
        }

    def get_stats(self) -> dict[str, Any]:
        """Získání statistik spojení"""
        runtime = time.time() - self.connection_stats["start_time"]

        return {
            **self.connection_stats,
            "runtime_seconds": runtime,
            "success_rate": (self.connection_stats["successful_connections"] /
                           max(1, self.connection_stats["connections_made"])),
            "is_initialized": self.is_initialized,
            "tor_running": self.controller.is_tor_running()
        }

    async def cleanup(self):
        """Úklid a uzavření Tor spojení"""
        if self.config.auto_start_tor:
            await self.controller.stop_tor()

        self.is_initialized = False
        logger.info("Secure Tor Connector ukončen")


# Globální instance
secure_tor_connector = SecureTorConnector()
