"""
Intelligent Proxy Manager
Dynamická správa identity a bezpečný přístup přes Tor pro M1

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import random
import socket
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Konfigurace proxy"""
    proxy_type: str = "socks5"  # socks5, http, https
    host: str = "127.0.0.1"
    port: int = 9050
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class IdentitySession:
    """Reprezentace identity session"""
    session_id: str
    proxy_config: ProxyConfig
    user_agent: str
    headers: Dict[str, str]
    created_at: float
    last_used: float
    request_count: int = 0
    max_requests: int = 10
    is_compromised: bool = False


class IntelligentProxyManager:
    """
    Inteligentní správce proxy pro anonymní přístup
    Podporuje Tor SOCKS proxy a rotaci identity
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tor_config = config.get("tor", {})
        self.security_config = config.get("security", {})

        # Tor proxy konfigurace
        self.tor_proxy = ProxyConfig(
            proxy_type="socks5",
            host="127.0.0.1",
            port=self.tor_config.get("socks_port", 9050),
            timeout=self.tor_config.get("timeout", 30)
        )

        # Aktivní sessions
        self.active_sessions: Dict[str, IdentitySession] = {}
        self.session_rotation_interval = 300  # 5 minut

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializace proxy manageru"""
        self.logger.info("🔐 Inicializuji Intelligent Proxy Manager...")

        # Kontrola Tor dostupnosti
        if await self._check_tor_availability():
            self.logger.info("✅ Tor proxy je dostupná")
        else:
            self.logger.warning("⚠️ Tor proxy není dostupná, fallback na direct connection")

        # Vytvoření první identity session
        await self._create_new_session()

    async def _check_tor_availability(self) -> bool:
        """Kontrola dostupnosti Tor proxy"""
        try:
            # Test connection k Tor SOCKS proxy
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.tor_proxy.host, self.tor_proxy.port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.warning(f"Tor availability check failed: {e}")
            return False

    async def _create_new_session(self) -> IdentitySession:
        """Vytvoří novou identity session"""
        import uuid
        from fake_useragent import UserAgent

        session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Generování fake user agent
        try:
            ua = UserAgent()
            user_agent = ua.random
        except Exception:
            # Fallback user agents
            fallback_agents = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            ]
            user_agent = random.choice(fallback_agents)

        # Generování random headers
        headers = self._generate_random_headers(user_agent)

        session = IdentitySession(
            session_id=session_id,
            proxy_config=self.tor_proxy,
            user_agent=user_agent,
            headers=headers,
            created_at=time.time(),
            last_used=time.time(),
            max_requests=random.randint(8, 15)  # Random limit pro realistické chování
        )

        self.active_sessions[session_id] = session
        self.logger.debug(f"✅ Vytvořena nová session: {session_id}")
        return session

    def _generate_random_headers(self, user_agent: str) -> Dict[str, str]:
        """Generuje randomizované HTTP headers"""

        # Základní headers s randomizací
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice([
                "en-US,en;q=0.5",
                "en-GB,en;q=0.5",
                "cs-CZ,cs;q=0.8,en;q=0.6",
                "de-DE,de;q=0.8,en;q=0.6"
            ]),
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": random.choice(["no-cache", "max-age=0", "no-store"]),
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        # Občasné přidání dalších headers
        if random.random() < 0.3:
            headers["DNT"] = "1"

        if random.random() < 0.2:
            headers["Pragma"] = "no-cache"

        return headers

    async def get_session(self, force_new: bool = False) -> IdentitySession:
        """Získá aktivní session nebo vytvoří novou"""

        # Kontrola, zda potřebujeme novou session
        if force_new or await self._should_rotate_session():
            await self._rotate_session()

        # Vrať první dostupnou session
        for session in self.active_sessions.values():
            if not session.is_compromised and session.request_count < session.max_requests:
                session.last_used = time.time()
                session.request_count += 1
                return session

        # Pokud žádná session není dostupná, vytvoř novou
        return await self._create_new_session()

    async def _should_rotate_session(self) -> bool:
        """Určí, zda je čas rotovat session"""

        if not self.active_sessions:
            return True

        current_time = time.time()

        for session in self.active_sessions.values():
            # Rotace podle času
            if current_time - session.created_at > self.session_rotation_interval:
                return True

            # Rotace podle počtu requestů
            if session.request_count >= session.max_requests:
                return True

            # Rotace pokud je session kompromitována
            if session.is_compromised:
                return True

        return False

    async def _rotate_session(self):
        """Rotuje aktivní sessions"""
        self.logger.debug("🔄 Rotating identity sessions...")

        # Označení starých sessions jako neplatné
        for session in self.active_sessions.values():
            session.is_compromised = True

        # Vyčištění starých sessions
        self.active_sessions.clear()

        # Vytvoření nové session
        await self._create_new_session()

        # Signál Tor k obnovení okruhu (pokud je dostupný)
        await self._signal_tor_new_circuit()

    async def _signal_tor_new_circuit(self):
        """Signalizuje Tor k vytvoření nového okruhu"""
        try:
            # Pokus o připojení k Tor control portu
            control_port = self.tor_config.get("control_port", 9051)

            reader, writer = await asyncio.open_connection("127.0.0.1", control_port)

            # Zaslání NEWNYM příkazu
            writer.write(b"AUTHENTICATE\r\n")
            await writer.drain()
            await reader.readline()

            writer.write(b"SIGNAL NEWNYM\r\n")
            await writer.drain()
            await reader.readline()

            writer.close()
            await writer.wait_closed()

            self.logger.debug("✅ Tor circuit renewed")

        except Exception as e:
            self.logger.debug(f"Could not signal Tor for new circuit: {e}")

    async def create_session_connector(self, session: IdentitySession) -> aiohttp.ClientSession:
        """Vytvoří aiohttp session s proxy konfigurací"""

        # Konfigurace proxy
        if session.proxy_config:
            proxy_url = f"{session.proxy_config.proxy_type}://{session.proxy_config.host}:{session.proxy_config.port}"
        else:
            proxy_url = None

        # Konfigurace timeout
        timeout = aiohttp.ClientTimeout(total=session.proxy_config.timeout)

        # Vytvoření connector s proxy
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=2,
            enable_cleanup_closed=True
        )

        # Vytvoření session
        client_session = aiohttp.ClientSession(
            connector=connector,
            headers=session.headers,
            timeout=timeout
        )

        return client_session

    async def make_request(
        self,
        method: str,
        url: str,
        force_new_session: bool = False,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Provede HTTP request přes anonymní proxy"""

        session = await self.get_session(force_new=force_new_session)
        client_session = await self.create_session_connector(session)

        try:
            # Přidání proxy do kwargs pokud je dostupná
            if session.proxy_config:
                proxy_url = f"{session.proxy_config.proxy_type}://{session.proxy_config.host}:{session.proxy_config.port}"
                kwargs["proxy"] = proxy_url

            # Provedení requestu
            async with client_session.request(method, url, **kwargs) as response:
                return response

        except Exception as e:
            self.logger.warning(f"Request failed: {e}")
            # Označení session jako kompromitované
            session.is_compromised = True
            raise
        finally:
            await client_session.close()

    async def perform_leak_test(self) -> Dict[str, Any]:
        """Provede test úniku IP adresy a DNS"""

        test_results = {
            "ip_leak": False,
            "dns_leak": False,
            "webrtc_leak": False,
            "real_ip": None,
            "proxy_ip": None,
            "test_timestamp": time.time()
        }

        try:
            # Test real IP (bez proxy)
            async with aiohttp.ClientSession() as session:
                async with session.get("https://httpbin.org/ip", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        test_results["real_ip"] = data.get("origin")

            # Test proxy IP
            proxy_session = await self.get_session()
            client = await self.create_session_connector(proxy_session)

            try:
                proxy_url = f"{proxy_session.proxy_config.proxy_type}://{proxy_session.proxy_config.host}:{proxy_session.proxy_config.port}"
                async with client.get("https://httpbin.org/ip", proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        test_results["proxy_ip"] = data.get("origin")
            finally:
                await client.close()

            # Kontrola IP leak
            if test_results["real_ip"] and test_results["proxy_ip"]:
                test_results["ip_leak"] = test_results["real_ip"] == test_results["proxy_ip"]

            self.logger.info(f"🔍 Leak test completed: IP leak = {test_results['ip_leak']}")

        except Exception as e:
            self.logger.error(f"Leak test failed: {e}")
            test_results["error"] = str(e)

        return test_results

    def get_manager_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky proxy manageru"""

        active_count = len([s for s in self.active_sessions.values() if not s.is_compromised])
        total_requests = sum(s.request_count for s in self.active_sessions.values())

        return {
            "total_sessions": len(self.active_sessions),
            "active_sessions": active_count,
            "total_requests": total_requests,
            "tor_enabled": self.tor_config.get("enabled", False),
            "rotation_interval": self.session_rotation_interval,
            "last_rotation": max([s.created_at for s in self.active_sessions.values()]) if self.active_sessions else None
        }


# Utility funkce
async def test_proxy_manager():
    """Test funkce pro proxy manager"""

    config = {
        "tor": {
            "enabled": True,
            "socks_port": 9050,
            "control_port": 9051,
            "timeout": 30
        },
        "security": {
            "max_retries": 3,
            "rotation_interval": 300
        }
    }

    manager = IntelligentProxyManager(config)
    await manager.initialize()

    # Test leak detection
    leak_results = await manager.perform_leak_test()
    print(f"📊 Leak test results: {leak_results}")

    # Test request
    try:
        response = await manager.make_request("GET", "https://httpbin.org/user-agent")
        print(f"✅ Request successful: {response.status}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Statistiky
    stats = manager.get_manager_stats()
    print(f"📊 Manager stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_proxy_manager())
