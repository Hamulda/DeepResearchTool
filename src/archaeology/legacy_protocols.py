"""
Legacy Protocol Detector pro DeepResearchTool
Implementuje detekci a extrakci dat ze zastaralých protokolů: Gopher, Finger, NNTP.
"""

import asyncio
import logging
import re
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class LegacyProtocolResult:
    """Výsledek skenování legacy protokolu"""
    protocol: str
    host: str
    port: int
    status: str  # "active", "inactive", "error"
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scan_timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None


@dataclass
class GopherMenuEntry:
    """Záznam z Gopher menu"""
    item_type: str
    display_string: str
    selector: str
    hostname: str
    port: int
    raw_line: str


@dataclass
class FingerInfo:
    """Informace získané přes Finger protokol"""
    user: str
    full_name: Optional[str] = None
    login_time: Optional[str] = None
    idle_time: Optional[str] = None
    terminal: Optional[str] = None
    remote_host: Optional[str] = None
    plan: Optional[str] = None
    project: Optional[str] = None


@dataclass
class UsenetGroup:
    """Informace o Usenet skupině"""
    name: str
    last_article: int
    first_article: int
    posting_allowed: bool
    description: Optional[str] = None


class GopherClient:
    """Klient pro Gopher protokol (RFC 1436)"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def connect_and_fetch(self, host: str, port: int = 70, selector: str = "") -> Optional[str]:
        """Připojení k Gopher serveru a získání obsahu"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Odeslání Gopher požadavku
            request = f"{selector}\r\n"
            writer.write(request.encode('utf-8'))
            await writer.drain()

            # Čtení odpovědi
            response_data = b""
            while True:
                try:
                    chunk = await asyncio.wait_for(reader.read(8192), timeout=5)
                    if not chunk:
                        break
                    response_data += chunk
                except asyncio.TimeoutError:
                    break

            writer.close()
            await writer.wait_closed()

            return response_data.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.debug(f"Gopher připojení selhalo pro {host}:{port} - {e}")
            return None

    def parse_gopher_menu(self, content: str) -> List[GopherMenuEntry]:
        """Parsování Gopher menu"""
        entries = []

        for line in content.split('\n'):
            line = line.strip()
            if not line or line == '.':
                continue

            try:
                # Gopher menu format: Type + Display + TAB + Selector + TAB + Host + TAB + Port
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        item_type = parts[0][0] if parts[0] else 'i'
                        display_string = parts[0][1:] if len(parts[0]) > 1 else ""
                        selector = parts[1]
                        hostname = parts[2]
                        port = int(parts[3]) if parts[3].isdigit() else 70

                        entry = GopherMenuEntry(
                            item_type=item_type,
                            display_string=display_string,
                            selector=selector,
                            hostname=hostname,
                            port=port,
                            raw_line=line
                        )
                        entries.append(entry)

            except Exception as e:
                logger.debug(f"Chyba při parsování Gopher menu řádku: {e}")

        return entries


class FingerClient:
    """Klient pro Finger protokol (RFC 742)"""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def finger_user(self, user: str, host: str, port: int = 79) -> Optional[FingerInfo]:
        """Finger dotaz na konkrétního uživatele"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Finger request format
            request = f"{user}\r\n"
            writer.write(request.encode('utf-8'))
            await writer.drain()

            # Čtení odpovědi
            response = await asyncio.wait_for(reader.read(4096), timeout=10)

            writer.close()
            await writer.wait_closed()

            if response:
                response_text = response.decode('utf-8', errors='ignore')
                return self._parse_finger_response(user, response_text)

        except Exception as e:
            logger.debug(f"Finger dotaz selhal pro {user}@{host} - {e}")

        return None

    async def finger_list_users(self, host: str, port: int = 79) -> List[str]:
        """Seznam uživatelů na serveru"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Prázdný dotaz pro seznam uživatelů
            writer.write(b"\r\n")
            await writer.drain()

            response = await asyncio.wait_for(reader.read(4096), timeout=10)

            writer.close()
            await writer.wait_closed()

            if response:
                response_text = response.decode('utf-8', errors='ignore')
                return self._extract_usernames(response_text)

        except Exception as e:
            logger.debug(f"Finger list users selhal pro {host} - {e}")

        return []

    def _parse_finger_response(self, user: str, response: str) -> FingerInfo:
        """Parsování Finger odpovědi"""
        info = FingerInfo(user=user)

        lines = response.split('\n')
        for line in lines:
            line = line.strip()

            # Hledání různých vzorů informací
            if 'Name:' in line or 'Real Name:' in line:
                info.full_name = re.search(r'(?:Name|Real Name):\s*(.+)', line)
                if info.full_name:
                    info.full_name = info.full_name.group(1).strip()

            elif 'Last login:' in line or 'Login:' in line:
                info.login_time = re.search(r'(?:Last login|Login):\s*(.+)', line)
                if info.login_time:
                    info.login_time = info.login_time.group(1).strip()

            elif 'Idle:' in line:
                info.idle_time = re.search(r'Idle:\s*(.+)', line)
                if info.idle_time:
                    info.idle_time = info.idle_time.group(1).strip()

            elif line.startswith('Plan:'):
                # Plán je obvykle na dalších řádcích
                plan_start = lines.index(line)
                plan_lines = lines[plan_start+1:]
                info.plan = '\n'.join(plan_lines).strip()
                break

        return info

    def _extract_usernames(self, response: str) -> List[str]:
        """Extrakce uživatelských jmen z odpovědi"""
        usernames = []

        # Různé formáty výstupu finger serveru
        lines = response.split('\n')
        for line in lines:
            line = line.strip()

            # Hledání vzorů uživatelských jmen
            username_match = re.search(r'^(\w+)\s+', line)
            if username_match:
                usernames.append(username_match.group(1))

        return list(set(usernames))  # Odstranění duplikátů


class NNTPClient:
    """Klient pro NNTP (Usenet) protokol (RFC 3977)"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def connect_and_probe(self, host: str, port: int = 119) -> Optional[Dict[str, Any]]:
        """Připojení k NNTP serveru a základní probe"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Čtení úvodní zprávy
            initial_response = await asyncio.wait_for(reader.readline(), timeout=10)
            initial_msg = initial_response.decode('utf-8', errors='ignore').strip()

            server_info = {
                "initial_response": initial_msg,
                "capabilities": [],
                "groups_count": 0,
                "server_software": None
            }

            # CAPABILITIES příkaz
            writer.write(b"CAPABILITIES\r\n")
            await writer.drain()

            capabilities = []
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                line_text = line.decode('utf-8', errors='ignore').strip()

                if line_text == '.' or not line_text:
                    break
                if line_text.startswith('101'):  # Capability list follows
                    continue

                capabilities.append(line_text)

            server_info["capabilities"] = capabilities

            # LIST příkaz pro počet skupin
            writer.write(b"LIST\r\n")
            await writer.drain()

            list_response = await asyncio.wait_for(reader.readline(), timeout=5)
            if list_response.startswith(b"215"):  # List follows
                groups_count = 0
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=5)
                    line_text = line.decode('utf-8', errors='ignore').strip()

                    if line_text == '.' or not line_text:
                        break
                    groups_count += 1

                    # Omezení počtu čtených skupin pro rychlost
                    if groups_count >= 100:
                        break

                server_info["groups_count"] = groups_count

            # QUIT
            writer.write(b"QUIT\r\n")
            await writer.drain()

            writer.close()
            await writer.wait_closed()

            return server_info

        except Exception as e:
            logger.debug(f"NNTP připojení selhalo pro {host}:{port} - {e}")
            return None

    async def list_newsgroups(self, host: str, port: int = 119, limit: int = 50) -> List[UsenetGroup]:
        """Seznam Usenet skupin"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Čtení úvodní zprávy
            await asyncio.wait_for(reader.readline(), timeout=10)

            # LIST příkaz
            writer.write(b"LIST\r\n")
            await writer.drain()

            list_response = await asyncio.wait_for(reader.readline(), timeout=5)

            groups = []
            if list_response.startswith(b"215"):  # List follows
                count = 0
                while count < limit:
                    line = await asyncio.wait_for(reader.readline(), timeout=5)
                    line_text = line.decode('utf-8', errors='ignore').strip()

                    if line_text == '.' or not line_text:
                        break

                    # Parsování: group_name last_article first_article posting_status
                    parts = line_text.split()
                    if len(parts) >= 4:
                        group = UsenetGroup(
                            name=parts[0],
                            last_article=int(parts[1]) if parts[1].isdigit() else 0,
                            first_article=int(parts[2]) if parts[2].isdigit() else 0,
                            posting_allowed=parts[3].lower() == 'y'
                        )
                        groups.append(group)
                        count += 1

            writer.write(b"QUIT\r\n")
            await writer.drain()

            writer.close()
            await writer.wait_closed()

            return groups

        except Exception as e:
            logger.debug(f"NNTP list groups selhal pro {host}:{port} - {e}")
            return []


class LegacyProtocolDetector:
    """
    Hlavní třída pro detekci a analýzu legacy protokolů:
    - Gopher (port 70)
    - Finger (port 79)
    - NNTP/Usenet (port 119)
    """

    def __init__(self,
                 timeout: int = 30,
                 max_concurrent_scans: int = 10):
        self.timeout = timeout
        self.gopher_client = GopherClient(timeout)
        self.finger_client = FingerClient(timeout)
        self.nntp_client = NNTPClient(timeout)

        # Semafór pro omezení souběžných skenů
        self._scan_semaphore = asyncio.Semaphore(max_concurrent_scans)

        logger.info("LegacyProtocolDetector inicializován")

    async def scan_gopher_server(self, host: str, port: int = 70) -> LegacyProtocolResult:
        """Skenování Gopher serveru"""
        async with self._scan_semaphore:
            start_time = datetime.now()

            try:
                content = await self.gopher_client.connect_and_fetch(host, port)

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000

                if content is not None:
                    # Parsování Gopher menu
                    menu_entries = self.gopher_client.parse_gopher_menu(content)

                    metadata = {
                        "menu_entries_count": len(menu_entries),
                        "menu_entries": [
                            {
                                "type": entry.item_type,
                                "display": entry.display_string,
                                "selector": entry.selector,
                                "hostname": entry.hostname,
                                "port": entry.port
                            }
                            for entry in menu_entries[:20]  # Omezení na prvních 20
                        ],
                        "content_length": len(content),
                        "has_subdirectories": any(entry.item_type == '1' for entry in menu_entries),
                        "has_files": any(entry.item_type == '0' for entry in menu_entries)
                    }

                    return LegacyProtocolResult(
                        protocol="gopher",
                        host=host,
                        port=port,
                        status="active",
                        content=content[:1000],  # Omezení velikosti
                        metadata=metadata,
                        response_time_ms=response_time
                    )
                else:
                    return LegacyProtocolResult(
                        protocol="gopher",
                        host=host,
                        port=port,
                        status="inactive",
                        response_time_ms=response_time
                    )

            except Exception as e:
                return LegacyProtocolResult(
                    protocol="gopher",
                    host=host,
                    port=port,
                    status="error",
                    metadata={"error": str(e)}
                )

    async def scan_finger_server(self, host: str, port: int = 79) -> LegacyProtocolResult:
        """Skenování Finger serveru"""
        async with self._scan_semaphore:
            start_time = datetime.now()

            try:
                # Zkusíme seznam uživatelů
                users = await self.finger_client.finger_list_users(host, port)

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000

                if users:
                    # Zkusíme finger na prvního uživatele pro více informací
                    user_info = None
                    if users:
                        user_info = await self.finger_client.finger_user(users[0], host, port)

                    metadata = {
                        "users_found": len(users),
                        "user_list": users[:10],  # Prvních 10 uživatelů
                        "sample_user_info": {
                            "user": user_info.user if user_info else None,
                            "full_name": user_info.full_name if user_info else None,
                            "login_time": user_info.login_time if user_info else None
                        } if user_info else None
                    }

                    return LegacyProtocolResult(
                        protocol="finger",
                        host=host,
                        port=port,
                        status="active",
                        content=f"Found {len(users)} users: {', '.join(users[:5])}",
                        metadata=metadata,
                        response_time_ms=response_time
                    )
                else:
                    # Server odpovídá, ale žádní uživatelé
                    return LegacyProtocolResult(
                        protocol="finger",
                        host=host,
                        port=port,
                        status="active",
                        content="No users found",
                        metadata={"users_found": 0},
                        response_time_ms=response_time
                    )

            except Exception as e:
                return LegacyProtocolResult(
                    protocol="finger",
                    host=host,
                    port=port,
                    status="error",
                    metadata={"error": str(e)}
                )

    async def scan_nntp_server(self, host: str, port: int = 119) -> LegacyProtocolResult:
        """Skenování NNTP/Usenet serveru"""
        async with self._scan_semaphore:
            start_time = datetime.now()

            try:
                server_info = await self.nntp_client.connect_and_probe(host, port)

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000

                if server_info:
                    # Získání seznamu skupin
                    groups = await self.nntp_client.list_newsgroups(host, port, limit=20)

                    metadata = {
                        "server_response": server_info["initial_response"],
                        "capabilities": server_info["capabilities"],
                        "estimated_groups_count": server_info["groups_count"],
                        "sample_groups": [
                            {
                                "name": group.name,
                                "articles_count": group.last_article - group.first_article,
                                "posting_allowed": group.posting_allowed
                            }
                            for group in groups
                        ]
                    }

                    return LegacyProtocolResult(
                        protocol="nntp",
                        host=host,
                        port=port,
                        status="active",
                        content=f"NNTP server with ~{server_info['groups_count']} groups",
                        metadata=metadata,
                        response_time_ms=response_time
                    )
                else:
                    return LegacyProtocolResult(
                        protocol="nntp",
                        host=host,
                        port=port,
                        status="inactive",
                        response_time_ms=response_time
                    )

            except Exception as e:
                return LegacyProtocolResult(
                    protocol="nntp",
                    host=host,
                    port=port,
                    status="error",
                    metadata={"error": str(e)}
                )

    async def comprehensive_legacy_scan(
        self,
        host: str,
        protocols: Optional[List[str]] = None
    ) -> Dict[str, LegacyProtocolResult]:
        """
        Komplexní skenování všech legacy protokolů na hostu
        """
        if protocols is None:
            protocols = ["gopher", "finger", "nntp"]

        results = {}

        # Definice portů pro protokoly
        protocol_ports = {
            "gopher": 70,
            "finger": 79,
            "nntp": 119
        }

        # Paralelní skenování všech protokolů
        tasks = []

        for protocol in protocols:
            if protocol in protocol_ports:
                port = protocol_ports[protocol]

                if protocol == "gopher":
                    task = self.scan_gopher_server(host, port)
                elif protocol == "finger":
                    task = self.scan_finger_server(host, port)
                elif protocol == "nntp":
                    task = self.scan_nntp_server(host, port)
                else:
                    continue

                tasks.append((protocol, task))

        # Spuštění všech skenů současně
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            for (protocol, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    results[protocol] = LegacyProtocolResult(
                        protocol=protocol,
                        host=host,
                        port=protocol_ports[protocol],
                        status="error",
                        metadata={"error": str(result)}
                    )
                else:
                    results[protocol] = result

        return results

    async def deep_gopher_exploration(
        self,
        host: str,
        port: int = 70,
        max_depth: int = 3,
        max_items: int = 100
    ) -> Dict[str, Any]:
        """
        Hluboké prohledávání Gopher hole s rekurzivním procházením
        """
        explored_items = []
        visited_selectors = set()

        async def explore_recursive(selector: str, depth: int):
            if depth > max_depth or len(explored_items) >= max_items:
                return

            if selector in visited_selectors:
                return

            visited_selectors.add(selector)

            content = await self.gopher_client.connect_and_fetch(host, port, selector)
            if content:
                menu_entries = self.gopher_client.parse_gopher_menu(content)

                for entry in menu_entries:
                    item_info = {
                        "selector": entry.selector,
                        "type": entry.item_type,
                        "display": entry.display_string,
                        "depth": depth,
                        "size": len(content) if content else 0
                    }

                    # Pokud je to soubor, zkusíme stáhnout obsah
                    if entry.item_type == '0':  # Text file
                        file_content = await self.gopher_client.connect_and_fetch(
                            entry.hostname, entry.port, entry.selector
                        )
                        if file_content:
                            item_info["content_preview"] = file_content[:200]
                            item_info["content_size"] = len(file_content)

                    # Pokud je to adresář, rekurzivně prozkoumáme
                    elif entry.item_type == '1' and depth < max_depth:
                        await explore_recursive(entry.selector, depth + 1)

                    explored_items.append(item_info)

        # Začneme od root
        await explore_recursive("", 0)

        return {
            "host": host,
            "port": port,
            "total_items_found": len(explored_items),
            "max_depth_reached": max([item["depth"] for item in explored_items]) if explored_items else 0,
            "items": explored_items,
            "file_types_found": list(set([item["type"] for item in explored_items])),
            "exploration_summary": {
                "text_files": len([i for i in explored_items if i["type"] == "0"]),
                "directories": len([i for i in explored_items if i["type"] == "1"]),
                "search_items": len([i for i in explored_items if i["type"] == "7"]),
                "other_items": len([i for i in explored_items if i["type"] not in ["0", "1", "7"]])
            }
        }

    def generate_legacy_report(self, scan_results: Dict[str, LegacyProtocolResult]) -> Dict[str, Any]:
        """Generování reportu z legacy skenování"""
        active_protocols = [p for p, r in scan_results.items() if r.status == "active"]

        report = {
            "scan_summary": {
                "total_protocols_scanned": len(scan_results),
                "active_protocols": len(active_protocols),
                "protocols_found": active_protocols,
                "scan_timestamp": datetime.now().isoformat()
            },
            "detailed_results": {},
            "security_implications": [],
            "recommendations": []
        }

        for protocol, result in scan_results.items():
            report["detailed_results"][protocol] = {
                "status": result.status,
                "response_time_ms": result.response_time_ms,
                "content_preview": result.content[:100] if result.content else None,
                "metadata": result.metadata
            }

            # Bezpečnostní implikace
            if result.status == "active":
                if protocol == "finger":
                    report["security_implications"].append(
                        f"Finger protokol na {result.host}:{result.port} odhaluje systémové informace"
                    )
                elif protocol == "gopher":
                    report["security_implications"].append(
                        f"Gopher server na {result.host}:{result.port} může obsahovat historická data"
                    )
                elif protocol == "nntp":
                    report["security_implications"].append(
                        f"NNTP server na {result.host}:{result.port} poskytuje přístup k diskusním skupinám"
                    )

        # Doporučení
        if active_protocols:
            report["recommendations"].extend([
                "Prověřte všechny aktivní legacy protokoly na citlivé informace",
                "Zvažte zakázání nepoužívaných legacy služeb",
                "Implementujte monitoring pro legacy protokoly"
            ])

        return report
