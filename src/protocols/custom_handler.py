"""
Custom Protocol Handler pro DeepResearchTool
Implementace klientů pro nestandardní protokoly: Gemini, IPFS, Matrix.
"""

import asyncio
import json
import logging
import socket
import ssl
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import httpx

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class ProtocolResponse:
    """Odpověď z custom protokolu"""
    protocol: str
    url: str
    status_code: Optional[int] = None
    content: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0
    error: Optional[str] = None


@dataclass
class GeminiDocument:
    """Gemini dokument s metadaty"""
    url: str
    content: str
    mime_type: str
    charset: str = "utf-8"
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IPFSContent:
    """IPFS obsah s metadaty"""
    hash: str
    content: bytes
    content_type: str
    size: int
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixEvent:
    """Matrix event z room"""
    event_id: str
    event_type: str
    sender: str
    content: Dict[str, Any]
    timestamp: int
    room_id: str


class GeminiClient:
    """Klient pro Gemini protokol (gemini://)"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.default_port = 1965

    async def fetch(self, url: str) -> GeminiDocument:
        """Načtení Gemini dokumentu"""
        parsed = urlparse(url)

        if parsed.scheme != "gemini":
            raise ValueError("Není Gemini URL")

        host = parsed.hostname
        port = parsed.port or self.default_port
        path = parsed.path or "/"

        if parsed.query:
            path += f"?{parsed.query}"

        # TLS připojení
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # Gemini používá self-signed certs

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=context),
                timeout=self.timeout
            )

            # Gemini request
            request = f"{url}\r\n"
            writer.write(request.encode('utf-8'))
            await writer.drain()

            # Čtení response
            response_line = await reader.readline()
            header = response_line.decode('utf-8').strip()

            # Parse status
            status_parts = header.split(' ', 1)
            status_code = int(status_parts[0])
            meta = status_parts[1] if len(status_parts) > 1 else ""

            content = ""
            if status_code == 20:  # Success
                # Čtení content
                content_bytes = await reader.read()
                content = content_bytes.decode('utf-8', errors='ignore')

            writer.close()
            await writer.wait_closed()

            # Parse Gemini content pro links
            links = self._extract_gemini_links(content)

            return GeminiDocument(
                url=url,
                content=content,
                mime_type=meta.split(';')[0] if ';' in meta else meta,
                links=links,
                metadata={
                    "status_code": status_code,
                    "meta": meta,
                    "host": host,
                    "port": port
                }
            )

        except Exception as e:
            raise Exception(f"Gemini fetch failed: {e}")

    def _extract_gemini_links(self, content: str) -> List[str]:
        """Extrakce linků z Gemini textu"""
        links = []

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('=>'):
                # Gemini link format: => URL [description]
                parts = line[2:].strip().split(' ', 1)
                if parts[0]:
                    links.append(parts[0])

        return links

    async def crawl_gemini_space(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 50
    ) -> List[GeminiDocument]:
        """Crawling Gemini space"""
        visited = set()
        to_visit = [(start_url, 0)]
        documents = []

        while to_visit and len(documents) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited or depth > max_depth:
                continue

            visited.add(url)

            try:
                doc = await self.fetch(url)
                documents.append(doc)

                # Přidání linků pro další crawling
                if depth < max_depth:
                    for link in doc.links:
                        if link.startswith('gemini://') and link not in visited:
                            to_visit.append((link, depth + 1))

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Gemini crawl error pro {url}: {e}")

        return documents


class IPFSClient:
    """Klient pro IPFS (InterPlanetary File System)"""

    def __init__(self, gateway_url: str = "https://ipfs.io"):
        self.gateway_url = gateway_url.rstrip('/')
        self.api_url = "http://127.0.0.1:5001"  # Local IPFS node

    async def fetch_by_hash(self, ipfs_hash: str) -> IPFSContent:
        """Načtení obsahu podle IPFS hash"""

        # Pokus o local node
        try:
            content = await self._fetch_from_local_node(ipfs_hash)
            if content:
                return content
        except Exception as e:
            logger.debug(f"Local IPFS node nedostupný: {e}")

        # Fallback na gateway
        return await self._fetch_from_gateway(ipfs_hash)

    async def _fetch_from_local_node(self, ipfs_hash: str) -> Optional[IPFSContent]:
        """Fetch z lokálního IPFS node"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_url}/api/v0/cat"
            params = {"arg": ipfs_hash}

            async with session.post(url, params=params) as response:
                if response.status == 200:
                    content = await response.read()
                    content_type = response.headers.get('content-type', 'application/octet-stream')

                    return IPFSContent(
                        hash=ipfs_hash,
                        content=content,
                        content_type=content_type,
                        size=len(content),
                        metadata={"source": "local_node"}
                    )

        return None

    async def _fetch_from_gateway(self, ipfs_hash: str) -> IPFSContent:
        """Fetch z IPFS gateway"""
        url = f"{self.gateway_url}/ipfs/{ipfs_hash}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    content_type = response.headers.get('content-type', 'application/octet-stream')

                    return IPFSContent(
                        hash=ipfs_hash,
                        content=content,
                        content_type=content_type,
                        size=len(content),
                        metadata={
                            "source": "gateway",
                            "gateway_url": self.gateway_url
                        }
                    )
                else:
                    raise Exception(f"IPFS gateway error: {response.status}")

    async def resolve_ipns(self, ipns_name: str) -> str:
        """Resolving IPNS name na IPFS hash"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_url}/api/v0/name/resolve"
            params = {"arg": ipns_name}

            async with session.post(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    path = data.get("Path", "")
                    if path.startswith("/ipfs/"):
                        return path[6:]  # Odstranění "/ipfs/" prefixu

        raise Exception(f"IPNS resolution failed for {ipns_name}")

    async def list_directory(self, ipfs_hash: str) -> List[Dict[str, Any]]:
        """Seznam souborů v IPFS adresáři"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_url}/api/v0/ls"
            params = {"arg": ipfs_hash}

            async with session.post(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    objects = data.get("Objects", [])

                    if objects:
                        links = objects[0].get("Links", [])
                        return [
                            {
                                "name": link.get("Name"),
                                "hash": link.get("Hash"),
                                "size": link.get("Size"),
                                "type": link.get("Type")
                            }
                            for link in links
                        ]

        return []

    async def search_content(self, query: str) -> List[str]:
        """Hledání obsahu v IPFS (basic implementation)"""
        # Toto by v produkci používalo specialized IPFS search services
        search_results = []

        # Placeholder pro search implementaci
        # V praxi by se napojilo na služby jako:
        # - IPFS Search (ipfs-search.com)
        # - Awesome IPFS lists
        # - DHT exploration

        return search_results


class MatrixClient:
    """Klient pro Matrix protokol"""

    def __init__(self, homeserver_url: str, access_token: Optional[str] = None):
        self.homeserver_url = homeserver_url.rstrip('/')
        self.access_token = access_token
        self.headers = {
            "Content-Type": "application/json"
        }

        if access_token:
            self.headers["Authorization"] = f"Bearer {access_token}"

    async def login(self, username: str, password: str) -> str:
        """Přihlášení k Matrix serveru"""
        url = f"{self.homeserver_url}/_matrix/client/r0/login"

        login_data = {
            "type": "m.login.password",
            "user": username,
            "password": password
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=login_data, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    return self.access_token
                else:
                    raise Exception(f"Matrix login failed: {response.status}")

    async def get_joined_rooms(self) -> List[str]:
        """Seznam připojených rooms"""
        url = f"{self.homeserver_url}/_matrix/client/r0/joined_rooms"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("joined_rooms", [])
                else:
                    return []

    async def get_room_messages(
        self,
        room_id: str,
        limit: int = 100,
        from_token: Optional[str] = None
    ) -> List[MatrixEvent]:
        """Načtení zpráv z room"""
        url = f"{self.homeserver_url}/_matrix/client/r0/rooms/{room_id}/messages"

        params = {
            "dir": "b",  # Backwards
            "limit": limit
        }

        if from_token:
            params["from"] = from_token

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    events = []

                    for event in data.get("chunk", []):
                        if event.get("type") == "m.room.message":
                            events.append(MatrixEvent(
                                event_id=event.get("event_id"),
                                event_type=event.get("type"),
                                sender=event.get("sender"),
                                content=event.get("content", {}),
                                timestamp=event.get("origin_server_ts"),
                                room_id=room_id
                            ))

                    return events
                else:
                    return []

    async def search_public_rooms(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Hledání veřejných rooms"""
        url = f"{self.homeserver_url}/_matrix/client/r0/publicRooms"

        search_params = {
            "limit": 100
        }

        if query:
            search_params["filter"] = {
                "generic_search_term": query
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=search_params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("chunk", [])
                else:
                    return []

    async def get_room_state(self, room_id: str) -> Dict[str, Any]:
        """Získání stavu room"""
        url = f"{self.homeserver_url}/_matrix/client/r0/rooms/{room_id}/state"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}


class CustomProtocolHandler:
    """
    Unified handler pro nestandardní protokoly:
    - Gemini protocol (gemini://)
    - IPFS (ipfs://, ipns://)
    - Matrix protocol (matrix://)
    """

    def __init__(self,
                 ipfs_gateway: str = "https://ipfs.io",
                 matrix_homeserver: Optional[str] = None):

        self.gemini_client = GeminiClient()
        self.ipfs_client = IPFSClient(ipfs_gateway)
        self.matrix_client = MatrixClient(matrix_homeserver) if matrix_homeserver else None

        logger.info("CustomProtocolHandler inicializován")

    async def fetch(self, url: str) -> ProtocolResponse:
        """
        Univerzální fetch pro custom protokoly
        """
        import time
        start_time = time.time()

        parsed = urlparse(url)
        protocol = parsed.scheme.lower()

        try:
            if protocol == "gemini":
                result = await self._handle_gemini(url)
            elif protocol == "ipfs":
                result = await self._handle_ipfs(url)
            elif protocol == "ipns":
                result = await self._handle_ipns(url)
            elif protocol == "matrix":
                result = await self._handle_matrix(url)
            else:
                raise ValueError(f"Nepodporovaný protokol: {protocol}")

            result.response_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            return ProtocolResponse(
                protocol=protocol,
                url=url,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )

    async def _handle_gemini(self, url: str) -> ProtocolResponse:
        """Handle Gemini protokolu"""
        doc = await self.gemini_client.fetch(url)

        return ProtocolResponse(
            protocol="gemini",
            url=url,
            status_code=doc.metadata.get("status_code"),
            content=doc.content,
            metadata={
                "mime_type": doc.mime_type,
                "links_found": len(doc.links),
                "links": doc.links[:10]  # První 10 linků
            }
        )

    async def _handle_ipfs(self, url: str) -> ProtocolResponse:
        """Handle IPFS protokolu"""
        parsed = urlparse(url)
        ipfs_hash = parsed.netloc or parsed.path.lstrip('/')

        content = await self.ipfs_client.fetch_by_hash(ipfs_hash)

        # Pokus o dekódování textu
        try:
            text_content = content.content.decode('utf-8')
        except:
            text_content = f"Binary content ({len(content.content)} bytes)"

        return ProtocolResponse(
            protocol="ipfs",
            url=url,
            status_code=200,
            content=text_content,
            metadata={
                "hash": content.hash,
                "content_type": content.content_type,
                "size": content.size,
                "source": content.metadata.get("source")
            }
        )

    async def _handle_ipns(self, url: str) -> ProtocolResponse:
        """Handle IPNS protokolu"""
        parsed = urlparse(url)
        ipns_name = parsed.netloc or parsed.path.lstrip('/')

        # Resolve IPNS na IPFS hash
        ipfs_hash = await self.ipfs_client.resolve_ipns(ipns_name)

        # Fetch obsahu
        content = await self.ipfs_client.fetch_by_hash(ipfs_hash)

        try:
            text_content = content.content.decode('utf-8')
        except:
            text_content = f"Binary content ({len(content.content)} bytes)"

        return ProtocolResponse(
            protocol="ipns",
            url=url,
            status_code=200,
            content=text_content,
            metadata={
                "ipns_name": ipns_name,
                "resolved_hash": ipfs_hash,
                "content_type": content.content_type,
                "size": content.size
            }
        )

    async def _handle_matrix(self, url: str) -> ProtocolResponse:
        """Handle Matrix protokolu"""
        if not self.matrix_client:
            raise ValueError("Matrix client není nakonfigurován")

        # Matrix URL format: matrix:roomid/event_id nebo matrix:@user:server
        parsed = urlparse(url)
        path = parsed.netloc + parsed.path

        if path.startswith("!"):  # Room ID
            room_id = path.split('/')[0]
            messages = await self.matrix_client.get_room_messages(room_id, limit=50)

            content = json.dumps([
                {
                    "sender": msg.sender,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ], indent=2)

            return ProtocolResponse(
                protocol="matrix",
                url=url,
                status_code=200,
                content=content,
                metadata={
                    "room_id": room_id,
                    "messages_count": len(messages)
                }
            )
        else:
            raise ValueError("Nepodporovaný Matrix URL formát")

    async def discover_content(self, protocol: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover obsahu pro různé protokoly
        """
        discoveries = []

        if protocol == "gemini":
            # Gemini space exploration
            known_servers = [
                "gemini://gemini.circumlunar.space/",
                "gemini://gus.guru/",
                "gemini://warmedal.se/~bjorn/",
                "gemini://geminiprotocol.net/"
            ]

            for server in known_servers:
                try:
                    doc = await self.gemini_client.fetch(server)
                    discoveries.append({
                        "protocol": "gemini",
                        "url": server,
                        "title": doc.content.split('\n')[0] if doc.content else "",
                        "links_count": len(doc.links)
                    })
                except Exception as e:
                    logger.debug(f"Gemini discovery error for {server}: {e}")

        elif protocol == "ipfs":
            # IPFS content discovery
            if "hash" in kwargs:
                try:
                    directory = await self.ipfs_client.list_directory(kwargs["hash"])
                    discoveries.extend([
                        {
                            "protocol": "ipfs",
                            "name": item["name"],
                            "hash": item["hash"],
                            "size": item["size"],
                            "type": item["type"]
                        }
                        for item in directory
                    ])
                except Exception as e:
                    logger.debug(f"IPFS discovery error: {e}")

        elif protocol == "matrix" and self.matrix_client:
            # Matrix public rooms discovery
            try:
                rooms = await self.matrix_client.search_public_rooms(kwargs.get("query"))
                discoveries.extend([
                    {
                        "protocol": "matrix",
                        "room_id": room.get("room_id"),
                        "name": room.get("name"),
                        "topic": room.get("topic"),
                        "members": room.get("num_joined_members", 0)
                    }
                    for room in rooms
                ])
            except Exception as e:
                logger.debug(f"Matrix discovery error: {e}")

        return discoveries

    def get_supported_protocols(self) -> List[str]:
        """Seznam podporovaných protokolů"""
        protocols = ["gemini", "ipfs", "ipns"]

        if self.matrix_client:
            protocols.append("matrix")

        return protocols

    async def batch_fetch(self, urls: List[str]) -> List[ProtocolResponse]:
        """Batch načítání více URLs"""
        semaphore = asyncio.Semaphore(5)  # Max 5 současných requestů

        async def fetch_with_semaphore(url: str) -> ProtocolResponse:
            async with semaphore:
                return await self.fetch(url)

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for result in results:
            if isinstance(result, ProtocolResponse):
                valid_results.append(result)
            else:
                logger.error(f"Batch fetch error: {result}")

        return valid_results
