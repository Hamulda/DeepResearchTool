#!/usr/bin/env python3
"""
Konektory pro těžko dostupné zdroje - Common Crawl, Memento, Ahmia, Legal, OA věda
Implementuje specialized retrievers pro deep research

Author: Senior IT Specialist
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from urllib.parse import urljoin, urlparse
import gzip
import io

import structlog
from cdx_toolkit import CDXFetcher
import mementoweb

logger = structlog.get_logger(__name__)

@dataclass
class ArchivalDocument:
    """Dokument z archivních zdrojů"""
    id: str
    title: str
    content: str
    url: str
    source: str
    timestamp: str
    citation: str
    metadata: Dict[str, Any]
    confidence: float = 0.8

class CommonCrawlConnector:
    """Konektor pro Common Crawl CDX/WARC"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("connectors", {}).get("common_crawl", {})
        self.cdx_server = self.config.get("cdx_server", "https://index.commoncrawl.org")
        self.max_results = self.config.get("max_results", 1000)
        self.timeout = self.config.get("timeout", 30)

        self.logger = structlog.get_logger(__name__)

    async def ccdx_lookup(self,
                         url: str,
                         filters: Optional[Dict[str, Any]] = None) -> List[ArchivalDocument]:
        """Lookup historických verzí URL v Common Crawl"""

        self.logger.info("Common Crawl lookup", url=url)

        try:
            # Použití cdx-toolkit pro vyhledávání
            cdx = CDXFetcher(source=self.cdx_server)

            # Filtry pro vyhledávání
            query_filters = filters or {}
            from_date = query_filters.get("from", "2010")
            to_date = query_filters.get("to", "2024")

            results = []

            # CDX query
            async with aiohttp.ClientSession() as session:
                cdx_url = f"{self.cdx_server}/cdx"
                params = {
                    "url": url,
                    "output": "json",
                    "from": from_date,
                    "to": to_date,
                    "limit": self.max_results
                }

                async with session.get(cdx_url, params=params) as response:
                    if response.status == 200:
                        cdx_lines = await response.text()

                        for line in cdx_lines.strip().split('\n'):
                            if line and not line.startswith('urlkey'):
                                try:
                                    parts = line.split(' ')
                                    if len(parts) >= 7:
                                        timestamp = parts[1]
                                        original_url = parts[2]
                                        status = parts[4]
                                        warc_filename = parts[9] if len(parts) > 9 else ""
                                        warc_offset = parts[10] if len(parts) > 10 else ""

                                        if status == "200":  # Pouze úspěšné captures
                                            # Fetch content z WARC
                                            content = await self.fetch_warc_segment(
                                                warc_filename, warc_offset
                                            )

                                            doc = ArchivalDocument(
                                                id=f"cc_{timestamp}_{hash(original_url)}",
                                                title=f"Common Crawl capture from {timestamp}",
                                                content=content[:5000],  # Truncate
                                                url=original_url,
                                                source="common_crawl",
                                                timestamp=self._parse_cc_timestamp(timestamp),
                                                citation=f"WARC:{warc_filename}:{warc_offset}",
                                                metadata={
                                                    "warc": {
                                                        "filename": warc_filename,
                                                        "offset": warc_offset
                                                    },
                                                    "status_code": status,
                                                    "capture_timestamp": timestamp
                                                }
                                            )
                                            results.append(doc)

                                except Exception as e:
                                    self.logger.warning("Chyba při parsování CDX záznamu", error=str(e))
                                    continue

            self.logger.info("Common Crawl lookup dokončen", results=len(results))
            return results[:self.max_results]

        except Exception as e:
            self.logger.error("Chyba při Common Crawl lookup", error=str(e))
            return []

    async def fetch_warc_segment(self, filename: str, offset: str) -> str:
        """Fetch konkrétního WARC segmentu"""

        if not filename or not offset:
            return ""

        try:
            # WARC URL konstrukce
            warc_base = "https://commoncrawl.s3.amazonaws.com/"
            warc_url = urljoin(warc_base, filename)

            # Range request pro specifický offset
            range_start = int(offset)
            range_end = range_start + 50000  # Max 50KB na segment

            async with aiohttp.ClientSession() as session:
                headers = {"Range": f"bytes={range_start}-{range_end}"}

                async with session.get(warc_url, headers=headers) as response:
                    if response.status in [200, 206]:
                        content = await response.read()

                        # Dekomprese pokud je gzipped
                        if filename.endswith('.gz'):
                            content = gzip.decompress(content)

                        # Extrakce HTML obsahu z WARC záznamu
                        text_content = self._extract_html_from_warc(content.decode('utf-8', errors='ignore'))
                        return text_content

        except Exception as e:
            self.logger.warning("Chyba při fetch WARC segmentu", filename=filename, error=str(e))

        return ""

    def _extract_html_from_warc(self, warc_content: str) -> str:
        """Extrakce HTML obsahu z WARC záznamu"""

        # Hledání HTML části v WARC záznamu
        html_start = warc_content.find('<html')
        if html_start == -1:
            html_start = warc_content.find('<!DOCTYPE')

        if html_start != -1:
            html_content = warc_content[html_start:]

            # Základní HTML cleaning
            from bs4 import BeautifulSoup
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                # Odstranění skriptů a stylů
                for script in soup(["script", "style"]):
                    script.decompose()

                return soup.get_text()[:3000]  # Max 3KB textu
            except:
                return html_content[:3000]

        return warc_content[:1000]

    def _parse_cc_timestamp(self, timestamp: str) -> str:
        """Parsování Common Crawl timestamp"""

        try:
            # CC timestamp formát: YYYYMMDDHHMMSS
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            return dt.isoformat()
        except:
            return timestamp

class MementoConnector:
    """Konektor pro Memento web archiving"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("connectors", {}).get("memento", {})
        self.timegate_url = self.config.get("timegate_url", "http://timetravel.mementoweb.org/timegate/")
        self.max_mementos = self.config.get("max_mementos", 10)

        self.logger = structlog.get_logger(__name__)

    async def time_travel(self, url: str, target_datetime: str) -> Optional[ArchivalDocument]:
        """Time travel k specifickému datu"""

        self.logger.info("Memento time travel", url=url, datetime=target_datetime)

        try:
            # Memento TimeGate request
            timegate_request = f"{self.timegate_url}{url}"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Accept-Datetime": target_datetime,
                    "User-Agent": "DeepResearchTool/1.0"
                }

                async with session.get(timegate_request, headers=headers) as response:
                    if response.status == 302:  # Redirect k memento
                        memento_url = response.headers.get('Location')
                        memento_datetime = response.headers.get('Memento-Datetime')

                        if memento_url:
                            # Fetch memento obsahu
                            content = await self._fetch_memento_content(memento_url)

                            return ArchivalDocument(
                                id=f"memento_{hash(url)}_{hash(target_datetime)}",
                                title=f"Memento capture from {memento_datetime}",
                                content=content,
                                url=url,
                                source="memento",
                                timestamp=memento_datetime or target_datetime,
                                citation=f"Memento:{memento_datetime}:{url}",
                                metadata={
                                    "memento": {
                                        "url": memento_url,
                                        "datetime": memento_datetime,
                                        "original_url": url
                                    }
                                }
                            )

        except Exception as e:
            self.logger.error("Chyba při Memento time travel", error=str(e))

        return None

    async def time_travel_diff(self,
                             url: str,
                             datetime1: str,
                             datetime2: str) -> Dict[str, Any]:
        """Porovnání změn mezi dvěma mementy"""

        self.logger.info("Memento diff", url=url, dt1=datetime1, dt2=datetime2)

        # Fetch obou verzí
        memento1 = await self.time_travel(url, datetime1)
        memento2 = await self.time_travel(url, datetime2)

        if not memento1 or not memento2:
            return {"error": "Nelze načíst oba mementa"}

        # Základní diff analýza
        diff_analysis = self._analyze_content_diff(
            memento1.content,
            memento2.content
        )

        return {
            "url": url,
            "memento1": {
                "datetime": memento1.timestamp,
                "content_length": len(memento1.content)
            },
            "memento2": {
                "datetime": memento2.timestamp,
                "content_length": len(memento2.content)
            },
            "diff_analysis": diff_analysis
        }

    async def _fetch_memento_content(self, memento_url: str) -> str:
        """Fetch obsahu memento"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(memento_url) as response:
                    if response.status == 200:
                        html_content = await response.text()

                        # Extrakce textu z HTML
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Odstranění navigation a archive bannerů
                        for elem in soup.find_all(['script', 'style', 'nav']):
                            elem.decompose()

                        return soup.get_text()[:5000]  # Max 5KB

        except Exception as e:
            self.logger.warning("Chyba při fetch memento", url=memento_url, error=str(e))

        return ""

    def _analyze_content_diff(self, content1: str, content2: str) -> Dict[str, Any]:
        """Analýza rozdílů mezi obsahy"""

        # Základní statistiky
        len_diff = len(content2) - len(content1)
        len_diff_pct = (len_diff / len(content1)) * 100 if len(content1) > 0 else 0

        # Word-level diff
        words1 = set(content1.split())
        words2 = set(content2.split())

        added_words = words2 - words1
        removed_words = words1 - words2
        common_words = words1 & words2

        return {
            "length_change": len_diff,
            "length_change_percent": len_diff_pct,
            "words_added": len(added_words),
            "words_removed": len(removed_words),
            "words_common": len(common_words),
            "similarity_score": len(common_words) / len(words1 | words2) if words1 | words2 else 0,
            "sample_added_words": list(added_words)[:10],
            "sample_removed_words": list(removed_words)[:10]
        }

class AhmiaConnector:
    """Konektor pro Ahmia (Tor OSINT)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("connectors", {}).get("ahmia", {})
        self.base_url = self.config.get("base_url", "https://ahmia.fi/search/")
        self.safety_checks = self.config.get("safety_checks", True)
        self.legal_filter = self.config.get("legal_filter", True)

        self.logger = structlog.get_logger(__name__)

    async def onion_discovery(self, query: str) -> List[ArchivalDocument]:
        """Discovery onion služeb přes Ahmia"""

        if not self.legal_filter:
            self.logger.warning("Legal filter vypnut - používej opatrně")

        self.logger.info("Ahmia onion discovery", query=query)

        try:
            # Ahmia search API
            search_url = f"{self.base_url}?q={query}"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "DeepResearchTool/1.0 (Research purposes)"
                }

                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()

                        # Parsování výsledků
                        results = self._parse_ahmia_results(html_content)

                        # Safety filtering
                        if self.safety_checks:
                            results = self._apply_safety_filters(results)

                        return results

        except Exception as e:
            self.logger.error("Chyba při Ahmia discovery", error=str(e))

        return []

    def _parse_ahmia_results(self, html_content: str) -> List[ArchivalDocument]:
        """Parsování Ahmia search výsledků"""

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        results = []

        # Hledání result divů (specifické pro Ahmia)
        for result_div in soup.find_all('div', class_='result'):
            try:
                title_elem = result_div.find('h4')
                title = title_elem.get_text() if title_elem else "Unknown"

                link_elem = result_div.find('a')
                onion_url = link_elem.get('href') if link_elem else ""

                desc_elem = result_div.find('p')
                description = desc_elem.get_text() if desc_elem else ""

                if onion_url and onion_url.endswith('.onion'):
                    doc = ArchivalDocument(
                        id=f"ahmia_{hash(onion_url)}",
                        title=title,
                        content=description,
                        url=onion_url,
                        source="ahmia_onion",
                        timestamp=datetime.now().isoformat(),
                        citation=f"Ahmia:{onion_url}",
                        metadata={
                            "onion_service": True,
                            "discovered_via": "ahmia",
                            "safety_checked": self.safety_checks
                        },
                        confidence=0.6  # Nižší confidence pro onion služby
                    )
                    results.append(doc)

            except Exception as e:
                self.logger.warning("Chyba při parsování Ahmia výsledku", error=str(e))
                continue

        return results

    def _apply_safety_filters(self, results: List[ArchivalDocument]) -> List[ArchivalDocument]:
        """Aplikace safety filtrů"""

        # Blacklist nebezpečných klíčových slov
        dangerous_keywords = [
            'weapon', 'drug', 'illegal', 'hack', 'crack', 'stolen',
            'child', 'abuse', 'violence', 'terrorist', 'bomb'
        ]

        filtered_results = []

        for result in results:
            content_lower = (result.title + " " + result.content).lower()

            # Kontrola nebezpečných klíčových slov
            if not any(keyword in content_lower for keyword in dangerous_keywords):
                filtered_results.append(result)
            else:
                self.logger.warning("Filtrován potenciálně nebezpečný obsah",
                                  url=result.url)

        return filtered_results

class LegalConnector:
    """Konektor pro právní zdroje - CourtListener/RECAP, SEC EDGAR"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("connectors", {}).get("legal", {})
        self.courtlistener_config = self.config.get("courtlistener", {})
        self.sec_config = self.config.get("sec_edgar", {})

        self.logger = structlog.get_logger(__name__)

    async def legal_monitor(self, queries: List[str]) -> List[ArchivalDocument]:
        """Monitoring právních změn"""

        self.logger.info("Legal monitoring", queries=queries)

        all_results = []

        # CourtListener monitoring
        if self.courtlistener_config.get("enabled", False):
            for query in queries:
                cl_results = await self._courtlistener_search(query)
                all_results.extend(cl_results)

        # SEC EDGAR monitoring
        if self.sec_config.get("enabled", False):
            for query in queries:
                sec_results = await self._sec_edgar_search(query)
                all_results.extend(sec_results)

        return all_results

    async def _courtlistener_search(self, query: str) -> List[ArchivalDocument]:
        """CourtListener API search"""

        api_key = self.courtlistener_config.get("api_key")
        if not api_key:
            self.logger.warning("CourtListener API key chybí")
            return []

        try:
            base_url = "https://www.courtlistener.com/api/rest/v3/search/"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token {api_key}",
                    "User-Agent": "DeepResearchTool/1.0"
                }

                params = {
                    "q": query,
                    "type": "o",  # Opinions
                    "format": "json",
                    "order_by": "-dateFiled"
                }

                async with session.get(base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for opinion in data.get("results", []):
                            doc = ArchivalDocument(
                                id=f"cl_{opinion.get('id')}",
                                title=opinion.get("caseName", ""),
                                content=opinion.get("text", "")[:3000],
                                url=opinion.get("absolute_url", ""),
                                source="courtlistener",
                                timestamp=opinion.get("dateFiled", ""),
                                citation=f"Court:{opinion.get('docket')}:{opinion.get('id')}",
                                metadata={
                                    "court": opinion.get("court"),
                                    "docket_id": opinion.get("docket"),
                                    "document_id": opinion.get("id"),
                                    "judges": opinion.get("judges", [])
                                }
                            )
                            results.append(doc)

                        return results

        except Exception as e:
            self.logger.error("Chyba při CourtListener search", error=str(e))

        return []

    async def _sec_edgar_search(self, query: str) -> List[ArchivalDocument]:
        """SEC EDGAR API search"""

        try:
            user_agent = self.sec_config.get("user_agent", "DeepResearchTool/1.0")

            # SEC Company search
            search_url = "https://www.sec.gov/cgi-bin/browse-edgar"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": user_agent,
                    "Accept": "application/json"
                }

                params = {
                    "action": "getcompany",
                    "CIK": query,
                    "output": "atom"
                }

                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        # SEC EDGAR parsing je složitější - zde zjednodušená implementace
                        content = await response.text()

                        # Parsování ATOM feedu
                        import xml.etree.ElementTree as ET
                        try:
                            root = ET.fromstring(content)

                            results = []
                            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                                link_elem = entry.find('.//{http://www.w3.org/2005/Atom}link')

                                if title_elem is not None and link_elem is not None:
                                    doc = ArchivalDocument(
                                        id=f"sec_{hash(link_elem.get('href', ''))}",
                                        title=title_elem.text or "",
                                        content="SEC filing - full content requires additional fetch",
                                        url=link_elem.get('href', ''),
                                        source="sec_edgar",
                                        timestamp=datetime.now().isoformat(),
                                        citation=f"SEC:{query}:unknown:unknown",
                                        metadata={
                                            "cik": query,
                                            "form_type": "unknown"
                                        }
                                    )
                                    results.append(doc)

                            return results

                        except ET.ParseError as e:
                            self.logger.warning("Chyba při parsování SEC XML", error=str(e))

        except Exception as e:
            self.logger.error("Chyba při SEC EDGAR search", error=str(e))

        return []

class SourceConnectorOrchestrator:
    """Orchestrátor pro všechny source konektory"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Inicializace konektorů
        self.common_crawl = CommonCrawlConnector(config)
        self.memento = MementoConnector(config)
        self.ahmia = AhmiaConnector(config)
        self.legal = LegalConnector(config)

        self.logger = structlog.get_logger(__name__)

    async def search_all_sources(self,
                                query: str,
                                url_hints: Optional[List[str]] = None) -> List[ArchivalDocument]:
        """Vyhledávání napříč všemi zdroji"""

        self.logger.info("Searching across all specialized sources", query=query)

        all_results = []

        # Common Crawl search (pokud máme URL hints)
        if url_hints:
            for url in url_hints[:3]:  # Limit na 3 URLs
                cc_results = await self.common_crawl.ccdx_lookup(url)
                all_results.extend(cc_results)

        # Ahmia onion discovery
        ahmia_results = await self.ahmia.onion_discovery(query)
        all_results.extend(ahmia_results)

        # Legal monitoring
        legal_results = await self.legal.legal_monitor([query])
        all_results.extend(legal_results)

        # Deduplikace a řazení
        unique_results = self._deduplicate_results(all_results)

        return unique_results

    def _deduplicate_results(self, results: List[ArchivalDocument]) -> List[ArchivalDocument]:
        """Deduplikace výsledků"""

        seen_urls = set()
        unique_results = []

        for result in results:
            url_key = result.url.lower().strip()
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(result)

        # Řazení podle confidence
        unique_results.sort(key=lambda x: x.confidence, reverse=True)

        return unique_results
