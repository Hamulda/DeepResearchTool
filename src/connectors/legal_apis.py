#!/usr/bin/env python3
"""Legal APIs konektor
CourtListener/RECAP a SEC EDGAR s přesnými identifikátory

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import aiofiles
import aiohttp


@dataclass
class LegalDocument:
    """Právní dokument s metadaty"""

    document_id: str
    document_type: str  # "court_opinion", "docket", "sec_filing"
    title: str
    court_name: str | None
    case_number: str | None
    filing_date: date
    parties: list[str]
    judges: list[str]
    content: str
    citations: list[str]
    docket_id: str | None
    filing_id: str | None
    char_offsets: dict[str, tuple[int, int]]  # key evidence -> (start, end)
    source_api: str
    confidence_score: float


class LegalAPIsConnector:
    """Legal APIs orchestrátor"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "research_cache/legal"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API credentials a endpoints
        self.courtlistener_token = config.get("courtlistener_token")
        self.courtlistener_base = "https://www.courtlistener.com/api/rest/v3"
        self.sec_edgar_base = "https://www.sec.gov/Archives/edgar/data"

        # Rate limits
        self.rate_limits = {"courtlistener": 1.0, "sec_edgar": 0.1}  # 1 req/s  # 10 req/s

        self.max_retries = config.get("max_retries", 3)
        self.user_agent = "DeepResearchTool/1.0 (+research@example.com)"

    async def search_legal_documents(
        self, query: str, document_types: list[str] = None, max_results: int = 50
    ) -> list[LegalDocument]:
        """Orchestrované vyhledávání právních dokumentů"""
        print(f"⚖️  Searching legal documents for: {query}")

        if document_types is None:
            document_types = ["court_opinion", "sec_filing"]

        all_documents = []

        # 1. CourtListener search
        if "court_opinion" in document_types or "docket" in document_types:
            court_docs = await self._search_courtlistener(query, max_results // 2)
            all_documents.extend(court_docs)

        # 2. SEC EDGAR search
        if "sec_filing" in document_types:
            sec_docs = await self._search_sec_edgar(query, max_results // 2)
            all_documents.extend(sec_docs)

        # 3. Deduplikace a seřazení
        unique_documents = self._deduplicate_documents(all_documents)
        unique_documents.sort(key=lambda x: x.confidence_score, reverse=True)

        print(f"✅ Found {len(unique_documents)} unique legal documents")
        return unique_documents[:max_results]

    async def _search_courtlistener(self, query: str, max_results: int) -> list[LegalDocument]:
        """Vyhledá v CourtListener"""
        if not self.courtlistener_token:
            print("⚠️  CourtListener token not configured")
            return []

        cache_key = hashlib.md5(f"courtlistener_{query}_{max_results}".encode()).hexdigest()
        cache_file = self.cache_dir / f"courtlistener_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file) as f:
                cached_data = json.loads(await f.read())
                documents = []
                for item in cached_data:
                    item["filing_date"] = date.fromisoformat(item["filing_date"])
                    documents.append(LegalDocument(**item))
                return documents

        documents = []

        # Search opinions
        opinions = await self._search_courtlistener_opinions(query, max_results // 2)
        documents.extend(opinions)

        # Search dockets
        dockets = await self._search_courtlistener_dockets(query, max_results // 2)
        documents.extend(dockets)

        # Cache výsledky
        cache_data = [self._legal_document_to_dict(doc) for doc in documents]
        async with aiofiles.open(cache_file, "w") as f:
            await f.write(json.dumps(cache_data, indent=2))

        return documents

    async def _search_courtlistener_opinions(
        self, query: str, max_results: int
    ) -> list[LegalDocument]:
        """Vyhledá court opinions"""
        await asyncio.sleep(self.rate_limits["courtlistener"])

        url = f"{self.courtlistener_base}/search/"
        params = {"q": query, "type": "o", "order_by": "score desc", "format": "json"}  # opinions

        headers = {
            "Authorization": f"Token {self.courtlistener_token}",
            "User-Agent": self.user_agent,
        }

        documents = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        for result in data.get("results", [])[:max_results]:
                            doc = await self._parse_courtlistener_opinion(result)
                            if doc:
                                documents.append(doc)
                    else:
                        print(f"❌ CourtListener opinions search failed: HTTP {response.status}")

        except Exception as e:
            print(f"❌ CourtListener opinions error: {e}")

        return documents

    async def _parse_courtlistener_opinion(self, result: dict[str, Any]) -> LegalDocument | None:
        """Parsuje CourtListener opinion"""
        try:
            # Fetch full opinion content
            opinion_url = result.get("resource_uri", "")
            if not opinion_url:
                return None

            full_url = f"{self.courtlistener_base.rstrip('/')}{opinion_url}"

            await asyncio.sleep(self.rate_limits["courtlistener"])

            headers = {
                "Authorization": f"Token {self.courtlistener_token}",
                "User-Agent": self.user_agent,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, headers=headers) as response:
                    if response.status != 200:
                        return None

                    opinion_data = await response.json()

            # Extract fields
            document_id = str(opinion_data.get("id", ""))
            title = opinion_data.get("case_name", "")
            court_name = opinion_data.get("court", "")

            # Date parsing
            date_filed = opinion_data.get("date_filed")
            if date_filed:
                filing_date = date.fromisoformat(date_filed)
            else:
                filing_date = date.today()

            # Extract parties and judges
            parties = []
            if title:
                # Simple extraction z case name
                vs_match = re.search(r"(.+?)\s+v\.?\s+(.+)", title)
                if vs_match:
                    parties = [vs_match.group(1).strip(), vs_match.group(2).strip()]

            judges = []
            author = opinion_data.get("author")
            if author:
                judges.append(author)

            # Content
            content = opinion_data.get("plain_text", "")
            if not content:
                content = opinion_data.get("html_with_citations", "")

            # Citations extraction
            citations = self._extract_citations_from_content(content)

            return LegalDocument(
                document_id=document_id,
                document_type="court_opinion",
                title=title,
                court_name=court_name,
                case_number=None,  # Would need additional parsing
                filing_date=filing_date,
                parties=parties,
                judges=judges,
                content=content,
                citations=citations,
                docket_id=str(opinion_data.get("docket", "")),
                filing_id=document_id,
                char_offsets={},  # Will be filled during evidence extraction
                source_api="courtlistener",
                confidence_score=0.9,
            )

        except Exception as e:
            print(f"❌ Failed to parse CourtListener opinion: {e}")
            return None

    async def _search_courtlistener_dockets(
        self, query: str, max_results: int
    ) -> list[LegalDocument]:
        """Vyhledá dockets"""
        await asyncio.sleep(self.rate_limits["courtlistener"])

        url = f"{self.courtlistener_base}/search/"
        params = {
            "q": query,
            "type": "r",  # RECAP/dockets
            "order_by": "score desc",
            "format": "json",
        }

        headers = {
            "Authorization": f"Token {self.courtlistener_token}",
            "User-Agent": self.user_agent,
        }

        documents = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        for result in data.get("results", [])[:max_results]:
                            doc = self._parse_courtlistener_docket(result)
                            if doc:
                                documents.append(doc)
                    else:
                        print(f"❌ CourtListener dockets search failed: HTTP {response.status}")

        except Exception as e:
            print(f"❌ CourtListener dockets error: {e}")

        return documents

    def _parse_courtlistener_docket(self, result: dict[str, Any]) -> LegalDocument | None:
        """Parsuje CourtListener docket"""
        try:
            document_id = str(result.get("id", ""))
            title = result.get("caseName", "")
            court_name = result.get("court", "")
            case_number = result.get("docketNumber", "")

            # Date
            date_filed = result.get("dateFiled")
            if date_filed:
                filing_date = date.fromisoformat(date_filed)
            else:
                filing_date = date.today()

            # Content z snippet
            content = result.get("snippet", "")

            return LegalDocument(
                document_id=document_id,
                document_type="docket",
                title=title,
                court_name=court_name,
                case_number=case_number,
                filing_date=filing_date,
                parties=[],  # Would need full docket fetch
                judges=[],
                content=content,
                citations=[],
                docket_id=document_id,
                filing_id=None,
                char_offsets={},
                source_api="courtlistener",
                confidence_score=0.7,
            )

        except Exception as e:
            print(f"❌ Failed to parse CourtListener docket: {e}")
            return None

    async def _search_sec_edgar(self, query: str, max_results: int) -> list[LegalDocument]:
        """Vyhledá v SEC EDGAR"""
        cache_key = hashlib.md5(f"sec_edgar_{query}_{max_results}".encode()).hexdigest()
        cache_file = self.cache_dir / f"sec_edgar_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file) as f:
                cached_data = json.loads(await f.read())
                documents = []
                for item in cached_data:
                    item["filing_date"] = date.fromisoformat(item["filing_date"])
                    documents.append(LegalDocument(**item))
                return documents

        documents = []

        # SEC EDGAR full-text search API
        await asyncio.sleep(self.rate_limits["sec_edgar"])

        search_url = "https://www.sec.gov/cgi-bin/srch-edgar"
        params = {
            "text": query,
            "first": "1990",
            "last": str(date.today().year),
            "count": str(min(max_results, 100)),
        }

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        documents = self._parse_sec_edgar_results(html_content)
                    else:
                        print(f"❌ SEC EDGAR search failed: HTTP {response.status}")

        except Exception as e:
            print(f"❌ SEC EDGAR search error: {e}")

        # Cache výsledky
        cache_data = [self._legal_document_to_dict(doc) for doc in documents]
        async with aiofiles.open(cache_file, "w") as f:
            await f.write(json.dumps(cache_data, indent=2))

        return documents

    def _parse_sec_edgar_results(self, html_content: str) -> list[LegalDocument]:
        """Parsuje SEC EDGAR výsledky"""
        documents = []

        # Simple HTML parsing pro SEC výsledky
        # V produkci by se použil BeautifulSoup

        # Extract filing links
        filing_pattern = r'<a href="([^"]*edgar/data[^"]*)"[^>]*>([^<]+)</a>'
        filings = re.findall(filing_pattern, html_content)

        for i, (filing_url, filing_title) in enumerate(filings[:20]):  # Limit
            try:
                # Extract filing info z URL a title
                filing_id = self._extract_filing_id_from_url(filing_url)
                company_name = self._extract_company_from_title(filing_title)

                document = LegalDocument(
                    document_id=filing_id,
                    document_type="sec_filing",
                    title=filing_title,
                    court_name=None,
                    case_number=None,
                    filing_date=date.today(),  # Would need additional parsing
                    parties=[company_name] if company_name else [],
                    judges=[],
                    content="",  # Would need full filing fetch
                    citations=[],
                    docket_id=None,
                    filing_id=filing_id,
                    char_offsets={},
                    source_api="sec_edgar",
                    confidence_score=0.8,
                )

                documents.append(document)

            except Exception as e:
                print(f"❌ Failed to parse SEC filing: {e}")
                continue

        return documents

    def _extract_filing_id_from_url(self, url: str) -> str:
        """Extrahuje filing ID z URL"""
        # Extract z cesty jako edgar/data/CIK/accession-number/file
        match = re.search(r"edgar/data/(\d+)/([^/]+)", url)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def _extract_company_from_title(self, title: str) -> str:
        """Extrahuje název společnosti z title"""
        # Simple extraction - první část před form type
        parts = title.split()
        if len(parts) > 0:
            return parts[0]
        return ""

    def _extract_citations_from_content(self, content: str) -> list[str]:
        """Extrahuje citace z obsahu dokumentu"""
        citations = []

        # Legal citation patterns
        citation_patterns = [
            r"\d+\s+U\.S\.?\s+\d+",  # Supreme Court
            r"\d+\s+F\.?\d*d?\s+\d+",  # Federal courts
            r"\d+\s+S\.Ct\.?\s+\d+",  # Supreme Court Reporter
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)

        return list(set(citations))  # Deduplicate

    def _deduplicate_documents(self, documents: list[LegalDocument]) -> list[LegalDocument]:
        """Deduplikuje documents podle ID"""
        seen_ids = set()
        unique_documents = []

        for doc in documents:
            if doc.document_id not in seen_ids:
                seen_ids.add(doc.document_id)
                unique_documents.append(doc)

        return unique_documents

    def _legal_document_to_dict(self, document: LegalDocument) -> dict[str, Any]:
        """Konvertuje LegalDocument na dict pro cache"""
        return {
            "document_id": document.document_id,
            "document_type": document.document_type,
            "title": document.title,
            "court_name": document.court_name,
            "case_number": document.case_number,
            "filing_date": document.filing_date.isoformat(),
            "parties": document.parties,
            "judges": document.judges,
            "content": document.content,
            "citations": document.citations,
            "docket_id": document.docket_id,
            "filing_id": document.filing_id,
            "char_offsets": document.char_offsets,
            "source_api": document.source_api,
            "confidence_score": document.confidence_score,
        }

    async def fetch_full_document_content(self, document: LegalDocument) -> LegalDocument:
        """Stáhne plný obsah dokumentu"""
        if document.source_api == "courtlistener" and document.content:
            return document  # Already have content

        if document.source_api == "sec_edgar":
            # Construct EDGAR document URL
            if document.filing_id:
                # Simplified - v produkci by byla komplexnější logika
                placeholder_content = f"SEC Filing: {document.title}\nFiling ID: {document.filing_id}\n[Full content would be fetched from EDGAR]"
                document.content = placeholder_content

        return document

    def extract_evidence_char_offsets(
        self, document: LegalDocument, evidence_phrases: list[str]
    ) -> LegalDocument:
        """Extrahuje char offsety pro evidence phrases"""
        char_offsets = {}
        content = document.content.lower()

        for phrase in evidence_phrases:
            phrase_lower = phrase.lower()
            start_pos = content.find(phrase_lower)

            if start_pos >= 0:
                end_pos = start_pos + len(phrase_lower)
                char_offsets[phrase] = (start_pos, end_pos)

        document.char_offsets = char_offsets
        return document
