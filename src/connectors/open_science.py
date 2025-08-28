#!/usr/bin/env python3
"""
Open Science konektor
OpenAlex→Crossref→Unpaywall→Europe PMC orchestrace s fallbacky

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import aiohttp
import aiofiles
from pathlib import Path
import re
from urllib.parse import quote


@dataclass
class ScientificPaper:
    """Vědecký článek s metadaty"""
    doi: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    journal: str
    open_access_status: str
    pdf_url: Optional[str]
    citations_count: int
    references: List[str]
    funding_info: List[str]
    source_api: str
    confidence_score: float


class OpenScienceConnector:
    """Open science API orchestrátor"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "research_cache/open_science"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.openalex_base = "https://api.openalex.org"
        self.crossref_base = "https://api.crossref.org"
        self.unpaywall_base = "https://api.unpaywall.org"
        self.europepmc_base = "https://www.ebi.ac.uk/europepmc/webservices/rest"

        # Rate limits (requests per second)
        self.rate_limits = {
            "openalex": 0.1,    # 10 req/s
            "crossref": 0.05,   # 20 req/s
            "unpaywall": 0.1,   # 10 req/s
            "europepmc": 0.1    # 10 req/s
        }

        self.email = config.get("email", "research@example.com")
        self.max_retries = config.get("max_retries", 3)

    async def search_scientific_papers(self,
                                     query: str,
                                     max_results: int = 50) -> List[ScientificPaper]:
        """Orchestrovaná vyhledávání napříč API"""
        print(f"🔬 Searching scientific papers for: {query}")

        all_papers = []

        # 1. OpenAlex search (primary)
        openalex_papers = await self._search_openalex(query, max_results // 2)
        all_papers.extend(openalex_papers)

        # 2. Europe PMC search (backup + complementary)
        europepmc_papers = await self._search_europepmc(query, max_results // 2)
        all_papers.extend(europepmc_papers)

        # 3. Deduplikace podle DOI
        unique_papers = self._deduplicate_papers(all_papers)

        # 4. Enhance s Crossref a Unpaywall
        enhanced_papers = await self._enhance_papers_metadata(unique_papers)

        # 5. Seřaď podle relevance a confidence
        enhanced_papers.sort(key=lambda x: x.confidence_score, reverse=True)

        print(f"✅ Found {len(enhanced_papers)} unique scientific papers")
        return enhanced_papers[:max_results]

    async def _search_openalex(self, query: str, max_results: int) -> List[ScientificPaper]:
        """Vyhledá v OpenAlex"""
        cache_key = hashlib.md5(f"openalex_{query}_{max_results}".encode()).hexdigest()
        cache_file = self.cache_dir / f"openalex_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                cached_data = json.loads(await f.read())
                return [ScientificPaper(**item) for item in cached_data]

        papers = []

        # Rate limiting
        await asyncio.sleep(self.rate_limits["openalex"])

        search_url = f"{self.openalex_base}/works"
        params = {
            "search": query,
            "per-page": min(max_results, 200),
            "mailto": self.email,
            "sort": "relevance_score:desc"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for work in data.get("results", []):
                            paper = self._parse_openalex_work(work)
                            if paper:
                                papers.append(paper)
                    else:
                        print(f"❌ OpenAlex search failed: HTTP {response.status}")

        except Exception as e:
            print(f"❌ OpenAlex search error: {e}")

        # Cache výsledky
        cache_data = [self._paper_to_dict(p) for p in papers]
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(cache_data, indent=2))

        return papers

    def _parse_openalex_work(self, work: Dict[str, Any]) -> Optional[ScientificPaper]:
        """Parsuje OpenAlex work do ScientificPaper"""
        try:
            # Extract basic info
            doi = work.get("doi", "").replace("https://doi.org/", "")
            if not doi:
                return None

            title = work.get("title", "")
            if not title:
                return None

            # Authors
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            # Abstract
            abstract = ""
            if work.get("abstract_inverted_index"):
                abstract = self._reconstruct_abstract(work["abstract_inverted_index"])

            # Publication info
            pub_date = work.get("publication_date", "")
            journal = ""
            if work.get("primary_location"):
                source = work["primary_location"].get("source", {})
                journal = source.get("display_name", "")

            # Open access
            oa_info = work.get("open_access", {})
            oa_status = oa_info.get("oa_type", "closed")

            # PDF URL
            pdf_url = None
            if work.get("best_oa_location"):
                pdf_url = work["best_oa_location"].get("pdf_url")

            return ScientificPaper(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                journal=journal,
                open_access_status=oa_status,
                pdf_url=pdf_url,
                citations_count=work.get("cited_by_count", 0),
                references=[],  # Will be filled by enhancement
                funding_info=[],
                source_api="openalex",
                confidence_score=0.9  # High confidence for OpenAlex
            )

        except Exception as e:
            print(f"❌ Failed to parse OpenAlex work: {e}")
            return None

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """Rekonstruuje abstract z inverted indexu"""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        return " ".join([word for _, word in word_positions])

    async def _search_europepmc(self, query: str, max_results: int) -> List[ScientificPaper]:
        """Vyhledá v Europe PMC"""
        cache_key = hashlib.md5(f"europepmc_{query}_{max_results}".encode()).hexdigest()
        cache_file = self.cache_dir / f"europepmc_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                cached_data = json.loads(await f.read())
                return [ScientificPaper(**item) for item in cached_data]

        papers = []

        # Rate limiting
        await asyncio.sleep(self.rate_limits["europepmc"])

        search_url = f"{self.europepmc_base}/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(max_results, 1000),
            "sort": "relevance"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for result in data.get("resultList", {}).get("result", []):
                            paper = self._parse_europepmc_result(result)
                            if paper:
                                papers.append(paper)
                    else:
                        print(f"❌ Europe PMC search failed: HTTP {response.status}")

        except Exception as e:
            print(f"❌ Europe PMC search error: {e}")

        # Cache výsledky
        cache_data = [self._paper_to_dict(p) for p in papers]
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(cache_data, indent=2))

        return papers

    def _parse_europepmc_result(self, result: Dict[str, Any]) -> Optional[ScientificPaper]:
        """Parsuje Europe PMC result do ScientificPaper"""
        try:
            # DOI
            doi = result.get("doi", "")
            if not doi:
                return None

            title = result.get("title", "")
            if not title:
                return None

            # Authors
            authors = []
            author_string = result.get("authorString", "")
            if author_string:
                authors = [a.strip() for a in author_string.split(",")]

            # Abstract
            abstract = result.get("abstractText", "")

            # Publication info
            pub_date = result.get("firstPublicationDate", "")
            journal = result.get("journalTitle", "")

            # Open access
            oa_status = "closed"
            if result.get("isOpenAccess") == "Y":
                oa_status = "gold"

            # PDF URL
            pdf_url = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
            pdf_url = pdf_url[0].get("url") if pdf_url else None

            return ScientificPaper(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                journal=journal,
                open_access_status=oa_status,
                pdf_url=pdf_url,
                citations_count=int(result.get("citedByCount", 0)),
                references=[],
                funding_info=[],
                source_api="europepmc",
                confidence_score=0.8  # Good confidence for Europe PMC
            )

        except Exception as e:
            print(f"❌ Failed to parse Europe PMC result: {e}")
            return None

    def _deduplicate_papers(self, papers: List[ScientificPaper]) -> List[ScientificPaper]:
        """Deduplikuje papers podle DOI"""
        seen_dois = set()
        unique_papers = []

        for paper in papers:
            if paper.doi and paper.doi not in seen_dois:
                seen_dois.add(paper.doi)
                unique_papers.append(paper)

        return unique_papers

    async def _enhance_papers_metadata(self, papers: List[ScientificPaper]) -> List[ScientificPaper]:
        """Enhance papers s Crossref a Unpaywall"""
        enhanced_papers = []

        for paper in papers:
            enhanced_paper = paper

            # Enhance s Crossref
            crossref_data = await self._get_crossref_metadata(paper.doi)
            if crossref_data:
                enhanced_paper = self._merge_crossref_data(enhanced_paper, crossref_data)

            # Enhance s Unpaywall
            unpaywall_data = await self._get_unpaywall_data(paper.doi)
            if unpaywall_data:
                enhanced_paper = self._merge_unpaywall_data(enhanced_paper, unpaywall_data)

            enhanced_papers.append(enhanced_paper)

        return enhanced_papers

    async def _get_crossref_metadata(self, doi: str) -> Optional[Dict[str, Any]]:
        """Získá metadata z Crossref"""
        if not doi:
            return None

        # Rate limiting
        await asyncio.sleep(self.rate_limits["crossref"])

        url = f"{self.crossref_base}/works/{doi}"
        headers = {
            "User-Agent": f"DeepResearchTool/1.0 (mailto:{self.email})"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message")

        except Exception as e:
            print(f"❌ Crossref lookup failed for {doi}: {e}")

        return None

    async def _get_unpaywall_data(self, doi: str) -> Optional[Dict[str, Any]]:
        """Získá open access info z Unpaywall"""
        if not doi:
            return None

        # Rate limiting
        await asyncio.sleep(self.rate_limits["unpaywall"])

        url = f"{self.unpaywall_base}/v2/{doi}"
        params = {"email": self.email}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()

        except Exception as e:
            print(f"❌ Unpaywall lookup failed for {doi}: {e}")

        return None

    def _merge_crossref_data(self,
                           paper: ScientificPaper,
                           crossref_data: Dict[str, Any]) -> ScientificPaper:
        """Merguje Crossref data do paper"""
        # Update references
        references = []
        for ref in crossref_data.get("reference", []):
            if ref.get("DOI"):
                references.append(ref["DOI"])

        # Update funding
        funding_info = []
        for funder in crossref_data.get("funder", []):
            funder_name = funder.get("name", "")
            if funder_name:
                funding_info.append(funder_name)

        # Create updated paper
        return ScientificPaper(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            journal=paper.journal,
            open_access_status=paper.open_access_status,
            pdf_url=paper.pdf_url,
            citations_count=paper.citations_count,
            references=references,
            funding_info=funding_info,
            source_api=paper.source_api,
            confidence_score=min(1.0, paper.confidence_score + 0.1)  # Bonus for enhanced
        )

    def _merge_unpaywall_data(self,
                            paper: ScientificPaper,
                            unpaywall_data: Dict[str, Any]) -> ScientificPaper:
        """Merguje Unpaywall data do paper"""
        # Update open access status
        oa_status = paper.open_access_status
        if unpaywall_data.get("is_oa"):
            oa_status = unpaywall_data.get("oa_type", "unknown")

        # Update PDF URL
        pdf_url = paper.pdf_url
        best_oa = unpaywall_data.get("best_oa_location")
        if best_oa and best_oa.get("url_for_pdf"):
            pdf_url = best_oa["url_for_pdf"]

        return ScientificPaper(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            journal=paper.journal,
            open_access_status=oa_status,
            pdf_url=pdf_url,
            citations_count=paper.citations_count,
            references=paper.references,
            funding_info=paper.funding_info,
            source_api=paper.source_api,
            confidence_score=min(1.0, paper.confidence_score + 0.05)  # Small bonus
        )

    def _paper_to_dict(self, paper: ScientificPaper) -> Dict[str, Any]:
        """Konvertuje ScientificPaper na dict pro cache"""
        return {
            "doi": paper.doi,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date,
            "journal": paper.journal,
            "open_access_status": paper.open_access_status,
            "pdf_url": paper.pdf_url,
            "citations_count": paper.citations_count,
            "references": paper.references,
            "funding_info": paper.funding_info,
            "source_api": paper.source_api,
            "confidence_score": paper.confidence_score
        }

    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky API usage"""
        # V produkci by se trackovala real usage
        return {
            "apis_available": ["openalex", "crossref", "unpaywall", "europepmc"],
            "rate_limits": self.rate_limits,
            "cache_dir": str(self.cache_dir),
            "email_contact": self.email
        }
