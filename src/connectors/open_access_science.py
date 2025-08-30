#!/usr/bin/env python3
"""Open Access Science Connectors - OpenAlex, Crossref, Unpaywall, Europe PMC, arXiv
Implementuje unified OA resolver pro vědecké zdroje

Author: Senior IT Specialist
"""

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ScientificDocument:
    """Vědecký dokument z OA zdrojů"""

    id: str
    title: str
    abstract: str
    full_text: str
    authors: list[str]
    doi: str
    url: str
    source: str
    publication_date: str
    journal: str
    citation_count: int
    open_access: bool
    pdf_url: str | None
    citation: str
    metadata: dict[str, Any]


class OpenAlexConnector:
    """Konektor pro OpenAlex"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("connectors", {}).get("open_access", {}).get("openalex", {})
        self.base_url = "https://api.openalex.org"
        self.mailto = self.config.get("mailto", "research@example.com")
        self.rate_limit = self.config.get("rate_limit", 100000)  # per day

        self.logger = structlog.get_logger(__name__)

    async def search_works(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[ScientificDocument]:
        """Vyhledávání vědeckých prací"""
        self.logger.info("OpenAlex search", query=query)

        try:
            # Konstrukce search URL
            encoded_query = quote(query)
            search_url = f"{self.base_url}/works"

            params = {
                "search": encoded_query,
                "mailto": self.mailto,
                "per-page": 25,
                "sort": "cited_by_count:desc",
            }

            # Přidání filtrů
            if filters:
                if filters.get("open_access_only"):
                    params["filter"] = "is_oa:true"
                if filters.get("publication_year"):
                    params["filter"] = f"publication_year:{filters['publication_year']}"

            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": f"DeepResearchTool/1.0 (mailto:{self.mailto})"}

                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for work in data.get("results", []):
                            doc = await self._parse_openalex_work(work)
                            if doc:
                                results.append(doc)

                        return results

        except Exception as e:
            self.logger.error("Chyba při OpenAlex search", error=str(e))

        return []

    async def _parse_openalex_work(self, work: dict[str, Any]) -> ScientificDocument | None:
        """Parsování OpenAlex work objektu"""
        try:
            # Základní metadata
            title = work.get("title", "")
            doi = work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else ""

            # Abstract/content
            abstract = ""
            if work.get("abstract_inverted_index"):
                abstract = self._reconstruct_abstract(work["abstract_inverted_index"])

            # Autoři
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            # Publication info
            pub_date = work.get("publication_date", "")
            journal = ""
            if work.get("primary_location", {}).get("source"):
                journal = work["primary_location"]["source"].get("display_name", "")

            # Open Access info
            oa_info = work.get("open_access", {})
            is_oa = oa_info.get("is_oa", False)
            oa_url = oa_info.get("oa_url", "")

            # PDF URL
            pdf_url = None
            for location in work.get("locations", []):
                if location.get("is_oa") and location.get("pdf_url"):
                    pdf_url = location["pdf_url"]
                    break

            return ScientificDocument(
                id=work.get("id", "").replace("https://openalex.org/", ""),
                title=title,
                abstract=abstract,
                full_text="",  # Bude načten později
                authors=authors,
                doi=doi,
                url=oa_url or work.get("id", ""),
                source="openalex",
                publication_date=pub_date,
                journal=journal,
                citation_count=work.get("cited_by_count", 0),
                open_access=is_oa,
                pdf_url=pdf_url,
                citation=f"DOI:{doi}" if doi else f"OpenAlex:{work.get('id', '')}",
                metadata={
                    "openalex_id": work.get("id"),
                    "type": work.get("type"),
                    "concepts": [c.get("display_name") for c in work.get("concepts", [])[:5]],
                    "institutions": [
                        inst.get("display_name")
                        for inst in [
                            auth.get("institutions", [{}])[0]
                            for auth in work.get("authorships", [])
                        ]
                        if inst.get("display_name")
                    ][:3],
                },
            )

        except Exception as e:
            self.logger.warning("Chyba při parsování OpenAlex work", error=str(e))
            return None

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]]) -> str:
        """Rekonstrukce abstract z inverted index"""
        try:
            # Vytvoření seznamu slov podle pozice
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))

            # Seřazení podle pozice a spojení
            word_positions.sort(key=lambda x: x[0])
            abstract = " ".join([word for _, word in word_positions])

            return abstract[:1000]  # Max 1000 znaků

        except Exception:
            return ""


class CrossrefConnector:
    """Konektor pro Crossref"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("connectors", {}).get("open_access", {}).get("crossref", {})
        self.base_url = "https://api.crossref.org"
        self.mailto = self.config.get("mailto", "research@example.com")

        self.logger = structlog.get_logger(__name__)

    async def get_doi_metadata(self, doi: str) -> dict[str, Any] | None:
        """Získání metadat pro DOI"""
        try:
            url = f"{self.base_url}/works/{doi}"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": f"DeepResearchTool/1.0 (mailto:{self.mailto})",
                    "Accept": "application/json",
                }

                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {})

        except Exception as e:
            self.logger.error("Chyba při Crossref lookup", doi=doi, error=str(e))

        return None

    async def search_works(self, query: str) -> list[ScientificDocument]:
        """Vyhledávání v Crossref"""
        try:
            url = f"{self.base_url}/works"
            params = {"query": query, "rows": 20, "sort": "score", "order": "desc"}

            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": f"DeepResearchTool/1.0 (mailto:{self.mailto})"}

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for item in data.get("message", {}).get("items", []):
                            doc = self._parse_crossref_work(item)
                            if doc:
                                results.append(doc)

                        return results

        except Exception as e:
            self.logger.error("Chyba při Crossref search", error=str(e))

        return []

    def _parse_crossref_work(self, work: dict[str, Any]) -> ScientificDocument | None:
        """Parsování Crossref work"""
        try:
            # DOI
            doi = work.get("DOI", "")

            # Title
            title = ""
            if work.get("title"):
                title = work["title"][0] if isinstance(work["title"], list) else work["title"]

            # Authors
            authors = []
            for author in work.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)

            # Journal
            journal = ""
            if work.get("container-title"):
                journal = (
                    work["container-title"][0]
                    if isinstance(work["container-title"], list)
                    else work["container-title"]
                )

            # Publication date
            pub_date = ""
            if work.get("published-print", {}).get("date-parts"):
                date_parts = work["published-print"]["date-parts"][0]
                if len(date_parts) >= 3:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 1:
                    pub_date = str(date_parts[0])

            return ScientificDocument(
                id=f"crossref_{doi}",
                title=title,
                abstract=work.get("abstract", ""),
                full_text="",
                authors=authors,
                doi=doi,
                url=f"https://doi.org/{doi}",
                source="crossref",
                publication_date=pub_date,
                journal=journal,
                citation_count=work.get("is-referenced-by-count", 0),
                open_access=False,  # Bude ověřeno přes Unpaywall
                pdf_url=None,
                citation=f"DOI:{doi}",
                metadata={
                    "type": work.get("type"),
                    "publisher": work.get("publisher"),
                    "issn": work.get("ISSN", []),
                    "volume": work.get("volume"),
                    "issue": work.get("issue"),
                    "page": work.get("page"),
                },
            )

        except Exception as e:
            self.logger.warning("Chyba při parsování Crossref work", error=str(e))
            return None


class UnpaywallConnector:
    """Konektor pro Unpaywall"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("connectors", {}).get("open_access", {}).get("unpaywall", {})
        self.base_url = "https://api.unpaywall.org/v2"
        self.mailto = self.config.get("mailto", "research@example.com")

        self.logger = structlog.get_logger(__name__)

    async def get_oa_status(self, doi: str) -> dict[str, Any] | None:
        """Získání OA statusu pro DOI"""
        try:
            url = f"{self.base_url}/{doi}?email={self.mailto}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()

        except Exception as e:
            self.logger.error("Chyba při Unpaywall lookup", doi=doi, error=str(e))

        return None

    async def find_oa_pdf(self, doi: str) -> str | None:
        """Hledání OA PDF pro DOI"""
        oa_data = await self.get_oa_status(doi)

        if oa_data and oa_data.get("is_oa"):
            # Preferované pořadí OA lokací
            preferred_hosts = ["pubmed", "arxiv", "biorxiv", "repository"]

            best_location = None
            for location in oa_data.get("oa_locations", []):
                if location.get("url_for_pdf"):
                    host = location.get("host_type", "")

                    # Preferované hosty
                    if any(pref in host.lower() for pref in preferred_hosts):
                        return location["url_for_pdf"]

                    # Backup location
                    if not best_location:
                        best_location = location["url_for_pdf"]

            return best_location

        return None


class EuropePMCConnector:
    """Konektor pro Europe PMC"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("connectors", {}).get("open_access", {}).get("europe_pmc", {})
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"
        self.rate_limit = self.config.get("rate_limit", 1000)  # per minute

        self.logger = structlog.get_logger(__name__)

    async def search_articles(self, query: str) -> list[ScientificDocument]:
        """Vyhledávání článků v Europe PMC"""
        try:
            url = f"{self.base_url}/search"
            params = {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": 25,
                "sort": "CITED_BY desc",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for article in data.get("resultList", {}).get("result", []):
                            doc = self._parse_europepmc_article(article)
                            if doc:
                                results.append(doc)

                        return results

        except Exception as e:
            self.logger.error("Chyba při Europe PMC search", error=str(e))

        return []

    def _parse_europepmc_article(self, article: dict[str, Any]) -> ScientificDocument | None:
        """Parsování Europe PMC článku"""
        try:
            # Základní info
            pmid = article.get("pmid", "")
            pmcid = article.get("pmcid", "")
            doi = article.get("doi", "")

            # Authors
            authors = []
            author_string = article.get("authorString", "")
            if author_string:
                authors = [a.strip() for a in author_string.split(",")][:5]  # Max 5 autorů

            # Full text availability
            has_fulltext = article.get("hasTextMinedTerms") == "Y"
            pdf_url = None

            if pmcid:
                # Europe PMC PDF URL
                pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"

            return ScientificDocument(
                id=f"pmc_{pmcid or pmid}",
                title=article.get("title", ""),
                abstract=article.get("abstractText", ""),
                full_text="",  # Bude načten později pokud dostupný
                authors=authors,
                doi=doi,
                url=f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                source="europe_pmc",
                publication_date=article.get("firstPublicationDate", ""),
                journal=article.get("journalTitle", ""),
                citation_count=int(article.get("citedByCount", 0)),
                open_access=has_fulltext,
                pdf_url=pdf_url,
                citation=f"PMCID:{pmcid}" if pmcid else f"PMID:{pmid}",
                metadata={
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "pubmed_central": bool(pmcid),
                    "publication_type": article.get("pubType"),
                    "mesh_terms": article.get("meshHeadingList", {}).get("meshHeading", [])[:5],
                },
            )

        except Exception as e:
            self.logger.warning("Chyba při parsování Europe PMC článku", error=str(e))
            return None


class ArxivConnector:
    """Konektor pro arXiv"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("connectors", {}).get("open_access", {}).get("arxiv", {})
        self.base_url = "http://export.arxiv.org/api/query"

        self.logger = structlog.get_logger(__name__)

    async def search_papers(self, query: str) -> list[ScientificDocument]:
        """Vyhledávání v arXiv"""
        try:
            params = {
                "search_query": f"all:{query}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": 20,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_arxiv_xml(xml_content)

        except Exception as e:
            self.logger.error("Chyba při arXiv search", error=str(e))

        return []

    def _parse_arxiv_xml(self, xml_content: str) -> list[ScientificDocument]:
        """Parsování arXiv XML response"""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_content)

            # XML namespaces
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            results = []

            for entry in root.findall("atom:entry", namespaces):
                try:
                    # Základní metadata
                    title = entry.find("atom:title", namespaces).text.strip()
                    summary = entry.find("atom:summary", namespaces).text.strip()

                    # arXiv ID a URL
                    arxiv_id = entry.find("atom:id", namespaces).text.split("/")[-1]
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                    # Autoři
                    authors = []
                    for author in entry.findall("atom:author", namespaces):
                        name = author.find("atom:name", namespaces).text
                        authors.append(name)

                    # Datum publikace
                    published = entry.find("atom:published", namespaces).text[:10]  # YYYY-MM-DD

                    # Kategorie
                    categories = []
                    for category in entry.findall("atom:category", namespaces):
                        categories.append(category.get("term"))

                    doc = ScientificDocument(
                        id=f"arxiv_{arxiv_id}",
                        title=title,
                        abstract=summary,
                        full_text="",  # PDF bude stažen později
                        authors=authors,
                        doi="",  # arXiv nemá DOI
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        source="arxiv",
                        publication_date=published,
                        journal="arXiv preprint",
                        citation_count=0,  # arXiv neposkytuje citation count
                        open_access=True,  # arXiv je vždy open access
                        pdf_url=pdf_url,
                        citation=f"arXiv:{arxiv_id}",
                        metadata={"arxiv_id": arxiv_id, "categories": categories, "preprint": True},
                    )

                    results.append(doc)

                except Exception as e:
                    self.logger.warning("Chyba při parsování arXiv entry", error=str(e))
                    continue

            return results

        except ET.ParseError as e:
            self.logger.error("Chyba při parsování arXiv XML", error=str(e))
            return []


class OAResolver:
    """Unified Open Access resolver"""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # Inicializace konektorů
        self.openalex = OpenAlexConnector(config)
        self.crossref = CrossrefConnector(config)
        self.unpaywall = UnpaywallConnector(config)
        self.europepmc = EuropePMCConnector(config)
        self.arxiv = ArxivConnector(config)

        self.logger = structlog.get_logger(__name__)

    async def oa_resolver(self, doi: str) -> ScientificDocument | None:
        """Sekvence OpenAlex → Crossref → Unpaywall → Europe PMC pro nejkvalitnější OA kopii"""
        self.logger.info("OA resolver", doi=doi)

        # 1. OpenAlex - nejkompletnější metadata
        try:
            # Hledání work by DOI v OpenAlex
            works = await self.openalex.search_works(f"doi:{doi}")
            if works:
                openalex_work = works[0]

                # Enrichment z dalších zdrojů
                enriched_work = await self._enrich_with_oa_sources(openalex_work)
                return enriched_work
        except Exception as e:
            self.logger.warning("OpenAlex lookup failed", error=str(e))

        # 2. Crossref fallback
        try:
            crossref_meta = await self.crossref.get_doi_metadata(doi)
            if crossref_meta:
                # Vytvoření dokumentu z Crossref
                doc = self.crossref._parse_crossref_work(crossref_meta)
                if doc:
                    enriched_doc = await self._enrich_with_oa_sources(doc)
                    return enriched_doc
        except Exception as e:
            self.logger.warning("Crossref lookup failed", error=str(e))

        # 3. Pouze Unpaywall
        try:
            oa_data = await self.unpaywall.get_oa_status(doi)
            if oa_data:
                return self._create_minimal_doc_from_unpaywall(oa_data)
        except Exception as e:
            self.logger.warning("Unpaywall lookup failed", error=str(e))

        return None

    async def _enrich_with_oa_sources(self, doc: ScientificDocument) -> ScientificDocument:
        """Enrichment dokumentu z dalších OA zdrojů"""
        # Unpaywall enrichment pro PDF
        if doc.doi and not doc.pdf_url:
            pdf_url = await self.unpaywall.find_oa_pdf(doc.doi)
            if pdf_url:
                doc.pdf_url = pdf_url
                doc.open_access = True

        # Europe PMC enrichment pro PubMed články
        if "pubmed" in doc.url.lower() or doc.metadata.get("pmid"):
            try:
                pmc_results = await self.europepmc.search_articles(doc.title[:100])
                if pmc_results:
                    pmc_doc = pmc_results[0]
                    # Merge užitečných metadat
                    if pmc_doc.metadata.get("pmcid") and not doc.metadata.get("pmcid"):
                        doc.metadata["pmcid"] = pmc_doc.metadata["pmcid"]
                        doc.citation = f"PMCID:{pmc_doc.metadata['pmcid']}"

                    if pmc_doc.pdf_url and not doc.pdf_url:
                        doc.pdf_url = pmc_doc.pdf_url
                        doc.open_access = True

            except Exception as e:
                self.logger.warning("Europe PMC enrichment failed", error=str(e))

        return doc

    def _create_minimal_doc_from_unpaywall(self, oa_data: dict[str, Any]) -> ScientificDocument:
        """Vytvoření minimálního dokumentu pouze z Unpaywall dat"""
        doi = oa_data.get("doi", "")

        return ScientificDocument(
            id=f"unpaywall_{doi}",
            title=oa_data.get("title", ""),
            abstract="",
            full_text="",
            authors=[],
            doi=doi,
            url=f"https://doi.org/{doi}",
            source="unpaywall",
            publication_date=str(oa_data.get("year", "")),
            journal=oa_data.get("journal_name", ""),
            citation_count=0,
            open_access=oa_data.get("is_oa", False),
            pdf_url=oa_data.get("best_oa_location", {}).get("url_for_pdf"),
            citation=f"DOI:{doi}",
            metadata={
                "oa_locations": oa_data.get("oa_locations", []),
                "publisher": oa_data.get("publisher"),
            },
        )

    async def search_all_oa_sources(self, query: str) -> list[ScientificDocument]:
        """Vyhledávání napříč všemi OA zdroji"""
        self.logger.info("Searching all OA sources", query=query)

        all_results = []

        # Paralelní vyhledávání
        search_tasks = [
            self.openalex.search_works(query),
            self.crossref.search_works(query),
            self.europepmc.search_articles(query),
            self.arxiv.search_papers(query),
        ]

        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning("Search task failed", error=str(result))

        except Exception as e:
            self.logger.error("Parallel OA search failed", error=str(e))

        # Deduplikace podle DOI/title
        unique_results = self._deduplicate_scientific_docs(all_results)

        # Enrichment top výsledků
        enriched_results = []
        for doc in unique_results[:10]:  # Top 10 pro enrichment
            enriched_doc = await self._enrich_with_oa_sources(doc)
            enriched_results.append(enriched_doc)

        # Přidání zbytku bez enrichment
        enriched_results.extend(unique_results[10:])

        return enriched_results

    def _deduplicate_scientific_docs(
        self, docs: list[ScientificDocument]
    ) -> list[ScientificDocument]:
        """Deduplikace vědeckých dokumentů"""
        seen_identifiers = set()
        unique_docs = []

        for doc in docs:
            # Klíč pro deduplikaci: DOI > title hash
            if doc.doi:
                identifier = f"doi:{doc.doi.lower()}"
            else:
                identifier = f"title:{hash(doc.title.lower())}"

            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_docs.append(doc)

        # Řazení podle citation count a OA status
        unique_docs.sort(key=lambda x: (x.open_access, x.citation_count), reverse=True)

        return unique_docs
