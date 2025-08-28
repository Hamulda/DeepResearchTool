#!/usr/bin/env python3
"""
Legal APIs Connector
CourtListener/RECAP a SEC EDGAR s přesnými docket/filing identifikacemi

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
import backoff

logger = logging.getLogger(__name__)


@dataclass
class CourtDocument:
    """Court document s přesnou identifikací"""
    docket_id: str
    document_id: str
    case_name: str
    court: str
    date_filed: datetime
    document_type: str
    description: str
    content: str
    citation: str
    download_url: str
    metadata: Dict[str, Any]


@dataclass
class SECFiling:
    """SEC filing s přesnou identifikací"""
    filing_id: str
    cik: str
    company_name: str
    form_type: str
    filing_date: datetime
    acceptance_datetime: str
    accession_number: str
    content: str
    edgar_url: str
    metadata: Dict[str, Any]


@dataclass
class LegalSearchResult:
    """Výsledek legal search"""
    query: str
    court_documents: List[CourtDocument]
    sec_filings: List[SECFiling]
    total_results: int
    processing_time: float
    api_status: Dict[str, Any]
    quality_metrics: Dict[str, float]


class LegalAPIsConnector:
    """Connector pro legal APIs s přesnými citacemi"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.legal_config = config.get("legal_apis", {})

        # CourtListener settings
        self.courtlistener_config = self.legal_config.get("courtlistener", {})
        self.cl_api_base = self.courtlistener_config.get("api_base", "https://www.courtlistener.com/api/rest/v3")
        self.cl_api_key = self.courtlistener_config.get("api_key")
        self.cl_enabled = self.courtlistener_config.get("enabled", True)

        # SEC EDGAR settings
        self.sec_config = self.legal_config.get("sec_edgar", {})
        self.sec_api_base = self.sec_config.get("api_base", "https://data.sec.gov")
        self.sec_user_agent = self.sec_config.get("user_agent", "DeepResearchTool research@example.com")
        self.sec_enabled = self.sec_config.get("enabled", True)

        # Search settings
        self.max_results_per_api = self.legal_config.get("max_results", 100)
        self.max_results = self.max_results_per_api  # Alias for compatibility
        self.date_range_years = self.legal_config.get("date_range_years", 5)

        # Request settings
        self.timeout = self.legal_config.get("timeout", 30)
        self.rate_limit_delay = self.legal_config.get("rate_limit_delay", 1.0)
        self.max_retries = self.legal_config.get("max_retries", 3)

        # Content filtering
        self.content_filters = self.legal_config.get("content_filters", {
            "min_content_length": 100,
            "max_content_length": 50000,
            "include_full_text": True
        })

        # Sessions
        self.session = None

    async def initialize(self):
        """Inicializace konektoru"""

        logger.info("Initializing Legal APIs Connector...")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            "User-Agent": self.sec_user_agent,
            "Accept": "application/json, text/html, */*"
        }

        # Add CourtListener API key if available
        if self.cl_api_key:
            headers["Authorization"] = f"Token {self.cl_api_key}"

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers
        )

        # Test API connectivity
        await self._test_api_connectivity()

        logger.info("✅ Legal APIs Connector initialized")

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()

    async def _test_api_connectivity(self):
        """Test connectivity pro legal APIs"""

        connectivity = {}

        # Test CourtListener
        if self.cl_enabled:
            try:
                test_url = f"{self.cl_api_base}/courts/"
                async with self.session.get(test_url) as response:
                    connectivity["courtlistener"] = response.status == 200
            except:
                connectivity["courtlistener"] = False

        # Test SEC EDGAR
        if self.sec_enabled:
            try:
                test_url = f"{self.sec_api_base}/submissions/CIK0000320193.json"  # Apple Inc.
                async with self.session.get(test_url) as response:
                    connectivity["sec_edgar"] = response.status == 200
            except:
                connectivity["sec_edgar"] = False

        logger.info(f"API connectivity: {connectivity}")

    async def search_legal_documents(self,
                                   query: str,
                                   date_from: Optional[datetime] = None,
                                   date_to: Optional[datetime] = None,
                                   court_type: Optional[str] = None,
                                   company_cik: Optional[str] = None) -> LegalSearchResult:
        """
        Hlavní search funkce pro legal documents

        Args:
            query: Search query
            date_from: Start date filter
            date_to: End date filter
            court_type: Specific court type for CourtListener
            company_cik: Specific company CIK for SEC

        Returns:
            LegalSearchResult s court documents a SEC filings
        """

        start_time = asyncio.get_event_loop().time()

        # Set default date range
        if not date_to:
            date_to = datetime.now()
        if not date_from:
            date_from = date_to - timedelta(days=self.date_range_years * 365)

        logger.info(f"Starting legal search for: {query}")

        try:
            # Concurrent search across APIs
            search_tasks = []

            if self.cl_enabled:
                search_tasks.append(self._search_courtlistener(query, date_from, date_to, court_type))

            if self.sec_enabled:
                search_tasks.append(self._search_sec_edgar(query, date_from, date_to, company_cik))

            # Execute searches concurrently
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            court_documents = []
            sec_filings = []

            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning(f"Search task {i} failed: {result}")
                elif isinstance(result, list):
                    if i == 0 and self.cl_enabled:  # CourtListener results
                        court_documents = result
                    elif (i == 1 and self.cl_enabled and self.sec_enabled) or (i == 0 and not self.cl_enabled):  # SEC results
                        sec_filings = result

            # Get API status
            api_status = await self._get_api_status()

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(court_documents, sec_filings)

            processing_time = asyncio.get_event_loop().time() - start_time

            result = LegalSearchResult(
                query=query,
                court_documents=court_documents,
                sec_filings=sec_filings,
                total_results=len(court_documents) + len(sec_filings),
                processing_time=processing_time,
                api_status=api_status,
                quality_metrics=quality_metrics
            )

            logger.info(f"Legal search completed: {len(court_documents)} court docs, {len(sec_filings)} SEC filings")

            return result

        except Exception as e:
            logger.error(f"Legal search failed: {e}")
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _search_courtlistener(self,
                                  query: str,
                                  date_from: datetime,
                                  date_to: datetime,
                                  court_type: Optional[str] = None) -> List[CourtDocument]:
        """Search CourtListener/RECAP"""

        logger.info("Searching CourtListener/RECAP...")

        court_documents = []

        # Build search parameters
        params = {
            "q": query,
            "filed_after": date_from.strftime("%Y-%m-%d"),
            "filed_before": date_to.strftime("%Y-%m-%d"),
            "order_by": "score desc",
            "format": "json"
        }

        if court_type:
            params["court"] = court_type

        try:
            # Search opinions
            opinions_url = f"{self.cl_api_base}/search/"
            async with self.session.get(opinions_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])

                    # Process opinion results
                    for result in results[:self.max_results_per_api]:
                        court_doc = await self._process_courtlistener_result(result, "opinion")
                        if court_doc:
                            court_documents.append(court_doc)

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            # Search dockets for additional documents
            dockets_params = params.copy()
            dockets_url = f"{self.cl_api_base}/dockets/"

            async with self.session.get(dockets_url, params=dockets_params) as response:
                if response.status == 200:
                    data = await response.json()
                    dockets = data.get("results", [])

                    # Process docket results (limited to avoid overload)
                    for docket in dockets[:20]:
                        docket_docs = await self._get_docket_documents(docket.get("id"))
                        court_documents.extend(docket_docs[:5])  # Limit per docket

        except Exception as e:
            logger.error(f"CourtListener search error: {e}")
            raise

        logger.info(f"Found {len(court_documents)} court documents")
        return court_documents

    async def _process_courtlistener_result(self, result: Dict[str, Any], doc_type: str) -> Optional[CourtDocument]:
        """Process jednotlivého CourtListener result"""

        try:
            # Extract document information
            docket_id = str(result.get("docket", {}).get("id", ""))
            document_id = str(result.get("id", ""))
            case_name = result.get("case_name", "")
            court = result.get("court", {}).get("full_name", "")

            # Parse date
            date_filed_str = result.get("date_filed")
            if date_filed_str:
                date_filed = datetime.strptime(date_filed_str, "%Y-%m-%d")
            else:
                date_filed = datetime.now()

            # Get document content
            content = ""
            if self.content_filters["include_full_text"]:
                content = await self._fetch_courtlistener_content(result)

            # Build citation
            citation = self._build_court_citation(result)

            # Build download URL
            download_url = result.get("absolute_url", "")
            if download_url and not download_url.startswith("http"):
                download_url = f"https://www.courtlistener.com{download_url}"

            court_doc = CourtDocument(
                docket_id=docket_id,
                document_id=document_id,
                case_name=case_name,
                court=court,
                date_filed=date_filed,
                document_type=doc_type,
                description=result.get("snippet", "")[:500],
                content=content,
                citation=citation,
                download_url=download_url,
                metadata={
                    "source": "courtlistener",
                    "judges": result.get("judges", []),
                    "disposition": result.get("disposition", ""),
                    "procedural_history": result.get("procedural_history", "")
                }
            )

            return court_doc

        except Exception as e:
            logger.warning(f"Failed to process CourtListener result: {e}")
            return None

    async def _fetch_courtlistener_content(self, result: Dict[str, Any]) -> str:
        """Fetch full content z CourtListener"""

        content = ""

        try:
            # Try to get plain text content
            text_url = result.get("plain_text")
            if text_url:
                async with self.session.get(text_url) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Apply content filtering
                        if len(content) < self.content_filters["min_content_length"]:
                            content = ""
                        elif len(content) > self.content_filters["max_content_length"]:
                            content = content[:self.content_filters["max_content_length"]]

                        await asyncio.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.warning(f"Failed to fetch CourtListener content: {e}")

        return content

    def _build_court_citation(self, result: Dict[str, Any]) -> str:
        """Build standardní court citation"""

        case_name = result.get("case_name", "")
        court = result.get("court", {}).get("citation_string", "")
        date_filed = result.get("date_filed", "")

        # Build basic citation
        citation_parts = []

        if case_name:
            citation_parts.append(case_name)

        if court:
            citation_parts.append(court)

        if date_filed:
            citation_parts.append(f"({date_filed})")

        # Add docket ID if available
        docket_id = result.get("docket", {}).get("docket_number")
        if docket_id:
            citation_parts.append(f"Docket {docket_id}")

        return ", ".join(citation_parts)

    async def _get_docket_documents(self, docket_id: str) -> List<CourtDocument]:
        """Získání documents pro specific docket"""

        documents = []

        try:
            docket_url = f"{self.cl_api_base}/dockets/{docket_id}/"
            async with self.session.get(docket_url) as response:
                if response.status == 200:
                    docket_data = await response.json()

                    # Get docket entries
                    docket_entries = docket_data.get("docket_entries", [])

                    for entry in docket_entries[:5]:  # Limit entries
                        entry_docs = entry.get("recap_documents", [])

                        for doc in entry_docs[:2]:  # Limit docs per entry
                            court_doc = await self._process_docket_document(doc, docket_data)
                            if court_doc:
                                documents.append(court_doc)

                    await asyncio.sleep(self.rate_limit_delay)

        except Exception as e:
            logger.warning(f"Failed to get docket documents: {e}")

        return documents

    async def _process_docket_document(self, doc: Dict[str, Any], docket_data: Dict[str, Any]) -> Optional[CourtDocument]:
        """Process docket document"""

        try:
            document_id = str(doc.get("id", ""))
            docket_id = str(docket_data.get("id", ""))
            case_name = docket_data.get("case_name", "")
            court = docket_data.get("court", {}).get("full_name", "")

            # Parse date
            date_filed_str = doc.get("date_created") or docket_data.get("date_filed")
            if date_filed_str:
                date_filed = datetime.fromisoformat(date_filed_str.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                date_filed = datetime.now()

            description = doc.get("description", "")[:500]

            # Build citation with document number
            doc_number = doc.get("document_number", "")
            citation = f"{case_name}, Docket {docket_data.get('docket_number', '')}, Doc. {doc_number} ({court})"

            court_doc = CourtDocument(
                docket_id=docket_id,
                document_id=document_id,
                case_name=case_name,
                court=court,
                date_filed=date_filed,
                document_type="docket_document",
                description=description,
                content="",  # Would need separate fetch
                citation=citation,
                download_url=doc.get("filepath_local", ""),
                metadata={
                    "source": "courtlistener_docket",
                    "document_number": doc_number,
                    "is_available": doc.get("is_available", False),
                    "page_count": doc.get("page_count", 0)
                }
            )

            return court_doc

        except Exception as e:
            logger.warning(f"Failed to process docket document: {e}")
            return None

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _search_sec_edgar(self,
                              query: str,
                              date_from: datetime,
                              date_to: datetime,
                              company_cik: Optional[str] = None) -> List[SECFiling]:
        """Search SEC EDGAR"""

        logger.info("Searching SEC EDGAR...")

        sec_filings = []

        try:
            if company_cik:
                # Search specific company
                filings = await self._search_company_filings(company_cik, date_from, date_to)
                sec_filings.extend(filings)
            else:
                # General search through company search
                companies = await self._search_companies(query)

                # Get filings for top companies
                for company in companies[:10]:  # Limit to avoid rate limits
                    cik = company.get("cik_str")
                    if cik:
                        company_filings = await self._search_company_filings(cik, date_from, date_to)
                        sec_filings.extend(company_filings[:5])  # Limit per company

                        await asyncio.sleep(self.rate_limit_delay)

        except Exception as e:
            logger.error(f"SEC EDGAR search error: {e}")
            raise

        logger.info(f"Found {len(sec_filings)} SEC filings")
        return sec_filings

    async def _search_companies(self, query: str) -> List[Dict[str, Any]]:
        """Search for companies matching query"""

        companies = []

        try:
            # Use company tickers endpoint
            tickers_url = f"{self.sec_api_base}/files/company_tickers.json"

            async with self.session.get(tickers_url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Search for companies matching query
                    for company_id, company_info in data.items():
                        company_name = company_info.get("title", "").lower()
                        ticker = company_info.get("ticker", "").lower()

                        if (query.lower() in company_name or
                            query.lower() in ticker or
                            any(term.lower() in company_name for term in query.split())):

                            companies.append(company_info)

                            if len(companies) >= 20:  # Limit results
                                break

        except Exception as e:
            logger.warning(f"Company search error: {e}")

        return companies

    async def _search_company_filings(self,
                                    cik: str,
                                    date_from: datetime,
                                    date_to: datetime) -> List[SECFiling]:
        """Search filings for specific company"""

        filings = []

        try:
            # Pad CIK to 10 digits
            cik_padded = str(cik).zfill(10)

            # Get company submissions
            submissions_url = f"{self.sec_api_base}/submissions/CIK{cik_padded}.json"

            async with self.session.get(submissions_url) as response:
                if response.status == 200:
                    data = await response.json()

                    company_name = data.get("name", "")
                    recent_filings = data.get("filings", {}).get("recent", {})

                    # Process recent filings
                    form_types = recent_filings.get("form", [])
                    filing_dates = recent_filings.get("filingDate", [])
                    accession_numbers = recent_filings.get("accessionNumber", [])

                    for i, form_type in enumerate(form_types):
                        if i >= len(filing_dates) or i >= len(accession_numbers):
                            break

                        filing_date_str = filing_dates[i]
                        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")

                        # Date filtering
                        if not (date_from <= filing_date <= date_to):
                            continue

                        accession_number = accession_numbers[i]

                        # Create SEC filing
                        sec_filing = await self._create_sec_filing(
                            cik, company_name, form_type, filing_date, accession_number, data
                        )

                        if sec_filing:
                            filings.append(sec_filing)

                            if len(filings) >= self.max_results_per_api:
                                break

        except Exception as e:
            logger.warning(f"Company filings search error for CIK {cik}: {e}")

        return filings

    async def _create_sec_filing(self,
                               cik: str,
                               company_name: str,
                               form_type: str,
                               filing_date: datetime,
                               accession_number: str,
                               company_data: Dict[str, Any]) -> Optional[SECFiling]:
        """Create SEC filing object"""

        try:
            # Generate filing ID
            filing_id = f"{cik}_{accession_number}"

            # Build EDGAR URL
            accession_clean = accession_number.replace("-", "")
            edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession_number}.txt"

            # Get filing content if enabled
            content = ""
            if self.content_filters["include_full_text"]:
                content = await self._fetch_sec_filing_content(edgar_url)

            sec_filing = SECFiling(
                filing_id=filing_id,
                cik=cik,
                company_name=company_name,
                form_type=form_type,
                filing_date=filing_date,
                acceptance_datetime=filing_date.isoformat(),
                accession_number=accession_number,
                content=content,
                edgar_url=edgar_url,
                metadata={
                    "source": "sec_edgar",
                    "company_sic": company_data.get("sic", ""),
                    "company_address": {
                        "city": company_data.get("addresses", {}).get("business", {}).get("city", ""),
                        "state": company_data.get("addresses", {}).get("business", {}).get("stateOrCountry", "")
                    }
                }
            )

            return sec_filing

        except Exception as e:
            logger.warning(f"Failed to create SEC filing: {e}")
            return None

    async def _fetch_sec_filing_content(self, edgar_url: str) -> str:
        """Fetch SEC filing content"""

        content = ""

        try:
            async with self.session.get(edgar_url) as response:
                if response.status == 200:
                    raw_content = await response.text()

                    # Extract main document content (simplified)
                    content = self._extract_sec_content(raw_content)

                    # Apply content filtering
                    if len(content) > self.content_filters["max_content_length"]:
                        content = content[:self.content_filters["max_content_length"]]

                    await asyncio.sleep(self.rate_limit_delay)

        except Exception as e:
            logger.warning(f"Failed to fetch SEC content: {e}")

        return content

    def _extract_sec_content(self, raw_content: str) -> str:
        """Extract main content from SEC filing"""

        try:
            # Simple extraction - in production would use proper SGML/XML parsing

            # Remove SGML headers and footers
            content = re.sub(r'<DOCUMENT>.*?</DOCUMENT>', '', raw_content, flags=re.DOTALL)
            content = re.sub(r'<.*?>', '', content)  # Remove remaining tags

            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()

            return content

        except Exception as e:
            logger.warning(f"SEC content extraction failed: {e}")
            return raw_content[:self.content_filters["max_content_length"]]

    async def _get_api_status(self) -> Dict[str, Any]:
        """Získání status všech APIs"""

        status = {
            "courtlistener": {"enabled": self.cl_enabled, "authenticated": bool(self.cl_api_key)},
            "sec_edgar": {"enabled": self.sec_enabled, "rate_limited": True}
        }

        return status

    def _calculate_quality_metrics(self,
                                 court_documents: List[CourtDocument],
                                 sec_filings: List[SECFiling]) -> Dict[str, float]:
        """Výpočet quality metrics"""

        total_docs = len(court_documents) + len(sec_filings)

        if total_docs == 0:
            return {"document_diversity": 0.0, "temporal_coverage": 0.0, "content_completeness": 0.0}

        # Document type diversity
        court_ratio = len(court_documents) / total_docs
        sec_ratio = len(sec_filings) / total_docs
        doc_diversity = 1.0 - abs(court_ratio - sec_ratio)  # More balanced = higher diversity

        # Temporal coverage
        all_dates = []
        all_dates.extend([doc.date_filed for doc in court_documents])
        all_dates.extend([filing.filing_date for filing in sec_filings])

        if len(all_dates) >= 2:
            date_span = (max(all_dates) - min(all_dates)).days
            temporal_coverage = min(date_span / (365 * self.date_range_years), 1.0)
        else:
            temporal_coverage = 0.0

        # Content completeness
        docs_with_content = sum(1 for doc in court_documents if doc.content)
        filings_with_content = sum(1 for filing in sec_filings if filing.content)
        content_completeness = (docs_with_content + filings_with_content) / total_docs

        return {
            "document_diversity": doc_diversity,
            "temporal_coverage": temporal_coverage,
            "content_completeness": content_completeness,
            "court_documents_ratio": court_ratio,
            "sec_filings_ratio": sec_ratio
        }

    async def get_connector_status(self) -> Dict[str, Any]:
        """Získání status konektoru"""

        api_status = await self._get_api_status()

        return {
            "connector_type": "legal_apis",
            "apis": api_status,
            "max_results_per_api": self.max_results_per_api,
            "date_range_years": self.date_range_years,
            "content_filters": self.content_filters,
            "rate_limiting": {
                "delay_seconds": self.rate_limit_delay,
                "max_retries": self.max_retries
            }
        }
