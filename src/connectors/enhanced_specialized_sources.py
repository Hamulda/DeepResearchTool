#!/usr/bin/env python3
"""
Enhanced Source Connectors with Authority Ranking
Open-science, legal/regulatory, and archive connectors with source preference

Author: Senior IT Specialist
"""

import asyncio
import aiohttp
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json

from .enhanced_evidence_binding import EnhancedEvidence, EnhancedEvidenceManager


@dataclass
class SourceAuthority:
    """Source authority and credibility scoring"""
    domain_authority: float = 0.5  # 0-1 scale
    peer_review_status: bool = False
    institutional_affiliation: bool = False
    citation_count: int = 0
    impact_factor: float = 0.0
    recency_score: float = 0.5  # Higher for more recent content
    legal_status: str = "unknown"  # official, unofficial, draft, repealed

    def calculate_overall_authority(self) -> float:
        """Calculate weighted overall authority score"""
        score = self.domain_authority * 0.3

        if self.peer_review_status:
            score += 0.2

        if self.institutional_affiliation:
            score += 0.1

        # Citation factor (logarithmic scale)
        if self.citation_count > 0:
            import math
            citation_factor = min(0.2, math.log10(self.citation_count + 1) / 10)
            score += citation_factor

        # Impact factor
        if self.impact_factor > 0:
            impact_factor = min(0.1, self.impact_factor / 50)  # Normalize to 0.1 max
            score += impact_factor

        # Recency
        score += self.recency_score * 0.1

        return min(1.0, score)


class OpenScienceConnector:
    """Enhanced open science connector with DOI resolution and preference ranking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evidence_manager = EnhancedEvidenceManager()

        # API endpoints
        self.crossref_api = "https://api.crossref.org/works/"
        self.unpaywall_api = "https://api.unpaywall.org/v2/"
        self.europepmc_api = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        self.openalex_api = "https://api.openalex.org/works"

        # Journal impact factors (subset for demo)
        self.journal_impact_factors = {
            "nature": 49.962,
            "science": 47.728,
            "cell": 41.582,
            "nejm": 91.245,
            "lancet": 79.321,
            "jama": 56.272
        }

        # Preprint server rankings
        self.preprint_rankings = {
            "arxiv.org": 0.8,
            "biorxiv.org": 0.7,
            "medrxiv.org": 0.7,
            "chemrxiv.org": 0.6,
            "preprints.org": 0.5
        }

    async def search_and_rank(self, query: str, max_results: int = 20) -> List[EnhancedEvidence]:
        """Search open science sources and rank by authority"""
        self.logger.info(f"Searching open science sources for: {query}")

        # Parallel search across sources
        tasks = [
            self._search_crossref(query, max_results // 4),
            self._search_europepmc(query, max_results // 4),
            self._search_openalex(query, max_results // 4),
            self._search_unpaywall_open_access(query, max_results // 4)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine and deduplicate results
        all_evidence = []
        seen_dois = set()

        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Search source failed: {result}")
                continue

            for evidence in result:
                # Deduplicate by DOI
                doi = evidence.persistent_ids.doi
                if doi and doi in seen_dois:
                    continue

                if doi:
                    seen_dois.add(doi)

                all_evidence.append(evidence)

        # Rank by authority with preference for peer-reviewed content
        ranked_evidence = self._rank_by_authority(all_evidence)

        self.logger.info(f"Found {len(ranked_evidence)} ranked open science results")
        return ranked_evidence[:max_results]

    async def _search_crossref(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search Crossref DOI database"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "rows": limit,
                    "sort": "relevance",
                    "filter": "type:journal-article"
                }

                async with session.get(self.crossref_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_crossref_results(data)
        except Exception as e:
            self.logger.error(f"Crossref search failed: {e}")

        return []

    async def _search_europepmc(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search Europe PMC database"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "pageSize": limit,
                    "format": "json"
                }

                url = urljoin(self.europepmc_api, "search")
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_europepmc_results(data)
        except Exception as e:
            self.logger.error(f"Europe PMC search failed: {e}")

        return []

    async def _search_openalex(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search OpenAlex database"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "search": query,
                    "per-page": limit,
                    "filter": "type:article"
                }

                async with session.get(self.openalex_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_openalex_results(data)
        except Exception as e:
            self.logger.error(f"OpenAlex search failed: {e}")

        return []

    async def _search_unpaywall_open_access(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search for open access versions via Unpaywall"""
        # This would typically work with DOIs from other searches
        # For demo, return empty list
        return []

    def _parse_crossref_results(self, data: Dict[str, Any]) -> List[EnhancedEvidence]:
        """Parse Crossref API results"""
        evidence_list = []

        for item in data.get("message", {}).get("items", []):
            try:
                # Extract basic info
                title = " ".join(item.get("title", ["Unknown Title"]))
                abstract = " ".join(item.get("abstract", [""])) if "abstract" in item else ""
                doi = item.get("DOI", "")

                # Extract journal info
                journal = ""
                if "container-title" in item:
                    journal = item["container-title"][0] if item["container-title"] else ""

                # Extract publication date
                pub_date = None
                if "published-print" in item:
                    date_parts = item["published-print"].get("date-parts", [[]])[0]
                    if len(date_parts) >= 3:
                        pub_date = datetime(date_parts[0], date_parts[1], date_parts[2], tzinfo=timezone.utc)

                # Create enhanced evidence
                source_data = {
                    "source_id": f"crossref_{doi}",
                    "canonical_url": f"https://doi.org/{doi}",
                    "title": title,
                    "snippet": abstract[:500] + "..." if len(abstract) > 500 else abstract
                }

                evidence = self.evidence_manager.create_enhanced_evidence(
                    source_data, content=abstract
                )

                # Set publication date
                if pub_date:
                    evidence.temporal_metadata.publication_date = pub_date

                # Calculate authority
                authority = self._calculate_journal_authority(journal, item)
                evidence.source_authority = authority.calculate_overall_authority()
                evidence.source_type = "academic"

                evidence_list.append(evidence)

            except Exception as e:
                self.logger.warning(f"Failed to parse Crossref item: {e}")
                continue

        return evidence_list

    def _parse_europepmc_results(self, data: Dict[str, Any]) -> List[EnhancedEvidence]:
        """Parse Europe PMC results"""
        evidence_list = []

        for item in data.get("resultList", {}).get("result", []):
            try:
                title = item.get("title", "Unknown Title")
                abstract = item.get("abstractText", "")
                pmid = item.get("pmid", "")
                doi = item.get("doi", "")

                source_data = {
                    "source_id": f"europepmc_{pmid or doi}",
                    "canonical_url": f"https://europepmc.org/article/MED/{pmid}" if pmid else f"https://doi.org/{doi}",
                    "title": title,
                    "snippet": abstract[:500] + "..." if len(abstract) > 500 else abstract
                }

                evidence = self.evidence_manager.create_enhanced_evidence(
                    source_data, content=abstract
                )

                # Set as academic and peer-reviewed (Europe PMC filters)
                evidence.source_type = "academic"
                authority = SourceAuthority(
                    domain_authority=0.85,  # Europe PMC is high quality
                    peer_review_status=True,
                    institutional_affiliation=True
                )
                evidence.source_authority = authority.calculate_overall_authority()

                evidence_list.append(evidence)

            except Exception as e:
                self.logger.warning(f"Failed to parse Europe PMC item: {e}")
                continue

        return evidence_list

    def _parse_openalex_results(self, data: Dict[str, Any]) -> List[EnhancedEvidence]:
        """Parse OpenAlex results"""
        evidence_list = []

        for item in data.get("results", []):
            try:
                title = item.get("title", "Unknown Title")
                abstract = item.get("abstract", "")
                doi = item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else ""

                source_data = {
                    "source_id": f"openalex_{item.get('id', '').split('/')[-1]}",
                    "canonical_url": item.get("doi", "") or item.get("id", ""),
                    "title": title,
                    "snippet": abstract[:500] + "..." if len(abstract) > 500 else abstract
                }

                evidence = self.evidence_manager.create_enhanced_evidence(
                    source_data, content=abstract
                )

                # Use OpenAlex metrics for authority
                citation_count = item.get("cited_by_count", 0)
                is_oa = item.get("open_access", {}).get("is_oa", False)

                authority = SourceAuthority(
                    domain_authority=0.8,  # OpenAlex is comprehensive
                    citation_count=citation_count,
                    peer_review_status=True  # Assume peer-reviewed
                )
                evidence.source_authority = authority.calculate_overall_authority()
                evidence.source_type = "academic"

                evidence_list.append(evidence)

            except Exception as e:
                self.logger.warning(f"Failed to parse OpenAlex item: {e}")
                continue

        return evidence_list

    def _calculate_journal_authority(self, journal: str, item: Dict[str, Any]) -> SourceAuthority:
        """Calculate authority for journal publication"""
        journal_lower = journal.lower()

        # Check impact factor
        impact_factor = 0.0
        for journal_key, factor in self.journal_impact_factors.items():
            if journal_key in journal_lower:
                impact_factor = factor
                break

        # Citation count
        citation_count = item.get("is-referenced-by-count", 0)

        # Base authority for peer-reviewed journal
        authority = SourceAuthority(
            domain_authority=0.8,  # Base for academic journals
            peer_review_status=True,
            institutional_affiliation=True,
            citation_count=citation_count,
            impact_factor=impact_factor
        )

        # Boost for high-impact journals
        if impact_factor > 30:
            authority.domain_authority = 0.95
        elif impact_factor > 10:
            authority.domain_authority = 0.9

        return authority

    def _rank_by_authority(self, evidence_list: List[EnhancedEvidence]) -> List[EnhancedEvidence]:
        """Rank evidence by authority with peer-review preference"""

        def authority_score(evidence: EnhancedEvidence) -> Tuple[int, float]:
            # First sort by peer-review status (1 for peer-reviewed, 0 for preprints)
            peer_reviewed = 1 if evidence.source_authority > 0.7 else 0

            # Check if it's a preprint
            for preprint_domain in self.preprint_rankings:
                if preprint_domain in evidence.domain:
                    peer_reviewed = 0
                    break

            # Return tuple for sorting: (peer_reviewed_priority, authority_score)
            return (peer_reviewed, evidence.source_authority)

        # Sort with peer-reviewed articles first, then by authority
        ranked = sorted(evidence_list, key=authority_score, reverse=True)

        self.logger.info(f"Ranked {len(ranked)} results by authority preference")
        return ranked


class LegalRegulatoryConnector:
    """Legal and regulatory document connector with precise citation matching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evidence_manager = EnhancedEvidenceManager()

        # Legal source authorities
        self.legal_authorities = {
            "sec.gov": {"authority": 0.95, "type": "federal_agency"},
            "eur-lex.europa.eu": {"authority": 0.95, "type": "eu_official"},
            "courtlistener.com": {"authority": 0.85, "type": "case_law"},
            "justia.com": {"authority": 0.7, "type": "legal_database"},
            "law.cornell.edu": {"authority": 0.8, "type": "academic_legal"}
        }

    async def search_legal_documents(self, query: str, max_results: int = 20) -> List[EnhancedEvidence]:
        """Search legal and regulatory sources"""
        self.logger.info(f"Searching legal sources for: {query}")

        tasks = [
            self._search_sec_edgar(query, max_results // 3),
            self._search_eur_lex(query, max_results // 3),
            self._search_courtlistener(query, max_results // 3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_evidence = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Legal search failed: {result}")
                continue
            all_evidence.extend(result)

        # Rank by legal authority
        ranked_evidence = self._rank_by_legal_authority(all_evidence)

        return ranked_evidence[:max_results]

    async def _search_sec_edgar(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search SEC EDGAR database"""
        # Simplified SEC search implementation
        try:
            # In a real implementation, this would use SEC EDGAR API
            # For demo, return mock results
            mock_results = [
                {
                    "source_id": "sec_10k_example",
                    "canonical_url": "https://sec.gov/Archives/edgar/data/example",
                    "title": f"SEC Filing related to {query}",
                    "snippet": f"Securities filing discussing {query} and regulatory compliance...",
                    "cik": "0001234567",
                    "filing_type": "10-K"
                }
            ]

            evidence_list = []
            for item in mock_results:
                evidence = self.evidence_manager.create_enhanced_evidence(item, content=item["snippet"])
                evidence.source_type = "legal"
                evidence.source_authority = 0.95  # SEC is authoritative

                # Set CIK identifier
                evidence.persistent_ids.cik = item.get("cik")

                evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            self.logger.error(f"SEC EDGAR search failed: {e}")
            return []

    async def _search_eur_lex(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search EUR-Lex legal database"""
        try:
            # Mock EU legal document
            mock_results = [
                {
                    "source_id": "eurlex_directive_example",
                    "canonical_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:example",
                    "title": f"EU Directive on {query}",
                    "snippet": f"European Union directive addressing {query} regulations...",
                    "celex_number": "32024L0001"
                }
            ]

            evidence_list = []
            for item in mock_results:
                evidence = self.evidence_manager.create_enhanced_evidence(item, content=item["snippet"])
                evidence.source_type = "legal"
                evidence.source_authority = 0.95  # EU official documents are authoritative

                evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            self.logger.error(f"EUR-Lex search failed: {e}")
            return []

    async def _search_courtlistener(self, query: str, limit: int) -> List[EnhancedEvidence]:
        """Search CourtListener case law database"""
        try:
            # Mock case law results
            mock_results = [
                {
                    "source_id": "courtlistener_case_example",
                    "canonical_url": "https://courtlistener.com/opinion/example/",
                    "title": f"Court Case: {query}",
                    "snippet": f"Legal precedent establishing {query} principles...",
                    "docket": "No. 2024-CV-001",
                    "court": "Supreme Court"
                }
            ]

            evidence_list = []
            for item in mock_results:
                evidence = self.evidence_manager.create_enhanced_evidence(item, content=item["snippet"])
                evidence.source_type = "legal"
                evidence.source_authority = 0.85  # Case law is authoritative but contextual

                # Set docket identifier
                evidence.persistent_ids.docket = item.get("docket")

                evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            self.logger.error(f"CourtListener search failed: {e}")
            return []

    def _rank_by_legal_authority(self, evidence_list: List[EnhancedEvidence]) -> List[EnhancedEvidence]:
        """Rank legal documents by authority hierarchy"""

        def legal_authority_score(evidence: EnhancedEvidence) -> Tuple[int, float]:
            domain = evidence.domain

            # Legal hierarchy scoring
            hierarchy_score = 0
            if domain in self.legal_authorities:
                auth_info = self.legal_authorities[domain]
                if auth_info["type"] == "federal_agency":
                    hierarchy_score = 4
                elif auth_info["type"] == "eu_official":
                    hierarchy_score = 4
                elif auth_info["type"] == "case_law":
                    hierarchy_score = 3
                elif auth_info["type"] == "academic_legal":
                    hierarchy_score = 2
                else:
                    hierarchy_score = 1

            return (hierarchy_score, evidence.source_authority)

        ranked = sorted(evidence_list, key=legal_authority_score, reverse=True)

        self.logger.info(f"Ranked {len(ranked)} legal documents by authority hierarchy")
        return ranked

    def extract_precise_citations(self, evidence: EnhancedEvidence, cited_passage: str) -> Dict[str, Any]:
        """Extract precise legal citations with string matching"""
        citations = {
            "exact_matches": [],
            "partial_matches": [],
            "confidence": 0.0
        }

        content = evidence.content or evidence.snippet

        # Exact string match
        if cited_passage.lower() in content.lower():
            citations["exact_matches"].append({
                "passage": cited_passage,
                "position": content.lower().find(cited_passage.lower()),
                "context_before": "",
                "context_after": ""
            })
            citations["confidence"] = 1.0

        # Partial matches (simplified)
        words = cited_passage.lower().split()
        if len(words) > 3:
            for i in range(len(words) - 2):
                partial = " ".join(words[i:i+3])
                if partial in content.lower():
                    citations["partial_matches"].append({
                        "fragment": partial,
                        "position": content.lower().find(partial)
                    })
                    citations["confidence"] = max(citations["confidence"], 0.6)

        return citations


class ArchiveMementoConnector:
    """Archive and Memento connector with CDX/snapshot tracking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evidence_manager = EnhancedEvidenceManager()

        # Archive endpoints
        self.wayback_api = "https://web.archive.org/cdx/search/cdx"
        self.memento_timegate = "http://timetravel.mementoweb.org/timegate/"

    async def search_archived_content(self, url: str, date_range: Tuple[str, str] = None) -> List[EnhancedEvidence]:
        """Search archived versions of content with diff analysis"""
        self.logger.info(f"Searching archived content for: {url}")

        try:
            # Get CDX records for URL
            cdx_records = await self._get_cdx_records(url, date_range)

            evidence_list = []
            previous_content = None

            for record in cdx_records:
                # Fetch archived content
                archived_evidence = await self._fetch_archived_content(record)

                if archived_evidence:
                    # Detect content changes
                    if previous_content:
                        changes = self._detect_content_changes(previous_content, archived_evidence.content)
                        if changes["significant_change"]:
                            self.logger.info(f"Significant content change detected at {record['timestamp']}")

                    evidence_list.append(archived_evidence)
                    previous_content = archived_evidence.content

            return evidence_list[:20]  # Limit results

        except Exception as e:
            self.logger.error(f"Archive search failed: {e}")
            return []

    async def _get_cdx_records(self, url: str, date_range: Tuple[str, str] = None) -> List[Dict[str, Any]]:
        """Get CDX records from Wayback Machine"""
        try:
            params = {
                "url": url,
                "output": "json",
                "limit": 50
            }

            if date_range:
                params["from"] = date_range[0]
                params["to"] = date_range[1]

            async with aiohttp.ClientSession() as session:
                async with session.get(self.wayback_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if not data:
                            return []

                        # Parse CDX format
                        headers = data[0]
                        records = []

                        for row in data[1:]:
                            record = dict(zip(headers, row))
                            records.append(record)

                        return records

        except Exception as e:
            self.logger.error(f"CDX search failed: {e}")
            return []

    async def _fetch_archived_content(self, cdx_record: Dict[str, Any]) -> Optional[EnhancedEvidence]:
        """Fetch content from archived snapshot"""
        try:
            wayback_url = f"https://web.archive.org/web/{cdx_record['timestamp']}/{cdx_record['original']}"

            # Create evidence with archive metadata
            source_data = {
                "source_id": f"archive_{cdx_record['timestamp']}_{hash(cdx_record['original'])}",
                "canonical_url": cdx_record["original"],
                "title": f"Archived: {cdx_record['original']}",
                "snippet": f"Archived content from {cdx_record['timestamp']}"
            }

            evidence = self.evidence_manager.create_enhanced_evidence(source_data)

            # Set archive-specific metadata
            evidence.source_type = "archive"
            evidence.temporal_metadata.wayback_timestamp = cdx_record["timestamp"]

            # Parse timestamp
            try:
                timestamp_dt = datetime.strptime(cdx_record["timestamp"], "%Y%m%d%H%M%S")
                evidence.temporal_metadata.memento_datetime = timestamp_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

            # Authority based on archive age (older = more stable)
            days_old = (datetime.now() - timestamp_dt).days if timestamp_dt else 0
            evidence.source_authority = min(0.8, 0.5 + (days_old / 3650))  # Max 0.8 for 10+ year old content

            return evidence

        except Exception as e:
            self.logger.error(f"Failed to fetch archived content: {e}")
            return None

    def _detect_content_changes(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Detect significant changes between content versions"""
        # Simple change detection
        old_words = set(old_content.lower().split())
        new_words = set(new_content.lower().split())

        added_words = new_words - old_words
        removed_words = old_words - new_words

        change_ratio = (len(added_words) + len(removed_words)) / max(len(old_words), 1)

        return {
            "significant_change": change_ratio > 0.1,  # 10% word change threshold
            "change_ratio": change_ratio,
            "words_added": len(added_words),
            "words_removed": len(removed_words),
            "content_hash_old": hash(old_content),
            "content_hash_new": hash(new_content)
        }


def create_enhanced_source_connectors(config: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function for enhanced source connectors"""
    return {
        "open_science": OpenScienceConnector(config.get("open_science", {})),
        "legal_regulatory": LegalRegulatoryConnector(config.get("legal", {})),
        "archive_memento": ArchiveMementoConnector(config.get("archive", {}))
    }
