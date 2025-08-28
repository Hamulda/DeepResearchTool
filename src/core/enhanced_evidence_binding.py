#!/usr/bin/env python3
"""
Enhanced Evidence Binding with Temporal Metadata
Time-aware evidence tracking with persistent identifiers and snapshot management

Author: Senior IT Specialist
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse
import json
import logging


@dataclass
class PersistentIdentifiers:
    """Collection of persistent identifiers for evidence"""
    doi: Optional[str] = None
    ecli: Optional[str] = None  # European Case Law Identifier
    cik: Optional[str] = None   # SEC Central Index Key
    docket: Optional[str] = None # Court docket number
    pmid: Optional[str] = None   # PubMed ID
    arxiv_id: Optional[str] = None
    issn: Optional[str] = None
    isbn: Optional[str] = None
    orcid: Optional[str] = None  # Author ORCID


@dataclass
class TemporalMetadata:
    """Time-related metadata for evidence"""
    # Source timestamps
    publication_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    access_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Archive/snapshot info
    memento_datetime: Optional[datetime] = None  # When content was archived
    snapshot_hash: Optional[str] = None  # Content hash for change detection
    wayback_timestamp: Optional[str] = None  # Wayback Machine timestamp

    # Version tracking
    content_version: Optional[str] = None
    revision_number: Optional[int] = None

    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None


@dataclass
class EnhancedEvidence:
    """Enhanced evidence with temporal metadata and persistent IDs"""
    # Core evidence data
    source_id: str
    canonical_url: str
    title: str
    snippet: str
    content: str = ""

    # Confidence and scoring
    relevance_score: float = 0.0
    confidence_score: float = 0.0

    # Enhanced metadata
    temporal_metadata: TemporalMetadata = field(default_factory=TemporalMetadata)
    persistent_ids: PersistentIdentifiers = field(default_factory=PersistentIdentifiers)

    # Source characteristics
    source_type: str = "web"  # web, academic, legal, archive, news
    domain: str = ""
    language: str = "en"
    source_authority: float = 0.5  # Authority/credibility score

    # Processing metadata
    extraction_method: str = "scraping"
    processing_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "temporal_metadata": {
                "publication_date": self.temporal_metadata.publication_date.isoformat() if self.temporal_metadata.publication_date else None,
                "last_modified": self.temporal_metadata.last_modified.isoformat() if self.temporal_metadata.last_modified else None,
                "access_date": self.temporal_metadata.access_date.isoformat(),
                "memento_datetime": self.temporal_metadata.memento_datetime.isoformat() if self.temporal_metadata.memento_datetime else None,
                "snapshot_hash": self.temporal_metadata.snapshot_hash,
                "wayback_timestamp": self.temporal_metadata.wayback_timestamp,
                "content_version": self.temporal_metadata.content_version,
                "revision_number": self.temporal_metadata.revision_number,
                "valid_from": self.temporal_metadata.valid_from.isoformat() if self.temporal_metadata.valid_from else None,
                "valid_until": self.temporal_metadata.valid_until.isoformat() if self.temporal_metadata.valid_until else None
            },
            "persistent_ids": {
                "doi": self.persistent_ids.doi,
                "ecli": self.persistent_ids.ecli,
                "cik": self.persistent_ids.cik,
                "docket": self.persistent_ids.docket,
                "pmid": self.persistent_ids.pmid,
                "arxiv_id": self.persistent_ids.arxiv_id,
                "issn": self.persistent_ids.issn,
                "isbn": self.persistent_ids.isbn,
                "orcid": self.persistent_ids.orcid
            },
            "source_type": self.source_type,
            "domain": self.domain,
            "language": self.language,
            "source_authority": self.source_authority,
            "extraction_method": self.extraction_method,
            "processing_timestamp": self.processing_timestamp.isoformat()
        }


class PersistentIdentifierExtractor:
    """Extracts persistent identifiers from content and URLs"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Regex patterns for various identifiers
        self.patterns = {
            "doi": re.compile(r"(?:doi:|DOI:)?\s*(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s\]]+)", re.IGNORECASE),
            "pmid": re.compile(r"(?:PMID:?\s*)?(\d{7,8})", re.IGNORECASE),
            "arxiv_id": re.compile(r"(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE),
            "ecli": re.compile(r"(ECLI:[A-Z]{2}:[A-Z0-9]+:\d{4}:[A-Z0-9]+)", re.IGNORECASE),
            "cik": re.compile(r"(?:CIK:?\s*)?(\d{10})", re.IGNORECASE),
            "issn": re.compile(r"(?:ISSN:?\s*)?(\d{4}-\d{3}[\dX])", re.IGNORECASE),
            "isbn": re.compile(r"(?:ISBN:?\s*)?(?:978-?|979-?)?(\d{1,5}-?\d{1,7}-?\d{1,7}-?[\dX])", re.IGNORECASE),
            "orcid": re.compile(r"(?:orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", re.IGNORECASE)
        }

    def extract_from_url(self, url: str) -> PersistentIdentifiers:
        """Extract identifiers from URL"""
        ids = PersistentIdentifiers()

        # DOI from URL
        if "doi.org" in url:
            doi_match = re.search(r"doi\.org/(10\.\d{4,}/[^?\s]+)", url)
            if doi_match:
                ids.doi = doi_match.group(1)

        # ArXiv from URL
        if "arxiv.org" in url:
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
            if arxiv_match:
                ids.arxiv_id = arxiv_match.group(1)

        # PubMed from URL
        if "pubmed.ncbi.nlm.nih.gov" in url:
            pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})", url)
            if pmid_match:
                ids.pmid = pmid_match.group(1)

        # SEC EDGAR
        if "sec.gov" in url:
            cik_match = re.search(r"CIK=(\d{10})", url, re.IGNORECASE)
            if cik_match:
                ids.cik = cik_match.group(1)

        return ids

    def extract_from_content(self, content: str, title: str = "") -> PersistentIdentifiers:
        """Extract identifiers from content text"""
        ids = PersistentIdentifiers()

        # Combine title and content for searching
        search_text = f"{title} {content}"

        # Extract each type of identifier
        for id_type, pattern in self.patterns.items():
            matches = pattern.findall(search_text)
            if matches:
                # Take the first valid match
                identifier = matches[0]

                # Validation and cleaning
                if id_type == "doi":
                    identifier = self._clean_doi(identifier)
                elif id_type == "pmid":
                    identifier = self._clean_pmid(identifier)
                elif id_type == "arxiv_id":
                    identifier = self._clean_arxiv_id(identifier)

                setattr(ids, id_type, identifier)
                self.logger.debug(f"Extracted {id_type}: {identifier}")

        return ids

    def _clean_doi(self, doi: str) -> str:
        """Clean and validate DOI"""
        # Remove common prefixes and clean up
        doi = re.sub(r"^(?:doi:|DOI:)\s*", "", doi, flags=re.IGNORECASE)
        doi = doi.strip()

        # Basic validation (starts with 10.)
        if not doi.startswith("10."):
            return None

        return doi

    def _clean_pmid(self, pmid: str) -> str:
        """Clean and validate PMID"""
        pmid = re.sub(r"[^\d]", "", pmid)

        # Validate length (7-8 digits)
        if len(pmid) < 7 or len(pmid) > 8:
            return None

        return pmid

    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and validate ArXiv ID"""
        # Remove arXiv: prefix if present
        arxiv_id = re.sub(r"^arXiv:", "", arxiv_id, flags=re.IGNORECASE)

        # Validate format (YYMM.NNNN or YYMM.NNNNNvN)
        if not re.match(r"\d{4}\.\d{4,5}(?:v\d+)?$", arxiv_id):
            return None

        return arxiv_id


class TemporalMetadataExtractor:
    """Extracts temporal metadata from content and headers"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Date patterns for content extraction
        self.date_patterns = [
            # ISO format
            re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"),
            # Standard formats
            re.compile(r"(\d{4}-\d{2}-\d{2})"),
            re.compile(r"(\d{1,2}/\d{1,2}/\d{4})"),
            re.compile(r"(\d{1,2}-\d{1,2}-\d{4})"),
            # Text dates
            re.compile(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})", re.IGNORECASE)
        ]

    def extract_from_headers(self, headers: Dict[str, str]) -> TemporalMetadata:
        """Extract temporal metadata from HTTP headers"""
        metadata = TemporalMetadata()

        # Last-Modified header
        if "last-modified" in headers:
            try:
                metadata.last_modified = datetime.strptime(
                    headers["last-modified"], "%a, %d %b %Y %H:%M:%S %Z"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                self.logger.warning(f"Could not parse Last-Modified: {headers['last-modified']}")

        # Date header
        if "date" in headers:
            try:
                access_date = datetime.strptime(
                    headers["date"], "%a, %d %b %Y %H:%M:%S %Z"
                ).replace(tzinfo=timezone.utc)
                metadata.access_date = access_date
            except ValueError:
                pass

        # ETag for version tracking
        if "etag" in headers:
            metadata.content_version = headers["etag"].strip('"')

        return metadata

    def extract_from_content(self, content: str, url: str = "") -> TemporalMetadata:
        """Extract temporal metadata from content"""
        metadata = TemporalMetadata()

        # Look for publication dates in content
        publication_date = self._find_publication_date(content)
        if publication_date:
            metadata.publication_date = publication_date

        # Generate snapshot hash
        metadata.snapshot_hash = self._generate_content_hash(content)

        # Extract Wayback Machine timestamp from URL if present
        if "web.archive.org" in url:
            wayback_match = re.search(r"web\.archive\.org/web/(\d{14})/", url)
            if wayback_match:
                metadata.wayback_timestamp = wayback_match.group(1)
                # Convert to datetime
                try:
                    wayback_dt = datetime.strptime(wayback_match.group(1), "%Y%m%d%H%M%S")
                    metadata.memento_datetime = wayback_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

        return metadata

    def _find_publication_date(self, content: str) -> Optional[datetime]:
        """Find publication date in content"""
        # Look for common publication date indicators
        pub_indicators = [
            r"published?:?\s*([^<\n]+)",
            r"publication\s+date:?\s*([^<\n]+)",
            r"date\s+published:?\s*([^<\n]+)",
            r"pub\s+date:?\s*([^<\n]+)",
            r"created:?\s*([^<\n]+)",
            r"posted:?\s*([^<\n]+)"
        ]

        for indicator in pub_indicators:
            matches = re.findall(indicator, content, re.IGNORECASE)
            for match in matches:
                date = self._parse_date_string(match.strip())
                if date:
                    return date

        # Fallback: look for any date pattern
        for pattern in self.date_patterns:
            matches = pattern.findall(content)
            for match in matches:
                date = self._parse_date_string(match)
                if date and date.year >= 1990:  # Reasonable cutoff
                    return date

        return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        date_str = date_str.strip()

        # Try various formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%B %d, %Y",
            "%B %d %Y",
            "%d %b %Y",
            "%d %B %Y"
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue

        return None

    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for change detection"""
        # Normalize content (remove extra whitespace, etc.)
        normalized = re.sub(r'\s+', ' ', content.strip())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


class EnhancedEvidenceManager:
    """Manager for enhanced evidence with temporal tracking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pid_extractor = PersistentIdentifierExtractor()
        self.temporal_extractor = TemporalMetadataExtractor()

        # Evidence change tracking
        self.change_history = {}

    def create_enhanced_evidence(self, source_data: Dict[str, Any],
                               content: str = "", headers: Dict[str, str] = None) -> EnhancedEvidence:
        """Create enhanced evidence from source data"""
        headers = headers or {}

        # Extract basic fields
        evidence = EnhancedEvidence(
            source_id=source_data.get("source_id", ""),
            canonical_url=source_data.get("canonical_url", ""),
            title=source_data.get("title", ""),
            snippet=source_data.get("snippet", ""),
            content=content,
            relevance_score=source_data.get("score", 0.0),
            domain=self._extract_domain(source_data.get("canonical_url", "")),
            source_type=self._determine_source_type(source_data.get("canonical_url", ""))
        )

        # Extract persistent identifiers
        url_ids = self.pid_extractor.extract_from_url(evidence.canonical_url)
        content_ids = self.pid_extractor.extract_from_content(content, evidence.title)

        # Merge identifiers (URL takes precedence)
        evidence.persistent_ids = self._merge_persistent_ids(url_ids, content_ids)

        # Extract temporal metadata
        header_temporal = self.temporal_extractor.extract_from_headers(headers)
        content_temporal = self.temporal_extractor.extract_from_content(content, evidence.canonical_url)

        # Merge temporal metadata
        evidence.temporal_metadata = self._merge_temporal_metadata(header_temporal, content_temporal)

        # Calculate source authority
        evidence.source_authority = self._calculate_source_authority(evidence)

        self.logger.debug(f"Created enhanced evidence: {evidence.source_id}")
        return evidence

    def track_evidence_changes(self, evidence: EnhancedEvidence,
                             previous_evidence: Optional[EnhancedEvidence] = None) -> Dict[str, Any]:
        """Track changes in evidence content"""
        changes = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_id": evidence.source_id,
            "changes_detected": False,
            "change_details": {}
        }

        if previous_evidence is None:
            # First time seeing this evidence
            changes["change_details"]["status"] = "new_evidence"
            self.change_history[evidence.source_id] = evidence.temporal_metadata.snapshot_hash
            return changes

        # Compare snapshot hashes
        current_hash = evidence.temporal_metadata.snapshot_hash
        previous_hash = previous_evidence.temporal_metadata.snapshot_hash

        if current_hash != previous_hash:
            changes["changes_detected"] = True
            changes["change_details"] = {
                "content_changed": True,
                "previous_hash": previous_hash,
                "current_hash": current_hash,
                "requires_reverification": True
            }

            # Update history
            self.change_history[evidence.source_id] = current_hash

            self.logger.info(f"Content change detected for evidence: {evidence.source_id}")

        # Check for temporal updates
        if (evidence.temporal_metadata.last_modified and
            previous_evidence.temporal_metadata.last_modified and
            evidence.temporal_metadata.last_modified > previous_evidence.temporal_metadata.last_modified):

            changes["changes_detected"] = True
            changes["change_details"]["last_modified_updated"] = True

        return changes

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def _determine_source_type(self, url: str) -> str:
        """Determine source type from URL"""
        domain = self._extract_domain(url)

        # Academic sources
        if any(x in domain for x in ["arxiv.org", "pubmed", "springer", "nature.com", "science.org"]):
            return "academic"

        # Legal sources
        if any(x in domain for x in ["sec.gov", "eur-lex.europa.eu", "courtlistener.com"]):
            return "legal"

        # Archive sources
        if any(x in domain for x in ["web.archive.org", "archive.org"]):
            return "archive"

        # News sources
        if any(x in domain for x in ["reuters.com", "bbc.com", "cnn.com", "nytimes.com"]):
            return "news"

        return "web"

    def _merge_persistent_ids(self, url_ids: PersistentIdentifiers,
                            content_ids: PersistentIdentifiers) -> PersistentIdentifiers:
        """Merge persistent IDs with URL taking precedence"""
        merged = PersistentIdentifiers()

        for field in ["doi", "ecli", "cik", "docket", "pmid", "arxiv_id", "issn", "isbn", "orcid"]:
            url_value = getattr(url_ids, field)
            content_value = getattr(content_ids, field)

            # URL value takes precedence
            final_value = url_value if url_value else content_value
            setattr(merged, field, final_value)

        return merged

    def _merge_temporal_metadata(self, header_meta: TemporalMetadata,
                               content_meta: TemporalMetadata) -> TemporalMetadata:
        """Merge temporal metadata from headers and content"""
        merged = TemporalMetadata()

        # Header data takes precedence for server-provided timestamps
        merged.last_modified = header_meta.last_modified
        merged.access_date = header_meta.access_date
        merged.content_version = header_meta.content_version

        # Content data for publication info and hashes
        merged.publication_date = content_meta.publication_date
        merged.snapshot_hash = content_meta.snapshot_hash
        merged.wayback_timestamp = content_meta.wayback_timestamp
        merged.memento_datetime = content_meta.memento_datetime

        return merged

    def _calculate_source_authority(self, evidence: EnhancedEvidence) -> float:
        """Calculate source authority score"""
        authority = 0.5  # Base score

        # Domain-based authority
        domain_scores = {
            "nature.com": 0.95,
            "science.org": 0.95,
            "nejm.org": 0.9,
            "arxiv.org": 0.8,
            "pubmed.ncbi.nlm.nih.gov": 0.85,
            "sec.gov": 0.9,
            "eur-lex.europa.eu": 0.9,
            "reuters.com": 0.8,
            "bbc.com": 0.8,
            "wikipedia.org": 0.7
        }

        for domain, score in domain_scores.items():
            if domain in evidence.domain:
                authority = max(authority, score)
                break

        # Boost for persistent identifiers
        if evidence.persistent_ids.doi:
            authority += 0.1
        if evidence.persistent_ids.pmid:
            authority += 0.1

        # Boost for recent content
        if evidence.temporal_metadata.publication_date:
            years_old = (datetime.now(timezone.utc) - evidence.temporal_metadata.publication_date).days / 365
            if years_old < 1:
                authority += 0.05
            elif years_old > 10:
                authority -= 0.05

        return min(1.0, max(0.1, authority))


def create_enhanced_evidence_manager() -> EnhancedEvidenceManager:
    """Factory function for enhanced evidence manager"""
    return EnhancedEvidenceManager()
