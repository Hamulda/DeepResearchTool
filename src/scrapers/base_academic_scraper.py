#!/usr/bin/env python3
"""BASE Bielefeld Academic Search Engine Integration
Access to 150+ million documents from 7000+ Deep Web sources

Author: Advanced IT Specialist
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import re
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AcademicDocument:
    """Represents an academic document from BASE"""

    document_id: str
    title: str
    authors: list[str]
    abstract: str
    publication_date: datetime | None
    publisher: str
    source_repository: str
    document_type: str  # article, thesis, book, conference_paper, etc.
    subjects: list[str]
    languages: list[str]
    doi: str | None
    url: str
    full_text_url: str | None
    citation_count: int = 0
    quality_score: float = 0.0
    access_rights: str = ""  # open_access, restricted, subscription
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Structured search query for BASE"""

    keywords: list[str]
    authors: list[str] = field(default_factory=list)
    title_keywords: list[str] = field(default_factory=list)
    subject_areas: list[str] = field(default_factory=list)
    date_range: tuple[datetime, datetime] | None = None
    document_types: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    access_type: str | None = None  # open_access, any
    repositories: list[str] = field(default_factory=list)


class BASESearchEngine:
    """Advanced integration with BASE Bielefeld Academic Search Engine"""

    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://api.base-search.net"
        self.oai_url = "https://oai.base-search.net/oai"
        self.search_url = "https://www.base-search.net/Search/Results"

        # Document type mappings
        self.document_types = {
            "article": "Articles",
            "thesis": "Theses",
            "book": "Books",
            "conference": "Conference Papers",
            "report": "Reports",
            "patent": "Patents",
            "dataset": "Research Data",
            "software": "Software",
        }

        # Subject classification (DDC - Dewey Decimal Classification)
        self.subject_areas = {
            "computer_science": ["004", "005", "006"],
            "mathematics": ["510", "511", "512", "513", "514", "515"],
            "physics": ["530", "531", "532", "533", "534", "535"],
            "chemistry": ["540", "541", "542", "543", "544", "545"],
            "biology": ["570", "571", "572", "573", "574", "575"],
            "medicine": ["610", "611", "612", "613", "614", "615"],
            "engineering": ["620", "621", "622", "623", "624", "625"],
            "economics": ["330", "331", "332", "333", "334", "335"],
            "sociology": ["301", "302", "303", "304", "305", "306"],
            "psychology": ["150", "152", "153", "154", "155", "158"],
            "history": ["900", "901", "902", "903", "904", "905"],
            "philosophy": ["100", "101", "102", "103", "104", "105"],
        }

        # Repository quality scores
        self.repository_scores = {
            "arxiv.org": 0.9,
            "pubmed": 0.95,
            "ieee": 0.9,
            "acm": 0.85,
            "springer": 0.85,
            "wiley": 0.8,
            "elsevier": 0.8,
            "plos": 0.85,
            "biomedcentral": 0.8,
            "nature": 0.95,
            "science": 0.95,
            "mit": 0.9,
            "harvard": 0.9,
            "stanford": 0.9,
        }

        # Language codes
        self.language_codes = {
            "english": "eng",
            "german": "ger",
            "french": "fre",
            "spanish": "spa",
            "italian": "ita",
            "russian": "rus",
            "chinese": "chi",
            "japanese": "jpn",
            "arabic": "ara",
        }

    async def search_comprehensive(
        self, query: SearchQuery, max_results: int = 1000, include_full_text: bool = True
    ) -> list[AcademicDocument]:
        """Comprehensive search across BASE database"""
        documents = []

        # Multiple search strategies
        search_strategies = [
            self._build_keyword_query(query),
            self._build_advanced_query(query),
            self._build_citation_query(query),
        ]

        for search_string in search_strategies:
            batch_docs = await self._execute_search(
                search_string, max_results // len(search_strategies), include_full_text
            )
            documents.extend(batch_docs)

            await self._rate_limit()

        # Remove duplicates and enhance
        unique_docs = self._deduplicate_documents(documents)
        enhanced_docs = await self._enhance_documents(unique_docs)

        # Sort by quality score
        enhanced_docs.sort(key=lambda x: x.quality_score, reverse=True)

        return enhanced_docs[:max_results]

    def _build_keyword_query(self, query: SearchQuery) -> str:
        """Build basic keyword search query"""
        query_parts = []

        # Main keywords
        if query.keywords:
            keywords_str = " OR ".join(f'"{kw}"' for kw in query.keywords)
            query_parts.append(f"({keywords_str})")

        # Author search
        if query.authors:
            authors_str = " OR ".join(f'author:"{author}"' for author in query.authors)
            query_parts.append(f"({authors_str})")

        # Title keywords
        if query.title_keywords:
            title_str = " OR ".join(f'title:"{kw}"' for kw in query.title_keywords)
            query_parts.append(f"({title_str})")

        # Subject areas
        if query.subject_areas:
            subject_codes = []
            for area in query.subject_areas:
                if area in self.subject_areas:
                    subject_codes.extend(self.subject_areas[area])

            if subject_codes:
                subject_str = " OR ".join(f'ddc:"{code}"' for code in subject_codes)
                query_parts.append(f"({subject_str})")

        return " AND ".join(query_parts) if query_parts else "*"

    def _build_advanced_query(self, query: SearchQuery) -> str:
        """Build advanced search query with filters"""
        base_query = self._build_keyword_query(query)
        filters = []

        # Date range filter
        if query.date_range:
            start_date = query.date_range[0].strftime("%Y-%m-%d")
            end_date = query.date_range[1].strftime("%Y-%m-%d")
            filters.append(f"dcdate:[{start_date} TO {end_date}]")

        # Document type filter
        if query.document_types:
            doc_types = " OR ".join(f'dctype:"{dt}"' for dt in query.document_types)
            filters.append(f"({doc_types})")

        # Language filter
        if query.languages:
            lang_codes = [self.language_codes.get(lang, lang) for lang in query.languages]
            lang_filter = " OR ".join(f'dclang:"{code}"' for code in lang_codes)
            filters.append(f"({lang_filter})")

        # Access type filter
        if query.access_type == "open_access":
            filters.append('dctype:"Open Access"')

        # Repository filter
        if query.repositories:
            repo_filter = " OR ".join(f'collname:"{repo}"' for repo in query.repositories)
            filters.append(f"({repo_filter})")

        if filters:
            return f'{base_query} AND {" AND ".join(filters)}'
        return base_query

    def _build_citation_query(self, query: SearchQuery) -> str:
        """Build query focused on highly cited papers"""
        base_query = self._build_keyword_query(query)

        # Add boost for high-impact repositories
        repo_boosts = []
        for repo, score in self.repository_scores.items():
            if score > 0.8:
                repo_boosts.append(f'collname:"{repo}"^{score}')

        if repo_boosts:
            boost_query = " OR ".join(repo_boosts)
            return f"({base_query}) AND ({boost_query})"
        return base_query

    async def _execute_search(
        self, query_string: str, max_results: int, include_full_text: bool
    ) -> list[AcademicDocument]:
        """Execute search against BASE API"""
        documents = []

        try:
            await self._rate_limit()

            # Prepare search parameters
            params = {
                "func": "find-b",
                "find_code": "WRD",
                "request": query_string,
                "format": "json",
                "start": "1",
                "count": str(min(max_results, 100)),  # API limit
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url + "/cgi-bin/BaseHttpSearchInterface.fcgi", params=params
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        documents = self._parse_search_results(data, include_full_text)
                    else:
                        logger.warning(f"BASE search failed with status {response.status}")
                        # Fallback to web scraping
                        documents = await self._fallback_web_search(query_string, max_results)

        except Exception as e:
            logger.error(f"Error executing BASE search: {e!s}")
            # Try fallback method
            documents = await self._fallback_web_search(query_string, max_results)

        return documents

    async def _fallback_web_search(self, query: str, max_results: int) -> list[AcademicDocument]:
        """Fallback web scraping method"""
        documents = []

        try:
            search_params = {
                "type": "all",
                "lookfor": query,
                "ling": "mix",
                "oaboost": "1",
                "newsearch": "1",
                "refid": "dcbasen",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_url, params=search_params) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        documents = self._parse_html_results(html_content)

        except Exception as e:
            logger.error(f"Fallback search failed: {e!s}")

        return documents[:max_results]

    def _parse_search_results(
        self, data: dict[str, Any], include_full_text: bool
    ) -> list[AcademicDocument]:
        """Parse JSON search results from BASE API"""
        documents = []

        try:
            if "response" in data and "docs" in data["response"]:
                for doc_data in data["response"]["docs"]:
                    document = self._create_document_from_json(doc_data, include_full_text)
                    if document:
                        documents.append(document)

        except Exception as e:
            logger.error(f"Error parsing search results: {e!s}")

        return documents

    def _parse_html_results(self, html_content: str) -> list[AcademicDocument]:
        """Parse HTML search results from BASE web interface"""
        documents = []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            # Find result entries
            result_entries = soup.find_all("div", class_=["result", "record"])

            for entry in result_entries:
                document = self._create_document_from_html(entry)
                if document:
                    documents.append(document)

        except Exception as e:
            logger.error(f"Error parsing HTML results: {e!s}")

        return documents

    def _create_document_from_json(
        self, doc_data: dict[str, Any], include_full_text: bool
    ) -> AcademicDocument | None:
        """Create AcademicDocument from JSON data"""
        try:
            # Extract basic information
            title = (
                doc_data.get("dctitle", ["Unknown Title"])[0]
                if isinstance(doc_data.get("dctitle"), list)
                else doc_data.get("dctitle", "Unknown Title")
            )

            authors = doc_data.get("dccreator", [])
            if isinstance(authors, str):
                authors = [authors]

            abstract = (
                doc_data.get("dcDescription", [""])[0]
                if isinstance(doc_data.get("dcDescription"), list)
                else doc_data.get("dcDescription", "")
            )

            # Parse publication date
            pub_date = None
            if "dcdate" in doc_data:
                date_str = (
                    doc_data["dcdate"][0]
                    if isinstance(doc_data["dcdate"], list)
                    else doc_data["dcdate"]
                )
                pub_date = self._parse_date(date_str)

            # Extract metadata
            publisher = (
                doc_data.get("dcpublisher", ["Unknown"])[0]
                if isinstance(doc_data.get("dcpublisher"), list)
                else doc_data.get("dcpublisher", "Unknown")
            )
            source_repo = (
                doc_data.get("collname", ["Unknown"])[0]
                if isinstance(doc_data.get("collname"), list)
                else doc_data.get("collname", "Unknown")
            )
            doc_type = (
                doc_data.get("dctype", ["Unknown"])[0]
                if isinstance(doc_data.get("dctype"), list)
                else doc_data.get("dctype", "Unknown")
            )

            subjects = doc_data.get("dcsubject", [])
            if isinstance(subjects, str):
                subjects = [subjects]

            languages = doc_data.get("dclang", [])
            if isinstance(languages, str):
                languages = [languages]

            # URLs
            url = (
                doc_data.get("dclink", [""])[0]
                if isinstance(doc_data.get("dclink"), list)
                else doc_data.get("dclink", "")
            )
            doi = (
                doc_data.get("dcdoi", [""])[0]
                if isinstance(doc_data.get("dcdoi"), list)
                else doc_data.get("dcdoi")
            )

            # Create document
            document = AcademicDocument(
                document_id=hashlib.md5(f"{title}{authors}".encode()).hexdigest()[:16],
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                publisher=publisher,
                source_repository=source_repo,
                document_type=doc_type,
                subjects=subjects,
                languages=languages,
                doi=doi,
                url=url,
                full_text_url=url if include_full_text else None,
            )

            return document

        except Exception as e:
            logger.error(f"Error creating document from JSON: {e!s}")
            return None

    def _create_document_from_html(self, entry) -> AcademicDocument | None:
        """Create AcademicDocument from HTML entry"""
        try:
            # Extract title
            title_elem = entry.find(["h2", "h3", "a"], class_=["title", "result-title"])
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"

            # Extract authors
            author_elem = entry.find(["span", "div"], class_=["author", "authors"])
            authors = []
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                authors = [auth.strip() for auth in author_text.split(";")]

            # Extract abstract/description
            desc_elem = entry.find(["p", "div"], class_=["description", "abstract"])
            abstract = desc_elem.get_text(strip=True) if desc_elem else ""

            # Extract URL
            url_elem = entry.find("a", href=True)
            url = url_elem["href"] if url_elem else ""

            # Extract other metadata
            metadata_elem = entry.find(["div", "span"], class_=["metadata", "details"])
            metadata_text = metadata_elem.get_text() if metadata_elem else ""

            # Parse metadata for additional information
            publisher = self._extract_from_metadata(metadata_text, ["Publisher:", "Source:"])
            source_repo = self._extract_from_metadata(metadata_text, ["Repository:", "Collection:"])
            doc_type = self._extract_from_metadata(metadata_text, ["Type:", "Document Type:"])

            document = AcademicDocument(
                document_id=hashlib.md5(f"{title}{authors}".encode()).hexdigest()[:16],
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=None,  # Would need more parsing
                publisher=publisher or "Unknown",
                source_repository=source_repo or "Unknown",
                document_type=doc_type or "Unknown",
                subjects=[],
                languages=[],
                doi=None,
                url=url,
                full_text_url=None,
            )

            return document

        except Exception as e:
            logger.error(f"Error creating document from HTML: {e!s}")
            return None

    def _extract_from_metadata(self, text: str, patterns: list[str]) -> str | None:
        """Extract specific information from metadata text"""
        for pattern in patterns:
            match = re.search(f"{pattern}\\s*([^\\n]+)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date from various formats"""
        if not date_str:
            return None

        # Common date formats
        formats = ["%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y", "%B %Y", "%B %d, %Y"]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # Extract year if full date parsing fails
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            try:
                year = int(year_match.group())
                return datetime(year, 1, 1)
            except ValueError:
                pass

        return None

    async def _enhance_documents(self, documents: list[AcademicDocument]) -> list[AcademicDocument]:
        """Enhance documents with additional metadata and scoring"""
        for doc in documents:
            # Calculate quality score
            doc.quality_score = self._calculate_quality_score(doc)

            # Determine access rights
            doc.access_rights = self._determine_access_rights(doc)

            # Extract additional metadata
            doc.metadata = await self._extract_additional_metadata(doc)

        return documents

    def _calculate_quality_score(self, document: AcademicDocument) -> float:
        """Calculate quality score for document"""
        score = 0.5  # Base score

        # Repository quality
        repo_lower = document.source_repository.lower()
        for repo, repo_score in self.repository_scores.items():
            if repo in repo_lower:
                score += (repo_score - 0.5) * 0.4
                break

        # DOI presence (indicates peer review)
        if document.doi:
            score += 0.1

        # Abstract quality
        if document.abstract and len(document.abstract) > 100:
            score += 0.1

        # Multiple authors (indicates collaboration)
        if len(document.authors) > 1:
            score += 0.05

        # Recent publication
        if document.publication_date:
            years_old = (datetime.now() - document.publication_date).days / 365.25
            if years_old < 5:
                score += 0.1
            elif years_old > 20:
                score -= 0.1

        # Subject classification
        if document.subjects:
            score += 0.05

        return min(1.0, max(0.0, score))

    def _determine_access_rights(self, document: AcademicDocument) -> str:
        """Determine access rights for document"""
        if "open access" in document.document_type.lower() or any(
            oa_indicator in document.url.lower() for oa_indicator in ["arxiv", "plos", "doaj"]
        ):
            return "open_access"
        if any(
            sub_indicator in document.publisher.lower()
            for sub_indicator in ["elsevier", "springer", "wiley"]
        ):
            return "subscription"
        return "unknown"

    async def _extract_additional_metadata(self, document: AcademicDocument) -> dict[str, Any]:
        """Extract additional metadata for document"""
        metadata = {
            "estimated_reading_time": max(5, len(document.abstract.split()) // 200),
            "complexity_score": self._assess_complexity(document),
            "interdisciplinary_score": self._assess_interdisciplinary(document),
            "practical_relevance": self._assess_practical_relevance(document),
        }

        return metadata

    def _assess_complexity(self, document: AcademicDocument) -> float:
        """Assess document complexity based on title and abstract"""
        complexity_indicators = [
            "algorithm",
            "mathematical",
            "theoretical",
            "quantum",
            "neural",
            "optimization",
            "statistical",
            "computational",
            "methodology",
        ]

        text = (document.title + " " + document.abstract).lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in text)

        return min(1.0, complexity_count / 5.0)

    def _assess_interdisciplinary(self, document: AcademicDocument) -> float:
        """Assess interdisciplinary nature of document"""
        if len(document.subjects) > 3:
            return 0.8
        if len(document.subjects) > 1:
            return 0.5
        return 0.2

    def _assess_practical_relevance(self, document: AcademicDocument) -> float:
        """Assess practical relevance of document"""
        practical_keywords = [
            "application",
            "implementation",
            "case study",
            "real-world",
            "practical",
            "empirical",
            "experiment",
            "evaluation",
        ]

        text = (document.title + " " + document.abstract).lower()
        practical_count = sum(1 for keyword in practical_keywords if keyword in text)

        return min(1.0, practical_count / 3.0)

    def _deduplicate_documents(self, documents: list[AcademicDocument]) -> list[AcademicDocument]:
        """Remove duplicate documents"""
        seen_docs = {}
        unique_docs = []

        for doc in documents:
            # Create a similarity key based on title and first author
            title_clean = re.sub(r"[^\w\s]", "", doc.title.lower())
            first_author = doc.authors[0] if doc.authors else "unknown"
            similarity_key = f"{title_clean}_{first_author.lower()}"

            if similarity_key not in seen_docs:
                seen_docs[similarity_key] = doc
                unique_docs.append(doc)
            else:
                # Merge information from duplicate
                existing_doc = seen_docs[similarity_key]
                if doc.doi and not existing_doc.doi:
                    existing_doc.doi = doc.doi
                if doc.full_text_url and not existing_doc.full_text_url:
                    existing_doc.full_text_url = doc.full_text_url

        return unique_docs

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

    def create_research_dashboard(self, documents: list[AcademicDocument]) -> dict[str, Any]:
        """Create comprehensive research dashboard"""
        dashboard = {
            "overview": {
                "total_documents": len(documents),
                "open_access_percentage": len(
                    [d for d in documents if d.access_rights == "open_access"]
                )
                / len(documents)
                * 100,
                "average_quality_score": sum(d.quality_score for d in documents) / len(documents),
                "date_range": self._get_date_range(documents),
            },
            "by_repository": self._analyze_by_repository(documents),
            "by_document_type": self._analyze_by_document_type(documents),
            "by_year": self._analyze_by_year(documents),
            "top_authors": self._analyze_top_authors(documents),
            "subject_distribution": self._analyze_subjects(documents),
            "quality_distribution": self._analyze_quality_distribution(documents),
            "recommendations": self._generate_recommendations(documents),
        }

        return dashboard

    def _get_date_range(self, documents: list[AcademicDocument]) -> dict[str, Any]:
        """Get date range of documents"""
        dates = [d.publication_date for d in documents if d.publication_date]
        if dates:
            return {
                "earliest": min(dates),
                "latest": max(dates),
                "span_years": (max(dates) - min(dates)).days / 365.25,
            }
        return {}

    def _analyze_by_repository(self, documents: list[AcademicDocument]) -> dict[str, Any]:
        """Analyze documents by repository"""
        repo_stats = {}
        for doc in documents:
            repo = doc.source_repository
            if repo not in repo_stats:
                repo_stats[repo] = {"count": 0, "quality_scores": []}
            repo_stats[repo]["count"] += 1
            repo_stats[repo]["quality_scores"].append(doc.quality_score)

        # Calculate averages
        for repo, stats in repo_stats.items():
            stats["avg_quality"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])

        return dict(sorted(repo_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10])

    def _analyze_by_document_type(self, documents: list[AcademicDocument]) -> dict[str, int]:
        """Analyze documents by type"""
        type_counts = {}
        for doc in documents:
            doc_type = doc.document_type
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_by_year(self, documents: list[AcademicDocument]) -> dict[int, int]:
        """Analyze documents by publication year"""
        year_counts = {}
        for doc in documents:
            if doc.publication_date:
                year = doc.publication_date.year
                year_counts[year] = year_counts.get(year, 0) + 1

        return dict(sorted(year_counts.items()))

    def _analyze_top_authors(self, documents: list[AcademicDocument]) -> dict[str, Any]:
        """Analyze top authors"""
        author_stats = {}
        for doc in documents:
            for author in doc.authors:
                if author not in author_stats:
                    author_stats[author] = {"count": 0, "quality_scores": []}
                author_stats[author]["count"] += 1
                author_stats[author]["quality_scores"].append(doc.quality_score)

        # Calculate averages and sort
        for author, stats in author_stats.items():
            stats["avg_quality"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])

        top_authors = sorted(author_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:20]
        return dict(top_authors)

    def _analyze_subjects(self, documents: list[AcademicDocument]) -> dict[str, int]:
        """Analyze subject distribution"""
        subject_counts = {}
        for doc in documents:
            for subject in doc.subjects:
                subject_counts[subject] = subject_counts.get(subject, 0) + 1

        return dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:20])

    def _analyze_quality_distribution(self, documents: list[AcademicDocument]) -> dict[str, int]:
        """Analyze quality score distribution"""
        quality_ranges = {
            "excellent (0.8-1.0)": 0,
            "good (0.6-0.8)": 0,
            "average (0.4-0.6)": 0,
            "poor (0.0-0.4)": 0,
        }

        for doc in documents:
            score = doc.quality_score
            if score >= 0.8:
                quality_ranges["excellent (0.8-1.0)"] += 1
            elif score >= 0.6:
                quality_ranges["good (0.6-0.8)"] += 1
            elif score >= 0.4:
                quality_ranges["average (0.4-0.6)"] += 1
            else:
                quality_ranges["poor (0.0-0.4)"] += 1

        return quality_ranges

    def _generate_recommendations(self, documents: list[AcademicDocument]) -> list[str]:
        """Generate research recommendations"""
        recommendations = []

        # Quality-based recommendations
        high_quality_docs = [d for d in documents if d.quality_score > 0.8]
        if len(high_quality_docs) > 5:
            recommendations.append(
                f"Focus on {len(high_quality_docs)} high-quality documents (score > 0.8)"
            )

        # Open access recommendations
        open_access_docs = [d for d in documents if d.access_rights == "open_access"]
        if len(open_access_docs) > len(documents) * 0.3:
            recommendations.append(
                f"Good open access coverage: {len(open_access_docs)} documents freely available"
            )

        # Recent research recommendations
        recent_docs = [
            d
            for d in documents
            if d.publication_date and (datetime.now() - d.publication_date).days < 365 * 2
        ]
        if len(recent_docs) > 10:
            recommendations.append(
                f"Strong recent research base: {len(recent_docs)} documents from last 2 years"
            )

        # Interdisciplinary recommendations
        interdisciplinary_docs = [
            d for d in documents if d.metadata.get("interdisciplinary_score", 0) > 0.5
        ]
        if len(interdisciplinary_docs) > 5:
            recommendations.append(
                f"Consider interdisciplinary approaches: {len(interdisciplinary_docs)} cross-domain documents found"
            )

        return recommendations
