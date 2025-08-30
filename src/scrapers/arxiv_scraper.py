#!/usr/bin/env python3
"""ArXiv Scraper for Deep Research Tool
Academic paper research with advanced filtering and analysis

Author: Advanced IT Specialist
"""

import asyncio
from datetime import datetime
import logging
import re
from typing import Any

import arxiv

logger = logging.getLogger(__name__)


class ArxivScraper:
    """Advanced scraper for ArXiv academic papers"""

    def __init__(self, rate_limit: float = 0.5, max_results: int = 100):
        """Initialize ArXiv scraper

        Args:
            rate_limit: Requests per second limit
            max_results: Maximum results per search

        """
        self.rate_limit = rate_limit
        self.max_results = max_results
        self.client = arxiv.Client()

        # ArXiv categories relevant to research
        self.relevant_categories = [
            "cs.AI",
            "cs.CL",
            "cs.IR",
            "cs.CY",
            "cs.SI",
            "stat.ML",
            "physics.soc-ph",
            "econ.GN",
            "q-bio",
        ]

    async def search_async(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Asynchronous search for academic papers

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of academic papers

        """
        results = []

        # Construct search query
        search_query = self._build_search_query(topic, time_range)

        try:
            # Execute search
            search = arxiv.Search(
                query=search_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Process results
            for paper in self.client.results(search):
                paper_data = await self._process_paper(paper)
                if paper_data:
                    results.append(paper_data)

                # Rate limiting
                await asyncio.sleep(1.0 / self.rate_limit)

        except Exception as e:
            logger.error(f"Error in ArXiv search: {e}")

        logger.info(f"ArXiv search found {len(results)} academic papers")
        return results

    def search(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    def _build_search_query(
        self, topic: str, time_range: tuple[datetime, datetime] | None
    ) -> str:
        """Build ArXiv search query with advanced filters"""
        # Clean and prepare search terms
        search_terms = self._extract_search_terms(topic)

        # Build query components
        query_parts = []

        # Main search in title, abstract, and comments
        main_query = " OR ".join([f'ti:"{term}"' for term in search_terms])
        main_query += " OR " + " OR ".join([f'abs:"{term}"' for term in search_terms])
        query_parts.append(f"({main_query})")

        # Add category filters
        category_filter = " OR ".join([f"cat:{cat}" for cat in self.relevant_categories])
        query_parts.append(f"({category_filter})")

        # Combine with AND
        query = " AND ".join(query_parts)

        return query

    def _extract_search_terms(self, topic: str) -> list[str]:
        """Extract meaningful search terms from topic"""
        # Remove common stop words and split
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r"\b\w{3,}\b", topic.lower())
        meaningful_words = [word for word in words if word not in stop_words]

        # Add the full topic as a phrase
        terms = [topic.strip()]
        terms.extend(meaningful_words)

        return terms[:10]  # Limit to prevent overly complex queries

    async def _process_paper(self, paper: arxiv.Result) -> dict[str, Any] | None:
        """Process individual ArXiv paper"""
        try:
            # Extract authors
            authors = [str(author) for author in paper.authors]

            # Extract categories
            categories = [str(cat) for cat in paper.categories]

            # Get full text if possible
            pdf_url = paper.pdf_url
            full_text = await self._extract_pdf_text(pdf_url) if pdf_url else ""

            # Combine abstract and full text
            content = paper.summary
            if full_text:
                content += "\n\n" + full_text[:5000]  # Limit full text length

            return {
                "title": paper.title,
                "content": content,
                "url": paper.entry_id,
                "pdf_url": pdf_url,
                "date": paper.published,
                "source": "arxiv",
                "source_type": "academic",
                "metadata": {
                    "authors": authors,
                    "categories": categories,
                    "arxiv_id": paper.get_short_id(),
                    "updated": paper.updated,
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                    "comment": paper.comment,
                    "primary_category": (
                        str(paper.primary_category) if paper.primary_category else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error processing paper {paper.entry_id}: {e}")
            return None

    async def _extract_pdf_text(self, pdf_url: str) -> str:
        """Extract text from PDF (simplified implementation)"""
        try:
            # For production, you'd want to implement actual PDF text extraction
            # using libraries like PyPDF2, pdfplumber, or similar
            # For now, return empty string to avoid complexity
            return ""
        except Exception as e:
            logger.error(f"Error extracting PDF text from {pdf_url}: {e}")
            return ""

    async def search_by_category(
        self, category: str, max_results: int = 50
    ) -> list[dict[str, Any]]:
        """Search papers by specific ArXiv category"""
        try:
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for paper in self.client.results(search):
                paper_data = await self._process_paper(paper)
                if paper_data:
                    results.append(paper_data)
                await asyncio.sleep(1.0 / self.rate_limit)

            return results

        except Exception as e:
            logger.error(f"Error searching by category {category}: {e}")
            return []

    async def search_by_author(self, author_name: str) -> list[dict[str, Any]]:
        """Search papers by specific author"""
        try:
            search = arxiv.Search(
                query=f'au:"{author_name}"',
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for paper in self.client.results(search):
                paper_data = await self._process_paper(paper)
                if paper_data:
                    results.append(paper_data)
                await asyncio.sleep(1.0 / self.rate_limit)

            return results

        except Exception as e:
            logger.error(f"Error searching by author {author_name}: {e}")
            return []
