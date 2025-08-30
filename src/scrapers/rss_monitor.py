#!/usr/bin/env python3
"""
RSS Monitor for Deep Research Tool
Real-time monitoring of RSS feeds for alternative media and research sources

Author: Advanced IT Specialist
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import feedparser
import aiohttp
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class RSSMonitor:
    """RSS feed monitoring for alternative media and research sources"""

    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize RSS monitor

        Args:
            rate_limit: Requests per second limit
        """
        self.rate_limit = rate_limit
        self.default_feeds = [
            # Alternative media
            "https://feeds.feedburner.com/zerohedge/feed",
            "https://www.activistpost.com/feed",
            "https://www.globalresearch.ca/rss",
            # Technology and research
            "https://feeds.feedburner.com/oreilly/radar",
            "https://arxiv.org/rss/cs.AI",
            "https://arxiv.org/rss/cs.CL",
            # News sources
            "https://rss.cnn.com/rss/edition.rss",
            "http://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.reuters.com/reuters/topNews",
        ]

    async def search_async(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search RSS feeds for topic-related content

        Args:
            topic: Search topic
            time_range: Optional time range for filtering

        Returns:
            List of relevant articles
        """
        results = []
        search_keywords = self._extract_keywords(topic)

        async with aiohttp.ClientSession() as session:
            for feed_url in self.default_feeds:
                try:
                    articles = await self._process_feed(
                        session, feed_url, search_keywords, time_range
                    )
                    results.extend(articles)

                    # Rate limiting
                    await asyncio.sleep(1.0 / self.rate_limit)

                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}: {e}")
                    continue

        # Sort by relevance and date
        results.sort(
            key=lambda x: (x.get("relevance_score", 0), x.get("date", datetime.min)), reverse=True
        )

        logger.info(f"RSS monitor found {len(results)} relevant articles")
        return results[:50]  # Limit results

    def search(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract search keywords from topic"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "a",
            "an",
        }
        words = re.findall(r"\b\w{3,}\b", topic.lower())
        keywords = [word for word in words if word not in stop_words]

        # Add the full topic as a phrase
        keywords.insert(0, topic.lower())

        return keywords[:10]

    async def _process_feed(
        self,
        session: aiohttp.ClientSession,
        feed_url: str,
        keywords: List[str],
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> List[Dict[str, Any]]:
        """Process individual RSS feed"""
        try:
            async with session.get(feed_url, timeout=30) as response:
                if response.status != 200:
                    return []

                content = await response.text()
                feed = feedparser.parse(content)

                articles = []
                for entry in feed.entries:
                    article = await self._process_entry(session, entry, keywords, time_range)
                    if article:
                        articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return []

    async def _process_entry(
        self,
        session: aiohttp.ClientSession,
        entry,
        keywords: List[str],
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> Optional[Dict[str, Any]]:
        """Process individual RSS entry"""
        try:
            # Extract basic information
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            # Parse publication date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])

            # Check time range
            if time_range and pub_date:
                if not (time_range[0] <= pub_date <= time_range[1]):
                    return None

            # Calculate relevance score
            relevance_score = self._calculate_relevance(title + " " + summary, keywords)

            # Skip if not relevant enough
            if relevance_score < 0.1:
                return None

            # Try to get full article content
            full_content = await self._fetch_full_article(session, link)
            content = full_content if full_content else summary

            return {
                "title": title,
                "content": content,
                "summary": summary,
                "url": link,
                "date": pub_date or datetime.now(),
                "source": "rss_feed",
                "source_type": "news",
                "relevance_score": relevance_score,
                "metadata": {
                    "feed_url": entry.get("id", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.term for tag in entry.get("tags", [])],
                    "content_length": len(content),
                },
            }

        except Exception as e:
            logger.error(f"Error processing RSS entry: {e}")
            return None

    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        text_lower = text.lower()
        total_score = 0.0

        for i, keyword in enumerate(keywords):
            # Weight earlier keywords more heavily
            weight = 1.0 / (i + 1)

            # Count occurrences
            count = text_lower.count(keyword.lower())
            if count > 0:
                # Logarithmic scoring to prevent keyword stuffing dominance
                score = weight * min(1.0, count / 5.0)
                total_score += score

        return min(1.0, total_score)

    async def _fetch_full_article(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        try:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    return None

                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer", "aside", "ad"]):
                    element.decompose()

                # Try to find main content
                main_content = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find("div", class_=re.compile(r"content|article|post"))
                    or soup.find("div", id=re.compile(r"content|article|post"))
                )

                if main_content:
                    text = main_content.get_text(separator=" ", strip=True)
                    # Clean up whitespace
                    text = re.sub(r"\s+", " ", text)
                    return text[:5000]  # Limit length

        except Exception as e:
            logger.debug(f"Could not fetch full article from {url}: {e}")

        return None

    async def monitor_feeds_realtime(self, keywords: List[str], callback) -> None:
        """Monitor feeds in real-time for new content"""
        seen_articles = set()

        while True:
            try:
                for feed_url in self.default_feeds:
                    async with aiohttp.ClientSession() as session:
                        articles = await self._process_feed(session, feed_url, keywords, None)

                        for article in articles:
                            article_id = article.get("url", "") + article.get("title", "")
                            if article_id not in seen_articles:
                                seen_articles.add(article_id)
                                await callback(article)

                        await asyncio.sleep(1.0 / self.rate_limit)

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    def add_custom_feed(self, feed_url: str):
        """Add custom RSS feed to monitoring list"""
        if feed_url not in self.default_feeds:
            self.default_feeds.append(feed_url)
            logger.info(f"Added custom feed: {feed_url}")

    def remove_feed(self, feed_url: str):
        """Remove RSS feed from monitoring list"""
        if feed_url in self.default_feeds:
            self.default_feeds.remove(feed_url)
            logger.info(f"Removed feed: {feed_url}")

    async def validate_feed(self, feed_url: str) -> bool:
        """Validate if RSS feed is accessible and parseable"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, timeout=15) as response:
                    if response.status != 200:
                        return False

                    content = await response.text()
                    feed = feedparser.parse(content)

                    # Check if feed has entries
                    return len(feed.entries) > 0

        except Exception as e:
            logger.error(f"Error validating feed {feed_url}: {e}")
            return False
