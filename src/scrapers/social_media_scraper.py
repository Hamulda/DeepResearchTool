#!/usr/bin/env python3
"""
Social Media & Alternative Platforms Scraper
Ethical scraping of public social media and alternative platforms for research

Author: Advanced IT Specialist
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin, quote
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


class SocialMediaScraper:
    """Ethical scraper for public social media and alternative platforms"""

    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize Social Media Scraper

        Args:
            rate_limit: Conservative rate limit for social platforms
        """
        self.rate_limit = rate_limit
        self.ua = UserAgent()

        # Alternative and decentralized platforms
        self.platforms = {
            "gab": {
                "name": "Gab",
                "base_url": "https://gab.com",
                "search_endpoint": "/search?q={query}",
                "type": "alternative_social",
                "public_only": True,
            },
            "minds": {
                "name": "Minds",
                "base_url": "https://www.minds.com",
                "search_endpoint": "/search?q={query}",
                "type": "blockchain_social",
                "public_only": True,
            },
            "parler": {
                "name": "Parler",
                "base_url": "https://parler.com",
                "search_endpoint": "/search?searchterm={query}",
                "type": "alternative_social",
                "public_only": True,
            },
            "telegram_public": {
                "name": "Telegram Public Channels",
                "base_url": "https://t.me",
                "search_endpoint": "/s/{query}",
                "type": "messaging_public",
                "public_only": True,
            },
            "reddit": {
                "name": "Reddit",
                "base_url": "https://www.reddit.com",
                "search_endpoint": "/search?q={query}&sort=relevance",
                "type": "forum_aggregator",
                "public_only": True,
            },
            "voat_archive": {
                "name": "Voat Archive",
                "base_url": "https://archive.today",
                "search_endpoint": "/search?q=site:voat.co+{query}",
                "type": "archived_forum",
                "public_only": True,
            },
            "bitchute": {
                "name": "BitChute",
                "base_url": "https://www.bitchute.com",
                "search_endpoint": "/search?query={query}",
                "type": "video_platform",
                "public_only": True,
            },
            "rumble": {
                "name": "Rumble",
                "base_url": "https://rumble.com",
                "search_endpoint": "/search/video?q={query}",
                "type": "video_platform",
                "public_only": True,
            },
            "odysee": {
                "name": "Odysee",
                "base_url": "https://odysee.com",
                "search_endpoint": "/$/search?q={query}",
                "type": "blockchain_video",
                "public_only": True,
            },
            "mastodon_public": {
                "name": "Mastodon Public Timeline",
                "base_url": "https://mastodon.social",
                "search_endpoint": "/api/v2/search?q={query}&type=statuses",
                "type": "federated_social",
                "public_only": True,
            },
        }

        # Content type patterns for filtering
        self.content_patterns = {
            "conspiracy": [
                r"\b(conspiracy|cover-up|false flag|deep state|illuminati)\b",
                r"\b(nwo|new world order|agenda|globalist)\b",
                r"\b(truth|wake up|sheep|redpill|awakening)\b",
            ],
            "leaked": [
                r"\b(leaked|whistleblower|insider|source)\b",
                r"\b(classified|secret|confidential)\b",
                r"\b(exposed|revealed|uncovered)\b",
            ],
            "research": [
                r"\b(research|study|investigation|analysis)\b",
                r"\b(evidence|proof|documentation|facts)\b",
                r"\b(archive|database|collection)\b",
            ],
        }

    async def search_async(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search across alternative social media platforms

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of social media posts and content
        """
        results = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # Search each platform
            for platform_id, platform_config in self.platforms.items():
                try:
                    logger.info(f"Searching {platform_config['name']}...")
                    platform_results = await self._search_platform(
                        session, platform_id, platform_config, topic, time_range
                    )
                    results.extend(platform_results)

                    # Rate limiting
                    await asyncio.sleep(1.0 / self.rate_limit)

                except Exception as e:
                    logger.error(f"Error searching {platform_config['name']}: {e}")
                    continue

        # Filter and sort results
        filtered_results = self._filter_results(results, topic)
        filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        logger.info(f"Social Media Scraper found {len(filtered_results)} posts")
        return filtered_results[:75]  # Limit results

    def search(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    async def _search_platform(
        self,
        session: aiohttp.ClientSession,
        platform_id: str,
        platform_config: Dict[str, Any],
        topic: str,
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> List[Dict[str, Any]]:
        """Search individual platform"""
        results = []

        try:
            # Construct search URL
            base_url = platform_config["base_url"]
            search_endpoint = platform_config["search_endpoint"]
            search_url = base_url + search_endpoint.format(query=quote(topic))

            headers = {
                "User-Agent": self.ua.random,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Referer": base_url,
            }

            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    platform_results = await self._parse_platform_results(
                        content, platform_id, platform_config, topic, base_url
                    )
                    results.extend(platform_results)

        except Exception as e:
            logger.debug(f"Failed to search {platform_id}: {e}")

        return results

    async def _parse_platform_results(
        self,
        html_content: str,
        platform_id: str,
        platform_config: Dict[str, Any],
        topic: str,
        base_url: str,
    ) -> List[Dict[str, Any]]:
        """Parse platform-specific search results"""
        results = []

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            platform_type = platform_config["type"]

            if platform_type == "alternative_social":
                results = self._parse_alternative_social(soup, platform_config, topic, base_url)
            elif platform_type == "video_platform":
                results = self._parse_video_platform(soup, platform_config, topic, base_url)
            elif platform_type == "forum_aggregator":
                results = self._parse_forum_aggregator(soup, platform_config, topic, base_url)
            elif platform_type == "federated_social":
                results = self._parse_federated_social(soup, platform_config, topic, base_url)
            else:
                results = self._parse_generic_social(soup, platform_config, topic, base_url)

        except Exception as e:
            logger.debug(f"Failed to parse {platform_id} results: {e}")

        return results

    def _parse_alternative_social(
        self, soup: BeautifulSoup, platform_config: Dict[str, Any], topic: str, base_url: str
    ) -> List[Dict[str, Any]]:
        """Parse alternative social media platforms (Gab, Minds, etc.)"""
        results = []

        # Look for post containers
        post_selectors = [
            'div[class*="post"]',
            'div[class*="status"]',
            'div[class*="activity"]',
            "article",
            'div[class*="item"]',
            'div[class*="content"]',
        ]

        posts = []
        for selector in post_selectors:
            posts.extend(soup.select(selector)[:20])

        for post in posts:
            try:
                # Extract post text
                text_elem = post.find(["p", "div", "span"], string=True)
                if not text_elem:
                    continue

                post_text = post.get_text(strip=True)
                if len(post_text) < 10:
                    continue

                # Extract author
                author_elem = post.find(
                    ["span", "div", "a"], class_=re.compile(r"author|user|name")
                )
                author = author_elem.get_text(strip=True) if author_elem else "Unknown"

                # Extract timestamp
                time_elem = post.find(["time", "span"], class_=re.compile(r"time|date"))
                timestamp = self._parse_timestamp(time_elem.get_text() if time_elem else "")

                # Calculate relevance
                relevance = self._calculate_social_relevance(post_text, topic)

                if relevance > 0.2:
                    # Try to find post link
                    post_link = base_url
                    link_elem = post.find("a", href=True)
                    if link_elem:
                        href = link_elem.get("href")
                        if href:
                            post_link = urljoin(base_url, href)

                    results.append(
                        {
                            "title": f"Post by {author}",
                            "content": post_text,
                            "url": post_link,
                            "date": timestamp,
                            "relevance_score": relevance,
                            "metadata": {
                                "platform": platform_config["name"],
                                "author": author,
                                "post_type": "social_post",
                                "content_length": len(post_text),
                            },
                        }
                    )

            except Exception as e:
                continue

        return results

    def _parse_video_platform(
        self, soup: BeautifulSoup, platform_config: Dict[str, Any], topic: str, base_url: str
    ) -> List[Dict[str, Any]]:
        """Parse video platforms (BitChute, Rumble, Odysee)"""
        results = []

        # Look for video containers
        video_containers = soup.find_all(["div", "article"], limit=15)

        for container in video_containers:
            try:
                # Look for video title
                title_elem = container.find(
                    ["h3", "h4", "a", "span"], class_=re.compile(r"title|name")
                )
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                if len(title) < 5:
                    continue

                # Look for description
                desc_elem = container.find(["p", "div"], class_=re.compile(r"desc|summary"))
                description = desc_elem.get_text(strip=True) if desc_elem else ""

                # Look for channel/author
                author_elem = container.find(
                    ["span", "a"], class_=re.compile(r"channel|author|uploader")
                )
                author = author_elem.get_text(strip=True) if author_elem else "Unknown"

                # Calculate relevance
                content = title + " " + description
                relevance = self._calculate_social_relevance(content, topic)

                if relevance > 0.25:
                    # Try to find video link
                    video_link = base_url
                    link_elem = container.find("a", href=True)
                    if link_elem:
                        href = link_elem.get("href")
                        if href:
                            video_link = urljoin(base_url, href)

                    results.append(
                        {
                            "title": title,
                            "content": description or title,
                            "url": video_link,
                            "date": datetime.now(),
                            "relevance_score": relevance,
                            "metadata": {
                                "platform": platform_config["name"],
                                "author": author,
                                "content_type": "video",
                                "description_length": len(description),
                            },
                        }
                    )

            except Exception as e:
                continue

        return results

    def _parse_forum_aggregator(
        self, soup: BeautifulSoup, platform_config: Dict[str, Any], topic: str, base_url: str
    ) -> List[Dict[str, Any]]:
        """Parse forum aggregators (Reddit)"""
        results = []

        # Look for post entries
        post_containers = soup.find_all(["div", "article"], limit=20)

        for container in post_containers:
            try:
                # Look for post title
                title_elem = container.find(["h3", "h2", "a"], class_=re.compile(r"title"))
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                if len(title) < 5:
                    continue

                # Look for subreddit/community
                community_elem = container.find(
                    ["a", "span"], class_=re.compile(r"subreddit|community")
                )
                community = community_elem.get_text(strip=True) if community_elem else ""

                # Look for score/upvotes
                score_elem = container.find(["span", "div"], class_=re.compile(r"score|vote|point"))
                score = score_elem.get_text(strip=True) if score_elem else "0"

                # Calculate relevance
                relevance = self._calculate_social_relevance(title, topic)

                if relevance > 0.3:
                    # Try to find post link
                    post_link = base_url
                    link_elem = container.find("a", href=True)
                    if link_elem:
                        href = link_elem.get("href")
                        if href:
                            post_link = urljoin(base_url, href)

                    results.append(
                        {
                            "title": title,
                            "content": title,
                            "url": post_link,
                            "date": datetime.now(),
                            "relevance_score": relevance,
                            "metadata": {
                                "platform": platform_config["name"],
                                "community": community,
                                "score": score,
                                "content_type": "forum_post",
                            },
                        }
                    )

            except Exception as e:
                continue

        return results

    def _parse_federated_social(
        self, soup: BeautifulSoup, platform_config: Dict[str, Any], topic: str, base_url: str
    ) -> List[Dict[str, Any]]:
        """Parse federated social networks (Mastodon)"""
        results = []

        # Look for status/toot containers
        status_containers = soup.find_all(["div", "article"], limit=15)

        for container in status_containers:
            try:
                # Look for status content
                content_elem = container.find(["div", "p"], class_=re.compile(r"content|status"))
                if not content_elem:
                    continue

                content = content_elem.get_text(strip=True)
                if len(content) < 10:
                    continue

                # Look for author
                author_elem = container.find(
                    ["span", "a"], class_=re.compile(r"author|username|display")
                )
                author = author_elem.get_text(strip=True) if author_elem else "Unknown"

                # Calculate relevance
                relevance = self._calculate_social_relevance(content, topic)

                if relevance > 0.25:
                    # Try to find status link
                    status_link = base_url
                    link_elem = container.find("a", href=True)
                    if link_elem:
                        href = link_elem.get("href")
                        if href:
                            status_link = urljoin(base_url, href)

                    results.append(
                        {
                            "title": f"Status by {author}",
                            "content": content,
                            "url": status_link,
                            "date": datetime.now(),
                            "relevance_score": relevance,
                            "metadata": {
                                "platform": platform_config["name"],
                                "author": author,
                                "content_type": "federated_post",
                                "content_length": len(content),
                            },
                        }
                    )

            except Exception as e:
                continue

        return results

    def _parse_generic_social(
        self, soup: BeautifulSoup, platform_config: Dict[str, Any], topic: str, base_url: str
    ) -> List[Dict[str, Any]]:
        """Generic parser for unknown social platforms"""
        results = []

        # Look for any content containers
        content_containers = soup.find_all(["div", "article", "section"], limit=25)

        for container in content_containers:
            try:
                text = container.get_text(strip=True)
                if len(text) < 15:
                    continue

                relevance = self._calculate_social_relevance(text, topic)

                if relevance > 0.2:
                    # Try to find associated link
                    link = base_url
                    link_elem = container.find("a", href=True)
                    if link_elem:
                        href = link_elem.get("href")
                        if href:
                            link = urljoin(base_url, href)

                    results.append(
                        {
                            "title": text[:80] + "..." if len(text) > 80 else text,
                            "content": text,
                            "url": link,
                            "date": datetime.now(),
                            "relevance_score": relevance,
                            "metadata": {
                                "platform": platform_config["name"],
                                "content_type": "generic_social",
                                "content_length": len(text),
                            },
                        }
                    )

            except Exception as e:
                continue

        return results[:10]  # Limit generic results

    def _calculate_social_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score for social media content"""
        if not text or not topic:
            return 0.0

        text_lower = text.lower()
        topic_lower = topic.lower()

        score = 0.0

        # Exact topic match
        if topic_lower in text_lower:
            score += 0.6

        # Individual word matches
        topic_words = topic_lower.split()
        text_words = text_lower.split()

        word_matches = sum(1 for word in topic_words if word in text_words)
        if word_matches > 0:
            score += (word_matches / len(topic_words)) * 0.3

        # Check for conspiracy/research patterns
        for pattern_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.1
                    break

        return min(1.0, score)

    def _parse_timestamp(self, time_str: str) -> datetime:
        """Parse timestamp from social media platforms"""
        if not time_str:
            return datetime.now()

        # Try different timestamp formats
        timestamp_patterns = [
            r"(\d{1,2})\s*(hours?|hrs?)\s*ago",
            r"(\d{1,2})\s*(days?|d)\s*ago",
            r"(\d{1,2})\s*(weeks?|w)\s*ago",
            r"(\d{1,2})\s*(months?|mo)\s*ago",
        ]

        for pattern in timestamp_patterns:
            match = re.search(pattern, time_str.lower())
            if match:
                value = int(match.group(1))
                unit = match.group(2)

                if "hour" in unit or "hr" in unit:
                    return datetime.now() - timedelta(hours=value)
                elif "day" in unit or unit == "d":
                    return datetime.now() - timedelta(days=value)
                elif "week" in unit or unit == "w":
                    return datetime.now() - timedelta(weeks=value)
                elif "month" in unit or unit == "mo":
                    return datetime.now() - timedelta(days=value * 30)

        return datetime.now()

    def _filter_results(self, results: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Filter results for quality and relevance"""
        filtered = []

        for result in results:
            content = result.get("content", "")

            # Skip very short content
            if len(content) < 10:
                continue

            # Skip spam-like content
            if self._is_spam_content(content):
                continue

            # Require minimum relevance
            if result.get("relevance_score", 0) < 0.15:
                continue

            # Add social media specific metadata
            result["source"] = "social_media"
            result["source_type"] = "social_post"

            filtered.append(result)

        return filtered

    def _is_spam_content(self, content: str) -> bool:
        """Detect spam-like content"""
        content_lower = content.lower()

        # Check for excessive repetition
        words = content_lower.split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            return True

        # Check for spam indicators
        spam_indicators = [
            "click here",
            "buy now",
            "limited time",
            "act fast",
            "make money",
            "work from home",
            "free gift",
        ]

        spam_count = sum(1 for indicator in spam_indicators if indicator in content_lower)
        if spam_count >= 2:
            return True

        return False

    async def search_specific_platform(
        self, platform_name: str, topic: str
    ) -> List[Dict[str, Any]]:
        """Search specific platform by name"""
        platform_id = None
        for pid, config in self.platforms.items():
            if config["name"].lower() == platform_name.lower():
                platform_id = pid
                break

        if not platform_id:
            logger.error(f"Unknown platform: {platform_name}")
            return []

        platform_config = self.platforms[platform_id]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            results = await self._search_platform(
                session, platform_id, platform_config, topic, None
            )

        return self._filter_results(results, topic)

    def get_supported_platforms(self) -> List[Dict[str, str]]:
        """Get list of supported platforms"""
        return [
            {
                "id": platform_id,
                "name": config["name"],
                "type": config["type"],
                "description": f"{config['type'].replace('_', ' ').title()} platform",
            }
            for platform_id, config in self.platforms.items()
        ]
