#!/usr/bin/env python3
"""
Wayback Machine Scraper for Deep Research Tool
Advanced Internet Archive integration for historical web research

Author: Advanced IT Specialist
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
import aiohttp
from waybackpy import WaybackMachine
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class WaybackMachineScraper:
    """Advanced scraper for Internet Archive Wayback Machine"""

    def __init__(self, rate_limit: float = 1.0, max_snapshots: int = 10):
        """
        Initialize Wayback Machine scraper

        Args:
            rate_limit: Requests per second limit
            max_snapshots: Maximum snapshots to retrieve per URL
        """
        self.rate_limit = rate_limit
        self.max_snapshots = max_snapshots
        self.ua = UserAgent()
        self.session = None
        self.last_request_time = 0

        # Wayback Machine API endpoints
        self.cdx_api = "http://web.archive.org/cdx/search/cdx"
        self.wayback_api = "http://archive.org/wayback/available"

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Asynchronous search for historical web content

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of historical documents
        """
        results = []

        # Generate search URLs based on topic
        search_urls = self._generate_search_urls(topic)

        async with aiohttp.ClientSession() as session:
            self.session = session

            for url in search_urls:
                try:
                    snapshots = await self._get_url_snapshots(url, time_range)
                    for snapshot in snapshots[:self.max_snapshots]:
                        content = await self._fetch_snapshot_content(snapshot)
                        if content:
                            results.append(content)

                    # Rate limiting
                    await asyncio.sleep(1.0 / self.rate_limit)

                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    continue

        logger.info(f"Wayback Machine search found {len(results)} historical documents")
        return results

    def search(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Synchronous search method for compatibility

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of historical documents
        """
        return asyncio.run(self.search_async(topic, time_range))

    def _generate_search_urls(self, topic: str) -> List[str]:
        """Generate potential URLs to search in Wayback Machine"""
        # Common domains that might have historical content about the topic
        domains = [
            "cnn.com", "bbc.com", "nytimes.com", "washingtonpost.com",
            "reuters.com", "ap.org", "npr.org", "time.com",
            "newsweek.com", "guardian.co.uk", "independent.co.uk",
            "wikipedia.org", "britannica.com"
        ]

        urls = []
        topic_slug = re.sub(r'[^a-zA-Z0-9\-]', '-', topic.lower()).strip('-')

        for domain in domains:
            # Different URL patterns to try
            patterns = [
                f"https://{domain}/*{topic_slug}*",
                f"https://{domain}/*/search?q={topic.replace(' ', '+')}"
            ]
            urls.extend(patterns)

        return urls[:20]  # Limit to prevent excessive requests

    async def _get_url_snapshots(self, url_pattern: str, time_range: Optional[Tuple[datetime, datetime]]) -> List[Dict[str, Any]]:
        """Get snapshots for a URL pattern from CDX API"""
        params = {
            'url': url_pattern,
            'output': 'json',
            'limit': self.max_snapshots,
            'filter': 'statuscode:200'
        }

        # Add time range filters if provided
        if time_range:
            start_date = time_range[0].strftime('%Y%m%d')
            end_date = time_range[1].strftime('%Y%m%d')
            params['from'] = start_date
            params['to'] = end_date

        try:
            async with self.session.get(self.cdx_api, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Parse CDX response
                    snapshots = []
                    if data and len(data) > 1:  # First row is headers
                        headers = data[0]
                        for row in data[1:]:
                            snapshot = dict(zip(headers, row))
                            snapshots.append({
                                'timestamp': snapshot.get('timestamp'),
                                'original_url': snapshot.get('original'),
                                'wayback_url': f"http://web.archive.org/web/{snapshot.get('timestamp')}/{snapshot.get('original')}",
                                'status_code': snapshot.get('statuscode'),
                                'mime_type': snapshot.get('mimetype'),
                                'length': snapshot.get('length')
                            })

                    return snapshots

        except Exception as e:
            logger.error(f"Error fetching snapshots for {url_pattern}: {e}")

        return []

    async def _fetch_snapshot_content(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch and parse content from a Wayback Machine snapshot"""
        wayback_url = snapshot['wayback_url']

        try:
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

            async with self.session.get(wayback_url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse content with BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')

                    # Remove Wayback Machine toolbar and scripts
                    self._clean_wayback_content(soup)

                    # Extract text content
                    text_content = self._extract_text_content(soup)

                    if len(text_content) < 100:  # Skip very short content
                        return None

                    # Extract metadata
                    title = self._extract_title(soup)
                    date = self._parse_wayback_timestamp(snapshot['timestamp'])

                    return {
                        'title': title,
                        'content': text_content,
                        'url': snapshot['original_url'],
                        'wayback_url': wayback_url,
                        'date': date,
                        'source': 'wayback_machine',
                        'source_type': 'archive',
                        'metadata': {
                            'timestamp': snapshot['timestamp'],
                            'status_code': snapshot['status_code'],
                            'mime_type': snapshot['mime_type'],
                            'content_length': len(text_content),
                            'original_length': snapshot.get('length')
                        }
                    }

        except Exception as e:
            logger.error(f"Error fetching content from {wayback_url}: {e}")

        return None

    def _clean_wayback_content(self, soup: BeautifulSoup):
        """Remove Wayback Machine specific elements"""
        # Remove Wayback Machine toolbar and overlays
        wayback_elements = soup.find_all(['div', 'script', 'iframe'],
                                       attrs={'id': re.compile(r'wm-|wayback|archive')})
        for element in wayback_elements:
            element.decompose()

        # Remove common ad and tracking elements
        ad_elements = soup.find_all(['div', 'script'],
                                  attrs={'class': re.compile(r'ad|advertisement|tracking|analytics')})
        for element in ad_elements:
            element.decompose()

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Focus on main content
        main_content = soup.find(['main', 'article', 'div']) or soup

        # Extract text
        text = main_content.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text().strip()
            # Remove common Wayback Machine additions
            title = re.sub(r'\s*-\s*Wayback Machine.*$', '', title, flags=re.IGNORECASE)
            return title

        # Fallback to h1
        h1_elem = soup.find('h1')
        if h1_elem:
            return h1_elem.get_text().strip()

        return "Untitled"

    def _parse_wayback_timestamp(self, timestamp: str) -> datetime:
        """Parse Wayback Machine timestamp to datetime"""
        try:
            # Wayback timestamp format: YYYYMMDDHHMMSS
            if len(timestamp) >= 8:
                year = int(timestamp[:4])
                month = int(timestamp[4:6])
                day = int(timestamp[6:8])
                hour = int(timestamp[8:10]) if len(timestamp) >= 10 else 0
                minute = int(timestamp[10:12]) if len(timestamp) >= 12 else 0
                second = int(timestamp[12:14]) if len(timestamp) >= 14 else 0

                return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

        return datetime.now()

    async def search_specific_urls(self, urls: List[str],
                                  time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Search for specific URLs in Wayback Machine

        Args:
            urls: List of specific URLs to search
            time_range: Optional time range

        Returns:
            List of historical documents
        """
        results = []

        async with aiohttp.ClientSession() as session:
            self.session = session

            for url in urls:
                try:
                    snapshots = await self._get_url_snapshots(url, time_range)
                    for snapshot in snapshots:
                        content = await self._fetch_snapshot_content(snapshot)
                        if content:
                            results.append(content)

                    await asyncio.sleep(1.0 / self.rate_limit)

                except Exception as e:
                    logger.error(f"Error processing specific URL {url}: {e}")

        return results

    async def find_changes_over_time(self, url: str,
                                   time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """
        Find how a specific URL changed over time

        Args:
            url: Specific URL to track
            time_range: Time range to analyze

        Returns:
            List of changes with analysis
        """
        snapshots = await self._get_url_snapshots(url, time_range)
        changes = []
        previous_content = None

        for snapshot in snapshots:
            content_data = await self._fetch_snapshot_content(snapshot)
            if not content_data:
                continue

            current_content = content_data['content']

            if previous_content:
                # Simple change detection
                similarity = self._calculate_text_similarity(previous_content, current_content)
                if similarity < 0.9:  # Significant change detected
                    changes.append({
                        'timestamp': snapshot['timestamp'],
                        'date': content_data['date'],
                        'similarity_to_previous': similarity,
                        'content_length': len(current_content),
                        'wayback_url': snapshot['wayback_url'],
                        'changes_detected': True
                    })

            previous_content = current_content

        return changes

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def get_available_years(self, url: str) -> List[int]:
        """Get years when snapshots are available for a URL"""
        try:
            wayback_machine = WaybackMachine(url)
            # This would require additional implementation
            # For now, return a placeholder
            return list(range(2000, datetime.now().year + 1))
        except Exception as e:
            logger.error(f"Error getting available years for {url}: {e}")
            return []
