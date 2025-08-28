#!/usr/bin/env python3
"""
IPFS and P2P Network Scraper for Deep Research Tool
Searches decentralized networks for censorship-resistant content

Author: Advanced IT Specialist
"""

import asyncio
import aiohttp
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import re

logger = logging.getLogger(__name__)

class IPFSScraper:
    """Scraper for IPFS (InterPlanetary File System) content"""

    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.ipfs_gateways = [
            "https://ipfs.io/ipfs/",
            "https://gateway.pinata.cloud/ipfs/",
            "https://cloudflare-ipfs.com/ipfs/",
            "https://dweb.link/ipfs/",
            "https://ipfs.infura.io/ipfs/"
        ]
        self.known_content_hashes = self._load_known_hashes()

    def _load_known_hashes(self) -> Dict[str, str]:
        """Load known IPFS content hashes for various topics"""
        return {
            # Example hashes - these would be populated from known sources
            'wikileaks_insurance': 'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
            'panama_papers': 'QmZFxhpfWE7VPdnqFZfzpVSPjN1NjvFu8hVhkQRJR6gPRC',
            'declassified_docs': 'QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn',
            'alternative_history': 'QmYFQGCdLQbiGTKYLKKG3JTB5sth7VPeqGNbD3VsZW2jDP'
        }

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Search IPFS networks for content related to topic"""
        await self._rate_limit()

        results = []

        # Search known content hashes
        topic_lower = topic.lower()
        for content_type, ipfs_hash in self.known_content_hashes.items():
            if any(keyword in content_type for keyword in topic_lower.split()):
                content = await self._fetch_ipfs_content(ipfs_hash)
                if content and self._content_matches_topic(content, topic):
                    results.append({
                        'title': f'IPFS Content: {content_type}',
                        'content': content[:2000],
                        'url': f'ipfs://{ipfs_hash}',
                        'source_type': 'ipfs_content',
                        'source_url': self.ipfs_gateways[0] + ipfs_hash,
                        'date': datetime.now(),
                        'metadata': {
                            'ipfs_hash': ipfs_hash,
                            'content_type': content_type,
                            'gateway_used': self.ipfs_gateways[0]
                        }
                    })

        # Search IPFS search engines/indexes
        search_results = await self._search_ipfs_indexes(topic)
        results.extend(search_results)

        logger.info(f"IPFS search for '{topic}' found {len(results)} results")
        return results

    async def _fetch_ipfs_content(self, ipfs_hash: str) -> Optional[str]:
        """Fetch content from IPFS using multiple gateways"""
        for gateway in self.ipfs_gateways:
            try:
                url = gateway + ipfs_hash
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            return content[:5000]  # Limit content size
            except Exception as e:
                logger.debug(f"Failed to fetch from gateway {gateway}: {e}")
                continue

        logger.warning(f"Failed to fetch IPFS content: {ipfs_hash}")
        return None

    async def _search_ipfs_indexes(self, topic: str) -> List[Dict[str, Any]]:
        """Search IPFS indexes and search engines"""
        results = []

        # Search using IPFS search APIs (if available)
        search_apis = [
            "https://search.ipfs.io/api/v0/search",
            # Add more IPFS search APIs as they become available
        ]

        for api_url in search_apis:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {'q': topic, 'limit': 20}
                    async with session.get(api_url, params=params, timeout=20) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Parse API response and extract results
                            api_results = self._parse_ipfs_search_results(data, topic)
                            results.extend(api_results)
            except Exception as e:
                logger.debug(f"IPFS search API error: {e}")
                continue

        return results

    def _parse_ipfs_search_results(self, data: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
        """Parse results from IPFS search APIs"""
        results = []

        # This would need to be adapted based on actual API response format
        items = data.get('results', data.get('items', []))

        for item in items:
            if isinstance(item, dict):
                ipfs_hash = item.get('hash', item.get('cid', ''))
                title = item.get('title', item.get('name', f'IPFS Document {ipfs_hash[:8]}'))
                description = item.get('description', item.get('content', ''))

                if ipfs_hash:
                    results.append({
                        'title': title,
                        'content': description[:1000],
                        'url': f'ipfs://{ipfs_hash}',
                        'source_type': 'ipfs_search',
                        'source_url': self.ipfs_gateways[0] + ipfs_hash,
                        'date': datetime.now(),
                        'metadata': {
                            'ipfs_hash': ipfs_hash,
                            'search_result': True
                        }
                    })

        return results

    def _content_matches_topic(self, content: str, topic: str) -> bool:
        """Check if content matches the search topic"""
        topic_words = topic.lower().split()
        content_lower = content.lower()

        matches = sum(1 for word in topic_words if word in content_lower)
        return matches >= len(topic_words) * 0.4  # 40% match threshold

    async def _rate_limit(self):
        """Apply rate limiting"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

class BitTorrentDHTScraper:
    """Scraper for BitTorrent DHT network"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.dht_search_apis = [
            "https://btdig.com/search",
            "https://bt4g.org/search",
            # Add more DHT search engines
        ]

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Search BitTorrent DHT network for topic-related content"""
        await self._rate_limit()

        results = []

        # Search DHT networks through public search engines
        for search_url in self.dht_search_apis:
            try:
                api_results = await self._search_dht_api(search_url, topic)
                results.extend(api_results)
            except Exception as e:
                logger.debug(f"DHT search error for {search_url}: {e}")
                continue

        # Filter for document/archive torrents
        filtered_results = []
        for result in results:
            if self._is_document_torrent(result):
                filtered_results.append(result)

        logger.info(f"BitTorrent DHT search for '{topic}' found {len(filtered_results)} document torrents")
        return filtered_results

    async def _search_dht_api(self, api_url: str, topic: str) -> List[Dict[str, Any]]:
        """Search specific DHT API"""
        results = []

        try:
            # Add document-related keywords
            search_query = f"{topic} documents OR archives OR leaks OR papers"

            async with aiohttp.ClientSession() as session:
                params = {'q': search_query}
                async with session.get(api_url, params=params, timeout=20) as response:
                    if response.status == 200:
                        # Parse response (would need to be adapted per API)
                        content = await response.text()
                        parsed_results = self._parse_dht_response(content, topic)
                        results.extend(parsed_results)

        except Exception as e:
            logger.error(f"Error searching DHT API {api_url}: {e}")

        return results

    def _parse_dht_response(self, html_content: str, topic: str) -> List[Dict[str, Any]]:
        """Parse HTML response from DHT search engines"""
        results = []

        # Extract torrent information using regex
        # This is a simplified example - real implementation would use proper HTML parsing
        torrent_pattern = r'magnet:\?xt=urn:btih:([a-fA-F0-9]{40})[^"]*'
        name_pattern = r'<title>([^<]+)</title>'

        magnet_links = re.findall(torrent_pattern, html_content)
        names = re.findall(name_pattern, html_content)

        for i, (hash_val, name) in enumerate(zip(magnet_links, names)):
            if topic.lower() in name.lower():
                results.append({
                    'title': f'Torrent: {name}',
                    'content': f'BitTorrent content related to {topic}',
                    'url': f'magnet:?xt=urn:btih:{hash_val}',
                    'source_type': 'bittorrent_dht',
                    'source_url': f'magnet:?xt=urn:btih:{hash_val}',
                    'date': datetime.now(),
                    'metadata': {
                        'info_hash': hash_val,
                        'torrent_name': name,
                        'search_engine': 'dht_search'
                    }
                })

        return results

    def _is_document_torrent(self, result: Dict[str, Any]) -> bool:
        """Check if torrent contains documents/archives"""
        title = result.get('title', '').lower()

        document_indicators = [
            'pdf', 'doc', 'documents', 'archive', 'papers', 'research',
            'leak', 'files', 'collection', 'library', 'database'
        ]

        return any(indicator in title for indicator in document_indicators)

    async def _rate_limit(self):
        """Apply rate limiting"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

class P2PNetworkScraper:
    """Combined P2P network scraper"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.ipfs_scraper = IPFSScraper(rate_limit / 2)
        self.bittorrent_scraper = BitTorrentDHTScraper(rate_limit / 2)

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Search across multiple P2P networks"""

        # Search IPFS and BitTorrent networks in parallel
        ipfs_task = asyncio.create_task(self.ipfs_scraper.search_async(topic, time_range))
        bt_task = asyncio.create_task(self.bittorrent_scraper.search_async(topic, time_range))

        ipfs_results, bt_results = await asyncio.gather(ipfs_task, bt_task, return_exceptions=True)

        results = []

        # Combine results
        if isinstance(ipfs_results, list):
            results.extend(ipfs_results)
        else:
            logger.error(f"IPFS search error: {ipfs_results}")

        if isinstance(bt_results, list):
            results.extend(bt_results)
        else:
            logger.error(f"BitTorrent search error: {bt_results}")

        # Deduplicate and prioritize
        deduplicated_results = self._deduplicate_results(results)

        logger.info(f"P2P network search for '{topic}' found {len(deduplicated_results)} total results")
        return deduplicated_results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results across P2P networks"""
        seen_titles = set()
        deduplicated = []

        for result in results:
            title_hash = hashlib.md5(result.get('title', '').encode()).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                deduplicated.append(result)

        return deduplicated

    def search(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Synchronous search wrapper"""
        return asyncio.run(self.search_async(topic, time_range))
