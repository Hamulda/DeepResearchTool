#!/usr/bin/env python3
"""
OSINT (Open Source Intelligence) Collector
Multi-source intelligence gathering and correlation

Author: Advanced IT Specialist
"""

import asyncio
import aiohttp
import logging
import re
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import time
from urllib.parse import urljoin, quote
import xml.etree.ElementTree as ET

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

@dataclass
class OSINTResult:
    """OSINT intelligence result"""
    source: str
    result_type: str  # social_media, public_record, threat_intel, etc.
    target: str  # searched entity
    data: Dict[str, Any]
    confidence_score: float
    collection_date: datetime
    source_reliability: str  # A, B, C, D, E, F (NATO standard)
    information_credibility: str  # 1-6 scale
    metadata: Dict[str, Any]

@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    indicator: str
    indicator_type: str  # ip, domain, hash, etc.
    threat_type: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    sources: List[str]
    context: Dict[str, Any]

@dataclass
class SocialMediaIntel:
    """Social media intelligence"""
    platform: str
    username: str
    profile_data: Dict[str, Any]
    posts: List[Dict[str, Any]]
    connections: List[str]
    activity_pattern: Dict[str, Any]
    risk_indicators: List[str]

class OSINTCollector(BaseScraper):
    """Open Source Intelligence collection and analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "osint_collector"

        # API keys and configuration
        self.social_platforms = config.get('osint', {}).get('professional_tools', {}).get('social_intelligence', {})
        self.threat_intel_config = config.get('osint', {}).get('professional_tools', {}).get('threat_intelligence', {})

        # Rate limiting per source
        self.rate_limits = {
            'twitter': 300,  # requests per 15 minutes
            'reddit': 60,    # requests per minute
            'telegram': 20,  # requests per minute
            'shodan': 100,   # requests per month (free tier)
            'virustotal': 4, # requests per minute (free tier)
            'alienvault': 10 # requests per minute
        }

        # Source reliability mapping
        self.source_reliability = {
            'official_gov': 'A',      # Completely reliable
            'verified_media': 'B',    # Usually reliable
            'social_verified': 'C',   # Fairly reliable
            'social_unverified': 'D', # Not usually reliable
            'anonymous': 'E',         # Unreliable
            'unknown': 'F'            # Reliability cannot be judged
        }

        # Request tracking
        self.request_counters = {}
        self.last_reset_time = {}

    async def search(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Comprehensive OSINT search across multiple sources"""
        target_type = kwargs.get('target_type', 'general')
        search_depth = kwargs.get('depth', 'standard')

        logger.info(f"Starting OSINT collection for: {query} (type: {target_type})")

        # Social Media Intelligence
        if self.social_platforms.get('enabled', False):
            async for result in self._collect_social_intelligence(query, **kwargs):
                yield result

        # Public Records
        async for result in self._collect_public_records(query, **kwargs):
            yield result

        # Threat Intelligence
        if target_type in ['domain', 'ip', 'hash', 'url']:
            async for result in self._collect_threat_intelligence(query, **kwargs):
                yield result

        # News and Media Monitoring
        async for result in self._collect_media_intelligence(query, **kwargs):
            yield result

        # Technical Intelligence (if applicable)
        if target_type in ['domain', 'ip', 'organization']:
            async for result in self._collect_technical_intelligence(query, **kwargs):
                yield result

    async def _collect_social_intelligence(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Collect intelligence from social media platforms"""
        try:
            platforms = kwargs.get('platforms', ['twitter', 'reddit', 'telegram'])

            for platform in platforms:
                if platform == 'twitter':
                    async for result in self._search_twitter(query, **kwargs):
                        yield result
                elif platform == 'reddit':
                    async for result in self._search_reddit(query, **kwargs):
                        yield result
                elif platform == 'telegram':
                    async for result in self._search_telegram(query, **kwargs):
                        yield result

        except Exception as e:
            logger.error(f"Error in social intelligence collection: {e}")

    async def _search_twitter(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Search Twitter for intelligence (using public APIs)"""
        try:
            # Using Twitter's public search (limited but available)
            search_url = "https://api.twitter.com/2/tweets/search/recent"

            # Note: Requires Twitter API v2 Bearer Token
            # For demo purposes, using a mock structure

            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'created_at,author_id,public_metrics,context_annotations',
                'user.fields': 'username,name,public_metrics,verified'
            }

            await self._respect_rate_limit('twitter')

            # Mock data structure for Twitter results
            twitter_data = {
                'tweets': [],
                'users': [],
                'metrics': {
                    'total_tweets': 0,
                    'sentiment_analysis': {},
                    'trending_keywords': [],
                    'user_types': {}
                }
            }

            yield OSINTResult(
                source="twitter",
                result_type="social_media",
                target=query,
                data=twitter_data,
                confidence_score=0.7,
                collection_date=datetime.now(),
                source_reliability='C',  # Fairly reliable for verified accounts
                information_credibility='3',  # Possibly true
                metadata={'platform': 'twitter', 'api_version': 'v2'}
            )

        except Exception as e:
            logger.error(f"Error searching Twitter: {e}")

    async def _search_reddit(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Search Reddit for intelligence"""
        try:
            # Reddit API search
            reddit_api = "https://www.reddit.com/search.json"

            params = {
                'q': query,
                'sort': 'relevance',
                'limit': 100,
                't': 'all'  # all time
            }

            await self._respect_rate_limit('reddit')

            async with aiohttp.ClientSession() as session:
                async with session.get(reddit_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        reddit_intel = {
                            'posts': [],
                            'subreddits': set(),
                            'sentiment_trends': {},
                            'user_activity': {},
                            'discussion_topics': []
                        }

                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})

                            reddit_intel['posts'].append({
                                'title': post_data.get('title'),
                                'content': post_data.get('selftext'),
                                'score': post_data.get('score'),
                                'subreddit': post_data.get('subreddit'),
                                'author': post_data.get('author'),
                                'created_utc': post_data.get('created_utc'),
                                'num_comments': post_data.get('num_comments'),
                                'url': post_data.get('url')
                            })

                            reddit_intel['subreddits'].add(post_data.get('subreddit'))

                        reddit_intel['subreddits'] = list(reddit_intel['subreddits'])

                        yield OSINTResult(
                            source="reddit",
                            result_type="social_media",
                            target=query,
                            data=reddit_intel,
                            confidence_score=0.6,
                            collection_date=datetime.now(),
                            source_reliability='D',  # Not usually reliable (anonymous)
                            information_credibility='4',  # Doubtfully true
                            metadata={'platform': 'reddit', 'posts_analyzed': len(reddit_intel['posts'])}
                        )

        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")

    async def _search_telegram(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Search Telegram channels for intelligence"""
        try:
            # Note: Telegram search requires special handling due to API restrictions
            # This would typically use Telegram's Bot API or third-party services

            # Mock structure for Telegram intelligence
            telegram_intel = {
                'channels': [],
                'messages': [],
                'user_mentions': [],
                'media_shared': [],
                'activity_timeline': {}
            }

            # In real implementation, would search through:
            # - Public channels
            # - Bot-accessible content
            # - Publicly shared messages

            yield OSINTResult(
                source="telegram",
                result_type="social_media",
                target=query,
                data=telegram_intel,
                confidence_score=0.5,
                collection_date=datetime.now(),
                source_reliability='E',  # Unreliable (often anonymous)
                information_credibility='5',  # Improbable
                metadata={'platform': 'telegram', 'search_method': 'public_channels'}
            )

        except Exception as e:
            logger.error(f"Error searching Telegram: {e}")

    async def _collect_public_records(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Collect public records and open data"""
        try:
            # Search various public record sources
            sources = [
                'government_databases',
                'business_registries',
                'court_records',
                'property_records',
                'professional_licenses'
            ]

            for source in sources:
                public_records = await self._search_public_source(source, query, **kwargs)
                if public_records:
                    yield OSINTResult(
                        source=f"public_records_{source}",
                        result_type="public_record",
                        target=query,
                        data=public_records,
                        confidence_score=0.8,  # Public records are generally reliable
                        collection_date=datetime.now(),
                        source_reliability='A',  # Completely reliable (official sources)
                        information_credibility='1',  # Confirmed by other sources
                        metadata={'record_type': source, 'jurisdiction': 'various'}
                    )

        except Exception as e:
            logger.error(f"Error collecting public records: {e}")

    async def _collect_threat_intelligence(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Collect threat intelligence data"""
        try:
            # Threat intelligence sources
            threat_sources = [
                'virustotal',
                'alienvault_otx',
                'shodan',
                'censys',
                'passive_dns'
            ]

            for source in threat_sources:
                threat_data = await self._query_threat_source(source, query, **kwargs)
                if threat_data:
                    yield OSINTResult(
                        source=f"threat_intel_{source}",
                        result_type="threat_intelligence",
                        target=query,
                        data=threat_data,
                        confidence_score=threat_data.get('confidence', 0.7),
                        collection_date=datetime.now(),
                        source_reliability='B',  # Usually reliable (security vendors)
                        information_credibility='2',  # Probably true
                        metadata={'threat_source': source, 'indicator_type': kwargs.get('target_type')}
                    )

        except Exception as e:
            logger.error(f"Error collecting threat intelligence: {e}")

    async def _collect_media_intelligence(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Collect media and news intelligence"""
        try:
            # News and media sources
            news_sources = [
                'google_news',
                'bing_news',
                'rss_feeds',
                'press_releases',
                'blog_mentions'
            ]

            for source in sources:
                media_data = await self._search_media_source(source, query, **kwargs)
                if media_data:
                    reliability = 'B' if 'verified_media' in source else 'C'

                    yield OSINTResult(
                        source=f"media_{source}",
                        result_type="media_intelligence",
                        target=query,
                        data=media_data,
                        confidence_score=0.6,
                        collection_date=datetime.now(),
                        source_reliability=reliability,
                        information_credibility='3',
                        metadata={'media_type': source, 'time_range': kwargs.get('time_range', '30d')}
                    )

        except Exception as e:
            logger.error(f"Error collecting media intelligence: {e}")

    async def _collect_technical_intelligence(self, query: str, **kwargs) -> AsyncGenerator[OSINTResult, None]:
        """Collect technical intelligence for domains/IPs"""
        try:
            technical_data = {
                'whois_data': await self._get_whois_data(query),
                'dns_records': await self._get_dns_records(query),
                'ssl_certificates': await self._get_ssl_info(query),
                'subdomain_enumeration': await self._enumerate_subdomains(query),
                'port_scan_results': await self._get_port_info(query),
                'technology_stack': await self._identify_technologies(query)
            }

            yield OSINTResult(
                source="technical_intelligence",
                result_type="technical_intel",
                target=query,
                data=technical_data,
                confidence_score=0.9,  # Technical data is highly reliable
                collection_date=datetime.now(),
                source_reliability='A',  # Completely reliable (technical facts)
                information_credibility='1',  # Confirmed
                metadata={'target_type': kwargs.get('target_type'), 'scan_depth': 'basic'}
            )

        except Exception as e:
            logger.error(f"Error collecting technical intelligence: {e}")

    async def _search_public_source(self, source: str, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search specific public record source"""
        # Mock implementation - would connect to actual APIs
        return {
            'source': source,
            'records_found': 0,
            'data': [],
            'search_query': query,
            'disclaimer': 'Mock data for demonstration'
        }

    async def _query_threat_source(self, source: str, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Query threat intelligence source"""
        # Mock implementation - would connect to actual threat intel APIs
        return {
            'source': source,
            'indicators': [],
            'threat_score': 0,
            'last_seen': None,
            'malware_families': [],
            'attribution': {}
        }

    async def _search_media_source(self, source: str, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search media source"""
        # Mock implementation - would connect to news APIs
        return {
            'source': source,
            'articles': [],
            'sentiment': 'neutral',
            'trending_topics': [],
            'coverage_timeline': {}
        }

    async def _get_whois_data(self, domain: str) -> Dict[str, Any]:
        """Get WHOIS information for domain"""
        # Mock implementation
        return {'registrar': '', 'creation_date': '', 'expiration_date': ''}

    async def _get_dns_records(self, domain: str) -> Dict[str, Any]:
        """Get DNS records for domain"""
        # Mock implementation
        return {'A': [], 'MX': [], 'NS': [], 'TXT': []}

    async def _get_ssl_info(self, domain: str) -> Dict[str, Any]:
        """Get SSL certificate information"""
        # Mock implementation
        return {'issuer': '', 'valid_from': '', 'valid_to': '', 'san': []}

    async def _enumerate_subdomains(self, domain: str) -> List[str]:
        """Enumerate subdomains"""
        # Mock implementation
        return []

    async def _get_port_info(self, target: str) -> Dict[str, Any]:
        """Get port scan information"""
        # Mock implementation
        return {'open_ports': [], 'services': {}}

    async def _identify_technologies(self, target: str) -> Dict[str, Any]:
        """Identify web technologies"""
        # Mock implementation
        return {'web_server': '', 'cms': '', 'frameworks': [], 'analytics': []}

    async def _respect_rate_limit(self, source: str):
        """Respect rate limits for different sources"""
        current_time = time.time()

        # Initialize tracking for new sources
        if source not in self.request_counters:
            self.request_counters[source] = 0
            self.last_reset_time[source] = current_time

        # Reset counter based on source-specific time windows
        reset_intervals = {
            'twitter': 900,  # 15 minutes
            'reddit': 60,    # 1 minute
            'telegram': 60,  # 1 minute
            'default': 60    # 1 minute
        }

        reset_interval = reset_intervals.get(source, reset_intervals['default'])

        if current_time - self.last_reset_time[source] >= reset_interval:
            self.request_counters[source] = 0
            self.last_reset_time[source] = current_time

        # Check rate limit
        rate_limit = self.rate_limits.get(source, 60)

        if self.request_counters[source] >= rate_limit:
            sleep_time = reset_interval - (current_time - self.last_reset_time[source])
            if sleep_time > 0:
                logger.info(f"Rate limit reached for {source}, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.request_counters[source] = 0
                self.last_reset_time[source] = time.time()

        self.request_counters[source] += 1

    def correlate_intelligence(self, results: List[OSINTResult]) -> Dict[str, Any]:
        """Correlate intelligence from multiple sources"""
        correlation = {
            'target': '',
            'confidence_aggregate': 0.0,
            'source_diversity': 0,
            'reliability_assessment': '',
            'timeline': [],
            'key_findings': [],
            'cross_references': [],
            'risk_assessment': {
                'threat_level': 'unknown',
                'indicators': [],
                'recommendations': []
            }
        }

        if not results:
            return correlation

        correlation['target'] = results[0].target
        correlation['source_diversity'] = len(set(r.source for r in results))

        # Calculate aggregate confidence
        total_confidence = sum(r.confidence_score for r in results)
        correlation['confidence_aggregate'] = total_confidence / len(results)

        # Assess overall reliability
        reliability_scores = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0}
        avg_reliability = sum(reliability_scores.get(r.source_reliability, 0) for r in results) / len(results)

        if avg_reliability >= 4:
            correlation['reliability_assessment'] = 'High'
        elif avg_reliability >= 3:
            correlation['reliability_assessment'] = 'Medium'
        else:
            correlation['reliability_assessment'] = 'Low'

        # Extract key findings
        for result in results:
            if result.confidence_score > 0.7:
                correlation['key_findings'].append({
                    'source': result.source,
                    'finding': str(result.data)[:200] + "..." if len(str(result.data)) > 200 else str(result.data),
                    'confidence': result.confidence_score
                })

        return correlation

    async def health_check(self) -> bool:
        """Check OSINT collector health"""
        return True  # Basic implementation

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'active_sources': len(self.request_counters),
            'total_requests': sum(self.request_counters.values()),
            'rate_limit_status': {
                source: f"{count}/{limit}"
                for source, count in self.request_counters.items()
                for limit in [self.rate_limits.get(source, 60)]
            }
        }
