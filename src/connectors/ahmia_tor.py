#!/usr/bin/env python3
"""
Ahmia/Tor konektor s legal-only režimem
Whitelist onion zdrojů, právní filtry a rate-limity

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import aiohttp
import aiofiles
from pathlib import Path
import re
from urllib.parse import urlparse, urljoin


@dataclass
class OnionSource:
    """Onion zdroj s metadaty"""
    onion_url: str
    title: str
    description: str
    category: str
    legal_status: str  # "verified", "questionable", "blacklisted"
    last_verified: datetime
    content_hash: str
    safety_score: float


class AhmiaConnector:
    """Ahmia/Tor konektor s právními filtry"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.legal_only = config.get("legal_only", True)
        self.cache_dir = Path(config.get("cache_dir", "research_cache/ahmia"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load whitelists and blacklists
        self.legal_whitelist = self._load_legal_whitelist()
        self.content_blacklist = self._load_content_blacklist()

        self.ahmia_base_url = "https://ahmia.fi"
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay = config.get("retry_delay", 3.0)
        self.rate_limit_delay = config.get("rate_limit_delay", 2.0)

    def _load_legal_whitelist(self) -> Set[str]:
        """Načte whitelist legálních onion domén"""
        whitelist_file = Path("configs/tor_legal_whitelist.json")

        if whitelist_file.exists():
            with open(whitelist_file, 'r') as f:
                data = json.load(f)
                return set(data.get("legal_domains", []))

        # Default legal whitelist (academic, news, NGO)
        return {
            "duckduckgogg42h6oa.onion",  # DuckDuckGo
            "facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion",  # Facebook
            "expyuzz4wqqyqhjn.onion",  # ProPublica
            "www.facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion",
            "3g2upl4pq6kufc4m.onion"  # DuckDuckGo old
        }

    def _load_content_blacklist(self) -> Set[str]:
        """Načte blacklist zakázaných kategorií"""
        return {
            "illegal", "drugs", "weapons", "fraud", "hacking",
            "violence", "extremism", "terrorism", "child",
            "pornography", "gambling", "stolen"
        }

    async def search_ahmia(self, query: str, max_results: int = 20) -> List[OnionSource]:
        """Vyhledá v Ahmia indexu"""
        print(f"🔍 Searching Ahmia for: {query}")

        if not self._is_legal_query(query):
            print("❌ Query rejected by legal filters")
            return []

        cache_key = hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
        cache_file = self.cache_dir / f"search_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                cached_data = json.loads(await f.read())
                sources = []
                for item in cached_data:
                    item['last_verified'] = datetime.fromisoformat(item['last_verified'])
                    sources.append(OnionSource(**item))
                return sources

        sources = []
        search_url = f"{self.ahmia_base_url}/search/"

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

                async with aiohttp.ClientSession() as session:
                    params = {
                        "q": query,
                        "redirect": "false"
                    }
                    headers = {
                        "User-Agent": "DeepResearchTool/1.0 (+research@example.com)",
                        "Accept": "text/html,application/xhtml+xml"
                    }

                    async with session.get(search_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            sources = self._parse_ahmia_results(html_content)

                            # Aplikuj právní filtry
                            if self.legal_only:
                                sources = self._filter_legal_sources(sources)

                            # Limituj výsledky
                            sources = sources[:max_results]
                            break
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"❌ Ahmia search failed: {e}")
                    return []
                await asyncio.sleep(self.retry_delay * retry_count)

        # Cache výsledky
        cache_data = []
        for source in sources:
            item = {
                "onion_url": source.onion_url,
                "title": source.title,
                "description": source.description,
                "category": source.category,
                "legal_status": source.legal_status,
                "last_verified": source.last_verified.isoformat(),
                "content_hash": source.content_hash,
                "safety_score": source.safety_score
            }
            cache_data.append(item)

        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(cache_data, indent=2))

        print(f"✅ Found {len(sources)} legal onion sources")
        return sources

    def _is_legal_query(self, query: str) -> bool:
        """Kontrola legálnosti dotazu"""
        query_lower = query.lower()

        # Blacklist keywords
        illegal_keywords = [
            "hack", "crack", "exploit", "weapon", "drug", "fraud",
            "stolen", "illegal", "child", "porn", "violence"
        ]

        for keyword in illegal_keywords:
            if keyword in query_lower:
                return False

        return True

    def _parse_ahmia_results(self, html_content: str) -> List[OnionSource]:
        """Parsuje výsledky z Ahmia"""
        sources = []

        # Simple HTML parsing pro Ahmia výsledky
        # V produkci by se použil BeautifulSoup
        import re

        # Extract result blocks
        result_pattern = r'<li class="result">.*?</li>'
        results = re.findall(result_pattern, html_content, re.DOTALL)

        for result in results:
            try:
                # Extract onion URL
                url_match = re.search(r'href="([^"]*\.onion[^"]*)"', result)
                if not url_match:
                    continue
                onion_url = url_match.group(1)

                # Extract title
                title_match = re.search(r'<h4[^>]*>(.*?)</h4>', result, re.DOTALL)
                title = title_match.group(1).strip() if title_match else "Unknown"

                # Extract description
                desc_match = re.search(r'<p class="excerpt">(.*?)</p>', result, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""

                # Determine category and safety
                category = self._categorize_source(title, description)
                safety_score = self._calculate_safety_score(title, description, onion_url)

                source = OnionSource(
                    onion_url=onion_url,
                    title=title,
                    description=description,
                    category=category,
                    legal_status=self._determine_legal_status(onion_url, category),
                    last_verified=datetime.now(),
                    content_hash="",
                    safety_score=safety_score
                )

                sources.append(source)

            except Exception as e:
                print(f"❌ Failed to parse result: {e}")
                continue

        return sources

    def _categorize_source(self, title: str, description: str) -> str:
        """Kategorizuje zdroj podle obsahu"""
        content = f"{title} {description}".lower()

        categories = {
            "news": ["news", "journal", "media", "press", "report"],
            "academic": ["research", "university", "academic", "science", "study"],
            "technology": ["tech", "software", "code", "development", "programming"],
            "privacy": ["privacy", "security", "anonymous", "tor", "vpn"],
            "social": ["forum", "discussion", "community", "social", "chat"],
            "commercial": ["shop", "store", "business", "commercial", "service"],
            "government": ["government", "official", "public", "agency"],
            "ngo": ["nonprofit", "charity", "organization", "foundation"]
        }

        for category, keywords in categories.items():
            if any(keyword in content for keyword in keywords):
                return category

        return "unknown"

    def _calculate_safety_score(self, title: str, description: str, url: str) -> float:
        """Vypočítá safety score zdroje"""
        score = 0.5  # Baseline

        content = f"{title} {description}".lower()

        # Positive indicators
        positive_indicators = [
            "official", "verified", "legitimate", "legal", "academic",
            "research", "news", "information", "education", "nonprofit"
        ]

        # Negative indicators
        negative_indicators = [
            "anonymous", "hidden", "secret", "underground", "illegal",
            "hack", "crack", "exploit", "fraud", "stolen"
        ]

        # URL whitelist bonus
        domain = urlparse(url).netloc
        if domain in self.legal_whitelist:
            score += 0.4

        # Content analysis
        for indicator in positive_indicators:
            if indicator in content:
                score += 0.1

        for indicator in negative_indicators:
            if indicator in content:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def _determine_legal_status(self, url: str, category: str) -> str:
        """Určí právní status zdroje"""
        domain = urlparse(url).netloc

        if domain in self.legal_whitelist:
            return "verified"

        if category in ["academic", "news", "government", "ngo"]:
            return "verified"

        if any(blacklisted in category.lower() for blacklisted in self.content_blacklist):
            return "blacklisted"

        return "questionable"

    def _filter_legal_sources(self, sources: List[OnionSource]) -> List[OnionSource]:
        """Filtruje pouze legální zdroje"""
        if not self.legal_only:
            return sources

        legal_sources = []
        for source in sources:
            if source.legal_status == "verified" and source.safety_score >= 0.6:
                legal_sources.append(source)

        return legal_sources

    async def fetch_onion_content(self,
                                source: OnionSource,
                                timeout: int = 30) -> Optional[str]:
        """Stáhne obsah z onion zdroje"""
        # Note: V produkci by bylo potřeba Tor proxy
        print(f"⚠️  Onion content fetching requires Tor proxy: {source.onion_url}")

        # Pro demo vrátíme placeholder
        placeholder_content = f"""
        Title: {source.title}
        Description: {source.description}
        Category: {source.category}
        Legal Status: {source.legal_status}
        Safety Score: {source.safety_score}
        
        [Content would be fetched through Tor proxy in production]
        """

        source.content_hash = hashlib.md5(placeholder_content.encode()).hexdigest()
        return placeholder_content

    def get_legal_compliance_report(self, sources: List[OnionSource]) -> Dict[str, Any]:
        """Generuje report legal compliance"""
        total_sources = len(sources)
        verified_sources = sum(1 for s in sources if s.legal_status == "verified")
        questionable_sources = sum(1 for s in sources if s.legal_status == "questionable")
        blacklisted_sources = sum(1 for s in sources if s.legal_status == "blacklisted")

        avg_safety_score = sum(s.safety_score for s in sources) / total_sources if total_sources > 0 else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_sources": total_sources,
            "legal_breakdown": {
                "verified": verified_sources,
                "questionable": questionable_sources,
                "blacklisted": blacklisted_sources
            },
            "compliance_rate": verified_sources / total_sources if total_sources > 0 else 0,
            "average_safety_score": avg_safety_score,
            "legal_only_mode": self.legal_only,
            "whitelist_size": len(self.legal_whitelist)
        }
