#!/usr/bin/env python3
"""
Temporal-aware Ranking
Time-sensitive scoring with recency boost and temporal decay

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class TemporalProfile(Enum):
    """Temporal profiles for different content types"""
    NEWS = "news"  # High decay, prefer very recent
    RESEARCH = "research"  # Medium decay, quality over recency
    REFERENCE = "reference"  # Low decay, timeless content
    POLICY = "policy"  # Step function, recent regulations critical
    FINANCIAL = "financial"  # Very high decay, real-time importance


@dataclass
class TemporalConfig:
    """Configuration for temporal ranking"""
    base_decay_days: float = 365.0  # Days for 1/e decay
    recency_boost_days: int = 30  # Days for recency boost
    recency_boost_factor: float = 1.5  # Multiplier for recent content
    temporal_weight: float = 0.3  # Weight of temporal score in final ranking
    profile: TemporalProfile = TemporalProfile.RESEARCH

    def get_decay_constant(self) -> float:
        """Get decay constant based on profile"""
        decay_multipliers = {
            TemporalProfile.NEWS: 0.1,      # Very fast decay (36 days)
            TemporalProfile.FINANCIAL: 0.05, # Extreme decay (18 days)
            TemporalProfile.POLICY: 0.5,     # Fast decay (183 days)
            TemporalProfile.RESEARCH: 1.0,   # Normal decay (365 days)
            TemporalProfile.REFERENCE: 3.0   # Slow decay (1095 days)
        }
        return self.base_decay_days * decay_multipliers.get(self.profile, 1.0)


class TemporalRanking:
    """Temporal-aware ranking with recency boost and domain-specific decay"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_config = self._load_temporal_config(config)

        # Temporal pattern detection
        self.temporal_keywords = {
            "high_temporal": ["latest", "recent", "current", "new", "updated", "breaking"],
            "date_specific": ["2024", "2023", "today", "yesterday", "this week", "this month"],
            "trend_keywords": ["trend", "emerging", "developing", "ongoing", "evolving"]
        }

        # Domain-specific temporal profiles
        self.domain_profiles = {
            "news": TemporalProfile.NEWS,
            "cnn.com": TemporalProfile.NEWS,
            "bbc.com": TemporalProfile.NEWS,
            "reuters.com": TemporalProfile.NEWS,
            "arxiv.org": TemporalProfile.RESEARCH,
            "pubmed.ncbi.nlm.nih.gov": TemporalProfile.RESEARCH,
            "sec.gov": TemporalProfile.FINANCIAL,
            "bloomberg.com": TemporalProfile.FINANCIAL,
            "europa.eu": TemporalProfile.POLICY,
            "whitehouse.gov": TemporalProfile.POLICY,
            "wikipedia.org": TemporalProfile.REFERENCE
        }

        logger.info("Temporal ranking initialized")

    def _load_temporal_config(self, config: Dict[str, Any]) -> TemporalConfig:
        """Load temporal configuration from config"""
        temporal_cfg = config.get("retrieval", {}).get("temporal", {})

        return TemporalConfig(
            base_decay_days=temporal_cfg.get("base_decay_days", 365.0),
            recency_boost_days=temporal_cfg.get("recency_boost_days", 30),
            recency_boost_factor=temporal_cfg.get("recency_boost_factor", 1.5),
            temporal_weight=temporal_cfg.get("temporal_weight", 0.3),
            profile=TemporalProfile(temporal_cfg.get("profile", "research"))
        )

    def detect_temporal_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Detect temporal intent in query (0.0 = no temporal intent, 1.0 = high temporal intent)"""

        query_lower = query.lower()

        # Check for explicit temporal keywords
        temporal_score = 0.0

        # High temporal keywords
        high_temporal_matches = sum(1 for kw in self.temporal_keywords["high_temporal"]
                                  if kw in query_lower)
        temporal_score += high_temporal_matches * 0.3

        # Date-specific keywords
        date_specific_matches = sum(1 for kw in self.temporal_keywords["date_specific"]
                                  if kw in query_lower)
        temporal_score += date_specific_matches * 0.4

        # Trend keywords
        trend_matches = sum(1 for kw in self.temporal_keywords["trend_keywords"]
                          if kw in query_lower)
        temporal_score += trend_matches * 0.2

        # Year patterns (2020-2024)
        year_pattern = r'\b(202[0-4])\b'
        if re.search(year_pattern, query_lower):
            temporal_score += 0.5

        # Context-based boosting
        if context:
            if context.get("query_type") == "temporal":
                temporal_score += 0.3
            if context.get("domain") in ["news", "finance", "policy"]:
                temporal_score += 0.2

        return min(temporal_score, 1.0)

    def get_temporal_profile(self, document: Dict[str, Any]) -> TemporalProfile:
        """Determine temporal profile for a document based on source"""

        source_url = document.get("source_url", "")
        domain = self._extract_domain(source_url)

        # Check domain-specific profiles
        if domain in self.domain_profiles:
            return self.domain_profiles[domain]

        # Check by source type in metadata
        metadata = document.get("metadata", {})
        source_type = metadata.get("source_type", "")

        if source_type in ["news", "newspaper"]:
            return TemporalProfile.NEWS
        elif source_type in ["academic", "journal"]:
            return TemporalProfile.RESEARCH
        elif source_type in ["government", "policy"]:
            return TemporalProfile.POLICY
        elif source_type in ["financial", "market"]:
            return TemporalProfile.FINANCIAL
        else:
            return TemporalProfile.REFERENCE

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return ""

    def calculate_temporal_score(
        self,
        document: Dict[str, Any],
        query_temporal_intent: float,
        reference_date: Optional[datetime] = None
    ) -> float:
        """Calculate temporal relevance score for a document"""

        if reference_date is None:
            reference_date = datetime.now()

        # Get document publication date
        pub_date = self._extract_publication_date(document)
        if not pub_date:
            return 0.5  # Neutral score for undated content

        # Get temporal profile for this document
        profile = self.get_temporal_profile(document)

        # Create config for this profile
        config = TemporalConfig(
            base_decay_days=self.temporal_config.base_decay_days,
            recency_boost_days=self.temporal_config.recency_boost_days,
            recency_boost_factor=self.temporal_config.recency_boost_factor,
            temporal_weight=self.temporal_config.temporal_weight,
            profile=profile
        )

        # Calculate age in days
        age_days = (reference_date - pub_date).days

        # Calculate base temporal score with profile-specific decay
        decay_constant = config.get_decay_constant()
        base_score = np.exp(-age_days / decay_constant)

        # Recency boost for very recent content
        if age_days <= config.recency_boost_days:
            boost_factor = config.recency_boost_factor * (1 - age_days / config.recency_boost_days)
            base_score *= boost_factor

        # Special handling for different profiles
        if profile == TemporalProfile.POLICY:
            # Step function for policy: recent regulations much more important
            if age_days <= 90:  # 3 months
                base_score = max(base_score, 0.9)
            elif age_days <= 365:  # 1 year
                base_score = max(base_score, 0.7)

        elif profile == TemporalProfile.FINANCIAL:
            # Very steep decay for financial information
            if age_days <= 1:
                base_score = 1.0
            elif age_days <= 7:
                base_score = max(base_score, 0.8)
            elif age_days <= 30:
                base_score = max(base_score, 0.6)

        elif profile == TemporalProfile.REFERENCE:
            # Boost classic/foundational content
            if age_days > 1095:  # > 3 years
                citation_count = document.get("metadata", {}).get("citation_count", 0)
                if citation_count > 100:  # Well-cited classic content
                    base_score = max(base_score, 0.7)

        # Apply query temporal intent weighting
        final_score = base_score * query_temporal_intent + 0.5 * (1 - query_temporal_intent)

        return min(final_score, 1.0)

    def _extract_publication_date(self, document: Dict[str, Any]) -> Optional[datetime]:
        """Extract publication date from document metadata"""

        metadata = document.get("metadata", {})

        # Try different date fields
        date_fields = [
            "publication_date", "pub_date", "date", "created_date",
            "published", "timestamp", "article_date"
        ]

        for field in date_fields:
            date_str = metadata.get(field)
            if date_str:
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    return parsed_date

        # Try to extract from content or URL
        content = document.get("content", "")
        url = document.get("source_url", "")

        # Look for date patterns in content (first 500 chars)
        date_from_content = self._extract_date_from_text(content[:500])
        if date_from_content:
            return date_from_content

        # Look for date in URL
        date_from_url = self._extract_date_from_url(url)
        if date_from_url:
            return date_from_url

        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format attempts"""

        if not date_str:
            return None

        # Clean date string
        date_str = str(date_str).strip()

        # Date formats to try
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date from text using regex patterns"""

        # Year-month-day patterns
        patterns = [
            r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b',
            r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](20\d{2})\b',
            r'\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](20\d{2})\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if pattern == patterns[0]:  # YYYY-MM-DD
                        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    elif pattern == patterns[1]:  # MM-DD-YYYY
                        return datetime(int(match.group(3)), int(match.group(1)), int(match.group(2)))
                    else:  # DD-MM-YYYY
                        return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
                except ValueError:
                    continue

        return None

    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        """Extract date from URL path"""

        # Pattern: /2024/03/15/ or /2024-03-15
        date_pattern = r'/(\d{4})[/-](\d{1,2})[/-](\d{1,2})/?'
        match = re.search(date_pattern, url)

        if match:
            try:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                pass

        return None

    def apply_temporal_ranking(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        base_scores: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Apply temporal ranking to document list"""

        if not documents:
            return documents

        # Detect temporal intent
        temporal_intent = self.detect_temporal_intent(query, context)

        # If no temporal intent, return original ranking
        if temporal_intent < 0.1:
            logger.info(f"Low temporal intent ({temporal_intent:.2f}), skipping temporal ranking")
            return documents

        logger.info(f"Applying temporal ranking (intent: {temporal_intent:.2f})")

        # Calculate temporal scores
        ranked_docs = []
        for i, doc in enumerate(documents):
            temporal_score = self.calculate_temporal_score(doc, temporal_intent)

            # Combine with base score if provided
            base_score = base_scores[i] if base_scores and i < len(base_scores) else 1.0

            # Weighted combination
            final_score = (
                base_score * (1 - self.temporal_config.temporal_weight) +
                temporal_score * self.temporal_config.temporal_weight
            )

            doc_copy = doc.copy()
            doc_copy["temporal_score"] = temporal_score
            doc_copy["base_score"] = base_score
            doc_copy["final_score"] = final_score

            ranked_docs.append(doc_copy)

        # Sort by final score
        ranked_docs.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(f"Temporal ranking completed: reordered {len(ranked_docs)} documents")
        return ranked_docs

    def get_temporal_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get temporal statistics for document set"""

        if not documents:
            return {}

        dates = []
        profiles = []

        for doc in documents:
            pub_date = self._extract_publication_date(doc)
            if pub_date:
                dates.append(pub_date)

            profile = self.get_temporal_profile(doc)
            profiles.append(profile.value)

        stats = {
            "total_documents": len(documents),
            "documents_with_dates": len(dates),
            "date_coverage_ratio": len(dates) / len(documents) if documents else 0
        }

        if dates:
            stats.update({
                "oldest_date": min(dates).isoformat(),
                "newest_date": max(dates).isoformat(),
                "date_span_days": (max(dates) - min(dates)).days,
                "avg_age_days": sum((datetime.now() - d).days for d in dates) / len(dates)
            })

        # Profile distribution
        from collections import Counter
        profile_counts = Counter(profiles)
        stats["profile_distribution"] = dict(profile_counts)

        return stats


def create_temporal_ranking(config: Dict[str, Any]) -> TemporalRanking:
    """Factory function for temporal ranking"""
    return TemporalRanking(config)


# Usage example
if __name__ == "__main__":
    config = {
        "retrieval": {
            "temporal": {
                "base_decay_days": 365.0,
                "recency_boost_days": 30,
                "temporal_weight": 0.4
            }
        }
    }

    ranking = TemporalRanking(config)

    # Test temporal intent detection
    queries = [
        "COVID-19 vaccine effectiveness",  # Low temporal
        "Latest COVID-19 research 2024",  # High temporal
        "Recent developments in AI",      # Medium temporal
        "Historical data on pandemics"    # Low temporal
    ]

    for query in queries:
        intent = ranking.detect_temporal_intent(query)
        print(f"Query: {query}")
        print(f"Temporal intent: {intent:.2f}")
        print()
