#!/usr/bin/env python3
"""
Credibility Assessor - Posuzovač důvěryhodnosti zdrojů
Výpočet váženého skóre důvěryhodnosti na základě více faktorů

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import re
import hashlib
import math
from datetime import datetime, timedelta
from urllib.parse import urlparse
import statistics

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SourceMetadata:
    """Metadata o zdroji informací"""

    source_id: str
    url: str
    domain: str
    title: str
    content_length: int
    publication_date: Optional[datetime]
    author: Optional[str]
    source_type: str  # news, blog, forum, social_media, academic, government
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentQualityMetrics:
    """Metriky kvality obsahu"""

    readability_score: float
    grammar_quality: float
    factual_density: float
    citation_count: int
    external_links_count: int
    spelling_errors: int
    sentiment_bias: float
    objectivity_score: float
    depth_score: float


@dataclass
class DomainReputationMetrics:
    """Metriky reputace domény"""

    domain_age: Optional[int]  # ve dnech
    alexa_rank: Optional[int]
    trustworthiness_score: float
    known_disinformation: bool
    fact_check_rating: Optional[str]
    blacklist_status: bool
    ssl_certificate: bool
    domain_category: str


@dataclass
class TemporalRelevanceMetrics:
    """Metriky časové relevance"""

    publication_age: Optional[int]  # ve dnech
    content_freshness: float
    update_frequency: float
    temporal_relevance: float
    historical_accuracy: float


@dataclass
class CredibilityAssessment:
    """Kompletní hodnocení důvěryhodnosti"""

    source_metadata: SourceMetadata
    content_quality: ContentQualityMetrics
    domain_reputation: DomainReputationMetrics
    temporal_relevance: TemporalRelevanceMetrics
    overall_credibility_score: float
    confidence: float
    risk_factors: List[str]
    positive_indicators: List[str]
    assessment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    detailed_breakdown: Dict[str, float] = field(default_factory=dict)


class CredibilityAssessor:
    """Pokročilý posuzovač důvěryhodnosti pro zpravodajské účely"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credibility_config = config.get("credibility", {})

        # Váhy pro různé faktory
        self.factor_weights = self.credibility_config.get(
            "factor_weights",
            {
                "domain_reputation": 0.25,
                "content_quality": 0.30,
                "temporal_relevance": 0.20,
                "source_authority": 0.15,
                "bias_detection": 0.10,
            },
        )

        # Threshold hodnoty
        self.credibility_thresholds = self.credibility_config.get(
            "thresholds",
            {"high_credibility": 0.8, "medium_credibility": 0.6, "low_credibility": 0.4},
        )

        # Databáze známých zdrojů
        self.domain_reputation_db = self._load_domain_reputation_db()
        self.blacklisted_domains = self._load_blacklisted_domains()
        self.trusted_domains = self._load_trusted_domains()

        # NLP komponenty
        self.sentiment_analyzer = None
        self.stopwords_set = set()

        # Cache a statistiky
        self.assessment_cache: Dict[str, CredibilityAssessment] = {}
        self.domain_stats = defaultdict(list)
        self.assessment_stats = defaultdict(int)

        self._initialize_nlp()

    def _initialize_nlp(self):
        """Inicializace NLP komponent"""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - sentiment analysis will be limited")
            return

        try:
            # Download required NLTK data
            import ssl

            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download("vader_lexicon", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stopwords_set = set(stopwords.words("english"))

            logger.info("NLTK components initialized successfully")

        except Exception as e:
            logger.warning(f"Error initializing NLTK: {e}")

    def _load_domain_reputation_db(self) -> Dict[str, Dict[str, Any]]:
        """Načtení databáze reputace domén"""
        # Předpřipravená databáze důvěryhodných domén
        trusted_sources = {
            # Zpravodajské weby
            "reuters.com": {"trustworthiness": 0.95, "category": "news", "bias": "minimal"},
            "ap.org": {"trustworthiness": 0.95, "category": "news", "bias": "minimal"},
            "bbc.com": {"trustworthiness": 0.90, "category": "news", "bias": "minimal"},
            "cnn.com": {"trustworthiness": 0.80, "category": "news", "bias": "left-center"},
            "foxnews.com": {"trustworthiness": 0.75, "category": "news", "bias": "right"},
            # Akademické zdroje
            "scholar.google.com": {
                "trustworthiness": 0.95,
                "category": "academic",
                "bias": "minimal",
            },
            "pubmed.ncbi.nlm.nih.gov": {
                "trustworthiness": 0.98,
                "category": "academic",
                "bias": "minimal",
            },
            "arxiv.org": {"trustworthiness": 0.90, "category": "academic", "bias": "minimal"},
            # Vládní weby
            "fbi.gov": {"trustworthiness": 0.95, "category": "government", "bias": "minimal"},
            "cia.gov": {"trustworthiness": 0.95, "category": "government", "bias": "minimal"},
            "nist.gov": {"trustworthiness": 0.95, "category": "government", "bias": "minimal"},
            # Fact-checking weby
            "snopes.com": {"trustworthiness": 0.85, "category": "fact_check", "bias": "minimal"},
            "factcheck.org": {"trustworthiness": 0.90, "category": "fact_check", "bias": "minimal"},
            "politifact.com": {
                "trustworthiness": 0.85,
                "category": "fact_check",
                "bias": "minimal",
            },
        }

        # Spojení s uživatelskou konfigurací
        user_domains = self.credibility_config.get("domain_reputation", {})
        return {**trusted_sources, **user_domains}

    def _load_blacklisted_domains(self) -> Set[str]:
        """Načtení seznamu blacklistovaných domén"""
        default_blacklist = {
            # Známé dezinformační weby
            "infowars.com",
            "breitbart.com",
            "naturalnews.com",
            "beforeitsnews.com",
            "activistpost.com",
            # Satirické weby často mylně brané vážně
            "theonion.com",
            "babylonbee.com",
            # Známé clickbait weby
            "buzzfeed.com",
            "upworthy.com",
        }

        user_blacklist = set(self.credibility_config.get("blacklisted_domains", []))
        return default_blacklist.union(user_blacklist)

    def _load_trusted_domains(self) -> Set[str]:
        """Načtení seznamu důvěryhodných domén"""
        default_trusted = {
            # Prestižní zpravodajské organizace
            "reuters.com",
            "ap.org",
            "bbc.com",
            "npr.org",
            # Akademické instituce
            "mit.edu",
            "harvard.edu",
            "stanford.edu",
            "cambridge.org",
            # Vládní instituce
            "gov",
            "edu",
            "mil",
            # Mezinárodní organizace
            "un.org",
            "who.int",
            "worldbank.org",
        }

        user_trusted = set(self.credibility_config.get("trusted_domains", []))
        return default_trusted.union(user_trusted)

    async def assess_content(self, content: str, url: str = "") -> float:
        """
        Veřejná metoda pro rychlé hodnocení důvěryhodnosti obsahu
        Používá se z AgenticLoop a dalších komponent

        Args:
            content: Text obsahu k hodnocení
            url: URL zdroje (volitelné)

        Returns:
            float: Skóre důvěryhodnosti 0.0-1.0
        """
        try:
            # Vytvoření základního SourceMetadata
            from urllib.parse import urlparse

            parsed_url = urlparse(url) if url else None

            source_metadata = SourceMetadata(
                source_id=f"temp_{hash(content) % 10000}",
                url=url,
                domain=parsed_url.netloc if parsed_url else "unknown",
                title="",
                content_length=len(content),
                publication_date=None,
                author=None,
                source_type="unknown",
                language="cs",
            )

            # Rychlé hodnocení kvality obsahu
            content_quality = await self._assess_content_quality(content, source_metadata)

            # Základní hodnocení domény pokud je URL dostupné
            if url:
                domain_reputation = await self._assess_domain_reputation(source_metadata)
                # Kombinace obsahu a domény
                final_score = (
                    content_quality.overall_score * 0.7 + domain_reputation.overall_score * 0.3
                )
            else:
                # Pouze kvalita obsahu
                final_score = content_quality.overall_score

            return min(1.0, max(0.0, final_score))

        except Exception as e:
            logger.error(f"Chyba při hodnocení důvěryhodnosti obsahu: {e}")
            return 0.5  # Neutrální skóre při chybě

    async def assess_source_credibility(
        self, content: str, source_metadata: SourceMetadata
    ) -> CredibilityAssessment:
        """Hlavní metoda pro hodnocení důvěryhodnosti zdroje"""

        # Cache kontrola
        cache_key = hashlib.md5(
            f"{source_metadata.source_id}:{source_metadata.url}".encode()
        ).hexdigest()
        if cache_key in self.assessment_cache:
            logger.info(f"Returning cached credibility assessment for {source_metadata.domain}")
            return self.assessment_cache[cache_key]

        start_time = datetime.now()

        try:
            # Analýza kvality obsahu
            content_quality = await self._assess_content_quality(content, source_metadata)

            # Analýza reputace domény
            domain_reputation = await self._assess_domain_reputation(source_metadata)

            # Analýza časové relevance
            temporal_relevance = await self._assess_temporal_relevance(source_metadata)

            # Výpočet celkového skóre
            overall_score, detailed_breakdown = await self._calculate_overall_credibility(
                content_quality, domain_reputation, temporal_relevance
            )

            # Identifikace rizikových faktorů a pozitivních indikátorů
            risk_factors = await self._identify_risk_factors(
                content_quality, domain_reputation, temporal_relevance, source_metadata
            )
            positive_indicators = await self._identify_positive_indicators(
                content_quality, domain_reputation, temporal_relevance, source_metadata
            )

            # Výpočet konfidence
            confidence = await self._calculate_assessment_confidence(
                content_quality, domain_reputation, temporal_relevance
            )

            # Vytvoření finálního hodnocení
            assessment = CredibilityAssessment(
                source_metadata=source_metadata,
                content_quality=content_quality,
                domain_reputation=domain_reputation,
                temporal_relevance=temporal_relevance,
                overall_credibility_score=overall_score,
                confidence=confidence,
                risk_factors=risk_factors,
                positive_indicators=positive_indicators,
                detailed_breakdown=detailed_breakdown,
            )

            # Cache uložení
            self.assessment_cache[cache_key] = assessment

            # Statistiky
            assessment_time = (datetime.now() - start_time).total_seconds()
            self.assessment_stats[f"assessment_time_{source_metadata.domain}"] = assessment_time
            self.assessment_stats["total_assessments"] += 1
            self.domain_stats[source_metadata.domain].append(overall_score)

            logger.info(
                f"Credibility assessment completed for {source_metadata.domain} in {assessment_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error assessing credibility for {source_metadata.domain}: {e}")
            # Fallback hodnocení
            assessment = self._create_fallback_assessment(source_metadata, str(e))

        return assessment

    async def _assess_content_quality(
        self, content: str, source_metadata: SourceMetadata
    ) -> ContentQualityMetrics:
        """Hodnocení kvality obsahu"""

        # Základní metriky
        content_length = len(content)
        word_count = len(content.split())

        # Readability score (Flesch Reading Ease aproximace)
        readability_score = await self._calculate_readability(content)

        # Grammar quality (aproximace založená na interpunkci a struktuře)
        grammar_quality = await self._assess_grammar_quality(content)

        # Factual density (počet faktických tvrzení)
        factual_density = await self._calculate_factual_density(content)

        # Citation count
        citation_count = await self._count_citations(content)

        # External links
        external_links_count = await self._count_external_links(content)

        # Spelling errors (aproximace)
        spelling_errors = await self._count_spelling_errors(content)

        # Sentiment bias
        sentiment_bias = await self._calculate_sentiment_bias(content)

        # Objectivity score
        objectivity_score = await self._calculate_objectivity_score(content)

        # Depth score (na základě délky a struktury)
        depth_score = await self._calculate_depth_score(content, word_count)

        return ContentQualityMetrics(
            readability_score=readability_score,
            grammar_quality=grammar_quality,
            factual_density=factual_density,
            citation_count=citation_count,
            external_links_count=external_links_count,
            spelling_errors=spelling_errors,
            sentiment_bias=sentiment_bias,
            objectivity_score=objectivity_score,
            depth_score=depth_score,
        )

    async def _assess_domain_reputation(
        self, source_metadata: SourceMetadata
    ) -> DomainReputationMetrics:
        """Hodnocení reputace domény"""
        domain = source_metadata.domain

        # Základní trustworthiness ze znalostní báze
        domain_info = self.domain_reputation_db.get(domain, {})
        trustworthiness_score = domain_info.get("trustworthiness", 0.5)  # Default střední

        # Kontrola blacklistu
        blacklist_status = domain in self.blacklisted_domains
        if blacklist_status:
            trustworthiness_score *= 0.2  # Drastické snížení

        # Kontrola trusted domén
        if any(trusted_domain in domain for trusted_domain in self.trusted_domains):
            trustworthiness_score = max(trustworthiness_score, 0.8)

        # Domain age (simulace - v reálné implementaci by se použilo WHOIS API)
        domain_age = await self._estimate_domain_age(domain)

        # SSL certificate check (simulace)
        ssl_certificate = await self._check_ssl_certificate(source_metadata.url)

        # Domain category
        domain_category = domain_info.get("category", await self._classify_domain_category(domain))

        # Fact-check rating
        fact_check_rating = domain_info.get("fact_check_rating")

        # Known disinformation check
        known_disinformation = domain in self.blacklisted_domains

        return DomainReputationMetrics(
            domain_age=domain_age,
            alexa_rank=None,  # Vyžaduje externí API
            trustworthiness_score=trustworthiness_score,
            known_disinformation=known_disinformation,
            fact_check_rating=fact_check_rating,
            blacklist_status=blacklist_status,
            ssl_certificate=ssl_certificate,
            domain_category=domain_category,
        )

    async def _assess_temporal_relevance(
        self, source_metadata: SourceMetadata
    ) -> TemporalRelevanceMetrics:
        """Hodnocení časové relevance"""
        now = datetime.now()

        # Publication age
        publication_age = None
        content_freshness = 0.5  # Default

        if source_metadata.publication_date:
            publication_age = (now - source_metadata.publication_date).days
            # Freshness klesá s časem
            content_freshness = max(0.1, 1.0 - (publication_age / 365.0))

        # Update frequency (simulace - v reálné implementaci by se trackovala)
        update_frequency = await self._estimate_update_frequency(source_metadata.domain)

        # Temporal relevance (kombinace freshness a aktuálnosti tématu)
        temporal_relevance = content_freshness * 0.7 + update_frequency * 0.3

        # Historical accuracy (simulace - hodnocení přesnosti historických informací)
        historical_accuracy = await self._assess_historical_accuracy(source_metadata)

        return TemporalRelevanceMetrics(
            publication_age=publication_age,
            content_freshness=content_freshness,
            update_frequency=update_frequency,
            temporal_relevance=temporal_relevance,
            historical_accuracy=historical_accuracy,
        )

    async def _calculate_readability(self, content: str) -> float:
        """Výpočet čitelnosti textu (Flesch Reading Ease aproximace)"""
        sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split(".")
        words = word_tokenize(content) if NLTK_AVAILABLE else content.split()

        if len(sentences) == 0 or len(words) == 0:
            return 0.5

        avg_sentence_length = len(words) / len(sentences)

        # Aproximace syllable count
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words) if len(words) > 0 else 0

        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Normalizace na 0-1
        return max(0.0, min(1.0, flesch_score / 100.0))

    def _count_syllables(self, word: str) -> int:
        """Aproximace počtu slabik v slově"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Silent e
        if word.endswith("e"):
            syllable_count -= 1

        return max(1, syllable_count)

    async def _assess_grammar_quality(self, content: str) -> float:
        """Hodnocení gramatické kvality (aproximace)"""
        # Základní kontroly
        sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split(".")

        if not sentences:
            return 0.0

        # Kontrola interpunkce
        punctuation_score = 0.0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[-1] in ".!?":
                punctuation_score += 1

        punctuation_score /= len(sentences)

        # Kontrola velkých písmen na začátku vět
        capitalization_score = 0.0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[0].isupper():
                capitalization_score += 1

        capitalization_score /= len(sentences)

        # Kontrola délky vět (příliš dlouhé nebo krátké věty snižují kvalitu)
        sentence_length_score = 0.0
        for sentence in sentences:
            words = sentence.split()
            length = len(words)
            if 5 <= length <= 25:  # Optimální délka věty
                sentence_length_score += 1
            elif length > 0:
                sentence_length_score += max(0.3, 1.0 - abs(length - 15) / 20.0)

        sentence_length_score /= len(sentences)

        # Kombinovaná grammar quality
        return (punctuation_score + capitalization_score + sentence_length_score) / 3.0

    async def _calculate_factual_density(self, content: str) -> float:
        """Výpočet hustoty faktických informací"""
        # Vzory pro faktická tvrzení
        factual_patterns = [
            r"\b\d+%\b",  # Procenta
            r"\b\d+[\s,]\d+\b",  # Čísla s čárkami
            r"\b(according to|study shows|research indicates|data shows)\b",  # Odkazování na zdroje
            r"\b(published|reported|announced|confirmed)\b",  # Oznámení faktů
            r"\b\d{4}\b",  # Roky
            r"\$\d+",  # Peněžní částky
        ]

        factual_matches = 0
        for pattern in factual_patterns:
            factual_matches += len(re.findall(pattern, content, re.IGNORECASE))

        words = content.split()
        factual_density = factual_matches / len(words) if words else 0

        return min(1.0, factual_density * 100)  # Normalizace

    async def _count_citations(self, content: str) -> int:
        """Počítání citací v textu"""
        citation_patterns = [
            r"\[\d+\]",  # [1], [2], etc.
            r"\(\d{4}\)",  # (2023)
            r"(et al\.|et al)",  # et al.
            r"\b(Source:|Reference:|According to:)\b",
            r"https?://[^\s]+",  # URL odkazy
        ]

        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content, re.IGNORECASE))

        return citation_count

    async def _count_external_links(self, content: str) -> int:
        """Počítání externích odkazů"""
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, content)
        return len(urls)

    async def _count_spelling_errors(self, content: str) -> int:
        """Odhad počtu pravopisných chyb"""
        # Jednoduchá heuristika - neobvyklé sekvence znaků
        words = content.split()
        error_count = 0

        for word in words:
            # Čistý word bez interpunkce
            clean_word = re.sub(r"[^\w]", "", word).lower()

            # Kontroly na chyby
            if len(clean_word) > 2:
                # Více než 3 stejné písmena za sebou
                if re.search(r"(.)\1{3,}", clean_word):
                    error_count += 1
                # Neobvyklé kombinace souhlásek
                elif re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", clean_word):
                    error_count += 1
                # Neobvyklé kombinace samohlásek
                elif re.search(r"[aeiou]{4,}", clean_word):
                    error_count += 1

        return error_count

    async def _calculate_sentiment_bias(self, content: str) -> float:
        """Výpočet sentimentové zaujatosti"""
        if not self.sentiment_analyzer:
            # Fallback bez NLTK
            emotional_words = [
                "amazing",
                "terrible",
                "fantastic",
                "awful",
                "incredible",
                "horrible",
                "outstanding",
                "disgusting",
                "wonderful",
                "pathetic",
                "brilliant",
                "stupid",
            ]

            content_lower = content.lower()
            emotional_count = sum(1 for word in emotional_words if word in content_lower)
            words = content.split()
            return min(1.0, emotional_count / len(words) * 100) if words else 0

        # NLTK sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(content)

        # Bias score jako absolutní hodnota compound score
        bias_score = abs(sentiment_scores["compound"])

        return bias_score

    async def _calculate_objectivity_score(self, content: str) -> float:
        """Výpočet objektivity textu"""
        # Subjektivní indikátory
        subjective_patterns = [
            r"\b(I think|I believe|In my opinion|It seems|appears to be)\b",
            r"\b(obviously|clearly|definitely|certainly|undoubtedly)\b",
            r"\b(should|must|need to|have to)\b",
            r"[!]{2,}",  # Více výkřičníků
            r"[?]{2,}",  # Více otazníků
        ]

        subjective_count = 0
        for pattern in subjective_patterns:
            subjective_count += len(re.findall(pattern, content, re.IGNORECASE))

        # Objektivní indikátory
        objective_patterns = [
            r"\b(according to|study shows|research indicates|data shows)\b",
            r"\b(reported|published|announced|confirmed)\b",
            r"\b\d+% of\b",
            r"\b(statistics show|surveys indicate)\b",
        ]

        objective_count = 0
        for pattern in objective_patterns:
            objective_count += len(re.findall(pattern, content, re.IGNORECASE))

        total_indicators = subjective_count + objective_count
        if total_indicators == 0:
            return 0.5  # Neutrální

        objectivity_score = objective_count / total_indicators
        return objectivity_score

    async def _calculate_depth_score(self, content: str, word_count: int) -> float:
        """Výpočet hloubky obsahu"""
        # Faktory pro hloubku
        depth_factors = 0.0

        # Délka obsahu
        if word_count >= 500:
            depth_factors += 0.3
        elif word_count >= 200:
            depth_factors += 0.2
        else:
            depth_factors += 0.1

        # Strukturované informace
        if re.search(
            r"\b(Background:|Introduction:|Method:|Results:|Conclusion:)\b", content, re.IGNORECASE
        ):
            depth_factors += 0.2

        # Důkazové materiály
        if re.search(r"\b(evidence|proof|data|statistics|survey|study)\b", content, re.IGNORECASE):
            depth_factors += 0.2

        # Více perspektiv
        if re.search(
            r"\b(however|on the other hand|alternatively|conversely)\b", content, re.IGNORECASE
        ):
            depth_factors += 0.2

        # Historický kontext
        if re.search(r"\b(historically|in the past|previously|earlier)\b", content, re.IGNORECASE):
            depth_factors += 0.1

        return min(1.0, depth_factors)

    async def _estimate_domain_age(self, domain: str) -> Optional[int]:
        """Odhad stáří domény (simulace)"""
        # V reálné implementaci by se použilo WHOIS API
        # Simulace na základě známých informací
        known_old_domains = {
            "bbc.com": 365 * 30,  # ~30 let
            "cnn.com": 365 * 25,  # ~25 let
            "reuters.com": 365 * 25,
            "ap.org": 365 * 20,
        }

        if domain in known_old_domains:
            return known_old_domains[domain]

        # Heuristiky pro odhad
        if any(tld in domain for tld in [".gov", ".edu", ".org"]):
            return 365 * 15  # Průměrně starší domény
        elif any(tld in domain for tld in [".com", ".net"]):
            return 365 * 10  # Střední věk
        else:
            return 365 * 5  # Novější domény

    async def _check_ssl_certificate(self, url: str) -> bool:
        """Kontrola SSL certifikátu"""
        # Jednoduchá kontrola podle URL
        return url.startswith("https://")

    async def _classify_domain_category(self, domain: str) -> str:
        """Klasifikace kategorie domény"""
        if any(tld in domain for tld in [".gov", ".mil"]):
            return "government"
        elif ".edu" in domain:
            return "academic"
        elif any(keyword in domain for keyword in ["news", "times", "post", "herald"]):
            return "news"
        elif any(keyword in domain for keyword in ["blog", "medium", "wordpress"]):
            return "blog"
        elif any(keyword in domain for keyword in ["facebook", "twitter", "instagram", "social"]):
            return "social_media"
        elif any(keyword in domain for keyword in ["forum", "reddit", "discussion"]):
            return "forum"
        else:
            return "general"

    async def _estimate_update_frequency(self, domain: str) -> float:
        """Odhad frekvence aktualizace"""
        # Simulace na základě typu domény
        domain_info = self.domain_reputation_db.get(domain, {})
        category = domain_info.get("category", "general")

        frequency_map = {
            "news": 0.9,  # Velmi často aktualizováno
            "government": 0.6,  # Středně často
            "academic": 0.4,  # Méně často
            "blog": 0.7,  # Často
            "forum": 0.8,  # Velmi často
            "social_media": 0.95,  # Neustále
            "general": 0.5,
        }

        return frequency_map.get(category, 0.5)

    async def _assess_historical_accuracy(self, source_metadata: SourceMetadata) -> float:
        """Hodnocení historické přesnosti"""
        # Simulace - v reálné implementaci by se použila databáze fact-checků
        domain = source_metadata.domain

        if domain in self.trusted_domains:
            return 0.9
        elif domain in self.blacklisted_domains:
            return 0.2
        else:
            domain_info = self.domain_reputation_db.get(domain, {})
            return domain_info.get("historical_accuracy", 0.6)

    async def _calculate_overall_credibility(
        self,
        content_quality: ContentQualityMetrics,
        domain_reputation: DomainReputationMetrics,
        temporal_relevance: TemporalRelevanceMetrics,
    ) -> Tuple[float, Dict[str, float]]:
        """Výpočet celkového skóre důvěryhodnosti"""

        # Normalizace jednotlivých komponent
        content_score = (
            content_quality.readability_score * 0.15
            + content_quality.grammar_quality * 0.15
            + content_quality.objectivity_score * 0.25
            + content_quality.factual_density * 0.20
            + min(1.0, content_quality.citation_count / 10) * 0.15
            + (1.0 - min(1.0, content_quality.sentiment_bias)) * 0.10
        )

        domain_score = (
            domain_reputation.trustworthiness_score * 0.6
            + (0.0 if domain_reputation.blacklist_status else 0.2)
            + (0.1 if domain_reputation.ssl_certificate else 0.0)
            + (0.1 if domain_reputation.domain_age and domain_reputation.domain_age > 365 else 0.05)
        )

        temporal_score = (
            temporal_relevance.content_freshness * 0.4
            + temporal_relevance.temporal_relevance * 0.3
            + temporal_relevance.historical_accuracy * 0.3
        )

        # Aplikace vah
        weights = self.factor_weights
        overall_score = (
            content_score * weights.get("content_quality", 0.3)
            + domain_score * weights.get("domain_reputation", 0.25)
            + temporal_score * weights.get("temporal_relevance", 0.2)
            + domain_score * weights.get("source_authority", 0.15)
            + (1.0 - content_quality.sentiment_bias) * weights.get("bias_detection", 0.1)
        )

        detailed_breakdown = {
            "content_quality_score": content_score,
            "domain_reputation_score": domain_score,
            "temporal_relevance_score": temporal_score,
            "bias_penalty": content_quality.sentiment_bias,
            "final_weighted_score": overall_score,
        }

        return min(1.0, max(0.0, overall_score)), detailed_breakdown

    async def _identify_risk_factors(
        self,
        content_quality: ContentQualityMetrics,
        domain_reputation: DomainReputationMetrics,
        temporal_relevance: TemporalRelevanceMetrics,
        source_metadata: SourceMetadata,
    ) -> List[str]:
        """Identifikace rizikových faktorů"""
        risk_factors = []

        # Domain-based risks
        if domain_reputation.blacklist_status:
            risk_factors.append("Domain is on blacklist of unreliable sources")

        if domain_reputation.known_disinformation:
            risk_factors.append("Domain known for spreading disinformation")

        if not domain_reputation.ssl_certificate:
            risk_factors.append("Website lacks SSL certificate (security risk)")

        if domain_reputation.trustworthiness_score < 0.3:
            risk_factors.append("Domain has very low trustworthiness rating")

        # Content-based risks
        if content_quality.sentiment_bias > 0.7:
            risk_factors.append("Content shows high emotional bias")

        if content_quality.objectivity_score < 0.3:
            risk_factors.append("Content lacks objectivity markers")

        if content_quality.grammar_quality < 0.5:
            risk_factors.append("Poor grammar quality detected")

        if content_quality.spelling_errors > 10:
            risk_factors.append(
                f"High number of spelling errors ({content_quality.spelling_errors})"
            )

        if content_quality.citation_count == 0 and content_quality.factual_density > 0.1:
            risk_factors.append("Factual claims without citations or sources")

        # Temporal risks
        if temporal_relevance.content_freshness < 0.2:
            risk_factors.append("Content is significantly outdated")

        if temporal_relevance.historical_accuracy < 0.4:
            risk_factors.append("Source has history of inaccurate reporting")

        return risk_factors

    async def _identify_positive_indicators(
        self,
        content_quality: ContentQualityMetrics,
        domain_reputation: DomainReputationMetrics,
        temporal_relevance: TemporalRelevanceMetrics,
        source_metadata: SourceMetadata,
    ) -> List[str]:
        """Identifikace pozitivních indikátorů"""
        positive_indicators = []

        # Domain-based positives
        if domain_reputation.trustworthiness_score > 0.8:
            positive_indicators.append("Highly trusted domain")

        if source_metadata.domain in self.trusted_domains:
            positive_indicators.append("Domain is on trusted sources list")

        if domain_reputation.domain_category in ["academic", "government"]:
            positive_indicators.append(
                f"Authoritative source type: {domain_reputation.domain_category}"
            )

        if domain_reputation.ssl_certificate:
            positive_indicators.append("Secure HTTPS connection")

        # Content-based positives
        if content_quality.citation_count > 5:
            positive_indicators.append(
                f"Well-cited content ({content_quality.citation_count} citations)"
            )

        if content_quality.objectivity_score > 0.7:
            positive_indicators.append("High objectivity markers")

        if content_quality.grammar_quality > 0.8:
            positive_indicators.append("High grammar quality")

        if content_quality.factual_density > 0.3:
            positive_indicators.append("High density of factual information")

        if content_quality.external_links_count > 3:
            positive_indicators.append("Contains external references")

        # Temporal positives
        if temporal_relevance.content_freshness > 0.8:
            positive_indicators.append("Recent and fresh content")

        if temporal_relevance.historical_accuracy > 0.8:
            positive_indicators.append("Source has strong accuracy track record")

        return positive_indicators

    async def _calculate_assessment_confidence(
        self,
        content_quality: ContentQualityMetrics,
        domain_reputation: DomainReputationMetrics,
        temporal_relevance: TemporalRelevanceMetrics,
    ) -> float:
        """Výpočet konfidence v hodnocení"""
        confidence_factors = []

        # Konfidence na základě dostupnosti dat
        if domain_reputation.trustworthiness_score > 0:
            confidence_factors.append(0.8)  # Máme data o doméně
        else:
            confidence_factors.append(0.4)  # Chybí data o doméně

        # Konfidence na základě délky obsahu
        if content_quality.depth_score > 0.5:
            confidence_factors.append(0.9)  # Dostatečný obsah pro analýzu
        else:
            confidence_factors.append(0.6)  # Omezený obsah

        # Konfidence na základě časových dat
        if temporal_relevance.publication_age is not None:
            confidence_factors.append(0.8)  # Známe datum publikace
        else:
            confidence_factors.append(0.5)  # Neznáme datum

        # Celková konfidence
        return statistics.mean(confidence_factors) if confidence_factors else 0.5

    def _create_fallback_assessment(
        self, source_metadata: SourceMetadata, error: str
    ) -> CredibilityAssessment:
        """Vytvoření fallback hodnocení při chybě"""
        return CredibilityAssessment(
            source_metadata=source_metadata,
            content_quality=ContentQualityMetrics(
                readability_score=0.5,
                grammar_quality=0.5,
                factual_density=0.5,
                citation_count=0,
                external_links_count=0,
                spelling_errors=0,
                sentiment_bias=0.5,
                objectivity_score=0.5,
                depth_score=0.5,
            ),
            domain_reputation=DomainReputationMetrics(
                domain_age=None,
                alexa_rank=None,
                trustworthiness_score=0.5,
                known_disinformation=False,
                fact_check_rating=None,
                blacklist_status=False,
                ssl_certificate=True,
                domain_category="unknown",
            ),
            temporal_relevance=TemporalRelevanceMetrics(
                publication_age=None,
                content_freshness=0.5,
                update_frequency=0.5,
                temporal_relevance=0.5,
                historical_accuracy=0.5,
            ),
            overall_credibility_score=0.5,
            confidence=0.3,
            risk_factors=[f"Assessment error: {error}"],
            positive_indicators=[],
            detailed_breakdown={"error": error},
        )

    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Získání statistik hodnocení"""
        domain_averages = {}
        for domain, scores in self.domain_stats.items():
            if scores:
                domain_averages[domain] = {
                    "average_score": statistics.mean(scores),
                    "assessment_count": len(scores),
                    "score_range": [min(scores), max(scores)],
                }

        return {
            "total_assessments": self.assessment_stats.get("total_assessments", 0),
            "cache_size": len(self.assessment_cache),
            "assessment_stats": dict(self.assessment_stats),
            "domain_averages": domain_averages,
            "trusted_domains_count": len(self.trusted_domains),
            "blacklisted_domains_count": len(self.blacklisted_domains),
            "known_domains_count": len(self.domain_reputation_db),
            "nltk_available": NLTK_AVAILABLE,
        }

    async def batch_assess_credibility(
        self, sources: List[Tuple[str, SourceMetadata]]
    ) -> List[CredibilityAssessment]:
        """Dávkové hodnocení více zdrojů"""
        assessments = []

        for content, metadata in sources:
            assessment = await self.assess_source_credibility(content, metadata)
            assessments.append(assessment)

        logger.info(f"Completed batch assessment of {len(sources)} sources")
        return assessments

    async def generate_credibility_report(
        self, assessments: List[CredibilityAssessment]
    ) -> Dict[str, Any]:
        """Generování komplexní zprávy o důvěryhodnosti"""
        if not assessments:
            return {"error": "No assessments provided"}

        # Kategorizace podle credibility score
        high_credibility = [
            a
            for a in assessments
            if a.overall_credibility_score >= self.credibility_thresholds["high_credibility"]
        ]
        medium_credibility = [
            a
            for a in assessments
            if self.credibility_thresholds["medium_credibility"]
            <= a.overall_credibility_score
            < self.credibility_thresholds["high_credibility"]
        ]
        low_credibility = [
            a
            for a in assessments
            if a.overall_credibility_score < self.credibility_thresholds["medium_credibility"]
        ]

        # Nejčastější rizikové faktory
        all_risk_factors = []
        for assessment in assessments:
            all_risk_factors.extend(assessment.risk_factors)
        risk_factor_counts = Counter(all_risk_factors)

        # Nejčastější pozitivní indikátory
        all_positive_indicators = []
        for assessment in assessments:
            all_positive_indicators.extend(assessment.positive_indicators)
        positive_indicator_counts = Counter(all_positive_indicators)

        # Domain analysis
        domain_scores = defaultdict(list)
        for assessment in assessments:
            domain_scores[assessment.source_metadata.domain].append(
                assessment.overall_credibility_score
            )

        domain_analysis = {}
        for domain, scores in domain_scores.items():
            domain_analysis[domain] = {
                "average_credibility": statistics.mean(scores),
                "assessment_count": len(scores),
                "credibility_range": [min(scores), max(scores)],
                "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_sources_assessed": len(assessments),
                "high_credibility_count": len(high_credibility),
                "medium_credibility_count": len(medium_credibility),
                "low_credibility_count": len(low_credibility),
                "average_credibility_score": statistics.mean(
                    [a.overall_credibility_score for a in assessments]
                ),
                "credibility_distribution": {
                    "high": len(high_credibility) / len(assessments),
                    "medium": len(medium_credibility) / len(assessments),
                    "low": len(low_credibility) / len(assessments),
                },
            },
            "most_credible_sources": [
                {
                    "domain": a.source_metadata.domain,
                    "credibility_score": a.overall_credibility_score,
                    "confidence": a.confidence,
                    "positive_indicators_count": len(a.positive_indicators),
                }
                for a in sorted(
                    assessments, key=lambda x: x.overall_credibility_score, reverse=True
                )[:5]
            ],
            "least_credible_sources": [
                {
                    "domain": a.source_metadata.domain,
                    "credibility_score": a.overall_credibility_score,
                    "confidence": a.confidence,
                    "risk_factors_count": len(a.risk_factors),
                }
                for a in sorted(assessments, key=lambda x: x.overall_credibility_score)[:5]
            ],
            "common_risk_factors": dict(risk_factor_counts.most_common(10)),
            "common_positive_indicators": dict(positive_indicator_counts.most_common(10)),
            "domain_analysis": domain_analysis,
            "assessment_confidence": {
                "average_confidence": statistics.mean([a.confidence for a in assessments]),
                "high_confidence_assessments": len([a for a in assessments if a.confidence > 0.8]),
                "low_confidence_assessments": len([a for a in assessments if a.confidence < 0.5]),
            },
        }

        return report
