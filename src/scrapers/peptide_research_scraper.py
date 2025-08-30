#!/usr/bin/env python3
"""Peptide Research Scraper for Deep Research Tool
Specialized scraper for peptide databases, hormone research, and biochemical pathways

Author: Advanced IT Specialist
"""

import asyncio
from datetime import datetime
import hashlib
import logging
import re
from typing import Any
from urllib.parse import quote, urljoin

# Try to import optional dependencies with fallbacks
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from fake_useragent import UserAgent
except ImportError:

    class UserAgent:
        @property
        def random(self):
            return "Mozilla/5.0 (compatible; DeepResearchTool/1.0)"


logger = logging.getLogger(__name__)


class PeptideResearchScraper:
    """Specialized scraper for peptide and hormone research"""

    def __init__(self, rate_limit: float = 0.6):
        """Initialize Peptide Research Scraper

        Args:
            rate_limit: Rate limit for peptide databases

        """
        if aiohttp is None:
            logger.warning("aiohttp not available, scraper will have limited functionality")

        self.rate_limit = rate_limit
        self.ua = UserAgent()

        # Specialized peptide and hormone databases
        self.peptide_sources = {
            "thpdb": {
                "name": "Therapeutic Peptides Database",
                "base_url": "https://webs.iiitd.edu.in/raghava/thpdb",
                "search_endpoint": "/browse.php",
                "api_endpoint": "/search_peptide.php?query={query}",
                "type": "therapeutic_peptides",
                "priority": "very_high",
            },
            "biopep": {
                "name": "BIOPEP-UWM Database",
                "base_url": "https://biochemia.uwm.edu.pl/biopep-uwm",
                "search_endpoint": "/search?query={query}",
                "type": "bioactive_peptides",
                "priority": "very_high",
            },
            "peptide_therapeutics": {
                "name": "Peptide Therapeutics Society",
                "base_url": "https://peptidetherapeutics.org",
                "search_endpoint": "/search?q={query}",
                "type": "peptide_therapeutics",
                "priority": "high",
            },
            "peptideatlas": {
                "name": "PeptideAtlas",
                "base_url": "https://peptideatlas.org",
                "search_endpoint": "/search/peptide?query={query}",
                "api_endpoint": "/api/peptide/{query}",
                "type": "peptide_atlas",
                "priority": "very_high",
            },
            "anticp": {
                "name": "AntiCP - Anticancer Peptides",
                "base_url": "https://webs.iiitd.edu.in/raghava/anticp",
                "search_endpoint": "/browse.php",
                "type": "anticancer_peptides",
                "priority": "high",
            },
            "camp": {
                "name": "CAMP - Antimicrobial Peptides",
                "base_url": "http://www.camp.bicnirrh.res.in",
                "search_endpoint": "/search?query={query}",
                "type": "antimicrobial_peptides",
                "priority": "high",
            },
            "peptide_database": {
                "name": "Peptide Database",
                "base_url": "https://peptides.biomedcentral.com",
                "search_endpoint": "/search?q={query}",
                "type": "peptide_research",
                "priority": "high",
            },
            "hormone_health_network": {
                "name": "Hormone Health Network",
                "base_url": "https://www.hormone.org",
                "search_endpoint": "/search?q={query}",
                "type": "hormone_information",
                "priority": "medium",
            },
            "endocrine_society": {
                "name": "Endocrine Society",
                "base_url": "https://www.endocrine.org",
                "search_endpoint": "/search?q={query}",
                "type": "endocrine_research",
                "priority": "high",
            },
            "growth_hormone_research": {
                "name": "Growth Hormone Research Society",
                "base_url": "https://www.ghresearch.org",
                "search_endpoint": "/search?query={query}",
                "type": "growth_hormone",
                "priority": "high",
            },
        }

        # Peptide-specific knowledge base
        self.peptide_database = {
            "growth_peptides": {
                "cjc-1295": {
                    "full_name": "CJC-1295",
                    "type": "Growth Hormone Releasing Hormone analog",
                    "mechanism": "GHRH receptor agonist",
                    "effects": [
                        "increased GH",
                        "increased IGF-1",
                        "improved sleep",
                        "muscle growth",
                    ],
                    "absorption": "subcutaneous injection",
                    "half_life": "6-8 days (with DAC)",
                    "related_terms": ["tesamorelin", "sermorelin", "GHRH", "growth hormone"],
                },
                "ipamorelin": {
                    "full_name": "Ipamorelin",
                    "type": "Growth Hormone Secretagogue",
                    "mechanism": "Ghrelin receptor agonist",
                    "effects": ["GH release", "no cortisol increase", "no prolactin increase"],
                    "absorption": "subcutaneous injection",
                    "half_life": "2 hours",
                    "related_terms": ["GHRP-2", "GHRP-6", "hexarelin", "ghrelin"],
                },
                "mk-677": {
                    "full_name": "Ibutamoren (MK-677)",
                    "type": "Growth Hormone Secretagogue",
                    "mechanism": "Ghrelin receptor agonist",
                    "effects": [
                        "increased GH",
                        "increased IGF-1",
                        "increased appetite",
                        "improved sleep",
                    ],
                    "absorption": "oral bioavailability",
                    "half_life": "24 hours",
                    "related_terms": [
                        "ibutamoren",
                        "growth hormone secretagogue",
                        "ghrelin mimetic",
                    ],
                },
            },
            "healing_peptides": {
                "bpc-157": {
                    "full_name": "Body Protection Compound-157",
                    "type": "Gastric peptide derivative",
                    "mechanism": "angiogenesis, tissue repair",
                    "effects": [
                        "tendon healing",
                        "gut protection",
                        "wound healing",
                        "neuroprotection",
                    ],
                    "absorption": "oral, subcutaneous, intramuscular",
                    "half_life": "short (stabilized forms available)",
                    "related_terms": [
                        "gastric juice peptide",
                        "pentadecapeptide",
                        "cytoprotective",
                    ],
                },
                "tb-500": {
                    "full_name": "Thymosin Beta-4",
                    "type": "Naturally occurring peptide",
                    "mechanism": "actin regulation, cell migration",
                    "effects": ["wound healing", "inflammation reduction", "tissue repair"],
                    "absorption": "subcutaneous, intramuscular injection",
                    "half_life": "2-3 hours",
                    "related_terms": ["thymosin", "actin-binding protein", "tissue repair"],
                },
            },
            "hormone_peptides": {
                "oxytocin": {
                    "full_name": "Oxytocin",
                    "type": "Neurohypophyseal hormone",
                    "mechanism": "oxytocin receptor binding",
                    "effects": ["social bonding", "uterine contractions", "milk ejection", "trust"],
                    "absorption": "intranasal, injection",
                    "half_life": "1-6 minutes",
                    "related_terms": ["love hormone", "bonding hormone", "vasopressin"],
                },
                "vasopressin": {
                    "full_name": "Arginine Vasopressin (AVP)",
                    "type": "Neurohypophyseal hormone",
                    "mechanism": "V1/V2 receptor binding",
                    "effects": ["water retention", "vasoconstriction", "social behavior"],
                    "absorption": "intranasal, injection",
                    "half_life": "10-20 minutes",
                    "related_terms": ["antidiuretic hormone", "ADH", "desmopressin"],
                },
            },
            "cosmetic_peptides": {
                "melanotan-2": {
                    "full_name": "Melanotan II",
                    "type": "Melanocortin receptor agonist",
                    "mechanism": "MC1R, MC4R activation",
                    "effects": ["skin tanning", "appetite suppression", "sexual function"],
                    "absorption": "subcutaneous injection",
                    "half_life": "1 hour",
                    "related_terms": ["α-MSH analog", "melanocortin", "bremelanotide", "pt-141"],
                },
                "pt-141": {
                    "full_name": "Bremelanotide (PT-141)",
                    "type": "Melanocortin receptor agonist",
                    "mechanism": "MC4R activation",
                    "effects": ["sexual arousal", "libido enhancement"],
                    "absorption": "subcutaneous injection, intranasal",
                    "half_life": "2-3 hours",
                    "related_terms": [
                        "bremelanotide",
                        "melanocortin",
                        "sexual dysfunction",
                        "libido",
                    ],
                },
            },
        }

        # Hormone interaction pathways
        self.hormone_pathways = {
            "growth_hormone_axis": [
                "hypothalamus",
                "GHRH",
                "somatostatin",
                "pituitary",
                "growth hormone",
                "liver",
                "IGF-1",
                "IGF-BP",
                "muscle",
                "bone",
                "fat",
            ],
            "hpa_axis": [
                "hypothalamus",
                "CRH",
                "pituitary",
                "ACTH",
                "adrenal",
                "cortisol",
                "stress response",
                "immune system",
                "metabolism",
            ],
            "reproductive_axis": [
                "hypothalamus",
                "GnRH",
                "pituitary",
                "LH",
                "FSH",
                "gonads",
                "testosterone",
                "estrogen",
                "progesterone",
            ],
        }

    async def search_async(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Search peptide and hormone databases

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of peptide research documents

        """
        results = []

        # Enhance query with peptide knowledge
        enhanced_queries = self._enhance_peptide_knowledge(topic)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            # Search each peptide source
            for source_id, source_config in self.peptide_sources.items():
                try:
                    logger.info(f"Searching {source_config['name']}...")

                    for query in enhanced_queries[:2]:  # Limit to 2 queries per source
                        source_results = await self._search_peptide_source(
                            session, source_id, source_config, query, time_range
                        )
                        results.extend(source_results)

                        # Rate limiting
                        await asyncio.sleep(1.0 / self.rate_limit)

                        if len(source_results) >= 15:  # Limit per source
                            break

                except Exception as e:
                    logger.error(f"Error searching {source_config['name']}: {e}")
                    continue

        # Add knowledge base information
        knowledge_results = self._extract_knowledge_base_info(topic)
        results.extend(knowledge_results)

        # Filter and prioritize results
        filtered_results = self._filter_peptide_results(results, topic)
        filtered_results.sort(
            key=lambda x: (
                x.get("priority_score", 0),
                x.get("peptide_relevance", 0),
                x.get("relevance_score", 0),
            ),
            reverse=True,
        )

        logger.info(f"Peptide Research Scraper found {len(filtered_results)} documents")
        return filtered_results[:80]  # Limit results

    def search(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    def _enhance_peptide_knowledge(self, topic: str) -> list[str]:
        """Enhance query using peptide knowledge base"""
        queries = [topic]
        topic_lower = topic.lower()

        # Check if topic matches known peptides
        for category, peptides in self.peptide_database.items():
            for peptide_name, info in peptides.items():
                if (
                    peptide_name in topic_lower
                    or info["full_name"].lower() in topic_lower
                    or any(term.lower() in topic_lower for term in info["related_terms"])
                ):

                    # Add enhanced queries based on peptide info
                    queries.extend(
                        [
                            f"{topic} {info['mechanism']}",
                            f"{topic} {info['type']}",
                            f"{topic} {' '.join(info['effects'][:2])}",
                            f"{topic} {info['absorption']}",
                            f"{info['full_name']} mechanism",
                            f"{info['full_name']} pharmacokinetics",
                        ]
                    )

                    # Add related terms
                    for related in info["related_terms"][:3]:
                        queries.append(f"{related} {topic}")

                    break

        # Add hormone pathway queries if relevant
        hormone_keywords = ["hormone", "growth", "insulin", "testosterone", "cortisol"]
        if any(keyword in topic_lower for keyword in hormone_keywords):
            for pathway_name, components in self.hormone_pathways.items():
                if any(comp in topic_lower for comp in components):
                    queries.append(f"{topic} {pathway_name}")
                    queries.append(f"{topic} pathway mechanism")
                    break

        # Add biochemical context
        biochem_terms = [
            "receptor binding",
            "bioavailability",
            "half life",
            "side effects",
            "dosage",
            "clinical study",
        ]

        for term in biochem_terms[:3]:
            queries.append(f"{topic} {term}")

        return list(set(queries[:8]))  # Remove duplicates and limit

    async def _search_peptide_source(
        self,
        session: aiohttp.ClientSession,
        source_id: str,
        source_config: dict[str, Any],
        query: str,
        time_range: tuple[datetime, datetime] | None,
    ) -> list[dict[str, Any]]:
        """Search individual peptide source"""
        results = []

        try:
            if source_id in ["thpdb", "anticp", "camp"]:
                results = await self._search_specialized_peptide_db(session, source_config, query)
            elif source_id == "peptideatlas":
                results = await self._search_peptide_atlas(session, source_config, query)
            elif source_id == "biopep":
                results = await self._search_biopep(session, source_config, query)
            else:
                results = await self._search_generic_peptide_source(session, source_config, query)

            # Add source metadata
            for result in results:
                result["peptide_source"] = source_config["name"]
                result["source_type"] = source_config["type"]
                result["source_priority"] = source_config["priority"]
                result["source"] = "peptide_research"

        except Exception as e:
            logger.debug(f"Failed to search {source_id}: {e}")

        return results

    async def _search_specialized_peptide_db(
        self, session: aiohttp.ClientSession, source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Search specialized peptide databases (THPDB, AntiCP, CAMP)"""
        results = []

        try:
            # Try API endpoint first if available
            if "api_endpoint" in source_config:
                api_url = source_config["base_url"] + source_config["api_endpoint"].format(
                    query=quote(query)
                )

                async with session.get(api_url) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            results = self._parse_peptide_api_response(data, source_config, query)
                        except:
                            # Fallback to HTML parsing
                            content = await response.text()
                            soup = BeautifulSoup(content, "html.parser")
                            results = self._parse_peptide_html(soup, source_config, query)

            # Fallback to web search
            if not results:
                search_url = source_config["base_url"] + source_config["search_endpoint"]
                if "{query}" in search_url:
                    search_url = search_url.format(query=quote(query))

                headers = {
                    "User-Agent": self.ua.random,
                    "Accept": "text/html,application/xhtml+xml",
                }

                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, "html.parser")
                        results = self._parse_peptide_html(soup, source_config, query)

        except Exception as e:
            logger.debug(f"Specialized peptide DB search failed: {e}")

        return results

    def _parse_peptide_api_response(
        self, data: dict[str, Any], source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Parse API response from peptide databases"""
        results = []

        try:
            # Handle different API response formats
            if isinstance(data, dict):
                if "results" in data:
                    peptide_data = data["results"]
                elif "peptides" in data:
                    peptide_data = data["peptides"]
                elif "data" in data:
                    peptide_data = data["data"]
                else:
                    peptide_data = [data]
            elif isinstance(data, list):
                peptide_data = data
            else:
                return results

            for item in peptide_data[:20]:  # Limit results
                if isinstance(item, dict):
                    result = {
                        "title": item.get(
                            "name", item.get("title", f"Peptide from {source_config['name']}")
                        ),
                        "url": item.get("url", item.get("link", source_config["base_url"])),
                        "content": self._extract_peptide_content(item),
                        "date": self._extract_date_from_item(item),
                        "source_url": source_config["base_url"],
                        "peptide_id": item.get("id", item.get("peptide_id")),
                        "sequence": item.get("sequence", ""),
                        "molecular_weight": item.get("molecular_weight", item.get("mw")),
                        "activity": item.get("activity", item.get("biological_activity")),
                        "peptide_relevance": self._calculate_peptide_relevance(item, query),
                        "mechanism": item.get("mechanism", ""),
                        "target": item.get("target", ""),
                        "source_organism": item.get("source_organism", ""),
                        "peptide_class": item.get("class", item.get("category")),
                    }

                    # Enhanced relevance scoring for peptides
                    result["priority_score"] = self._score_peptide_priority(result, query)
                    results.append(result)

        except Exception as e:
            logger.debug(f"Error parsing peptide API response: {e}")

        return results

    def _extract_peptide_content(self, item: dict[str, Any]) -> str:
        """Extract comprehensive content from peptide data"""
        content_parts = []

        # Basic information
        if item.get("description"):
            content_parts.append(f"Description: {item['description']}")

        # Sequence information
        if item.get("sequence"):
            content_parts.append(f"Sequence: {item['sequence']}")

        # Biological activity
        if item.get("activity") or item.get("biological_activity"):
            activity = item.get("activity") or item.get("biological_activity")
            content_parts.append(f"Biological Activity: {activity}")

        # Mechanism of action
        if item.get("mechanism"):
            content_parts.append(f"Mechanism: {item['mechanism']}")

        # Target information
        if item.get("target"):
            content_parts.append(f"Target: {item['target']}")

        # Pharmacological properties
        properties = []
        if item.get("molecular_weight"):
            properties.append(f"MW: {item['molecular_weight']}")
        if item.get("half_life"):
            properties.append(f"Half-life: {item['half_life']}")
        if item.get("bioavailability"):
            properties.append(f"Bioavailability: {item['bioavailability']}")

        if properties:
            content_parts.append(f"Properties: {', '.join(properties)}")

        # Clinical information
        if item.get("clinical_phase"):
            content_parts.append(f"Clinical Phase: {item['clinical_phase']}")

        # Safety profile
        if item.get("side_effects"):
            content_parts.append(f"Side Effects: {item['side_effects']}")

        return " | ".join(content_parts) if content_parts else "Peptide research data"

    def _calculate_peptide_relevance(self, item: dict[str, Any], query: str) -> float:
        """Calculate relevance score specific to peptide research"""
        relevance_score = 0.0
        query_lower = query.lower()

        # Title/name matching
        title = (item.get("title", "") + " " + item.get("name", "")).lower()
        if any(word in title for word in query_lower.split()):
            relevance_score += 0.3

        # Sequence matching (for specific peptide queries)
        sequence = item.get("sequence", "").lower()
        if sequence and any(word in sequence for word in query_lower.split() if len(word) > 2):
            relevance_score += 0.2

        # Activity/mechanism matching
        activity = (item.get("activity", "") + " " + item.get("mechanism", "")).lower()
        if any(word in activity for word in query_lower.split()):
            relevance_score += 0.25

        # Target matching
        target = item.get("target", "").lower()
        if target and any(word in target for word in query_lower.split()):
            relevance_score += 0.15

        # Class/category matching
        peptide_class = item.get("peptide_class", "").lower()
        if peptide_class and any(word in peptide_class for word in query_lower.split()):
            relevance_score += 0.1

        return min(1.0, relevance_score)

    def _score_peptide_priority(self, result: dict[str, Any], query: str) -> float:
        """Score peptide research priority based on multiple factors"""
        priority_score = 0.0

        # Base relevance
        priority_score += result.get("peptide_relevance", 0) * 0.3

        # Clinical significance
        if result.get("activity"):
            activity_lower = result["activity"].lower()
            if any(
                term in activity_lower for term in ["therapeutic", "clinical", "approved", "trial"]
            ):
                priority_score += 0.2

        # Mechanism clarity
        if result.get("mechanism"):
            priority_score += 0.15

        # Sequence availability (important for research)
        if result.get("sequence"):
            priority_score += 0.1

        # Target specificity
        if result.get("target"):
            priority_score += 0.1

        # Molecular properties
        if result.get("molecular_weight"):
            priority_score += 0.05

        # Source authority
        source_priority = result.get("source_priority", "medium")
        priority_multipliers = {"very_high": 1.2, "high": 1.1, "medium": 1.0, "low": 0.9}
        priority_score *= priority_multipliers.get(source_priority, 1.0)

        return min(1.0, priority_score)

    def _parse_peptide_html(
        self, soup: BeautifulSoup, source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Parse HTML content from peptide sources"""
        results = []

        try:
            # Try different common HTML structures for peptide databases
            entries = (
                soup.find_all("div", class_=re.compile(r"(peptide|entry|result)", re.IGNORECASE))
                or soup.find_all("tr", class_=re.compile(r"(peptide|entry|result)", re.IGNORECASE))
                or soup.find_all("article")
                or soup.find_all("div", class_=re.compile(r"item", re.IGNORECASE))
            )

            for entry in entries[:15]:  # Limit results
                title_elem = (
                    entry.find("h1")
                    or entry.find("h2")
                    or entry.find("h3")
                    or entry.find("a", href=True)
                    or entry.find(class_=re.compile(r"title", re.IGNORECASE))
                )

                title = (
                    title_elem.get_text(strip=True)
                    if title_elem
                    else f"Peptide from {source_config['name']}"
                )

                # Extract URL
                link_elem = entry.find("a", href=True)
                url = (
                    urljoin(source_config["base_url"], link_elem["href"])
                    if link_elem
                    else source_config["base_url"]
                )

                # Extract content
                content = self._extract_html_content(entry)

                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "date": self._extract_date_from_html(entry),
                    "source_url": source_config["base_url"],
                    "peptide_relevance": self._calculate_html_relevance(content, query),
                    "priority_score": 0.5,  # Base score for HTML results
                }

                results.append(result)

        except Exception as e:
            logger.debug(f"Error parsing peptide HTML: {e}")

        return results

    def _extract_html_content(self, entry) -> str:
        """Extract meaningful content from HTML entry"""
        content_parts = []

        # Get text content, excluding navigation and footer elements
        for elem in entry.find_all(["p", "div", "span", "td"]):
            text = elem.get_text(strip=True)
            if text and len(text) > 10 and not self._is_navigation_text(text):
                content_parts.append(text)

        content = " | ".join(content_parts[:5])  # Limit content
        return content if content else entry.get_text(strip=True)[:500]

    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/menu content"""
        nav_indicators = ["home", "about", "contact", "login", "search", "menu", "navigation"]
        return any(indicator in text.lower() for indicator in nav_indicators) and len(text) < 50

    def _calculate_html_relevance(self, content: str, query: str) -> float:
        """Calculate relevance for HTML-extracted content"""
        if not content:
            return 0.0

        content_lower = content.lower()
        query_words = query.lower().split()

        matches = sum(1 for word in query_words if word in content_lower)
        return min(1.0, matches / len(query_words))

    def _extract_date_from_item(self, item: dict[str, Any]) -> datetime | None:
        """Extract date from API item"""
        date_fields = ["date", "created_date", "publication_date", "updated", "timestamp"]

        for field in date_fields:
            if field in item:
                try:
                    date_str = item[field]
                    if isinstance(date_str, str):
                        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if isinstance(date_str, datetime):
                        return date_str
                except:
                    continue

        return None

    def _extract_date_from_html(self, entry) -> datetime | None:
        """Extract date from HTML entry"""
        # Look for date patterns in HTML
        date_patterns = [r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"]

        text = entry.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if "-" in match.group(0):
                        return datetime.strptime(match.group(0), "%Y-%m-%d")
                    return datetime.strptime(match.group(0), "%m/%d/%Y")
                except:
                    continue

        return None

    async def _search_peptide_atlas(
        self, session: aiohttp.ClientSession, source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Search PeptideAtlas specifically"""
        return await self._search_specialized_peptide_db(session, source_config, query)

    async def _search_biopep(
        self, session: aiohttp.ClientSession, source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Search BIOPEP-UWM specifically"""
        return await self._search_specialized_peptide_db(session, source_config, query)

    async def _search_generic_peptide_source(
        self, session: aiohttp.ClientSession, source_config: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        """Search generic peptide sources"""
        return await self._search_specialized_peptide_db(session, source_config, query)

    def _extract_knowledge_base_info(self, topic: str) -> list[dict[str, Any]]:
        """Extract relevant information from peptide knowledge base"""
        results = []
        topic_lower = topic.lower()

        for category, peptides in self.peptide_database.items():
            for peptide_name, info in peptides.items():
                # Check if this peptide is relevant to the query
                if (
                    peptide_name in topic_lower
                    or info["full_name"].lower() in topic_lower
                    or any(term.lower() in topic_lower for term in info["related_terms"])
                ):

                    # Create comprehensive content
                    content_parts = [
                        f"Full Name: {info['full_name']}",
                        f"Type: {info['type']}",
                        f"Mechanism: {info['mechanism']}",
                        f"Effects: {', '.join(info['effects'])}",
                        f"Absorption: {info['absorption']}",
                        f"Half-life: {info['half_life']}",
                        f"Related Terms: {', '.join(info['related_terms'])}",
                    ]

                    result = {
                        "title": f"{info['full_name']} - Peptide Research Overview",
                        "url": f"knowledge_base://{category}/{peptide_name}",
                        "content": " | ".join(content_parts),
                        "date": datetime.now(),
                        "source_url": "internal_knowledge_base",
                        "peptide_source": "Internal Knowledge Base",
                        "source_type": "knowledge_base",
                        "source_priority": "very_high",
                        "peptide_relevance": 1.0,  # Perfect match from knowledge base
                        "priority_score": 0.9,
                        "peptide_class": category,
                        "mechanism": info["mechanism"],
                        "effects": info["effects"],
                    }

                    results.append(result)

        return results

    def _filter_peptide_results(
        self, results: list[dict[str, Any]], topic: str
    ) -> list[dict[str, Any]]:
        """Filter and enhance peptide research results"""
        if not results:
            return []

        # Remove duplicates based on content similarity
        unique_results = self._remove_duplicate_peptides(results)

        # Apply quality filters
        filtered_results = []
        for result in unique_results:
            # Skip results with very low relevance
            if result.get("peptide_relevance", 0) < 0.1:
                continue

            # Skip results with insufficient content
            content = result.get("content", "")
            if len(content.strip()) < 20:
                continue

            # Enhance with additional metadata
            result["content_type"] = "peptide"
            result["research_domain"] = self._classify_research_domain(result)
            result["clinical_relevance"] = self._assess_clinical_relevance(result)

            filtered_results.append(result)

        return filtered_results

    def _remove_duplicate_peptides(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate peptide entries"""
        seen_content = set()
        unique_results = []

        for result in results:
            # Create content hash for duplicate detection
            content_key = self._create_content_hash(result)

            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        return unique_results

    def _create_content_hash(self, result: dict[str, Any]) -> str:
        """Create hash for duplicate detection"""
        # Use title and first part of content for hashing
        title = result.get("title", "").lower().strip()
        content = result.get("content", "")[:100].lower().strip()
        sequence = result.get("sequence", "").lower().strip()

        hash_content = f"{title}|{content}|{sequence}"
        return hashlib.md5(hash_content.encode()).hexdigest()

    def _classify_research_domain(self, result: dict[str, Any]) -> str:
        """Classify the research domain of the peptide"""
        content = result.get("content", "").lower()
        title = result.get("title", "").lower()

        combined_text = f"{title} {content}"

        if any(
            term in combined_text
            for term in ["growth hormone", "gh", "igf-1", "muscle", "anti-aging"]
        ):
            return "growth_hormone_research"
        if any(term in combined_text for term in ["healing", "repair", "wound", "regeneration"]):
            return "tissue_repair"
        if any(term in combined_text for term in ["cancer", "tumor", "oncology", "anticancer"]):
            return "oncology"
        if any(term in combined_text for term in ["antimicrobial", "antibiotic", "infection"]):
            return "antimicrobial"
        if any(term in combined_text for term in ["cosmetic", "skin", "tanning", "beauty"]):
            return "cosmetic"
        if any(
            term in combined_text for term in ["hormone", "endocrine", "oxytocin", "vasopressin"]
        ):
            return "endocrinology"
        return "general_peptide_research"

    def _assess_clinical_relevance(self, result: dict[str, Any]) -> str:
        """Assess clinical relevance of peptide research"""
        content = result.get("content", "").lower()

        if any(
            term in content for term in ["approved", "fda approved", "clinical trial", "phase iii"]
        ):
            return "high"
        if any(term in content for term in ["clinical", "therapeutic", "treatment", "therapy"]):
            return "medium"
        if any(term in content for term in ["research", "study", "investigation"]):
            return "research"
        return "experimental"
