#!/usr/bin/env python3
"""Research Orchestrator for Deep Research Tool
Coordinates multi-source research pipeline with AI analysis

Author: Advanced IT Specialist
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from typing import Any

import yaml

from ..analysis.enhanced_analyzer import EnhancedResearchAnalyzer
from ..scrapers.archive_hunter import ArchiveHunter
from ..scrapers.arxiv_scraper import ArxivScraper
from ..scrapers.medical_research_scraper import MedicalResearchScraper
from ..scrapers.pdf_scraper import PDFScraper
from ..scrapers.peptide_research_scraper import PeptideResearchScraper
from ..scrapers.rss_monitor import RSSMonitor
from ..scrapers.social_media_scraper import SocialMediaScraper
from ..scrapers.tor_scraper import TorDeepWebScraper
from ..scrapers.wayback_scraper import WaybackMachineScraper
from ..storage.cache import ResearchCache
from ..utils.compliance import ComplianceChecker
from .context_manager import ContextManager
from .ollama_agent import AnalysisResult, OllamaResearchAgent
from .priority_scorer import InformationPriorityScorer

logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """Research query configuration"""

    topic: str
    time_range: tuple[datetime, datetime] | None = None
    sources: list[str] = None
    analysis_type: str = "general_research"
    max_results: int = 100
    priority_threshold: float = 0.5


@dataclass
class ResearchResults:
    """Complete research results with enhanced analysis"""

    query: ResearchQuery
    sources_found: int
    analysis: AnalysisResult
    enhanced_analysis: Any  # ResearchSummary from enhanced analyzer
    source_breakdown: dict[str, int]
    unique_sources: list[dict[str, Any]]
    cross_references: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    execution_time: float
    timestamp: datetime


class CrossReferenceAnalyzer:
    """Analyzer for finding connections between sources"""

    def find_connections(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find cross-references and connections between sources"""
        connections = []

        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i + 1 :], i + 1):
                similarity = self._calculate_similarity(source1, source2)
                if similarity > 0.3:  # Threshold for meaningful connection
                    connections.append(
                        {
                            "source1": source1.get("title", "Unknown"),
                            "source2": source2.get("title", "Unknown"),
                            "similarity": similarity,
                            "connection_type": self._classify_connection(source1, source2),
                            "url1": source1.get("url", ""),
                            "url2": source2.get("url", ""),
                        }
                    )

        return sorted(connections, key=lambda x: x["similarity"], reverse=True)[:10]

    def _calculate_similarity(self, source1: dict[str, Any], source2: dict[str, Any]) -> float:
        """Calculate similarity between two sources"""
        content1 = (source1.get("title", "") + " " + source1.get("content", "")).lower()
        content2 = (source2.get("title", "") + " " + source2.get("content", "")).lower()

        # Simple word overlap similarity
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _classify_connection(self, source1: dict[str, Any], source2: dict[str, Any]) -> str:
        """Classify the type of connection between sources"""
        # Check if same author/organization
        if source1.get("source_url", "") == source2.get("source_url", ""):
            return "same_source"

        # Check temporal relationship
        date1 = source1.get("date")
        date2 = source2.get("date")
        if date1 and date2:
            if isinstance(date1, str):
                try:
                    date1 = datetime.fromisoformat(date1)
                except:
                    date1 = None
            if isinstance(date2, str):
                try:
                    date2 = datetime.fromisoformat(date2)
                except:
                    date2 = None

            if date1 and date2:
                time_diff = abs((date1 - date2).days)
                if time_diff < 30:
                    return "temporal_cluster"

        return "content_similarity"


class ResearchOrchestrator:
    """Main orchestrator for coordinating research across multiple sources"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the research orchestrator

        Args:
            config_path: Path to configuration file

        """
        self.config = self._load_config(config_path)
        self.ollama_agent = OllamaResearchAgent(
            model_name=self.config["research_config"]["ollama"]["model"],
            host=self.config["research_config"]["ollama"]["host"],
        )
        self.context_manager = ContextManager()
        self.cache = ResearchCache()
        self.compliance_checker = ComplianceChecker()
        self.priority_scorer = InformationPriorityScorer()
        self.enhanced_analyzer = EnhancedResearchAnalyzer()

        # Initialize scrapers
        self.scrapers = {}
        self._initialize_scrapers()

        # Cross-reference analyzer
        self.cross_ref_analyzer = CrossReferenceAnalyzer()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration"""
        return {
            "research_config": {
                "ollama": {"model": "llama3.2:8b", "host": "http://localhost:11434"},
                "sources": {
                    "wayback_machine": {"enabled": True},
                    "arxiv": {"enabled": True},
                    "rss_feeds": {"enabled": True},
                    "pdf_archives": {"enabled": True},
                    "medical_research": {"enabled": True},
                    "peptide_research": {"enabled": True},
                },
                "rate_limits": {
                    "wayback_machine": 1.0,
                    "arxiv": 0.5,
                    "rss_feeds": 2.0,
                    "pdf_sources": 1.0,
                },
            }
        }

    def _initialize_scrapers(self):
        """Initialize all enabled scrapers"""
        sources_config = self.config["research_config"]["sources"]

        if sources_config.get("wayback_machine", {}).get("enabled", True):
            self.scrapers["wayback"] = WaybackMachineScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["wayback_machine"]
            )

        if sources_config.get("arxiv", {}).get("enabled", True):
            self.scrapers["arxiv"] = ArxivScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["arxiv"]
            )

        if sources_config.get("rss_feeds", {}).get("enabled", True):
            self.scrapers["rss_feeds"] = RSSMonitor(
                rate_limit=self.config["research_config"]["rate_limits"]["rss_feeds"]
            )

        if sources_config.get("pdf_archives", {}).get("enabled", True):
            self.scrapers["pdf_archives"] = PDFScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        if sources_config.get("archive_hunter", {}).get("enabled", True):
            self.scrapers["archive_hunter"] = ArchiveHunter(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        if sources_config.get("tor_scraper", {}).get("enabled", True):
            self.scrapers["tor_scraper"] = TorDeepWebScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        if sources_config.get("social_media_scraper", {}).get("enabled", True):
            self.scrapers["social_media_scraper"] = SocialMediaScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        if sources_config.get("medical_research_scraper", {}).get("enabled", True):
            self.scrapers["medical_research_scraper"] = MedicalResearchScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        if sources_config.get("peptide_research_scraper", {}).get("enabled", True):
            self.scrapers["peptide_research_scraper"] = PeptideResearchScraper(
                rate_limit=self.config["research_config"]["rate_limits"]["pdf_sources"]
            )

        # NEW ADVANCED SCRAPERS FROM PERPLEXITY ANALYSIS

        # CIA CREST declassified documents
        if sources_config.get("cia_crest", {}).get("enabled", True):
            from ..scrapers.declassified_scraper import CIACRESTScraper

            self.scrapers["cia_crest"] = CIACRESTScraper(
                rate_limit=self.config["research_config"]["rate_limits"].get("declassified", 0.1)
            )

        # BASE Academic Search Engine (150M+ documents)
        if sources_config.get("base_search", {}).get("enabled", True):
            from ..scrapers.base_scraper import BASEScraper

            self.scrapers["base_search"] = BASEScraper(
                rate_limit=self.config["research_config"]["rate_limits"].get("academic", 0.5)
            )

        # P2P Networks (IPFS + BitTorrent)
        if sources_config.get("p2p_networks", {}).get("enabled", True):
            from ..scrapers.p2p_scraper import P2PNetworkScraper

            self.scrapers["p2p_networks"] = P2PNetworkScraper(
                rate_limit=self.config["research_config"]["rate_limits"].get("p2p", 1.0)
            )

        # OSINT Multi-Source Intelligence
        if sources_config.get("osint_collector", {}).get("enabled", True):
            from ..scrapers.osint_collector import OSINTCollector

            self.scrapers["osint_collector"] = OSINTCollector(
                rate_limit=self.config["research_config"]["rate_limits"].get("osint", 1.0)
            )

        logger.info(f"Initialized {len(self.scrapers)} scrapers: {list(self.scrapers.keys())}")

    # Add specialized research method for advanced sources
    async def advanced_research(
        self,
        topic: str,
        time_range: tuple[datetime, datetime] = None,
        analysis_type: str = "advanced_research",
        specialized_sources: list[str] = None,
    ) -> ResearchResults:
        """Perform advanced research using specialized sources like CIA CREST, BASE, P2P networks

        Args:
            topic: Research topic
            time_range: Optional time range for historical research
            analysis_type: Type of analysis to perform
            specialized_sources: Specific advanced sources to use

        Returns:
            Complete research results with enhanced analysis

        """
        start_time = datetime.now()

        # Default to all advanced sources if none specified
        if specialized_sources is None:
            specialized_sources = ["cia_crest", "base_search", "p2p_networks", "osint_collector"]

        # Create research query with specialized sources
        query = ResearchQuery(
            topic=topic,
            time_range=time_range,
            analysis_type=analysis_type,
            sources=specialized_sources,
        )

        logger.info(f"Starting advanced research for: {topic} using sources: {specialized_sources}")

        try:
            # Step 1: Parallel scraping across specialized sources
            raw_sources = await self._orchestrate_advanced_scraping(query)

            # Step 2: Enhanced priority scoring for specialized content
            scored_sources = self._score_specialized_sources(raw_sources, query)

            # Step 3: Deduplicate and filter sources
            unique_sources = self._deduplicate_sources(scored_sources)

            # Step 4: Advanced cross-reference analysis
            cross_references = self.cross_ref_analyzer.find_connections(unique_sources)

            # Step 5: Enhanced timeline construction
            timeline = self._construct_timeline(unique_sources)

            # Step 6: Specialized content analysis
            enhanced_analysis = self.enhanced_analyzer.analyze_research_data(
                unique_sources, topic, self.priority_scorer
            )

            # Step 7: Advanced AI analysis with specialized prompts
            context = self.context_manager.build_context(unique_sources, cross_references, timeline)

            ai_analysis = await self.ollama_agent.analyze_research(topic, context, analysis_type)

            # Step 8: Source breakdown analysis with specialized categories
            source_breakdown = self._analyze_specialized_source_breakdown(unique_sources)

            execution_time = (datetime.now() - start_time).total_seconds()

            results = ResearchResults(
                query=query,
                sources_found=len(unique_sources),
                analysis=ai_analysis,
                enhanced_analysis=enhanced_analysis,
                source_breakdown=source_breakdown,
                unique_sources=unique_sources,
                cross_references=cross_references,
                timeline=timeline,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )

            logger.info(
                f"Advanced research completed in {execution_time:.2f}s, found {len(unique_sources)} specialized sources"
            )
            return results

        except Exception as e:
            logger.error(f"Error in advanced research: {e}")
            raise

    async def _orchestrate_advanced_scraping(self, query: ResearchQuery) -> list[dict[str, Any]]:
        """Orchestrate parallel scraping across advanced/specialized sources"""
        tasks = []

        for scraper_name in query.sources:
            if scraper_name in self.scrapers:
                scraper = self.scrapers[scraper_name]

                # Handle OSINT collector differently (it takes a target parameter)
                if scraper_name == "osint_collector":
                    task = asyncio.create_task(
                        self._safe_osint_collection(scraper, query.topic, query.time_range),
                        name=scraper_name,
                    )
                else:
                    task = asyncio.create_task(
                        self._safe_scrape(scraper, query.topic, query.time_range), name=scraper_name
                    )
                tasks.append(task)

        # Wait for all scraping tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_sources = []
        for i, result in enumerate(results):
            scraper_name = tasks[i].get_name()
            if isinstance(result, Exception):
                logger.error(f"Advanced scraper {scraper_name} failed: {result}")
                continue

            if isinstance(result, list):
                all_sources.extend(result)
            else:
                logger.warning(f"Unexpected result type from {scraper_name}: {type(result)}")

        logger.info(f"Collected {len(all_sources)} sources from {len(tasks)} advanced scrapers")
        return all_sources

    async def _safe_osint_collection(
        self, osint_collector, topic: str, time_range: tuple[datetime, datetime] | None
    ) -> list[dict[str, Any]]:
        """Safely execute OSINT collection"""
        try:
            osint_result = await osint_collector.collect_intelligence(topic)

            # Convert OSINT result to standard format
            return [
                {
                    "title": f"OSINT Intelligence: {topic}",
                    "content": json.dumps(osint_result.intelligence_data, default=str)[:2000],
                    "url": f"osint://intelligence/{topic}",
                    "source_type": "osint_intelligence",
                    "source_url": f"osint://intelligence/{topic}",
                    "date": osint_result.collection_timestamp,
                    "metadata": {
                        "osint_target": osint_result.target,
                        "correlation_score": osint_result.correlation_score,
                        "verification_status": osint_result.verification_status,
                        "source_reliability": osint_result.source_reliability,
                        "intelligence_data": osint_result.intelligence_data,
                    },
                }
            ]
        except Exception as e:
            logger.error(f"OSINT collection error: {e}")
            return []

    def _score_specialized_sources(
        self, sources: list[dict[str, Any]], query: ResearchQuery
    ) -> list[dict[str, Any]]:
        """Enhanced scoring for specialized sources"""
        scored_sources = []

        for source in sources:
            # Standard priority scoring
            content = source.get("content", "")
            metadata = {
                "source_url": source.get("source_url", ""),
                "source_type": source.get("source_type", "unknown"),
                "date": source.get("date"),
                "content_type": source.get("content_type", "general"),
            }

            priority_score = self.priority_scorer.score_information(content, metadata)
            detailed_analysis = self.priority_scorer.get_detailed_analysis(content, metadata)

            # Enhanced scoring for specialized sources
            specialized_boost = self._calculate_specialized_source_boost(source)
            final_score = min(1.0, priority_score + specialized_boost)

            # Add scoring information to source
            source["priority_score"] = final_score
            source["detailed_scores"] = detailed_analysis["component_scores"]
            source["confidence_level"] = detailed_analysis["confidence_level"]
            source["key_insights"] = detailed_analysis["key_insights"]
            source["reliability_assessment"] = detailed_analysis["reliability_assessment"]
            source["specialized_boost"] = specialized_boost

            # Filter by threshold (lower threshold for specialized sources)
            specialized_threshold = max(0.3, query.priority_threshold - 0.2)
            if final_score >= specialized_threshold:
                scored_sources.append(source)

        # Sort by priority score
        scored_sources.sort(key=lambda x: x["priority_score"], reverse=True)

        logger.info(
            f"Filtered {len(scored_sources)} high-priority specialized sources from {len(sources)} total"
        )
        return scored_sources[: query.max_results * 2]  # Allow more results for specialized sources

    def _calculate_specialized_source_boost(self, source: dict[str, Any]) -> float:
        """Calculate boost score for specialized sources"""
        boost = 0.0
        source_type = source.get("source_type", "")

        # Boost scores for different specialized source types
        specialized_boosts = {
            "cia_declassified": 0.3,  # High boost for declassified CIA documents
            "base_academic": 0.2,  # Medium boost for academic sources
            "ipfs_content": 0.25,  # High boost for censorship-resistant content
            "bittorrent_dht": 0.2,  # Medium boost for P2P archives
            "osint_intelligence": 0.35,  # Highest boost for OSINT intelligence
            "tor_hidden_service": 0.15,  # Lower boost but still valuable
            "tor_deep_crawl": 0.2,  # Medium boost for deep web content
        }

        boost += specialized_boosts.get(source_type, 0.0)

        # Additional boost based on metadata
        metadata = source.get("metadata", {})

        # CIA documents: boost for classification level
        if source_type == "cia_declassified":
            classification = metadata.get("classification", {})
            original_level = classification.get("original_level", 0)
            boost += original_level * 0.05  # Higher classification = higher boost

            if metadata.get("redaction_analysis", {}).get("heavily_redacted", False):
                boost += 0.1  # Heavily redacted documents often more significant

        # Academic sources: boost for peer review and citations
        elif source_type == "base_academic":
            if metadata.get("peer_reviewed", False):
                boost += 0.1

            citation_count = metadata.get("citation_count", 0)
            if citation_count > 10:
                boost += 0.05

        # OSINT: boost for high correlation and verification
        elif source_type == "osint_intelligence":
            correlation_score = metadata.get("correlation_score", 0.0)
            boost += correlation_score * 0.1

            verification = metadata.get("verification_status", "")
            if verification == "high_confidence":
                boost += 0.1

        return min(0.4, boost)  # Cap boost at 0.4

    def _analyze_specialized_source_breakdown(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze breakdown of specialized sources"""
        breakdown = {}
        specialized_breakdown = {}

        for source in sources:
            source_type = source.get("source_type", "unknown")
            breakdown[source_type] = breakdown.get(source_type, 0) + 1

            # Group by specialized categories
            if source_type in ["cia_declassified"]:
                specialized_breakdown["government_declassified"] = (
                    specialized_breakdown.get("government_declassified", 0) + 1
                )
            elif source_type in ["base_academic"]:
                specialized_breakdown["deep_web_academic"] = (
                    specialized_breakdown.get("deep_web_academic", 0) + 1
                )
            elif source_type in ["ipfs_content", "bittorrent_dht"]:
                specialized_breakdown["p2p_networks"] = (
                    specialized_breakdown.get("p2p_networks", 0) + 1
                )
            elif source_type in ["osint_intelligence"]:
                specialized_breakdown["osint_intelligence"] = (
                    specialized_breakdown.get("osint_intelligence", 0) + 1
                )
            elif source_type in ["tor_hidden_service", "tor_deep_crawl"]:
                specialized_breakdown["dark_web"] = specialized_breakdown.get("dark_web", 0) + 1

        # Combine both breakdowns
        breakdown.update(specialized_breakdown)
        return breakdown

    async def deep_research(
        self,
        topic: str,
        time_range: tuple[datetime, datetime] = None,
        analysis_type: str = "general_research",
    ) -> ResearchResults:
        """Perform comprehensive research across all sources with enhanced analysis

        Args:
            topic: Research topic
            time_range: Optional time range for historical research
            analysis_type: Type of analysis to perform

        Returns:
            Complete research results with enhanced analysis

        """
        start_time = datetime.now()

        # Create research query
        query = ResearchQuery(
            topic=topic,
            time_range=time_range,
            analysis_type=analysis_type,
            sources=list(self.scrapers.keys()),
        )

        logger.info(f"Starting enhanced deep research for: {topic}")

        try:
            # Step 1: Parallel scraping across all sources
            raw_sources = await self._orchestrate_scraping(query)

            # Step 2: Priority scoring and filtering
            scored_sources = self._score_and_filter_sources(raw_sources, query)

            # Step 3: Deduplicate and filter sources
            unique_sources = self._deduplicate_sources(scored_sources)

            # Step 4: Cross-reference analysis
            cross_references = self.cross_ref_analyzer.find_connections(unique_sources)

            # Step 5: Timeline construction
            timeline = self._construct_timeline(unique_sources)

            # Step 6: Enhanced analysis using new analyzer
            enhanced_analysis = self.enhanced_analyzer.analyze_research_data(
                unique_sources, topic, self.priority_scorer
            )

            # Step 7: Context-aware AI analysis
            context = self.context_manager.build_context(unique_sources, cross_references, timeline)

            ai_analysis = await self.ollama_agent.analyze_research(topic, context, analysis_type)

            # Step 8: Source breakdown analysis
            source_breakdown = self._analyze_source_breakdown(unique_sources)

            execution_time = (datetime.now() - start_time).total_seconds()

            results = ResearchResults(
                query=query,
                sources_found=len(unique_sources),
                analysis=ai_analysis,
                enhanced_analysis=enhanced_analysis,
                source_breakdown=source_breakdown,
                unique_sources=unique_sources,
                cross_references=cross_references,
                timeline=timeline,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )

            logger.info(
                f"Deep research completed in {execution_time:.2f}s, found {len(unique_sources)} sources"
            )
            return results

        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            raise

    def _score_and_filter_sources(
        self, sources: list[dict[str, Any]], query: ResearchQuery
    ) -> list[dict[str, Any]]:
        """Score sources using priority scorer and filter by threshold"""
        scored_sources = []

        for source in sources:
            # Score using priority scorer
            content = source.get("content", "")
            metadata = {
                "source_url": source.get("source_url", ""),
                "source_type": source.get("source_type", "unknown"),
                "date": source.get("date"),
                "content_type": source.get("content_type", "general"),
            }

            priority_score = self.priority_scorer.score_information(content, metadata)
            detailed_analysis = self.priority_scorer.get_detailed_analysis(content, metadata)

            # Add scoring information to source
            source["priority_score"] = priority_score
            source["detailed_scores"] = detailed_analysis["component_scores"]
            source["confidence_level"] = detailed_analysis["confidence_level"]
            source["key_insights"] = detailed_analysis["key_insights"]
            source["reliability_assessment"] = detailed_analysis["reliability_assessment"]

            # Filter by threshold
            if priority_score >= query.priority_threshold:
                scored_sources.append(source)

        # Sort by priority score
        scored_sources.sort(key=lambda x: x["priority_score"], reverse=True)

        logger.info(
            f"Filtered {len(scored_sources)} high-priority sources from {len(sources)} total"
        )
        return scored_sources[: query.max_results]

    async def _orchestrate_scraping(self, query: ResearchQuery) -> list[dict[str, Any]]:
        """Orchestrate parallel scraping across all sources"""
        tasks = []

        for scraper_name, scraper in self.scrapers.items():
            if scraper_name in query.sources:
                task = asyncio.create_task(
                    self._safe_scrape(scraper, query.topic, query.time_range), name=scraper_name
                )
                tasks.append(task)

        # Wait for all scraping tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_sources = []
        for i, result in enumerate(results):
            scraper_name = tasks[i].get_name()
            if isinstance(result, Exception):
                logger.error(f"Scraper {scraper_name} failed: {result}")
                continue

            if isinstance(result, list):
                all_sources.extend(result)
            else:
                logger.warning(f"Unexpected result type from {scraper_name}: {type(result)}")

        logger.info(f"Collected {len(all_sources)} sources from {len(tasks)} scrapers")
        return all_sources

    async def _safe_scrape(
        self, scraper, topic: str, time_range: tuple[datetime, datetime] | None
    ) -> list[dict[str, Any]]:
        """Safely execute scraper with error handling"""
        try:
            if hasattr(scraper, "search_async"):
                return await scraper.search_async(topic, time_range)
            # Run synchronous scrapers in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, scraper.search, topic, time_range)
        except Exception as e:
            logger.error(f"Scraper error: {e}")
            return []

    def _deduplicate_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate sources based on content similarity"""
        seen_hashes = set()
        unique_sources = []

        for source in sources:
            # Create content hash
            content_hash = self._create_content_hash(source)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_sources.append(source)

        logger.info(f"Deduplicated to {len(unique_sources)} unique sources from {len(sources)}")
        return unique_sources

    def _create_content_hash(self, source: dict[str, Any]) -> str:
        """Create hash for duplicate detection"""
        title = source.get("title", "").strip().lower()
        content = source.get("content", "")[:200].strip().lower()  # First 200 chars
        url = source.get("url", "").strip().lower()

        hash_content = f"{title}|{content}|{url}"
        return hashlib.md5(hash_content.encode()).hexdigest()

    def _construct_timeline(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Construct timeline from sources"""
        timeline_events = []

        for source in sources:
            date = source.get("date")
            if date:
                if isinstance(date, str):
                    try:
                        date = datetime.fromisoformat(date)
                    except:
                        continue

                timeline_events.append(
                    {
                        "date": date.isoformat(),
                        "title": source.get("title", "Unknown"),
                        "url": source.get("url", ""),
                        "source_type": source.get("source_type", "unknown"),
                        "priority_score": source.get("priority_score", 0.5),
                    }
                )

        # Sort by date
        timeline_events.sort(key=lambda x: x["date"])

        return timeline_events

    def _analyze_source_breakdown(self, sources: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze breakdown of sources by type"""
        breakdown = {}

        for source in sources:
            source_type = source.get("source_type", "unknown")
            breakdown[source_type] = breakdown.get(source_type, 0) + 1

        return breakdown

    async def interactive_research(self, topic: str):
        """Generator for streaming research results"""
        yield f"🎯 Zahajuji interaktivní výzkum pro téma: {topic}\n"
        yield "=" * 50 + "\n\n"

        # Start research
        start_time = datetime.now()
        query = ResearchQuery(topic=topic, sources=list(self.scrapers.keys()))

        yield "🔍 Krok 1: Spouštím paralelní scraping všech zdrojů...\n"
        raw_sources = await self._orchestrate_scraping(query)
        yield f"✅ Nalezeno {len(raw_sources)} zdrojů\n\n"

        yield "📊 Krok 2: Vyhodnocuji prioritu a filtruju zdroje...\n"
        scored_sources = self._score_and_filter_sources(raw_sources, query)
        yield f"✅ Vybráno {len(scored_sources)} vysoce prioritních zdrojů\n\n"

        yield "🔗 Krok 3: Analyzuji křížové odkazy...\n"
        unique_sources = self._deduplicate_sources(scored_sources)
        cross_references = self.cross_ref_analyzer.find_connections(unique_sources)
        yield f"✅ Nalezeno {len(cross_references)} křížových odkazů\n\n"

        yield "📅 Krok 4: Konstruuji časovou linii...\n"
        timeline = self._construct_timeline(unique_sources)
        yield f"✅ Vytvořena časová linie s {len(timeline)} událostmi\n\n"

        yield "🤖 Krok 5: AI analýza výsledků...\n"
        context = self.context_manager.build_context(unique_sources, cross_references, timeline)
        ai_analysis = await self.ollama_agent.analyze_research(topic, context, "general_research")
        yield f"✅ AI analýza dokončena (jistota: {ai_analysis.confidence:.1%})\n\n"

        execution_time = (datetime.now() - start_time).total_seconds()
        yield f"🎉 Výzkum dokončen za {execution_time:.2f} sekund!\n"
        yield f"📊 Celkový počet zdrojů: {len(unique_sources)}\n"
        yield f"🎯 Výsledná jistota: {ai_analysis.confidence:.1%}\n\n"

        yield "🔍 KLÍČOVÁ ZJIŠTĚNÍ:\n"
        yield "-" * 30 + "\n"
        for finding in ai_analysis.key_findings[:5]:
            yield f"• {finding}\n"

    async def verify_setup(self) -> dict[str, bool]:
        """Verify system setup and component status"""
        status = {}

        # Check Ollama connection
        try:
            await self.ollama_agent.verify_connection()
            status["ollama_connection"] = True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            status["ollama_connection"] = False

        # Check scrapers initialization
        status["scrapers_initialized"] = len(self.scrapers) > 0

        # Check cache
        try:
            self.cache.get_stats()
            status["cache_system"] = True
        except Exception as e:
            logger.error(f"Cache system error: {e}")
            status["cache_system"] = False

        # Check compliance checker
        try:
            self.compliance_checker.display_legal_disclaimer()
            status["compliance_checker"] = True
        except Exception as e:
            logger.error(f"Compliance checker error: {e}")
            status["compliance_checker"] = False

        # Check enhanced analyzer
        try:
            # Test with dummy data
            test_sources = [{"title": "test", "content": "test content"}]
            self.enhanced_analyzer.analyze_research_data(test_sources, "test", self.priority_scorer)
            status["enhanced_analyzer"] = True
        except Exception as e:
            logger.error(f"Enhanced analyzer error: {e}")
            status["enhanced_analyzer"] = False

        return status

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics"""
        stats = {
            "scrapers_count": len(self.scrapers),
            "enabled_scrapers": list(self.scrapers.keys()),
            "cache_stats": self.cache.get_stats() if hasattr(self.cache, "get_stats") else {},
            "config_sources": len(self.config["research_config"]["sources"]),
            "ollama_model": self.config["research_config"]["ollama"]["model"],
            "ollama_host": self.config["research_config"]["ollama"]["host"],
        }
        return stats
