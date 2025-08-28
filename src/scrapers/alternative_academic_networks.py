#!/usr/bin/env python3
"""
Alternative Academic Networks Integration
Access to independent journals, radical archives, and specialized research networks

Author: Advanced IT Specialist
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class AlternativeAcademicDocument:
    """Represents a document from alternative academic sources"""
    document_id: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: Optional[datetime]
    source_network: str  # 'independent_journal', 'radical_archive', 'environmental_research'
    journal_name: str
    document_type: str
    subjects: List[str]
    keywords: List[str]
    methodology: str
    political_stance: Optional[str]  # 'radical', 'progressive', 'critical', 'activist'
    accessibility: str  # 'open_access', 'free', 'community_funded'
    url: str
    full_text_url: Optional[str] = None
    impact_metrics: Dict[str, Any] = field(default_factory=dict)
    community_engagement: Dict[str, Any] = field(default_factory=dict)
    alternative_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchNetwork:
    """Represents an alternative research network"""
    network_id: str
    name: str
    description: str
    focus_areas: List[str]
    ideology: str
    founding_year: int
    member_count: int
    publication_count: int
    access_model: str
    website_url: str
    api_endpoint: Optional[str] = None

class IndependentJournalsCollector:
    """Collector for independent academic journals"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Independent and radical journals
        self.independent_journals = {
            'acme_journal': {
                'name': 'ACME: An International Journal for Critical Geographies',
                'url': 'https://acme-journal.org',
                'focus': ['critical_geography', 'social_justice', 'political_ecology'],
                'ideology': 'critical',
                'access': 'open_access'
            },
            'journal_peer_production': {
                'name': 'Journal of Peer Production',
                'url': 'http://peerproduction.net',
                'focus': ['commons', 'peer_production', 'digital_commons'],
                'ideology': 'commons_based',
                'access': 'open_access'
            },
            'radical_housing_journal': {
                'name': 'Radical Housing Journal',
                'url': 'https://radicalhousingjournal.org',
                'focus': ['housing_justice', 'urban_studies', 'anti_gentrification'],
                'ideology': 'radical',
                'access': 'open_access'
            },
            'degrowth_info': {
                'name': 'Degrowth.info',
                'url': 'https://degrowth.info',
                'focus': ['degrowth', 'post_growth', 'ecological_economics'],
                'ideology': 'post_growth',
                'access': 'community_funded'
            },
            'uneven_earth': {
                'name': 'Uneven Earth',
                'url': 'https://unevenearth.org',
                'focus': ['political_ecology', 'environmental_justice', 'climate_politics'],
                'ideology': 'environmental_justice',
                'access': 'open_access'
            },
            'ephemera_journal': {
                'name': 'Ephemera: Theory & Politics in Organization',
                'url': 'http://www.ephemerajournal.org',
                'focus': ['organization_theory', 'critical_management', 'anarchist_studies'],
                'ideology': 'critical',
                'access': 'open_access'
            },
            'interface_journal': {
                'name': 'Interface: A Journal For and About Social Movements',
                'url': 'https://www.interfacejournal.net',
                'focus': ['social_movements', 'activism', 'protest_studies'],
                'ideology': 'activist',
                'access': 'open_access'
            }
        }

        # Research methodology classifications
        self.methodologies = {
            'participatory_action_research': 'Participatory Action Research',
            'critical_ethnography': 'Critical Ethnography',
            'militant_research': 'Militant Research',
            'community_based_research': 'Community-Based Research',
            'decolonial_methodology': 'Decolonial Methodology',
            'feminist_methodology': 'Feminist Methodology'
        }

    async def search_independent_journals(self,
                                        query: str,
                                        focus_areas: Optional[List[str]] = None,
                                        ideology_filter: Optional[str] = None,
                                        max_results: int = 100) -> List[AlternativeAcademicDocument]:
        """Search across independent academic journals"""

        documents = []

        # Filter journals based on criteria
        target_journals = self._filter_journals(focus_areas, ideology_filter)

        for journal_id, journal_info in target_journals.items():
            try:
                logger.info(f"Searching {journal_info['name']}...")

                journal_docs = await self._search_single_journal(
                    journal_id, journal_info, query, max_results // len(target_journals)
                )
                documents.extend(journal_docs)

                await self._rate_limit()

            except Exception as e:
                logger.error(f"Error searching {journal_info['name']}: {str(e)}")
                continue

        # Sort by relevance and alternative metrics
        documents.sort(key=lambda x: x.alternative_metrics.get('community_score', 0), reverse=True)

        logger.info(f"Found {len(documents)} documents from independent journals")
        return documents[:max_results]

    def _filter_journals(self, focus_areas: Optional[List[str]], ideology_filter: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Filter journals based on focus areas and ideology"""
        if not focus_areas and not ideology_filter:
            return self.independent_journals

        filtered_journals = {}

        for journal_id, journal_info in self.independent_journals.items():
            include_journal = True

            # Filter by focus areas
            if focus_areas:
                journal_focus = journal_info.get('focus', [])
                if not any(area in journal_focus for area in focus_areas):
                    include_journal = False

            # Filter by ideology
            if ideology_filter and journal_info.get('ideology') != ideology_filter:
                include_journal = False

            if include_journal:
                filtered_journals[journal_id] = journal_info

        return filtered_journals

    async def _search_single_journal(self,
                                   journal_id: str,
                                   journal_info: Dict[str, Any],
                                   query: str,
                                   max_results: int) -> List[AlternativeAcademicDocument]:
        """Search a single independent journal"""

        documents = []

        try:
            # Build search URL based on journal
            search_url = self._build_journal_search_url(journal_id, journal_info, query)

            if not search_url:
                return documents

            # Fetch search results
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        documents = self._parse_journal_results(
                            journal_id, journal_info, content, max_results
                        )
                    else:
                        logger.warning(f"Search failed for {journal_info['name']} with status {response.status}")

        except Exception as e:
            logger.error(f"Error searching {journal_info['name']}: {str(e)}")

        return documents

    def _build_journal_search_url(self, journal_id: str, journal_info: Dict[str, Any], query: str) -> Optional[str]:
        """Build search URL for specific journal"""
        base_url = journal_info['url']

        # Journal-specific search patterns
        search_patterns = {
            'acme_journal': f"{base_url}/index.php/acme/search/search?query={quote(query)}",
            'journal_peer_production': f"{base_url}/?s={quote(query)}",
            'radical_housing_journal': f"{base_url}/?s={quote(query)}",
            'degrowth_info': f"{base_url}/?s={quote(query)}",
            'uneven_earth': f"{base_url}/?s={quote(query)}",
            'ephemera_journal': f"{base_url}/index.php/ephemera/search/search?query={quote(query)}",
            'interface_journal': f"{base_url}/?s={quote(query)}"
        }

        return search_patterns.get(journal_id)

    def _parse_journal_results(self,
                             journal_id: str,
                             journal_info: Dict[str, Any],
                             html_content: str,
                             max_results: int) -> List[AlternativeAcademicDocument]:
        """Parse search results from journal HTML"""

        documents = []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find article entries (patterns vary by journal)
            article_selectors = [
                'article', '.article', '.post', '.entry',
                '.search-result', '.publication', '.paper'
            ]

            articles = []
            for selector in article_selectors:
                found_articles = soup.select(selector)
                if found_articles:
                    articles = found_articles
                    break

            for article in articles[:max_results]:
                doc = self._extract_journal_document(journal_id, journal_info, article)
                if doc:
                    documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing results for {journal_info['name']}: {str(e)}")

        return documents

    def _extract_journal_document(self,
                                journal_id: str,
                                journal_info: Dict[str, Any],
                                article_element) -> Optional[AlternativeAcademicDocument]:
        """Extract document information from article element"""

        try:
            # Extract title
            title_selectors = ['h1', 'h2', 'h3', '.title', '.entry-title', '.article-title']
            title = self._extract_text_by_selectors(article_element, title_selectors)

            if not title:
                return None

            # Extract authors
            author_selectors = ['.author', '.authors', '.byline', '.by-author']
            authors_text = self._extract_text_by_selectors(article_element, author_selectors)
            authors = self._parse_authors(authors_text)

            # Extract abstract/excerpt
            abstract_selectors = ['.abstract', '.excerpt', '.summary', '.content', 'p']
            abstract = self._extract_text_by_selectors(article_element, abstract_selectors)

            # Extract publication date
            date_selectors = ['.date', '.published', '.post-date', 'time']
            date_text = self._extract_text_by_selectors(article_element, date_selectors)
            publication_date = self._parse_date(date_text)

            # Extract URL
            url_element = article_element.find('a', href=True)
            url = urljoin(journal_info['url'], url_element['href']) if url_element else journal_info['url']

            # Extract keywords and subjects
            keywords, subjects = self._extract_keywords_and_subjects(title, abstract, journal_info)

            # Determine methodology
            methodology = self._classify_methodology(title, abstract)

            # Create document
            document = AlternativeAcademicDocument(
                document_id=f"{journal_id}_{hashlib.md5((url + title).encode()).hexdigest()[:12]}",
                title=title,
                authors=authors,
                abstract=abstract or "",
                publication_date=publication_date,
                source_network='independent_journal',
                journal_name=journal_info['name'],
                document_type='article',
                subjects=subjects,
                keywords=keywords,
                methodology=methodology,
                political_stance=journal_info.get('ideology'),
                accessibility=journal_info.get('access', 'open_access'),
                url=url,
                full_text_url=url,  # Most independent journals provide full text
                alternative_metrics=self._calculate_alternative_metrics(journal_id, journal_info)
            )

            return document

        except Exception as e:
            logger.error(f"Error extracting document: {str(e)}")
            return None

    def _extract_text_by_selectors(self, element, selectors: List[str]) -> str:
        """Extract text using CSS selectors"""
        for selector in selectors:
            found_element = element.select_one(selector)
            if found_element:
                return found_element.get_text(strip=True)
        return ""

    def _parse_authors(self, authors_text: str) -> List[str]:
        """Parse author names from text"""
        if not authors_text:
            return []

        # Common author separators
        separators = [',', ';', ' and ', ' & ', '\n']

        authors = [authors_text]
        for sep in separators:
            if sep in authors_text:
                authors = authors_text.split(sep)
                break

        # Clean author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) > 2:
                cleaned_authors.append(author)

        return cleaned_authors

    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse publication date from text"""
        if not date_text:
            return None

        # Common date formats
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}\.\d{1,2}\.\d{4})',  # DD.MM.YYYY
            r'(\d{4})',  # Year only
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                date_str = match.group(1)

                # Try different parsing formats
                formats = ['%Y-%m-%d', '%m/%d/%Y', '%d.%m.%Y', '%Y']

                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

        return None

    def _extract_keywords_and_subjects(self, title: str, abstract: str, journal_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract keywords and subjects from content"""
        text = f"{title} {abstract}".lower()

        # Subject area keywords
        subject_keywords = {
            'political_ecology': ['political ecology', 'environmental politics', 'nature-society'],
            'critical_geography': ['critical geography', 'spatial justice', 'urban geography'],
            'social_movements': ['social movement', 'activism', 'protest', 'resistance'],
            'commons': ['commons', 'commoning', 'collective', 'community ownership'],
            'degrowth': ['degrowth', 'post-growth', 'steady state', 'limits to growth'],
            'environmental_justice': ['environmental justice', 'environmental racism', 'climate justice'],
            'housing_justice': ['housing justice', 'gentrification', 'displacement', 'housing rights'],
            'feminist_studies': ['feminist', 'gender', 'patriarchy', 'women'],
            'decolonial_studies': ['decolonial', 'postcolonial', 'indigenous', 'colonial'],
            'anarchist_studies': ['anarchist', 'anarchism', 'mutual aid', 'self-organization']
        }

        keywords = []
        subjects = []

        # Extract based on journal focus
        journal_focus = journal_info.get('focus', [])
        subjects.extend(journal_focus)

        # Extract from text content
        for subject, subject_keywords_list in subject_keywords.items():
            if any(keyword in text for keyword in subject_keywords_list):
                subjects.append(subject)
                keywords.extend([kw for kw in subject_keywords_list if kw in text])

        return list(set(keywords)), list(set(subjects))

    def _classify_methodology(self, title: str, abstract: str) -> str:
        """Classify research methodology"""
        text = f"{title} {abstract}".lower()

        methodology_indicators = {
            'participatory_action_research': ['participatory', 'action research', 'community participation'],
            'critical_ethnography': ['ethnography', 'ethnographic', 'participant observation'],
            'militant_research': ['militant research', 'activist research', 'research justice'],
            'community_based_research': ['community based', 'community research', 'grassroots research'],
            'decolonial_methodology': ['decolonial', 'indigenous methodology', 'traditional knowledge'],
            'feminist_methodology': ['feminist methodology', 'standpoint theory', 'situated knowledge']
        }

        for methodology, indicators in methodology_indicators.items():
            if any(indicator in text for indicator in indicators):
                return methodology

        return 'qualitative_research'

    def _calculate_alternative_metrics(self, journal_id: str, journal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate alternative academic metrics"""

        # Alternative metrics focus on community impact rather than citations
        metrics = {
            'community_score': 0.0,
            'accessibility_score': 1.0,  # All independent journals are accessible
            'social_impact_score': 0.0,
            'political_relevance_score': 0.0
        }

        # Community score based on journal characteristics
        if journal_info.get('access') == 'open_access':
            metrics['community_score'] += 0.3

        if journal_info.get('ideology') in ['radical', 'critical', 'activist']:
            metrics['political_relevance_score'] += 0.4

        # Social impact based on focus areas
        focus_areas = journal_info.get('focus', [])
        social_impact_keywords = ['justice', 'movement', 'community', 'environmental', 'housing']

        if any(keyword in ' '.join(focus_areas) for keyword in social_impact_keywords):
            metrics['social_impact_score'] += 0.5

        return metrics

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

class EnvironmentalResearchCollector:
    """Collector for environmental and degrowth research networks"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Environmental research networks
        self.environmental_networks = {
            'degrowth_info': {
                'name': 'Degrowth.info',
                'url': 'https://degrowth.info',
                'focus': ['degrowth', 'post_growth_economics', 'ecological_economics'],
                'type': 'research_network'
            },
            'uneven_earth': {
                'name': 'Uneven Earth',
                'url': 'https://unevenearth.org',
                'focus': ['political_ecology', 'environmental_justice', 'climate_politics'],
                'type': 'publication_platform'
            },
            'resilience_org': {
                'name': 'Resilience.org',
                'url': 'https://www.resilience.org',
                'focus': ['community_resilience', 'transition_towns', 'permaculture'],
                'type': 'community_platform'
            },
            'post_carbon_institute': {
                'name': 'Post Carbon Institute',
                'url': 'https://www.postcarbon.org',
                'focus': ['post_carbon_society', 'energy_transition', 'local_resilience'],
                'type': 'research_institute'
            },
            'global_ecovillage_network': {
                'name': 'Global Ecovillage Network',
                'url': 'https://ecovillage.org',
                'focus': ['ecovillages', 'sustainable_communities', 'regenerative_living'],
                'type': 'network_organization'
            }
        }

    async def search_environmental_research(self,
                                          query: str,
                                          focus_areas: Optional[List[str]] = None,
                                          max_results: int = 50) -> List[AlternativeAcademicDocument]:
        """Search environmental research networks"""

        documents = []

        # Filter networks based on focus areas
        target_networks = self._filter_environmental_networks(focus_areas)

        for network_id, network_info in target_networks.items():
            try:
                logger.info(f"Searching {network_info['name']}...")

                network_docs = await self._search_environmental_network(
                    network_id, network_info, query, max_results // len(target_networks)
                )
                documents.extend(network_docs)

                await self._rate_limit()

            except Exception as e:
                logger.error(f"Error searching {network_info['name']}: {str(e)}")
                continue

        logger.info(f"Found {len(documents)} environmental research documents")
        return documents[:max_results]

    def _filter_environmental_networks(self, focus_areas: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
        """Filter environmental networks based on focus areas"""
        if not focus_areas:
            return self.environmental_networks

        filtered_networks = {}

        for network_id, network_info in self.environmental_networks.items():
            network_focus = network_info.get('focus', [])
            if any(area in network_focus for area in focus_areas):
                filtered_networks[network_id] = network_info

        return filtered_networks

    async def _search_environmental_network(self,
                                          network_id: str,
                                          network_info: Dict[str, Any],
                                          query: str,
                                          max_results: int) -> List[AlternativeAcademicDocument]:
        """Search a specific environmental research network"""

        documents = []

        try:
            # Build search URL
            search_url = f"{network_info['url']}/?s={quote(query)}"

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        documents = self._parse_environmental_results(
                            network_id, network_info, content, max_results
                        )

        except Exception as e:
            logger.error(f"Error searching {network_info['name']}: {str(e)}")

        return documents

    def _parse_environmental_results(self,
                                   network_id: str,
                                   network_info: Dict[str, Any],
                                   html_content: str,
                                   max_results: int) -> List[AlternativeAcademicDocument]:
        """Parse environmental research results"""

        documents = []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find content entries
            entries = soup.find_all(['article', '.post', '.entry', '.result'])

            for entry in entries[:max_results]:
                doc = self._extract_environmental_document(network_id, network_info, entry)
                if doc:
                    documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing environmental results: {str(e)}")

        return documents

    def _extract_environmental_document(self,
                                      network_id: str,
                                      network_info: Dict[str, Any],
                                      entry) -> Optional[AlternativeAcademicDocument]:
        """Extract environmental research document"""

        try:
            # Extract title
            title_elem = entry.find(['h1', 'h2', 'h3', 'a'])
            title = title_elem.get_text(strip=True) if title_elem else ""

            if not title:
                return None

            # Extract content/abstract
            content_elem = entry.find(['p', '.content', '.excerpt'])
            content = content_elem.get_text(strip=True) if content_elem else ""

            # Extract URL
            url_elem = entry.find('a', href=True)
            url = urljoin(network_info['url'], url_elem['href']) if url_elem else network_info['url']

            # Extract date
            date_elem = entry.find(['.date', 'time', '.published'])
            date_text = date_elem.get_text(strip=True) if date_elem else ""
            publication_date = self._parse_environmental_date(date_text)

            # Classify as environmental research
            subjects = network_info.get('focus', [])
            keywords = self._extract_environmental_keywords(title, content)

            document = AlternativeAcademicDocument(
                document_id=f"{network_id}_{hashlib.md5((url + title).encode()).hexdigest()[:12]}",
                title=title,
                authors=[],  # Often not specified in environmental platforms
                abstract=content,
                publication_date=publication_date,
                source_network='environmental_research',
                journal_name=network_info['name'],
                document_type='article',
                subjects=subjects,
                keywords=keywords,
                methodology='community_based_research',
                political_stance='environmental_justice',
                accessibility='open_access',
                url=url,
                full_text_url=url,
                alternative_metrics={
                    'environmental_impact_score': 0.8,
                    'community_relevance': 0.7,
                    'practical_application': 0.9
                }
            )

            return document

        except Exception as e:
            logger.error(f"Error extracting environmental document: {str(e)}")
            return None

    def _parse_environmental_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from environmental platform"""
        # Similar to journal date parsing but with environmental platform specifics
        return None  # Simplified for this implementation

    def _extract_environmental_keywords(self, title: str, content: str) -> List[str]:
        """Extract environmental keywords"""
        text = f"{title} {content}".lower()

        environmental_keywords = [
            'sustainability', 'climate change', 'biodiversity', 'ecosystem',
            'renewable energy', 'carbon footprint', 'circular economy',
            'permaculture', 'degrowth', 'resilience', 'transition',
            'ecological', 'environmental justice', 'green technology'
        ]

        found_keywords = [kw for kw in environmental_keywords if kw in text]
        return found_keywords

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

class AlternativeAcademicOrchestrator:
    """Orchestrator for alternative academic networks"""

    def __init__(self):
        self.independent_journals = IndependentJournalsCollector()
        self.environmental_research = EnvironmentalResearchCollector()

    async def search_all_alternative_sources(self,
                                           query: str,
                                           source_types: Optional[List[str]] = None,
                                           focus_areas: Optional[List[str]] = None,
                                           ideology_filter: Optional[str] = None,
                                           max_results_per_source: int = 50) -> Dict[str, List[AlternativeAcademicDocument]]:
        """Search all alternative academic sources"""

        results = {}

        # Default to all source types if not specified
        if not source_types:
            source_types = ['independent_journals', 'environmental_research']

        # Search independent journals
        if 'independent_journals' in source_types:
            logger.info("Searching independent journals...")
            independent_docs = await self.independent_journals.search_independent_journals(
                query=query,
                focus_areas=focus_areas,
                ideology_filter=ideology_filter,
                max_results=max_results_per_source
            )
            results['independent_journals'] = independent_docs

        # Search environmental research networks
        if 'environmental_research' in source_types:
            logger.info("Searching environmental research networks...")
            environmental_docs = await self.environmental_research.search_environmental_research(
                query=query,
                focus_areas=focus_areas,
                max_results=max_results_per_source
            )
            results['environmental_research'] = environmental_docs

        return results

    def analyze_alternative_collection(self, results: Dict[str, List[AlternativeAcademicDocument]]) -> Dict[str, Any]:
        """Analyze collection of alternative academic documents"""

        analysis = {
            'total_documents': 0,
            'by_source_network': {},
            'political_stance_distribution': {},
            'methodology_distribution': {},
            'subject_analysis': {},
            'accessibility_analysis': {},
            'temporal_distribution': {},
            'alternative_metrics_summary': {},
            'recommendations': []
        }

        all_docs = []
        for source_type, docs in results.items():
            analysis['by_source_network'][source_type] = len(docs)
            analysis['total_documents'] += len(docs)
            all_docs.extend(docs)

        if not all_docs:
            return analysis

        # Political stance analysis
        stances = {}
        for doc in all_docs:
            stance = doc.political_stance or 'neutral'
            stances[stance] = stances.get(stance, 0) + 1
        analysis['political_stance_distribution'] = stances

        # Methodology analysis
        methodologies = {}
        for doc in all_docs:
            method = doc.methodology
            methodologies[method] = methodologies.get(method, 0) + 1
        analysis['methodology_distribution'] = methodologies

        # Subject analysis
        all_subjects = []
        for doc in all_docs:
            all_subjects.extend(doc.subjects)

        subject_freq = {}
        for subject in all_subjects:
            subject_freq[subject] = subject_freq.get(subject, 0) + 1

        analysis['subject_analysis'] = dict(sorted(subject_freq.items(), key=lambda x: x[1], reverse=True)[:10])

        # Accessibility analysis
        accessibility = {}
        for doc in all_docs:
            access_type = doc.accessibility
            accessibility[access_type] = accessibility.get(access_type, 0) + 1
        analysis['accessibility_analysis'] = accessibility

        # Temporal analysis
        years = {}
        for doc in all_docs:
            if doc.publication_date:
                year = doc.publication_date.year
                years[year] = years.get(year, 0) + 1
        analysis['temporal_distribution'] = dict(sorted(years.items()))

        # Alternative metrics summary
        if all_docs:
            community_scores = [doc.alternative_metrics.get('community_score', 0) for doc in all_docs]
            social_impact_scores = [doc.alternative_metrics.get('social_impact_score', 0) for doc in all_docs]

            analysis['alternative_metrics_summary'] = {
                'average_community_score': sum(community_scores) / len(community_scores),
                'average_social_impact': sum(social_impact_scores) / len(social_impact_scores),
                'high_community_engagement': len([s for s in community_scores if s > 0.7]),
                'high_social_impact': len([s for s in social_impact_scores if s > 0.7])
            }

        # Generate recommendations
        analysis['recommendations'] = self._generate_alternative_recommendations(analysis, all_docs)

        return analysis

    def _generate_alternative_recommendations(self, analysis: Dict[str, Any], documents: List[AlternativeAcademicDocument]) -> List[str]:
        """Generate recommendations for alternative academic research"""
        recommendations = []

        # Political stance recommendations
        stances = analysis['political_stance_distribution']
        if 'radical' in stances and stances['radical'] > 5:
            recommendations.append("Strong representation of radical perspectives - consider for critical analysis frameworks")

        # Methodology recommendations
        methodologies = analysis['methodology_distribution']
        if 'participatory_action_research' in methodologies:
            recommendations.append("Participatory research approaches available - suitable for community-engaged projects")

        # Subject diversity recommendations
        subject_count = len(analysis['subject_analysis'])
        if subject_count > 10:
            recommendations.append("High subject diversity found - opportunities for interdisciplinary analysis")

        # Accessibility recommendations
        open_access_count = analysis['accessibility_analysis'].get('open_access', 0)
        if open_access_count > analysis['total_documents'] * 0.8:
            recommendations.append("Excellent open access coverage - full texts readily available for analysis")

        # Alternative metrics recommendations
        metrics = analysis.get('alternative_metrics_summary', {})
        if metrics.get('average_community_score', 0) > 0.6:
            recommendations.append("High community engagement scores - sources have strong grassroots connections")

        return recommendations
