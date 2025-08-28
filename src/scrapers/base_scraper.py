#!/usr/bin/env python3
"""
BASE (Bielefeld Academic Search Engine) Scraper for Deep Research Tool
Accesses 150+ million documents from 7000+ Deep Web sources

Author: Advanced IT Specialist
"""

import asyncio
import aiohttp
import logging
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import quote, urlencode
import time
import re

logger = logging.getLogger(__name__)

class BASEScraper:
    """Scraper for BASE (Bielefeld Academic Search Engine) - Deep Web academic content"""

    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"

        # Document type classifications
        self.document_types = {
            '1': 'article',
            '2': 'bachelor_thesis',
            '3': 'master_thesis',
            '4': 'doctoral_thesis',
            '5': 'habilitation',
            '6': 'book',
            '7': 'bookpart',
            '8': 'conference_object',
            '9': 'patent',
            '10': 'preprint',
            '11': 'report',
            '12': 'review',
            '13': 'annotation',
            '14': 'contribution_to_journal',
            '15': 'other'
        }

        # Subject classifications for targeted searching
        self.subject_areas = {
            'social_sciences': ['300', '301', '302', '303', '304', '305'],
            'history': ['900', '901', '902', '903', '904', '905'],
            'political_science': ['320', '321', '322', '323', '324', '325'],
            'psychology': ['150', '151', '152', '153', '154', '155'],
            'medicine': ['610', '611', '612', '613', '614', '615'],
            'technology': ['600', '601', '602', '603', '604', '605']
        }

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Search BASE database for academic and research documents"""
        await self._rate_limit()

        results = []

        # Create multiple search strategies
        search_strategies = self._create_search_strategies(topic)

        for strategy in search_strategies:
            try:
                search_results = await self._execute_base_search(strategy, time_range)
                results.extend(search_results)

                # Add delay between searches
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in BASE search strategy '{strategy['name']}': {e}")
                continue

        # Deduplicate and enhance results
        unique_results = self._deduplicate_results(results)
        enhanced_results = self._enhance_document_metadata(unique_results)

        logger.info(f"BASE search for '{topic}' found {len(enhanced_results)} academic documents")
        return enhanced_results

    def _create_search_strategies(self, topic: str) -> List[Dict[str, Any]]:
        """Create multiple search strategies for comprehensive coverage"""
        strategies = []
        topic_lower = topic.lower()

        # Strategy 1: Direct topic search
        strategies.append({
            'name': 'direct_search',
            'query': topic,
            'filters': {},
            'boost': 'relevance'
        })

        # Strategy 2: Academic terminology search
        academic_terms = self._generate_academic_terms(topic)
        strategies.append({
            'name': 'academic_terms',
            'query': ' OR '.join(academic_terms),
            'filters': {'doctype': ['1', '4', '11']},  # Articles, theses, reports
            'boost': 'academic'
        })

        # Strategy 3: Subject-specific search
        relevant_subjects = self._identify_relevant_subjects(topic_lower)
        if relevant_subjects:
            strategies.append({
                'name': 'subject_specific',
                'query': topic,
                'filters': {'dcsubject': relevant_subjects},
                'boost': 'subject'
            })

        # Strategy 4: Historical/temporal search for certain topics
        if any(term in topic_lower for term in ['history', 'historical', 'cold war', 'vietnam', 'wwii']):
            strategies.append({
                'name': 'historical',
                'query': f'"{topic}" history historical',
                'filters': {'dcsubject': self.subject_areas['history']},
                'boost': 'temporal'
            })

        # Strategy 5: Research methodology search
        if any(term in topic_lower for term in ['research', 'study', 'analysis', 'investigation']):
            strategies.append({
                'name': 'research_focused',
                'query': f'{topic} research study analysis',
                'filters': {'doctype': ['1', '11', '12']},  # Articles, reports, reviews
                'boost': 'methodology'
            })

        return strategies[:3]  # Limit to top 3 strategies

    def _generate_academic_terms(self, topic: str) -> List[str]:
        """Generate academic variations of search terms"""
        terms = [topic]

        # Add academic synonyms
        academic_mappings = {
            'conspiracy': ['conspiracy theory', 'alternative narrative', 'unofficial account'],
            'assassination': ['political assassination', 'targeted killing', 'political violence'],
            'government': ['state apparatus', 'political institution', 'public administration'],
            'surveillance': ['monitoring', 'intelligence gathering', 'observation'],
            'propaganda': ['information warfare', 'media manipulation', 'public relations']
        }

        topic_lower = topic.lower()
        for key, synonyms in academic_mappings.items():
            if key in topic_lower:
                terms.extend(synonyms)

        # Add methodological terms
        if any(word in topic_lower for word in ['theory', 'analysis', 'study']):
            terms.extend([f'{topic} methodology', f'{topic} framework'])

        return terms[:5]  # Limit to 5 terms

    def _identify_relevant_subjects(self, topic: str) -> List[str]:
        """Identify relevant subject classifications"""
        relevant_subjects = []

        subject_keywords = {
            'social_sciences': ['social', 'society', 'community', 'culture'],
            'history': ['history', 'historical', 'past', 'chronicle'],
            'political_science': ['political', 'government', 'policy', 'state'],
            'psychology': ['psychology', 'behavior', 'mental', 'cognitive'],
            'medicine': ['medical', 'health', 'disease', 'treatment'],
            'technology': ['technology', 'technical', 'engineering', 'digital']
        }

        for subject_area, keywords in subject_keywords.items():
            if any(keyword in topic for keyword in keywords):
                relevant_subjects.extend(self.subject_areas.get(subject_area, []))

        return relevant_subjects[:10]  # Limit to 10 subject codes

    async def _execute_base_search(self, strategy: Dict[str, Any], time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Execute BASE search with specific strategy"""
        results = []

        try:
            # Build search parameters
            params = {
                'func': 'PerformSearch',
                'query': strategy['query'],
                'format': 'json',
                'hits': 50,  # Maximum results per request
                'offset': 0
            }

            # Add filters
            filters = strategy.get('filters', {})
            if filters:
                filter_strings = []
                for filter_type, values in filters.items():
                    if isinstance(values, list):
                        filter_strings.append(f"{filter_type}:({' OR '.join(values)})")
                    else:
                        filter_strings.append(f"{filter_type}:{values}")

                if filter_strings:
                    params['filter'] = ' AND '.join(filter_strings)

            # Add date range if specified
            if time_range:
                start_year = time_range[0].year
                end_year = time_range[1].year
                params['daterange'] = f"{start_year}-{end_year}"

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        documents = data.get('response', {}).get('docs', [])

                        for doc in documents:
                            parsed_doc = self._parse_base_document(doc, strategy)
                            if parsed_doc:
                                results.append(parsed_doc)
                    else:
                        logger.warning(f"BASE API returned status {response.status}")

        except Exception as e:
            logger.error(f"Error executing BASE search: {e}")

        return results

    def _parse_base_document(self, doc: Dict[str, Any], strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse BASE document response"""
        try:
            title = doc.get('dctitle', doc.get('title', 'Unknown Document'))
            if isinstance(title, list):
                title = title[0] if title else 'Unknown Document'

            # Extract document metadata
            authors = self._extract_authors(doc)
            abstract = self._extract_abstract(doc)
            subjects = doc.get('dcsubject', [])
            if isinstance(subjects, str):
                subjects = [subjects]

            # Determine document type
            doc_type_code = doc.get('dctypenorm', ['15'])[0] if doc.get('dctypenorm') else '15'
            doc_type = self.document_types.get(doc_type_code, 'other')

            # Extract publication info
            publication_info = self._extract_publication_info(doc)

            # Get document URL and source
            doc_url = self._extract_document_url(doc)
            source_repository = doc.get('collname', 'Unknown Repository')

            # Extract date
            doc_date = self._extract_document_date(doc)

            return {
                'title': title,
                'content': abstract,
                'url': doc_url,
                'source_type': 'base_academic',
                'source_url': doc_url,
                'date': doc_date,
                'metadata': {
                    'authors': authors,
                    'document_type': doc_type,
                    'subjects': subjects,
                    'publication_info': publication_info,
                    'source_repository': source_repository,
                    'search_strategy': strategy['name'],
                    'languages': doc.get('dclang', []),
                    'rights': doc.get('dcrights', ''),
                    'identifier': doc.get('dcidentifier', ''),
                    'base_id': doc.get('dcid', ''),
                    'relevance_score': doc.get('score', 0.0),
                    'open_access': doc.get('oa', False),
                    'peer_reviewed': self._detect_peer_review(doc),
                    'citation_count': doc.get('citationcount', 0)
                }
            }

        except Exception as e:
            logger.error(f"Error parsing BASE document: {e}")
            return None

    def _extract_authors(self, doc: Dict[str, Any]) -> List[str]:
        """Extract author information"""
        authors = []

        # Try different author fields
        author_fields = ['dccreator', 'dccontributor', 'author']

        for field in author_fields:
            author_data = doc.get(field, [])
            if isinstance(author_data, str):
                authors.append(author_data)
            elif isinstance(author_data, list):
                authors.extend(author_data)

        # Clean and deduplicate authors
        cleaned_authors = []
        for author in authors:
            if isinstance(author, str) and author.strip():
                cleaned_author = author.strip()
                if cleaned_author not in cleaned_authors:
                    cleaned_authors.append(cleaned_author)

        return cleaned_authors[:10]  # Limit to 10 authors

    def _extract_abstract(self, doc: Dict[str, Any]) -> str:
        """Extract document abstract or description"""
        abstract_fields = ['dcdescription', 'dcabstract', 'description', 'abstract']

        for field in abstract_fields:
            abstract = doc.get(field, '')
            if isinstance(abstract, list):
                abstract = ' '.join(abstract)

            if abstract and len(abstract.strip()) > 50:
                return abstract.strip()[:2000]  # Limit to 2000 characters

        return ''

    def _extract_publication_info(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract publication information"""
        return {
            'publisher': doc.get('dcpublisher', ''),
            'journal': doc.get('dcjournal', ''),
            'volume': doc.get('dcvolume', ''),
            'issue': doc.get('dcissue', ''),
            'pages': doc.get('dcpages', ''),
            'isbn': doc.get('dcisbn', ''),
            'issn': doc.get('dcissn', ''),
            'doi': doc.get('dcdoi', '')
        }

    def _extract_document_url(self, doc: Dict[str, Any]) -> str:
        """Extract document URL"""
        # Try different URL fields
        url_fields = ['dclink', 'dcidentifier', 'link', 'url']

        for field in url_fields:
            url_data = doc.get(field, '')
            if isinstance(url_data, list):
                url_data = url_data[0] if url_data else ''

            if url_data and url_data.startswith('http'):
                return url_data

        # Fallback to BASE record URL
        base_id = doc.get('dcid', '')
        if base_id:
            return f"https://www.base-search.net/Record/{base_id}"

        return ''

    def _extract_document_date(self, doc: Dict[str, Any]) -> Optional[datetime]:
        """Extract document date"""
        date_fields = ['dcdate', 'dcyear', 'date', 'year']

        for field in date_fields:
            date_str = doc.get(field, '')
            if isinstance(date_str, list):
                date_str = date_str[0] if date_str else ''

            if date_str:
                try:
                    # Try various date formats
                    if len(date_str) == 4 and date_str.isdigit():  # Year only
                        return datetime(int(date_str), 1, 1)
                    elif '-' in date_str:  # YYYY-MM-DD format
                        return datetime.strptime(date_str[:10], '%Y-%m-%d')
                    elif '/' in date_str:  # MM/DD/YYYY format
                        return datetime.strptime(date_str, '%m/%d/%Y')
                except Exception:
                    continue

        return None

    def _detect_peer_review(self, doc: Dict[str, Any]) -> bool:
        """Detect if document is peer-reviewed"""
        # Check document type
        doc_type_code = doc.get('dctypenorm', ['15'])[0] if doc.get('dctypenorm') else '15'
        if doc_type_code in ['1', '12']:  # Articles and reviews are typically peer-reviewed
            return True

        # Check for peer-review indicators in metadata
        peer_review_indicators = [
            'peer-reviewed', 'peer reviewed', 'refereed', 'reviewed',
            'journal article', 'academic journal'
        ]

        text_fields = [
            doc.get('dctype', ''),
            doc.get('dcdescription', ''),
            doc.get('dcpublisher', ''),
            doc.get('dcjournal', '')
        ]

        combined_text = ' '.join(str(field) for field in text_fields).lower()

        return any(indicator in combined_text for indicator in peer_review_indicators)

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents"""
        seen_identifiers = set()
        unique_results = []

        for result in results:
            # Create unique identifier from multiple fields
            metadata = result.get('metadata', {})
            identifier = (
                metadata.get('base_id', '') or
                metadata.get('identifier', '') or
                metadata.get('publication_info', {}).get('doi', '') or
                result.get('title', '')
            )

            if identifier and identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_results.append(result)

        return unique_results

    def _enhance_document_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance document metadata with additional analysis"""
        enhanced_results = []

        for result in results:
            try:
                # Calculate academic credibility score
                credibility_score = self._calculate_academic_credibility(result)
                result['metadata']['credibility_score'] = credibility_score

                # Add research impact indicators
                impact_indicators = self._assess_research_impact(result)
                result['metadata']['impact_indicators'] = impact_indicators

                # Add topical relevance score
                relevance_score = self._calculate_topical_relevance(result)
                result['metadata']['topical_relevance'] = relevance_score

                enhanced_results.append(result)

            except Exception as e:
                logger.error(f"Error enhancing document metadata: {e}")
                enhanced_results.append(result)

        return enhanced_results

    def _calculate_academic_credibility(self, result: Dict[str, Any]) -> float:
        """Calculate academic credibility score"""
        score = 0.5  # Base score
        metadata = result.get('metadata', {})

        # Peer review boost
        if metadata.get('peer_reviewed', False):
            score += 0.2

        # Document type credibility
        doc_type = metadata.get('document_type', 'other')
        type_scores = {
            'article': 0.3,
            'doctoral_thesis': 0.25,
            'report': 0.2,
            'review': 0.3,
            'book': 0.15,
            'master_thesis': 0.1
        }
        score += type_scores.get(doc_type, 0.05)

        # Citation count influence
        citation_count = metadata.get('citation_count', 0)
        if citation_count > 10:
            score += 0.1
        elif citation_count > 5:
            score += 0.05

        # Open access availability
        if metadata.get('open_access', False):
            score += 0.05

        # Author count (collaborative work often more credible)
        author_count = len(metadata.get('authors', []))
        if 2 <= author_count <= 5:
            score += 0.05

        return min(1.0, score)

    def _assess_research_impact(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess research impact indicators"""
        metadata = result.get('metadata', {})

        return {
            'has_doi': bool(metadata.get('publication_info', {}).get('doi')),
            'open_access': metadata.get('open_access', False),
            'citation_count': metadata.get('citation_count', 0),
            'peer_reviewed': metadata.get('peer_reviewed', False),
            'institutional_affiliation': bool(metadata.get('authors')),
            'recent_publication': self._is_recent_publication(result.get('date')),
            'subject_coverage': len(metadata.get('subjects', [])),
            'international_scope': self._detect_international_scope(result)
        }

    def _calculate_topical_relevance(self, result: Dict[str, Any]) -> float:
        """Calculate topical relevance score"""
        # This would ideally use NLP similarity to the original search query
        # For now, use a simplified approach based on metadata

        score = result.get('metadata', {}).get('relevance_score', 0.0)

        # Normalize BASE relevance score (usually 0-100) to 0-1
        if score > 1:
            score = score / 100

        return min(1.0, max(0.0, score))

    def _is_recent_publication(self, doc_date: Optional[datetime]) -> bool:
        """Check if publication is recent (within last 10 years)"""
        if not doc_date:
            return False

        cutoff_date = datetime.now().replace(year=datetime.now().year - 10)
        return doc_date >= cutoff_date

    def _detect_international_scope(self, result: Dict[str, Any]) -> bool:
        """Detect if research has international scope"""
        authors = result.get('metadata', {}).get('authors', [])

        # Simple heuristic: multiple authors often indicate broader collaboration
        return len(authors) >= 3

    async def _rate_limit(self):
        """Apply rate limiting for BASE API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

    def search(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Synchronous search wrapper"""
        return asyncio.run(self.search_async(topic, time_range))
