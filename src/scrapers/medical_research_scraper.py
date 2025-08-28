#!/usr/bin/env python3
"""
Medical Research Scraper for Deep Research Tool
Advanced scraper for medical databases, peptide research, and biochemical sources

Author: Advanced IT Specialist
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin, quote, urlparse
from fake_useragent import UserAgent
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class MedicalResearchScraper:
    """Advanced scraper for medical databases and peptide research"""

    def __init__(self, rate_limit: float = 0.8):
        """
        Initialize Medical Research Scraper

        Args:
            rate_limit: Conservative rate limit for medical databases
        """
        self.rate_limit = rate_limit
        self.ua = UserAgent()

        # Medical and biochemical databases
        self.medical_sources = {
            'pubmed': {
                'name': 'PubMed/MEDLINE',
                'base_url': 'https://pubmed.ncbi.nlm.nih.gov',
                'api_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'search_endpoint': '/esearch.fcgi?db=pubmed&term={query}&retmax=50&retmode=json',
                'fetch_endpoint': '/efetch.fcgi?db=pubmed&id={ids}&retmode=xml',
                'type': 'medical_database',
                'priority': 'high'
            },
            'pmc': {
                'name': 'PMC (PubMed Central)',
                'base_url': 'https://www.ncbi.nlm.nih.gov/pmc',
                'api_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'search_endpoint': '/esearch.fcgi?db=pmc&term={query}&retmax=30&retmode=json',
                'type': 'open_access_medical',
                'priority': 'high'
            },
            'peptideatlas': {
                'name': 'PeptideAtlas',
                'base_url': 'https://peptideatlas.org',
                'search_endpoint': '/search/peptide?query={query}',
                'type': 'peptide_database',
                'priority': 'very_high'
            },
            'uniprot': {
                'name': 'UniProt',
                'base_url': 'https://www.uniprot.org',
                'api_url': 'https://rest.uniprot.org',
                'search_endpoint': '/uniprotkb/search?query={query}&format=json&size=50',
                'type': 'protein_database',
                'priority': 'high'
            },
            'chembl': {
                'name': 'ChEMBL',
                'base_url': 'https://www.ebi.ac.uk/chembl',
                'api_url': 'https://www.ebi.ac.uk/chembl/api/data',
                'search_endpoint': '/molecule/search?q={query}&format=json',
                'type': 'chemical_bioactivity',
                'priority': 'high'
            },
            'drugbank': {
                'name': 'DrugBank',
                'base_url': 'https://go.drugbank.com',
                'search_endpoint': '/search?q={query}&type=drugs',
                'type': 'drug_database',
                'priority': 'high'
            },
            'clinicaltrials': {
                'name': 'ClinicalTrials.gov',
                'base_url': 'https://clinicaltrials.gov',
                'api_url': 'https://clinicaltrials.gov/api',
                'search_endpoint': '/query/study_fields?expr={query}&fields=NCTId,BriefTitle,Condition,InterventionName&max_rnk=50&fmt=json',
                'type': 'clinical_trials',
                'priority': 'high'
            },
            'cochrane': {
                'name': 'Cochrane Library',
                'base_url': 'https://www.cochranelibrary.com',
                'search_endpoint': '/search?q={query}&searchBy=1',
                'type': 'systematic_reviews',
                'priority': 'high'
            },
            'bioportal': {
                'name': 'NCBO BioPortal',
                'base_url': 'https://bioportal.bioontology.org',
                'api_url': 'https://data.bioontology.org',
                'search_endpoint': '/search?q={query}&require_exact_match=false',
                'type': 'biomedical_ontology',
                'priority': 'medium'
            },
            'bindingdb': {
                'name': 'BindingDB',
                'base_url': 'https://www.bindingdb.org',
                'search_endpoint': '/bind/chemsearch/marvin/MolStructure.jsp?monomerid={query}',
                'type': 'binding_affinity',
                'priority': 'high'
            },
            'reactome': {
                'name': 'Reactome',
                'base_url': 'https://reactome.org',
                'search_endpoint': '/content/query?q={query}&species=9606&types=Pathway,Reaction,PhysicalEntity',
                'type': 'pathway_database',
                'priority': 'medium'
            },
            'pharmgkb': {
                'name': 'PharmGKB',
                'base_url': 'https://www.pharmgkb.org',
                'search_endpoint': '/search?query={query}',
                'type': 'pharmacogenomics',
                'priority': 'medium'
            },
            'peptide_therapeutics': {
                'name': 'Peptide Therapeutics Foundation',
                'base_url': 'https://peptidetherapeutics.org',
                'search_endpoint': '/search?q={query}',
                'type': 'peptide_therapeutics',
                'priority': 'very_high'
            },
            'therapeutic_peptides_db': {
                'name': 'Therapeutic Peptides Database',
                'base_url': 'https://webs.iiitd.edu.in/raghava/thpdb',
                'search_endpoint': '/search.php?query={query}',
                'type': 'therapeutic_peptides',
                'priority': 'very_high'
            },
            'bioactive_peptides': {
                'name': 'BIOPEP Database',
                'base_url': 'https://biochemia.uwm.edu.pl/biopep-uwm',
                'search_endpoint': '/search?query={query}',
                'type': 'bioactive_peptides',
                'priority': 'very_high'
            }
        }

        # Peptide-specific search patterns
        self.peptide_patterns = {
            'hormonal_effects': [
                r'\b(growth hormone|insulin|glucagon|leptin|ghrelin)\b',
                r'\b(testosterone|estrogen|cortisol|thyroid|adrenaline)\b',
                r'\b(oxytocin|vasopressin|endorphin|enkephalin)\b'
            ],
            'neurotransmitter_effects': [
                r'\b(dopamine|serotonin|norepinephrine|acetylcholine)\b',
                r'\b(GABA|glutamate|histamine|melatonin)\b',
                r'\b(neuropeptide|neurotransmitter|neuromodulator)\b'
            ],
            'absorption_pathways': [
                r'\b(oral|sublingual|intranasal|subcutaneous|intravenous)\b',
                r'\b(bioavailability|absorption|metabolism|half-life)\b',
                r'\b(peptide transport|carrier|receptor|binding)\b'
            ],
            'mechanisms': [
                r'\b(receptor binding|signal transduction|pathway)\b',
                r'\b(agonist|antagonist|modulator|inhibitor)\b',
                r'\b(pharmacokinetics|pharmacodynamics|mechanism)\b'
            ]
        }

    async def search_async(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Search across medical and biochemical databases

        Args:
            topic: Search topic (preferably peptide-related)
            time_range: Optional time range for search

        Returns:
            List of medical research documents
        """
        results = []

        # Enhance query with peptide-specific terms if relevant
        enhanced_queries = self._enhance_peptide_query(topic)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            # Search each medical source
            for source_id, source_config in self.medical_sources.items():
                try:
                    logger.info(f"Searching {source_config['name']}...")

                    for query in enhanced_queries[:2]:  # Limit to 2 enhanced queries per source
                        source_results = await self._search_medical_source(
                            session, source_id, source_config, query, time_range
                        )
                        results.extend(source_results)

                        # Rate limiting
                        await asyncio.sleep(1.0 / self.rate_limit)

                        if len(source_results) >= 20:  # Limit per source
                            break

                except Exception as e:
                    logger.error(f"Error searching {source_config['name']}: {e}")
                    continue

        # Filter and prioritize results
        filtered_results = self._filter_medical_results(results, topic)
        filtered_results.sort(key=lambda x: (
            x.get('priority_score', 0),
            x.get('citation_count', 0),
            x.get('relevance_score', 0)
        ), reverse=True)

        logger.info(f"Medical Research Scraper found {len(filtered_results)} documents")
        return filtered_results[:100]  # Limit results

    def search(self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    def _enhance_peptide_query(self, topic: str) -> List[str]:
        """Enhance query with peptide-specific terms"""
        queries = [topic]

        # Add peptide if not present
        if 'peptide' not in topic.lower():
            queries.append(f"{topic} peptide")

        # Add specific enhancement based on topic
        enhancements = []

        # Check for specific peptide names and add context
        common_peptides = {
            'bpc-157': ['healing', 'regeneration', 'gut health', 'tendon repair'],
            'tb-500': ['thymosin', 'healing', 'muscle repair', 'inflammation'],
            'cjc-1295': ['growth hormone', 'releasing hormone', 'anti-aging'],
            'ipamorelin': ['growth hormone', 'secretagogue', 'ghrp'],
            'mk-677': ['ibutamoren', 'growth hormone', 'secretagogue', 'ghrelin'],
            'pt-141': ['bremelanotide', 'melanocortin', 'sexual function'],
            'melanotan': ['melanocortin', 'tanning', 'alpha-msh'],
            'oxytocin': ['hormone', 'bonding', 'social', 'uterine'],
            'vasopressin': ['antidiuretic', 'hormone', 'water retention'],
            'ghrp': ['growth hormone releasing', 'peptide', 'secretagogue'],
            'sermorelin': ['growth hormone', 'releasing hormone', 'ghrh']
        }

        topic_lower = topic.lower()
        for peptide, contexts in common_peptides.items():
            if peptide in topic_lower:
                for context in contexts:
                    enhancements.append(f"{topic} {context}")
                break

        # Add mechanism-focused queries
        if any(term in topic_lower for term in ['peptide', 'hormone', 'growth']):
            enhancements.extend([
                f"{topic} mechanism of action",
                f"{topic} pharmacokinetics",
                f"{topic} bioavailability",
                f"{topic} side effects",
                f"{topic} clinical trial"
            ])

        queries.extend(enhancements[:3])  # Limit enhancements
        return queries[:5]  # Total limit

    async def _search_medical_source(self, session: aiohttp.ClientSession, source_id: str,
                                   source_config: Dict[str, Any], query: str,
                                   time_range: Optional[Tuple[datetime, datetime]]) -> List[Dict[str, Any]]:
        """Search individual medical source"""
        results = []

        try:
            if source_id == 'pubmed':
                results = await self._search_pubmed(session, source_config, query, time_range)
            elif source_id == 'uniprot':
                results = await self._search_uniprot(session, source_config, query)
            elif source_id == 'chembl':
                results = await self._search_chembl(session, source_config, query)
            elif source_id == 'clinicaltrials':
                results = await self._search_clinicaltrials(session, source_config, query)
            else:
                results = await self._search_generic_medical(session, source_config, query)

            # Add source metadata
            for result in results:
                result['medical_source'] = source_config['name']
                result['source_type'] = source_config['type']
                result['source_priority'] = source_config['priority']
                result['source'] = 'medical_research'

        except Exception as e:
            logger.debug(f"Failed to search {source_id}: {e}")

        return results

    async def _search_pubmed(self, session: aiohttp.ClientSession, source_config: Dict[str, Any],
                           query: str, time_range: Optional[Tuple[datetime, datetime]]) -> List[Dict[str, Any]]:
        """Search PubMed using E-utilities API"""
        results = []

        try:
            # Step 1: Search for article IDs
            search_url = source_config['api_url'] + source_config['search_endpoint'].format(
                query=quote(query)
            )

            # Add date filter if provided
            if time_range:
                start_date = time_range[0].strftime('%Y/%m/%d')
                end_date = time_range[1].strftime('%Y/%m/%d')
                search_url += f"&datetype=pdat&mindate={start_date}&maxdate={end_date}"

            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    ids = data.get('esearchresult', {}).get('idlist', [])

                    if ids:
                        # Step 2: Fetch article details
                        fetch_url = source_config['api_url'] + source_config['fetch_endpoint'].format(
                            ids=','.join(ids[:20])  # Limit to 20 articles
                        )

                        await asyncio.sleep(0.5)  # NCBI rate limiting

                        async with session.get(fetch_url) as fetch_response:
                            if fetch_response.status == 200:
                                xml_content = await fetch_response.text()
                                results = self._parse_pubmed_xml(xml_content, query)

        except Exception as e:
            logger.debug(f"PubMed search failed: {e}")

        return results

    def _parse_pubmed_xml(self, xml_content: str, query: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        results = []

        try:
            root = ET.fromstring(xml_content)

            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else 'Untitled'

                    # Extract abstract
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ''

                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")

                    # Extract journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ''

                    # Extract publication date
                    pub_date = self._extract_pubmed_date(article)

                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''

                    # Calculate relevance
                    content = f"{title} {abstract}"
                    relevance = self._calculate_medical_relevance(content, query)

                    if relevance > 0.2:
                        results.append({
                            'title': title,
                            'content': abstract or title,
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else '',
                            'date': pub_date,
                            'relevance_score': relevance,
                            'metadata': {
                                'pmid': pmid,
                                'authors': authors,
                                'journal': journal,
                                'abstract_length': len(abstract),
                                'content_type': 'research_article',
                                'database': 'PubMed'
                            }
                        })

                except Exception as e:
                    continue

        except ET.ParseError as e:
            logger.debug(f"XML parsing error: {e}")

        return results

    def _extract_pubmed_date(self, article) -> datetime:
        """Extract publication date from PubMed article"""
        try:
            # Try PubDate first
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                month_elem = pub_date.find('Month')
                day_elem = pub_date.find('Day')

                year = int(year_elem.text) if year_elem is not None else datetime.now().year
                month = self._parse_month(month_elem.text) if month_elem is not None else 1
                day = int(day_elem.text) if day_elem is not None else 1

                return datetime(year, month, day)
        except:
            pass

        return datetime.now()

    def _parse_month(self, month_str: str) -> int:
        """Parse month from various formats"""
        if month_str.isdigit():
            return int(month_str)

        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        return month_map.get(month_str.lower()[:3], 1)

    async def _search_uniprot(self, session: aiohttp.ClientSession, source_config: Dict[str, Any],
                            query: str) -> List[Dict[str, Any]]:
        """Search UniProt protein database"""
        results = []

        try:
            search_url = source_config['api_url'] + source_config['search_endpoint'].format(
                query=quote(query)
            )

            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'application/json'
            }

            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    for protein in data.get('results', []):
                        try:
                            protein_name = protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
                            gene_name = protein.get('genes', [{}])[0].get('geneName', {}).get('value', '') if protein.get('genes') else ''
                            organism = protein.get('organism', {}).get('scientificName', '')
                            uniprot_id = protein.get('primaryAccession', '')

                            # Build content
                            content = f"Protein: {protein_name}"
                            if gene_name:
                                content += f", Gene: {gene_name}"
                            if organism:
                                content += f", Organism: {organism}"

                            relevance = self._calculate_medical_relevance(content, query)

                            if relevance > 0.3:
                                results.append({
                                    'title': f"{protein_name} ({gene_name})" if gene_name else protein_name,
                                    'content': content,
                                    'url': f"https://www.uniprot.org/uniprotkb/{uniprot_id}",
                                    'date': datetime.now(),
                                    'relevance_score': relevance,
                                    'metadata': {
                                        'uniprot_id': uniprot_id,
                                        'gene_name': gene_name,
                                        'organism': organism,
                                        'content_type': 'protein_entry',
                                        'database': 'UniProt'
                                    }
                                })

                        except Exception as e:
                            continue

        except Exception as e:
            logger.debug(f"UniProt search failed: {e}")

        return results

    async def _search_chembl(self, session: aiohttp.ClientSession, source_config: Dict[str, Any],
                           query: str) -> List[Dict[str, Any]]:
        """Search ChEMBL chemical bioactivity database"""
        results = []

        try:
            search_url = source_config['api_url'] + source_config['search_endpoint'].format(
                query=quote(query)
            )

            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'application/json'
            }

            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    for molecule in data.get('molecules', []):
                        try:
                            molecule_name = molecule.get('pref_name', molecule.get('molecule_chembl_id', 'Unknown'))
                            chembl_id = molecule.get('molecule_chembl_id', '')
                            max_phase = molecule.get('max_phase', 0)
                            molecular_weight = molecule.get('molecular_weight', '')

                            content = f"Compound: {molecule_name}, ChEMBL ID: {chembl_id}"
                            if max_phase:
                                content += f", Development Phase: {max_phase}"
                            if molecular_weight:
                                content += f", MW: {molecular_weight}"

                            relevance = self._calculate_medical_relevance(content, query)

                            if relevance > 0.25:
                                results.append({
                                    'title': molecule_name,
                                    'content': content,
                                    'url': f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}",
                                    'date': datetime.now(),
                                    'relevance_score': relevance,
                                    'metadata': {
                                        'chembl_id': chembl_id,
                                        'max_phase': max_phase,
                                        'molecular_weight': molecular_weight,
                                        'content_type': 'chemical_compound',
                                        'database': 'ChEMBL'
                                    }
                                })

                        except Exception as e:
                            continue

        except Exception as e:
            logger.debug(f"ChEMBL search failed: {e}")

        return results

    async def _search_clinicaltrials(self, session: aiohttp.ClientSession, source_config: Dict[str, Any],
                                   query: str) -> List[Dict[str, Any]]:
        """Search ClinicalTrials.gov"""
        results = []

        try:
            search_url = source_config['api_url'] + source_config['search_endpoint'].format(
                query=quote(query)
            )

            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()

                    studies = data.get('StudyFieldsResponse', {}).get('StudyFields', [])

                    for study in studies:
                        try:
                            nct_id = study.get('NCTId', [''])[0]
                            title = study.get('BriefTitle', ['Untitled'])[0]
                            conditions = study.get('Condition', [])
                            interventions = study.get('InterventionName', [])

                            content = f"Clinical Trial: {title}"
                            if conditions:
                                content += f", Conditions: {', '.join(conditions[:3])}"
                            if interventions:
                                content += f", Interventions: {', '.join(interventions[:3])}"

                            relevance = self._calculate_medical_relevance(content, query)

                            if relevance > 0.3:
                                results.append({
                                    'title': title,
                                    'content': content,
                                    'url': f"https://clinicaltrials.gov/study/{nct_id}",
                                    'date': datetime.now(),
                                    'relevance_score': relevance,
                                    'metadata': {
                                        'nct_id': nct_id,
                                        'conditions': conditions,
                                        'interventions': interventions,
                                        'content_type': 'clinical_trial',
                                        'database': 'ClinicalTrials.gov'
                                    }
                                })

                        except Exception as e:
                            continue

        except Exception as e:
            logger.debug(f"ClinicalTrials search failed: {e}")

        return results

    async def _search_generic_medical(self, session: aiohttp.ClientSession, source_config: Dict[str, Any],
                                    query: str) -> List[Dict[str, Any]]:
        """Generic search for other medical databases"""
        results = []

        try:
            search_url = source_config['base_url'] + source_config['search_endpoint'].format(
                query=quote(query)
            )

            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Look for relevant content based on source type
                    source_type = source_config['type']

                    if 'peptide' in source_type:
                        results = self._parse_peptide_database(soup, source_config, query)
                    elif 'pathway' in source_type:
                        results = self._parse_pathway_database(soup, source_config, query)
                    else:
                        results = self._parse_generic_medical_results(soup, source_config, query)

        except Exception as e:
            logger.debug(f"Generic medical search failed for {source_config['name']}: {e}")

        return results

    def _parse_peptide_database(self, soup: BeautifulSoup, source_config: Dict[str, Any],
                              query: str) -> List[Dict[str, Any]]:
        """Parse peptide-specific database results"""
        results = []

        # Look for peptide entries
        peptide_elements = soup.find_all(['div', 'tr', 'article'], limit=20)

        for element in peptide_elements:
            try:
                text = element.get_text(strip=True)

                # Filter for peptide-relevant content
                if (len(text) > 20 and
                    any(term in text.lower() for term in ['peptide', 'amino acid', 'sequence', 'bioactive'])):

                    relevance = self._calculate_medical_relevance(text, query)

                    if relevance > 0.3:
                        # Try to find associated link
                        link = source_config['base_url']
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            href = link_elem.get('href')
                            if href:
                                link = urljoin(source_config['base_url'], href)

                        title = text[:100] + "..." if len(text) > 100 else text

                        results.append({
                            'title': title,
                            'content': text,
                            'url': link,
                            'date': datetime.now(),
                            'relevance_score': relevance,
                            'metadata': {
                                'content_type': 'peptide_entry',
                                'database': source_config['name']
                            }
                        })

            except Exception as e:
                continue

        return results

    def _parse_pathway_database(self, soup: BeautifulSoup, source_config: Dict[str, Any],
                              query: str) -> List[Dict[str, Any]]:
        """Parse pathway database results"""
        results = []

        # Look for pathway entries
        pathway_elements = soup.find_all(['div', 'section', 'article'], limit=15)

        for element in pathway_elements:
            try:
                title_elem = element.find(['h1', 'h2', 'h3', 'h4'])
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                content = element.get_text(strip=True)

                if (title and len(title) > 5 and
                    any(term in content.lower() for term in ['pathway', 'reaction', 'enzyme', 'protein'])):

                    relevance = self._calculate_medical_relevance(title + " " + content, query)

                    if relevance > 0.25:
                        link = source_config['base_url']
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            href = link_elem.get('href')
                            if href:
                                link = urljoin(source_config['base_url'], href)

                        results.append({
                            'title': title,
                            'content': content[:500],
                            'url': link,
                            'date': datetime.now(),
                            'relevance_score': relevance,
                            'metadata': {
                                'content_type': 'pathway_entry',
                                'database': source_config['name']
                            }
                        })

            except Exception as e:
                continue

        return results

    def _parse_generic_medical_results(self, soup: BeautifulSoup, source_config: Dict[str, Any],
                                     query: str) -> List[Dict[str, Any]]:
        """Parse generic medical database results"""
        results = []

        # Look for research entries
        content_elements = soup.find_all(['article', 'div', 'section'], limit=20)

        for element in content_elements:
            try:
                text = element.get_text(strip=True)

                if len(text) > 30:
                    relevance = self._calculate_medical_relevance(text, query)

                    if relevance > 0.2:
                        link = source_config['base_url']
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            href = link_elem.get('href')
                            if href:
                                link = urljoin(source_config['base_url'], href)

                        title = text[:80] + "..." if len(text) > 80 else text

                        results.append({
                            'title': title,
                            'content': text[:400],
                            'url': link,
                            'date': datetime.now(),
                            'relevance_score': relevance,
                            'metadata': {
                                'content_type': 'medical_entry',
                                'database': source_config['name']
                            }
                        })

            except Exception as e:
                continue

        return results[:10]  # Limit generic results

    def _calculate_medical_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score for medical content"""
        if not text or not query:
            return 0.0

        text_lower = text.lower()
        query_lower = query.lower()

        score = 0.0

        # Exact query match
        if query_lower in text_lower:
            score += 0.5

        # Individual word matches
        query_words = query_lower.split()
        text_words = text_lower.split()

        word_matches = sum(1 for word in query_words if word in text_words)
        if word_matches > 0:
            score += (word_matches / len(query_words)) * 0.3

        # Medical/peptide relevance boost
        medical_terms = [
            'peptide', 'hormone', 'protein', 'amino acid', 'receptor',
            'clinical', 'therapeutic', 'bioactive', 'pharmacokinetics',
            'mechanism', 'pathway', 'binding', 'absorption', 'metabolism'
        ]

        medical_matches = sum(1 for term in medical_terms if term in text_lower)
        if medical_matches > 0:
            score += min(0.3, medical_matches * 0.05)

        # Check for peptide patterns
        for pattern_type, patterns in self.peptide_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.1
                    break

        return min(1.0, score)

    def _filter_medical_results(self, results: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Filter and enhance medical results"""
        filtered = []

        for result in results:
            # Skip very short content
            content = result.get('content', '')
            if len(content) < 20:
                continue

            # Require minimum relevance
            if result.get('relevance_score', 0) < 0.15:
                continue

            # Add priority score based on source
            priority_map = {'very_high': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4}
            priority = result.get('source_priority', 'medium')
            result['priority_score'] = priority_map.get(priority, 0.5)

            # Estimate citation count (placeholder - would need real API for this)
            result['citation_count'] = 0

            # Add medical research specific metadata
            result['source'] = 'medical_research'
            if 'source_type' not in result:
                result['source_type'] = 'medical_document'

            filtered.append(result)

        return filtered

    async def search_specific_database(self, database_name: str, topic: str) -> List[Dict[str, Any]]:
        """Search specific medical database by name"""
        database_id = None
        for db_id, config in self.medical_sources.items():
            if config['name'].lower() == database_name.lower():
                database_id = db_id
                break

        if not database_id:
            logger.error(f"Unknown medical database: {database_name}")
            return []

        source_config = self.medical_sources[database_id]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            results = await self._search_medical_source(
                session, database_id, source_config, topic, None
            )

        return self._filter_medical_results(results, topic)

    def get_supported_databases(self) -> List[Dict[str, str]]:
        """Get list of supported medical databases"""
        return [
            {
                'id': db_id,
                'name': config['name'],
                'type': config['type'],
                'priority': config['priority'],
                'description': f"{config['type'].replace('_', ' ').title()} database"
            }
            for db_id, config in self.medical_sources.items()
        ]
