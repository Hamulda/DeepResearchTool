#!/usr/bin/env python3
"""
Advanced Cross-Source Correlation Engine
Multi-dimensional correlation analysis across specialized sources

Author: Advanced IT Specialist
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CorrelationDimension:
    """Represents a dimension of correlation analysis"""
    dimension_name: str
    weight: float
    correlation_method: str
    threshold: float
    confidence_modifier: float = 1.0

@dataclass
class MultiDimensionalCorrelation:
    """Advanced correlation result with multiple dimensions"""
    correlation_id: str
    source_combinations: List[Tuple[str, str]]
    correlation_dimensions: Dict[str, float]
    overall_strength: float
    confidence_score: float
    evidence_chains: List[Dict[str, Any]]
    temporal_coherence: float
    geographical_coherence: float
    semantic_coherence: float
    entity_overlap_score: float
    methodological_alignment: float
    validation_status: str

@dataclass
class CorrelationCluster:
    """Cluster of related correlations"""
    cluster_id: str
    correlations: List[MultiDimensionalCorrelation]
    central_theme: str
    cluster_strength: float
    member_sources: Set[str]
    temporal_span: Tuple[datetime, datetime]
    key_entities: List[str]

class AdvancedCorrelationEngine:
    """Multi-dimensional correlation analysis engine"""

    def __init__(self):
        # Correlation dimensions with weights
        self.correlation_dimensions = {
            'temporal': CorrelationDimension(
                dimension_name='temporal',
                weight=0.2,
                correlation_method='time_proximity',
                threshold=0.6,
                confidence_modifier=1.0
            ),
            'semantic': CorrelationDimension(
                dimension_name='semantic',
                weight=0.25,
                correlation_method='tfidf_cosine',
                threshold=0.7,
                confidence_modifier=1.2
            ),
            'entity_based': CorrelationDimension(
                dimension_name='entity_based',
                weight=0.2,
                correlation_method='entity_overlap',
                threshold=0.5,
                confidence_modifier=1.1
            ),
            'geographical': CorrelationDimension(
                dimension_name='geographical',
                weight=0.15,
                correlation_method='location_proximity',
                threshold=0.6,
                confidence_modifier=0.9
            ),
            'methodological': CorrelationDimension(
                dimension_name='methodological',
                weight=0.1,
                correlation_method='method_similarity',
                threshold=0.5,
                confidence_modifier=0.8
            ),
            'citation_network': CorrelationDimension(
                dimension_name='citation_network',
                weight=0.1,
                correlation_method='citation_analysis',
                threshold=0.4,
                confidence_modifier=1.3
            )
        }

        # Time windows for temporal correlation
        self.time_windows = {
            'immediate': timedelta(days=1),
            'short_term': timedelta(days=7),
            'medium_term': timedelta(days=30),
            'long_term': timedelta(days=365),
            'historical': timedelta(days=365*5)
        }

        # Semantic similarity thresholds
        self.semantic_thresholds = {
            'high_similarity': 0.8,
            'medium_similarity': 0.6,
            'low_similarity': 0.4
        }

    async def perform_multidimensional_correlation(self,
                                                 multi_source_data: Dict[str, List[Any]],
                                                 correlation_types: Optional[List[str]] = None) -> List[MultiDimensionalCorrelation]:
        """Perform comprehensive multi-dimensional correlation analysis"""

        logger.info("Starting multi-dimensional correlation analysis...")

        if not correlation_types:
            correlation_types = list(self.correlation_dimensions.keys())

        correlations = []

        # Prepare data for analysis
        prepared_data = await self._prepare_correlation_data(multi_source_data)

        # Generate all source pair combinations
        source_pairs = self._generate_source_combinations(list(multi_source_data.keys()))

        for source_pair in source_pairs:
            source1, source2 = source_pair

            # Get documents from both sources
            docs1 = multi_source_data.get(source1, [])
            docs2 = multi_source_data.get(source2, [])

            if not docs1 or not docs2:
                continue

            # Perform correlation analysis across all dimensions
            correlation = await self._analyze_source_pair_correlation(
                source_pair, docs1, docs2, prepared_data, correlation_types
            )

            if correlation and correlation.overall_strength >= 0.5:
                correlations.append(correlation)

        # Sort correlations by strength
        correlations.sort(key=lambda x: x.overall_strength, reverse=True)

        logger.info(f"Found {len(correlations)} significant multi-dimensional correlations")
        return correlations

    async def _prepare_correlation_data(self, multi_source_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Prepare data structures for correlation analysis"""

        prepared_data = {
            'entity_extraction': {},
            'temporal_mapping': {},
            'text_corpus': {},
            'geographical_mapping': {},
            'methodology_mapping': {},
            'citation_networks': {}
        }

        # Extract entities and metadata from all sources
        for source_type, documents in multi_source_data.items():

            # Entity extraction
            all_entities = set()
            temporal_events = []
            text_corpus = []
            geographical_locations = set()
            methodologies = set()

            for doc in documents:
                # Extract entities
                doc_entities = self._extract_document_entities(doc)
                all_entities.update(doc_entities)

                # Extract temporal information
                temporal_info = self._extract_temporal_info(doc)
                if temporal_info:
                    temporal_events.append(temporal_info)

                # Extract text for semantic analysis
                doc_text = self._extract_document_text(doc)
                if doc_text:
                    text_corpus.append(doc_text)

                # Extract geographical information
                geo_info = self._extract_geographical_info(doc)
                geographical_locations.update(geo_info)

                # Extract methodology information
                methodology = self._extract_methodology_info(doc)
                if methodology:
                    methodologies.add(methodology)

            # Store prepared data
            prepared_data['entity_extraction'][source_type] = all_entities
            prepared_data['temporal_mapping'][source_type] = temporal_events
            prepared_data['text_corpus'][source_type] = text_corpus
            prepared_data['geographical_mapping'][source_type] = geographical_locations
            prepared_data['methodology_mapping'][source_type] = methodologies

        return prepared_data

    def _generate_source_combinations(self, sources: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible source pair combinations"""
        combinations = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                combinations.append((sources[i], sources[j]))
        return combinations

    async def _analyze_source_pair_correlation(self,
                                             source_pair: Tuple[str, str],
                                             docs1: List[Any],
                                             docs2: List[Any],
                                             prepared_data: Dict[str, Any],
                                             correlation_types: List[str]) -> Optional[MultiDimensionalCorrelation]:
        """Analyze correlation between a pair of sources"""

        source1, source2 = source_pair
        correlation_scores = {}
        evidence_chains = []

        # Temporal correlation
        if 'temporal' in correlation_types:
            temporal_score = self._calculate_temporal_correlation(
                prepared_data['temporal_mapping'].get(source1, []),
                prepared_data['temporal_mapping'].get(source2, [])
            )
            correlation_scores['temporal'] = temporal_score

            if temporal_score > self.correlation_dimensions['temporal'].threshold:
                evidence_chains.append({
                    'type': 'temporal',
                    'strength': temporal_score,
                    'evidence': 'Significant temporal overlap in document publication dates'
                })

        # Semantic correlation
        if 'semantic' in correlation_types:
            semantic_score = await self._calculate_semantic_correlation(
                prepared_data['text_corpus'].get(source1, []),
                prepared_data['text_corpus'].get(source2, [])
            )
            correlation_scores['semantic'] = semantic_score

            if semantic_score > self.correlation_dimensions['semantic'].threshold:
                evidence_chains.append({
                    'type': 'semantic',
                    'strength': semantic_score,
                    'evidence': 'High semantic similarity in document content'
                })

        # Entity-based correlation
        if 'entity_based' in correlation_types:
            entity_score = self._calculate_entity_correlation(
                prepared_data['entity_extraction'].get(source1, set()),
                prepared_data['entity_extraction'].get(source2, set())
            )
            correlation_scores['entity_based'] = entity_score

            if entity_score > self.correlation_dimensions['entity_based'].threshold:
                evidence_chains.append({
                    'type': 'entity_based',
                    'strength': entity_score,
                    'evidence': 'Significant overlap in named entities'
                })

        # Geographical correlation
        if 'geographical' in correlation_types:
            geo_score = self._calculate_geographical_correlation(
                prepared_data['geographical_mapping'].get(source1, set()),
                prepared_data['geographical_mapping'].get(source2, set())
            )
            correlation_scores['geographical'] = geo_score

            if geo_score > self.correlation_dimensions['geographical'].threshold:
                evidence_chains.append({
                    'type': 'geographical',
                    'strength': geo_score,
                    'evidence': 'Geographic overlap in document coverage'
                })

        # Methodological correlation
        if 'methodological' in correlation_types:
            method_score = self._calculate_methodological_correlation(
                prepared_data['methodology_mapping'].get(source1, set()),
                prepared_data['methodology_mapping'].get(source2, set())
            )
            correlation_scores['methodological'] = method_score

            if method_score > self.correlation_dimensions['methodological'].threshold:
                evidence_chains.append({
                    'type': 'methodological',
                    'strength': method_score,
                    'evidence': 'Similar research methodologies employed'
                })

        # Calculate overall correlation strength
        overall_strength = self._calculate_overall_correlation_strength(correlation_scores)

        if overall_strength < 0.5:  # Minimum threshold for significance
            return None

        # Calculate confidence score
        confidence_score = self._calculate_correlation_confidence(correlation_scores, evidence_chains)

        # Create correlation object
        correlation_id = hashlib.md5(f"{source1}_{source2}_{overall_strength}".encode()).hexdigest()[:12]

        correlation = MultiDimensionalCorrelation(
            correlation_id=correlation_id,
            source_combinations=[source_pair],
            correlation_dimensions=correlation_scores,
            overall_strength=overall_strength,
            confidence_score=confidence_score,
            evidence_chains=evidence_chains,
            temporal_coherence=correlation_scores.get('temporal', 0.0),
            geographical_coherence=correlation_scores.get('geographical', 0.0),
            semantic_coherence=correlation_scores.get('semantic', 0.0),
            entity_overlap_score=correlation_scores.get('entity_based', 0.0),
            methodological_alignment=correlation_scores.get('methodological', 0.0),
            validation_status='pending'
        )

        return correlation

    def _calculate_temporal_correlation(self, events1: List[Dict], events2: List[Dict]) -> float:
        """Calculate temporal correlation between two event sets"""
        if not events1 or not events2:
            return 0.0

        correlation_count = 0
        total_comparisons = 0

        for event1 in events1:
            date1 = event1.get('date')
            if not date1:
                continue

            for event2 in events2:
                date2 = event2.get('date')
                if not date2:
                    continue

                total_comparisons += 1

                # Check proximity within different time windows
                time_diff = abs((date1 - date2).total_seconds())

                for window_name, window_duration in self.time_windows.items():
                    if time_diff <= window_duration.total_seconds():
                        correlation_count += 1
                        break

        return correlation_count / total_comparisons if total_comparisons > 0 else 0.0

    async def _calculate_semantic_correlation(self, corpus1: List[str], corpus2: List[str]) -> float:
        """Calculate semantic correlation using TF-IDF and cosine similarity"""
        if not corpus1 or not corpus2:
            return 0.0

        try:
            # Combine corpora for TF-IDF fitting
            combined_corpus = corpus1 + corpus2

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(combined_corpus)

            # Split back into two groups
            matrix1 = tfidf_matrix[:len(corpus1)]
            matrix2 = tfidf_matrix[len(corpus1):]

            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(matrix1, matrix2)

            # Return average similarity
            return float(np.mean(similarities))

        except Exception as e:
            logger.error(f"Error calculating semantic correlation: {str(e)}")
            return 0.0

    def _calculate_entity_correlation(self, entities1: Set[str], entities2: Set[str]) -> float:
        """Calculate entity-based correlation"""
        if not entities1 or not entities2:
            return 0.0

        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)

        # Jaccard similarity
        jaccard_score = len(intersection) / len(union) if union else 0.0

        # Weighted by entity importance (simplified)
        overlap_score = len(intersection) / min(len(entities1), len(entities2))

        return (jaccard_score + overlap_score) / 2.0

    def _calculate_geographical_correlation(self, locations1: Set[str], locations2: Set[str]) -> float:
        """Calculate geographical correlation"""
        if not locations1 or not locations2:
            return 0.0

        intersection = locations1.intersection(locations2)
        union = locations1.union(locations2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_methodological_correlation(self, methods1: Set[str], methods2: Set[str]) -> float:
        """Calculate methodological correlation"""
        if not methods1 or not methods2:
            return 0.0

        intersection = methods1.intersection(methods2)
        union = methods1.union(methods2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_overall_correlation_strength(self, correlation_scores: Dict[str, float]) -> float:
        """Calculate weighted overall correlation strength"""
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension_name, score in correlation_scores.items():
            if dimension_name in self.correlation_dimensions:
                dimension = self.correlation_dimensions[dimension_name]
                weight = dimension.weight
                confidence_modifier = dimension.confidence_modifier

                weighted_sum += score * weight * confidence_modifier
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_correlation_confidence(self, correlation_scores: Dict[str, float], evidence_chains: List[Dict]) -> float:
        """Calculate confidence score for correlation"""
        base_confidence = 0.5

        # Factor in number of correlation dimensions
        dimension_bonus = len(correlation_scores) * 0.1

        # Factor in evidence strength
        evidence_bonus = len(evidence_chains) * 0.05

        # Factor in high-scoring dimensions
        high_score_bonus = sum(0.1 for score in correlation_scores.values() if score > 0.8)

        confidence = base_confidence + dimension_bonus + evidence_bonus + high_score_bonus
        return min(1.0, confidence)

    # Helper methods for data extraction
    def _extract_document_entities(self, document: Any) -> Set[str]:
        """Extract entities from document"""
        entities = set()

        # Try different entity fields
        entity_fields = ['entities_mentioned', 'entities', 'named_entities']
        for field in entity_fields:
            if hasattr(document, field):
                field_entities = getattr(document, field, [])
                if isinstance(field_entities, list):
                    entities.update(field_entities)

        # Extract from text using patterns if no pre-extracted entities
        if not entities:
            text = self._extract_document_text(document)
            if text:
                entities.update(self._extract_entities_from_text(text))

        return entities

    def _extract_temporal_info(self, document: Any) -> Optional[Dict[str, Any]]:
        """Extract temporal information from document"""
        date_fields = ['publication_date', 'date_created', 'original_date', 'date']

        for field in date_fields:
            if hasattr(document, field):
                date_value = getattr(document, field)
                if date_value:
                    return {
                        'date': date_value,
                        'field': field,
                        'document_id': getattr(document, 'document_id', 'unknown')
                    }

        return None

    def _extract_document_text(self, document: Any) -> str:
        """Extract text content from document"""
        text_fields = ['content', 'abstract', 'description', 'title']

        text_parts = []
        for field in text_fields:
            if hasattr(document, field):
                field_text = getattr(document, field, '')
                if field_text:
                    text_parts.append(str(field_text))

        return ' '.join(text_parts)

    def _extract_geographical_info(self, document: Any) -> Set[str]:
        """Extract geographical information from document"""
        locations = set()

        geo_fields = ['geographical_coverage', 'location', 'countries', 'regions']
        for field in geo_fields:
            if hasattr(document, field):
                field_locations = getattr(document, field, [])
                if isinstance(field_locations, list):
                    locations.update(field_locations)
                elif isinstance(field_locations, str):
                    locations.add(field_locations)

        return locations

    def _extract_methodology_info(self, document: Any) -> Optional[str]:
        """Extract methodology information from document"""
        method_fields = ['methodology', 'research_method', 'approach']

        for field in method_fields:
            if hasattr(document, field):
                method = getattr(document, field)
                if method:
                    return str(method)

        return None

    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract entities from text using basic patterns"""
        entities = set()

        # Basic entity patterns
        patterns = {
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'organization': r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',
            'location': r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b'
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:  # Filter very short matches
                    entities.add(f"{match}:{entity_type}")

        return entities

    async def cluster_correlations(self, correlations: List[MultiDimensionalCorrelation]) -> List[CorrelationCluster]:
        """Cluster related correlations"""

        if not correlations:
            return []

        logger.info("Clustering correlations...")

        # Build correlation network
        correlation_graph = nx.Graph()

        for correlation in correlations:
            correlation_graph.add_node(correlation.correlation_id, correlation=correlation)

        # Add edges between similar correlations
        for i, corr1 in enumerate(correlations):
            for j, corr2 in enumerate(correlations[i+1:], i+1):
                similarity = self._calculate_correlation_similarity(corr1, corr2)
                if similarity > 0.6:  # Similarity threshold
                    correlation_graph.add_edge(corr1.correlation_id, corr2.correlation_id, weight=similarity)

        # Find communities/clusters
        try:
            communities = nx.community.greedy_modularity_communities(correlation_graph)
        except:
            # Fallback to simple clustering if networkx community detection fails
            communities = self._simple_clustering(correlations)

        # Create correlation clusters
        clusters = []
        for i, community in enumerate(communities):
            if len(community) >= 2:  # Minimum cluster size
                cluster_correlations = [
                    correlation_graph.nodes[node_id]['correlation']
                    for node_id in community
                ]

                cluster = self._create_correlation_cluster(f"cluster_{i}", cluster_correlations)
                clusters.append(cluster)

        logger.info(f"Created {len(clusters)} correlation clusters")
        return clusters

    def _calculate_correlation_similarity(self, corr1: MultiDimensionalCorrelation, corr2: MultiDimensionalCorrelation) -> float:
        """Calculate similarity between two correlations"""

        # Compare correlation dimensions
        dimension_similarity = 0.0
        common_dimensions = 0

        for dim in corr1.correlation_dimensions:
            if dim in corr2.correlation_dimensions:
                score1 = corr1.correlation_dimensions[dim]
                score2 = corr2.correlation_dimensions[dim]
                dimension_similarity += 1.0 - abs(score1 - score2)
                common_dimensions += 1

        if common_dimensions == 0:
            return 0.0

        dimension_similarity /= common_dimensions

        # Compare source overlaps
        sources1 = set(sum(corr1.source_combinations, ()))
        sources2 = set(sum(corr2.source_combinations, ()))
        source_overlap = len(sources1.intersection(sources2)) / len(sources1.union(sources2))

        # Overall similarity
        return (dimension_similarity + source_overlap) / 2.0

    def _simple_clustering(self, correlations: List[MultiDimensionalCorrelation]) -> List[List[str]]:
        """Simple clustering fallback method"""
        clusters = []
        used_correlations = set()

        for i, corr1 in enumerate(correlations):
            if corr1.correlation_id in used_correlations:
                continue

            cluster = [corr1.correlation_id]
            used_correlations.add(corr1.correlation_id)

            for j, corr2 in enumerate(correlations[i+1:], i+1):
                if corr2.correlation_id in used_correlations:
                    continue

                similarity = self._calculate_correlation_similarity(corr1, corr2)
                if similarity > 0.6:
                    cluster.append(corr2.correlation_id)
                    used_correlations.add(corr2.correlation_id)

            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    def _create_correlation_cluster(self, cluster_id: str, correlations: List[MultiDimensionalCorrelation]) -> CorrelationCluster:
        """Create a correlation cluster"""

        # Calculate cluster strength (average of member correlations)
        cluster_strength = sum(c.overall_strength for c in correlations) / len(correlations)

        # Collect all member sources
        member_sources = set()
        for correlation in correlations:
            for source_pair in correlation.source_combinations:
                member_sources.update(source_pair)

        # Determine central theme (most common evidence type)
        evidence_types = []
        for correlation in correlations:
            for evidence in correlation.evidence_chains:
                evidence_types.append(evidence['type'])

        if evidence_types:
            central_theme = Counter(evidence_types).most_common(1)[0][0]
        else:
            central_theme = "unknown"

        # Calculate temporal span (if available)
        temporal_span = (datetime.min, datetime.max)  # Default span

        # Collect key entities (placeholder implementation)
        key_entities = []

        return CorrelationCluster(
            cluster_id=cluster_id,
            correlations=correlations,
            central_theme=central_theme,
            cluster_strength=cluster_strength,
            member_sources=member_sources,
            temporal_span=temporal_span,
            key_entities=key_entities
        )

    def analyze_correlation_patterns(self, correlations: List[MultiDimensionalCorrelation]) -> Dict[str, Any]:
        """Analyze patterns in correlation results"""

        analysis = {
            'total_correlations': len(correlations),
            'strength_distribution': {},
            'dimension_analysis': {},
            'source_pair_frequency': {},
            'confidence_analysis': {},
            'evidence_type_frequency': {},
            'temporal_patterns': {},
            'recommendations': []
        }

        if not correlations:
            return analysis

        # Strength distribution
        strength_ranges = {'weak (0.5-0.6)': 0, 'moderate (0.6-0.8)': 0, 'strong (0.8-1.0)': 0}
        for corr in correlations:
            strength = corr.overall_strength
            if 0.5 <= strength < 0.6:
                strength_ranges['weak (0.5-0.6)'] += 1
            elif 0.6 <= strength < 0.8:
                strength_ranges['moderate (0.6-0.8)'] += 1
            else:
                strength_ranges['strong (0.8-1.0)'] += 1

        analysis['strength_distribution'] = strength_ranges

        # Dimension analysis
        dimension_scores = defaultdict(list)
        for corr in correlations:
            for dim, score in corr.correlation_dimensions.items():
                dimension_scores[dim].append(score)

        analysis['dimension_analysis'] = {
            dim: {
                'average_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'active_correlations': len([s for s in scores if s > 0.5])
            }
            for dim, scores in dimension_scores.items()
        }

        # Source pair frequency
        source_pairs = []
        for corr in correlations:
            source_pairs.extend(corr.source_combinations)

        pair_counter = Counter(source_pairs)
        analysis['source_pair_frequency'] = dict(pair_counter.most_common(5))

        # Evidence type frequency
        evidence_types = []
        for corr in correlations:
            for evidence in corr.evidence_chains:
                evidence_types.append(evidence['type'])

        analysis['evidence_type_frequency'] = dict(Counter(evidence_types))

        # Generate recommendations
        analysis['recommendations'] = self._generate_correlation_recommendations(analysis)

        return analysis

    def _generate_correlation_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []

        # Strength recommendations
        strong_correlations = analysis['strength_distribution'].get('strong (0.8-1.0)', 0)
        if strong_correlations > 5:
            recommendations.append(f"Strong correlation network detected ({strong_correlations} strong links) - investigate for causal relationships")

        # Dimension recommendations
        dimension_analysis = analysis['dimension_analysis']

        if 'semantic' in dimension_analysis and dimension_analysis['semantic']['average_score'] > 0.7:
            recommendations.append("High semantic correlation - documents share similar themes and concepts")

        if 'temporal' in dimension_analysis and dimension_analysis['temporal']['average_score'] > 0.6:
            recommendations.append("Significant temporal patterns - events may be causally related")

        if 'entity_based' in dimension_analysis and dimension_analysis['entity_based']['average_score'] > 0.6:
            recommendations.append("Strong entity overlap - focus on shared actors and organizations")

        # Source pair recommendations
        frequent_pairs = analysis['source_pair_frequency']
        if len(frequent_pairs) > 0:
            most_frequent = list(frequent_pairs.keys())[0]
            recommendations.append(f"Strongest correlation between {most_frequent[0]} and {most_frequent[1]} sources")

        return recommendations
