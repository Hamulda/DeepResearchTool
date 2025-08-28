#!/usr/bin/env python3
        date_patterns = [
            r'(?:on|at|during)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})'
        ]

        # Event extraction around dates
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                start_pos = match.start()

                # Extract context around the date
                context_start = max(0, start_pos - 200)
                context_end = min(len(content), start_pos + 200)
                context = content[context_start:context_end]

                # Try to parse the date
                try:
                    parsed_date = self._parse_flexible_date(date_str)
                    if parsed_date:
                        # Extract event description
                        event_desc = self._extract_event_description(context, date_str)

                        events.append(TimelineEvent(
                            event_id=f"{document.get('document_id', 'unknown')}_{len(events)}",
                            timestamp=parsed_date,
                            description=event_desc,
                            sources=[document.get('source_url', '')],
                            confidence=0.7,  # Base confidence
                            event_type=self._classify_event_type(event_desc),
                            entities_involved=self._extract_entities_from_context(context),
                            location=self._extract_location_from_context(context),
                            significance_score=self._calculate_event_significance(event_desc)
                        ))
                except:
                    continue

        return events

    async def _correlate_sources(self, source1: Dict[str, Any], source2: Dict[str, Any]) -> CrossSourceCorrelation:
        """Correlate two sources for consistency and overlap"""
        content1 = source1.get('content', '')
        content2 = source2.get('content', '')

        # Semantic similarity using spaCy
        doc1 = self.nlp(content1[:1000000])  # Limit for performance
        doc2 = self.nlp(content2[:1000000])

        semantic_similarity = doc1.similarity(doc2) if doc1.vector_norm and doc2.vector_norm else 0.0

        # Entity overlap analysis
        entities1 = await self._extract_specialized_entities(source1)
        entities2 = await self._extract_specialized_entities(source2)

        matching_entities = []
        for entity_type in entities1.keys():
            if entity_type in entities2:
                matches = set(entities1[entity_type]) & set(entities2[entity_type])
                matching_entities.extend(matches)

        # Temporal alignment
        dates1 = self._extract_all_dates(content1)
        dates2 = self._extract_all_dates(content2)
        temporal_alignment = self._calculate_temporal_overlap(dates1, dates2)

        # Factual consistency check
        factual_consistency = await self._check_factual_consistency(source1, source2)

        # Overall correlation strength
        correlation_strength = (
            semantic_similarity * 0.3 +
            (len(matching_entities) / max(len(set(sum(entities1.values(), []))), 1)) * 0.3 +
            temporal_alignment * 0.2 +
            (1.0 if factual_consistency else 0.0) * 0.2
        )

        return CrossSourceCorrelation(
            correlation_id=f"{source1.get('id', 'unknown')}_{source2.get('id', 'unknown')}",
            sources=[source1.get('source', ''), source2.get('source', '')],
            correlation_strength=correlation_strength,
            matching_entities=matching_entities,
            temporal_alignment=temporal_alignment,
            semantic_similarity=semantic_similarity,
            factual_consistency=factual_consistency,
            conflicting_information=await self._identify_source_conflicts(source1, source2)
        )

    async def _assess_source_credibility(self, source: Dict[str, Any], all_sources: List[Dict[str, Any]]) -> SourceCredibilityAssessment:
        """Assess the credibility of a single source"""
        source_type = source.get('source_type', 'unknown')
        content = source.get('content', '')

        # Base reliability scores by source type
        base_reliability = {
            'government_official': 0.9,
            'academic_paper': 0.8,
            'verified_news': 0.7,
            'declassified_document': 0.9,
            'social_media_verified': 0.5,
            'social_media_unverified': 0.3,
            'anonymous': 0.2,
            'unknown': 0.4
        }

        reliability_score = base_reliability.get(source_type, 0.4)

        # Consistency with other sources
        consistency_scores = []
        for other_source in all_sources:
            if other_source != source:
                correlation = await self._correlate_sources(source, other_source)
                consistency_scores.append(correlation.correlation_strength)

        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5

        # Bias detection
        bias_indicators = self._detect_bias_indicators(content)

        # Temporal consistency
        temporal_consistency = self._check_temporal_consistency(source)

        # Cross-reference validation
        cross_reference_validation = await self._validate_cross_references(source, all_sources)

        return SourceCredibilityAssessment(
            source_id=source.get('id', 'unknown'),
            reliability_score=reliability_score,
            consistency_score=consistency_score,
            corroboration_count=len([s for s in consistency_scores if s > 0.6]),
            bias_indicators=bias_indicators,
            temporal_consistency=temporal_consistency,
            cross_reference_validation=cross_reference_validation,
            reputation_metrics=self._calculate_source_reputation(source)
        )

    def _detect_bias_indicators(self, content: str) -> List[str]:
        """Detect potential bias indicators in content"""
        bias_indicators = []

        # Emotional language patterns
        emotional_patterns = [
            r'\b(obviously|clearly|definitely|undoubtedly|certainly)\b',
            r'\b(shocking|outrageous|unbelievable|incredible)\b',
            r'\b(always|never|everyone|no one|all|none)\b'  # Absolutes
        ]

        for pattern in emotional_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                bias_indicators.append(f"emotional_language: {pattern}")

        # Political bias indicators
        political_terms = [
            'liberal', 'conservative', 'leftist', 'right-wing',
            'mainstream media', 'fake news', 'propaganda'
        ]

        for term in political_terms:
            if term.lower() in content.lower():
                bias_indicators.append(f"political_bias: {term}")

        # Source citation analysis
        citation_count = len(re.findall(r'(?:according to|source:|citation:|reference:)', content, re.IGNORECASE))
        if citation_count == 0 and len(content) > 1000:
            bias_indicators.append("lack_of_citations")

        return bias_indicators

    def _calculate_source_reputation(self, source: Dict[str, Any]) -> Dict[str, float]:
        """Calculate reputation metrics for a source"""
        return {
            'domain_authority': 0.5,  # Would be calculated from actual domain metrics
            'publication_frequency': 0.5,
            'fact_check_history': 0.5,
            'expert_citations': 0.5,
            'transparency_score': 0.5
        }

    # Additional helper methods would continue here...

    def _classify_redaction_type(self, redaction_text: str, context_before: str, context_after: str) -> str:
        """Classify the type of redaction based on context"""
        context = f"{context_before} {context_after}".lower()

        if any(word in context for word in ['name', 'person', 'individual']):
            return 'personal_information'
        elif any(word in context for word in ['location', 'address', 'facility']):
            return 'location_information'
        elif any(word in context for word in ['method', 'technique', 'operation']):
            return 'operational_details'
        elif any(word in context for word in ['source', 'informant', 'asset']):
            return 'source_protection'
        else:
            return 'unknown'

    def _infer_classification_rationale(self, content: str) -> List[str]:
        """Infer reasons for classification based on content"""
        rationales = []
        content_lower = content.lower()

        rationale_patterns = {
            'national_security': ['national security', 'foreign intelligence', 'defense'],
            'source_protection': ['source', 'informant', 'asset', 'agent'],
            'operational_security': ['operation', 'mission', 'plan', 'strategy'],
            'diplomatic_relations': ['diplomatic', 'foreign relations', 'embassy'],
            'intelligence_methods': ['collection method', 'surveillance', 'technique']
        }

        for rationale, keywords in rationale_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                rationales.append(rationale)

        return rationales

    def _extract_historical_context(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract historical context from document"""
        content = document.get('content', '')
        creation_date = document.get('creation_date')

        # Identify historical events and periods
        historical_events = []
        event_patterns = [
            r'\b(World War|Cold War|Vietnam War|Korean War)\b',
            r'\b(Berlin Wall|Cuban Missile Crisis|Watergate)\b',
            r'\b(\d{4}s?)\b'  # Decades
        ]

        for pattern in event_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            historical_events.extend(matches)

        return {
            'time_period': self._determine_time_period(creation_date),
            'historical_events_mentioned': list(set(historical_events)),
            'geopolitical_context': self._extract_geopolitical_context(content),
            'relevant_administrations': self._identify_administrations(content, creation_date)
        }

    async def health_check(self) -> bool:
        """Check analyzer health"""
        try:
            # Test spaCy model
            test_doc = self.nlp("Test document for health check.")
            return len(test_doc) > 0
        except Exception as e:
            logger.error(f"Analyzer health check failed: {e}")
            return False

        return True

    async def _analyze_onion_network(self, sites: List[Any]) -> Dict[str, Any]:
        """Analyze network topology of onion sites"""
        network_analysis = {
            'total_sites': len(sites),
            'categories': defaultdict(int),
            'interconnections': 0,
            'isolated_sites': 0,
            'hub_sites': []
        }

        # Build site network graph
        site_graph = nx.Graph()

        for site in sites:
            site_graph.add_node(site.onion_url)

            # Count categories
            if hasattr(site, 'category'):
                network_analysis['categories'][site.category] += 1

            # Add edges based on shared links
            if hasattr(site, 'links_found'):
                for link in site.links_found:
                    if link.endswith('.onion'):
                        site_graph.add_edge(site.onion_url, link)

        # Analyze network properties
        network_analysis['interconnections'] = site_graph.number_of_edges()
        network_analysis['isolated_sites'] = len(list(nx.isolates(site_graph)))

        # Find hub sites (high degree centrality)
        if site_graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(site_graph)
            hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            network_analysis['hub_sites'] = hubs

        return dict(network_analysis)

    async def _detect_threat_indicators(self, sites: List[Any]) -> Dict[str, Any]:
        """Detect potential threat indicators in onion sites"""
        threat_indicators = {
            'suspicious_keywords': defaultdict(int),
            'malicious_patterns': [],
            'security_concerns': [],
            'recommendation': 'proceed_with_caution'
        }

        suspicious_patterns = [
            r'\b(?:hack|exploit|breach|ddos|botnet)\b',
            r'\b(?:stolen|fraud|scam|phishing)\b',
            r'\b(?:weapon|explosive|drug|narcotic)\b'
        ]

        for site in sites:
            if hasattr(site, 'title'):
                content = site.title.lower()

                for pattern in suspicious_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        threat_indicators['suspicious_keywords'][match] += 1

        # Risk assessment
        total_threats = sum(threat_indicators['suspicious_keywords'].values())
        if total_threats > len(sites) * 0.3:  # More than 30% of sites have threats
            threat_indicators['recommendation'] = 'high_risk_environment'
        elif total_threats > 0:
            threat_indicators['recommendation'] = 'proceed_with_caution'
        else:
            threat_indicators['recommendation'] = 'relatively_safe'

        return dict(threat_indicators)

    async def _find_cross_source_entity_correlations(self, all_entities: Dict[str, Set[str]]) -> List[CorrelationResult]:
        """Find entity correlations across different sources"""
        correlations = []
        source_pairs = []

        # Generate all source pairs
        sources = list(all_entities.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source_pairs.append((sources[i], sources[j]))

        for source1, source2 in source_pairs:
            entities1 = all_entities[source1]
            entities2 = all_entities[source2]

            # Find overlapping entities
            overlap = entities1.intersection(entities2)

            if len(overlap) > 0:
                # Calculate confidence based on overlap size and source reliability
                confidence = min(1.0, len(overlap) / 5.0)  # Normalize to max 5 entities
                confidence *= (self.source_weights.get(source1, 0.5) + self.source_weights.get(source2, 0.5)) / 2

                if confidence >= self.correlation_threshold:
                    correlation = CorrelationResult(
                        entities=list(overlap),
                        confidence_score=confidence,
                        correlation_type='entity_based',
                        sources=[source1, source2],
                        timeline=[],
                        evidence=[f"Shared entities: {', '.join(list(overlap)[:5])}"],
                        credibility_assessment={
                            source1: self.source_weights.get(source1, 0.5),
                            source2: self.source_weights.get(source2, 0.5)
                        }
                    )
                    correlations.append(correlation)

        return correlations

    async def _find_temporal_correlations(self, source_timelines: Dict[str, List[Dict[str, Any]]]) -> List[CorrelationResult]:
        """Find temporal correlations between sources"""
        correlations = []

        # Look for events happening within time windows
        for window_name, window_size in self.time_windows.items():
            window_correlations = []

            # Compare all timeline pairs
            sources = list(source_timelines.keys())
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    source1, source2 = sources[i], sources[j]
                    timeline1, timeline2 = source_timelines[source1], source_timelines[source2]

                    # Find events within time window
                    for event1 in timeline1:
                        for event2 in timeline2:
                            time_diff = abs((event1['date'] - event2['date']).total_seconds())

                            if time_diff <= window_size.total_seconds():
                                # Check for shared entities
                                shared_entities = set(event1['entities']).intersection(set(event2['entities']))

                                if len(shared_entities) > 0:
                                    confidence = min(1.0, len(shared_entities) / 3.0)
                                    confidence *= (self.source_weights.get(source1, 0.5) + self.source_weights.get(source2, 0.5)) / 2

                                    if confidence >= self.correlation_threshold:
                                        correlation = CorrelationResult(
                                            entities=list(shared_entities),
                                            confidence_score=confidence,
                                            correlation_type='temporal',
                                            sources=[source1, source2],
                                            timeline=[event1, event2],
                                            evidence=[f"Events within {window_name} window: {event1['title']} and {event2['title']}"],
                                            credibility_assessment={
                                                source1: self.source_weights.get(source1, 0.5),
                                                source2: self.source_weights.get(source2, 0.5)
                                            }
                                        )
                                        window_correlations.append(correlation)

            correlations.extend(window_correlations)

        return correlations

    async def _find_content_correlations(self, multi_source_data: Dict[str, List[Any]]) -> List[CorrelationResult]:
        """Find content-based correlations using text similarity"""
        correlations = []

        # Extract text content from all sources
        source_texts = {}
        for source_type, documents in multi_source_data.items():
            texts = []
            for doc in documents:
                # Combine available text fields
                doc_text = ""
                for field in ['title', 'abstract', 'content', 'description']:
                    if hasattr(doc, field):
                        text = getattr(doc, field, '')
                        if text:
                            doc_text += f" {text}"

                if doc_text.strip():
                    texts.append(doc_text.strip())

            source_texts[source_type] = texts

        # Compare content similarity between sources
        sources = list(source_texts.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                texts1, texts2 = source_texts[source1], source_texts[source2]

                # Find similar content
                for text1 in texts1:
                    for text2 in texts2:
                        similarity = SequenceMatcher(None, text1[:1000], text2[:1000]).ratio()

                        if similarity > 0.7:  # High text similarity
                            confidence = similarity * (self.source_weights.get(source1, 0.5) + self.source_weights.get(source2, 0.5)) / 2

                            if confidence >= self.correlation_threshold:
                                correlation = CorrelationResult(
                                    entities=[],
                                    confidence_score=confidence,
                                    correlation_type='content_based',
                                    sources=[source1, source2],
                                    timeline=[],
                                    evidence=[f"High content similarity ({similarity:.2f}): '{text1[:100]}...' and '{text2[:100]}...'"],
                                    credibility_assessment={
                                        source1: self.source_weights.get(source1, 0.5),
                                        source2: self.source_weights.get(source2, 0.5)
                                    }
                                )
                                correlations.append(correlation)

        return correlations

    async def generate_intelligence_report(self,
                                         topic: str,
                                         correlations: List[CorrelationResult],
                                         source_data: Dict[str, List[Any]]) -> IntelligenceReport:
        """Generate comprehensive intelligence report"""

        # Extract key findings
        key_findings = []
        high_confidence_correlations = [c for c in correlations if c.confidence_score > 0.8]

        if high_confidence_correlations:
            key_findings.append(f"Found {len(high_confidence_correlations)} high-confidence correlations across sources")

        # Most correlated entities
        all_entities = []
        for correlation in correlations:
            all_entities.extend(correlation.entities)

        if all_entities:
            entity_counts = Counter(all_entities)
            most_common = entity_counts.most_common(5)
            key_findings.append(f"Most frequently correlated entities: {', '.join([entity for entity, count in most_common])}")

        # Source reliability assessment
        source_reliability = {}
        for source_type in source_data.keys():
            reliability = self.source_weights.get(source_type, 0.5)
            source_reliability[source_type] = reliability

        # Generate timeline
        timeline = []
        for correlation in correlations:
            if correlation.timeline:
                timeline.extend(correlation.timeline)

        timeline.sort(key=lambda x: x.get('date', datetime.now()))

        # Confidence level assessment
        avg_confidence = sum(c.confidence_score for c in correlations) / len(correlations) if correlations else 0
        if avg_confidence > 0.8:
            confidence_level = "HIGH"
        elif avg_confidence > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Generate recommendations
        recommendations = self._generate_analysis_recommendations(correlations, source_data)

        # Summary generation
        summary = self._generate_executive_summary(topic, correlations, key_findings)

        return IntelligenceReport(
            topic=topic,
            summary=summary,
            key_findings=key_findings,
            correlations=correlations,
            source_reliability=source_reliability,
            confidence_level=confidence_level,
            recommendations=recommendations,
            timeline=timeline[-20:],  # Last 20 events
            entity_network={},  # Would be populated with network analysis
            anomalies=[],  # Would be populated with anomaly detection
            verification_status={}  # Would be populated with verification results
        )

    def _generate_analysis_recommendations(self, correlations: List[CorrelationResult], source_data: Dict[str, List[Any]]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Source diversification recommendations
        if len(source_data) < 3:
            recommendations.append("Consider expanding to additional source types for better coverage")

        # Correlation strength recommendations
        strong_correlations = [c for c in correlations if c.confidence_score > 0.8]
        if len(strong_correlations) > 5:
            recommendations.append("Multiple strong correlations found - investigate shared entities for deeper insights")

        # Temporal analysis recommendations
        temporal_correlations = [c for c in correlations if c.correlation_type == 'temporal']
        if len(temporal_correlations) > 3:
            recommendations.append("Significant temporal patterns detected - consider timeline-based analysis")

        # Source reliability recommendations
        low_reliability_sources = [source for source, data in source_data.items()
                                 if self.source_weights.get(source, 0.5) < 0.6]
        if low_reliability_sources:
            recommendations.append(f"Verify findings from lower-reliability sources: {', '.join(low_reliability_sources)}")

        return recommendations

    def _generate_executive_summary(self, topic: str, correlations: List[CorrelationResult], key_findings: List[str]) -> str:
        """Generate executive summary of analysis"""
        summary_parts = [
            f"Analysis of '{topic}' across multiple intelligence sources.",
            f"Identified {len(correlations)} correlations across {len(set(c.sources[0] for c in correlations))} source types."
        ]

        if key_findings:
            summary_parts.append("Key findings include:")
            summary_parts.extend([f"• {finding}" for finding in key_findings[:3]])

        high_confidence = [c for c in correlations if c.confidence_score > 0.8]
        if high_confidence:
            summary_parts.append(f"Analysis confidence is high with {len(high_confidence)} strong correlations identified.")

        return " ".join(summary_parts)
