#!/usr/bin/env python3
"""Timeline Reconstruction Engine
Advanced temporal analysis and event correlation across multiple sources

Author: Advanced IT Specialist
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re
from typing import Any

from dateutil import parser as date_parser
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """Represents an event in the timeline"""

    event_id: str
    timestamp: datetime
    title: str
    description: str
    sources: list[str]
    confidence_score: float
    event_type: str
    entities_involved: list[str]
    locations: list[str]
    significance_score: float
    related_events: list[str] = field(default_factory=list)
    contradictory_sources: list[str] = field(default_factory=list)
    verification_status: str = "unverified"


@dataclass
class TimelineConflict:
    """Represents conflicting information about timeline events"""

    conflict_id: str
    conflicting_events: list[TimelineEvent]
    conflict_type: str  # temporal, factual, source_disagreement
    resolution_confidence: float
    recommended_resolution: TimelineEvent | None


@dataclass
class TemporalPattern:
    """Represents patterns in temporal data"""

    pattern_id: str
    pattern_type: str  # cyclical, seasonal, trend, anomaly
    start_date: datetime
    end_date: datetime | None
    frequency: str | None
    confidence: float
    supporting_events: list[str]


class TimelineReconstructor:
    """Advanced timeline reconstruction and temporal analysis"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = "timeline_reconstructor"

        # Timeline storage
        self.events = {}  # event_id -> TimelineEvent
        self.timeline_graph = nx.DiGraph()  # For event relationships
        self.temporal_index = {}  # timestamp -> [event_ids]

        # Analysis parameters
        self.confidence_threshold = config.get("timeline", {}).get("confidence_threshold", 0.6)
        self.temporal_tolerance = timedelta(
            days=config.get("timeline", {}).get("temporal_tolerance_days", 7)
        )

        # Pattern recognition
        self.known_patterns = {}
        self.conflict_resolution_rules = self._load_conflict_resolution_rules()

    async def reconstruct_timeline(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Reconstruct comprehensive timeline from multiple sources"""
        logger.info(f"Reconstructing timeline from {len(sources)} sources")

        # Phase 1: Extract events from all sources
        extracted_events = []
        for source in sources:
            events = await self._extract_events_from_source(source)
            extracted_events.extend(events)

        logger.info(f"Extracted {len(extracted_events)} events")

        # Phase 2: Deduplicate and merge similar events
        merged_events = await self._merge_similar_events(extracted_events)
        logger.info(f"Merged to {len(merged_events)} unique events")

        # Phase 3: Establish temporal relationships
        await self._establish_temporal_relationships(merged_events)

        # Phase 4: Detect and resolve conflicts
        conflicts = await self._detect_timeline_conflicts(merged_events)
        resolved_events = await self._resolve_conflicts(merged_events, conflicts)

        # Phase 5: Validate timeline consistency
        validation_results = await self._validate_timeline_consistency(resolved_events)

        # Phase 6: Identify temporal patterns
        patterns = await self._identify_temporal_patterns(resolved_events)

        # Phase 7: Calculate significance scores
        await self._calculate_significance_scores(resolved_events)

        # Phase 8: Build final timeline
        final_timeline = await self._build_final_timeline(resolved_events)

        return {
            "timeline": final_timeline,
            "total_events": len(resolved_events),
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": len([c for c in conflicts if c.recommended_resolution]),
            "temporal_patterns": patterns,
            "validation_results": validation_results,
            "confidence_metrics": await self._calculate_timeline_confidence(resolved_events),
            "methodology": {
                "extraction_methods": [
                    "date_pattern_matching",
                    "nlp_temporal_extraction",
                    "metadata_analysis",
                ],
                "merge_criteria": ["temporal_proximity", "semantic_similarity", "entity_overlap"],
                "conflict_resolution": [
                    "source_reliability",
                    "evidence_weight",
                    "consensus_analysis",
                ],
            },
        }

    async def _extract_events_from_source(self, source: dict[str, Any]) -> list[TimelineEvent]:
        """Extract timeline events from a single source"""
        events = []
        content = source.get("content", "")
        source_id = source.get("id", f"source_{hash(str(source))}")
        source_reliability = source.get("reliability_score", 0.5)

        try:
            # Method 1: Date pattern extraction
            date_events = await self._extract_events_by_date_patterns(
                content, source_id, source_reliability
            )
            events.extend(date_events)

            # Method 2: Temporal phrase extraction
            phrase_events = await self._extract_events_by_temporal_phrases(
                content, source_id, source_reliability
            )
            events.extend(phrase_events)

            # Method 3: Metadata-based extraction
            metadata_events = await self._extract_events_from_metadata(source, source_reliability)
            events.extend(metadata_events)

            # Method 4: Context-based event extraction
            context_events = await self._extract_contextual_events(
                content, source_id, source_reliability
            )
            events.extend(context_events)

        except Exception as e:
            logger.error(f"Error extracting events from source {source_id}: {e}")

        return events

    async def _extract_events_by_date_patterns(
        self, content: str, source_id: str, reliability: float
    ) -> list[TimelineEvent]:
        """Extract events using date pattern matching"""
        events = []

        # Comprehensive date patterns
        date_patterns = [
            # Full dates
            r"(?:on|at|during|in)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
            # Year-only patterns
            r"(?:in|during|since|until)\s+(\d{4})",
            # Relative dates
            r"(yesterday|today|tomorrow)",
            r"(\d+)\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|after|before)",
            # Historical periods
            r"(?:during|in)\s+(the\s+)?([A-Z][a-z]+\s+(?:War|Crisis|Revolution|Period|Era))",
            r"(?:in|during)\s+(the\s+)?(\d{4}s?)",  # Decades
        ]

        for pattern in date_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))

            for match in matches:
                date_str = match.group(1)
                match_start = match.start()

                # Extract context around the date
                context_start = max(0, match_start - 300)
                context_end = min(len(content), match_start + 300)
                context = content[context_start:context_end]

                # Parse the date
                parsed_date = await self._parse_flexible_date(date_str)
                if not parsed_date:
                    continue

                # Extract event description from context
                event_description = await self._extract_event_from_context(context, date_str)

                # Identify entities and locations
                entities = await self._extract_entities_from_text(context)
                locations = await self._extract_locations_from_text(context)

                # Create event
                event = TimelineEvent(
                    event_id=f"{source_id}_event_{len(events)}",
                    timestamp=parsed_date,
                    title=await self._generate_event_title(event_description),
                    description=event_description,
                    sources=[source_id],
                    confidence_score=reliability
                    * self._calculate_extraction_confidence(context, date_str),
                    event_type=await self._classify_event_type(event_description),
                    entities_involved=entities,
                    locations=locations,
                    significance_score=0.5,  # Will be calculated later
                )

                events.append(event)

        return events

    async def _extract_events_by_temporal_phrases(
        self, content: str, source_id: str, reliability: float
    ) -> list[TimelineEvent]:
        """Extract events using temporal phrases and indicators"""
        events = []

        # Temporal indicators that suggest events
        temporal_phrases = [
            r"(when|after|before|during|following|preceding)\s+([^.]{10,100})",
            r"(then|subsequently|later|earlier|meanwhile)\s+([^.]{10,100})",
            r"(first|second|third|finally|initially|ultimately)\s+([^.]{10,100})",
            r"(happened|occurred|took place|began|started|ended|concluded)\s+([^.]{10,100})",
        ]

        for pattern in temporal_phrases:
            matches = re.finditer(pattern, content, re.IGNORECASE)

            for match in matches:
                temporal_marker = match.group(1)
                event_text = match.group(2)

                # Try to find associated dates in nearby text
                context_start = max(0, match.start() - 200)
                context_end = min(len(content), match.end() + 200)
                context = content[context_start:context_end]

                # Look for dates in the context
                date_in_context = await self._find_date_in_context(context)

                if date_in_context:
                    entities = await self._extract_entities_from_text(event_text)
                    locations = await self._extract_locations_from_text(event_text)

                    event = TimelineEvent(
                        event_id=f"{source_id}_temporal_{len(events)}",
                        timestamp=date_in_context,
                        title=await self._generate_event_title(event_text),
                        description=event_text.strip(),
                        sources=[source_id],
                        confidence_score=reliability * 0.7,  # Lower confidence for inferred dates
                        event_type=await self._classify_event_type(event_text),
                        entities_involved=entities,
                        locations=locations,
                        significance_score=0.5,
                    )

                    events.append(event)

        return events

    async def _merge_similar_events(self, events: list[TimelineEvent]) -> list[TimelineEvent]:
        """Merge similar events from different sources"""
        if not events:
            return []

        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)

        merged_events = []
        i = 0

        while i < len(events):
            current_event = events[i]
            similar_events = [current_event]

            # Look for similar events within temporal tolerance
            j = i + 1
            while j < len(events):
                other_event = events[j]

                # Check if events are too far apart temporally
                if other_event.timestamp - current_event.timestamp > self.temporal_tolerance:
                    break

                # Check similarity
                if await self._are_events_similar(current_event, other_event):
                    similar_events.append(other_event)
                    j += 1
                else:
                    j += 1

            # Merge similar events
            merged_event = await self._merge_event_group(similar_events)
            merged_events.append(merged_event)

            # Move to next unprocessed event
            i += len(similar_events)

        return merged_events

    async def _are_events_similar(self, event1: TimelineEvent, event2: TimelineEvent) -> bool:
        """Determine if two events are similar enough to merge"""
        # Temporal proximity check
        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
        if time_diff > self.temporal_tolerance.total_seconds():
            return False

        # Semantic similarity check
        semantic_similarity = await self._calculate_semantic_similarity(
            event1.description, event2.description
        )

        # Entity overlap check
        entities1 = set(event1.entities_involved)
        entities2 = set(event2.entities_involved)
        entity_overlap = len(entities1 & entities2) / max(len(entities1 | entities2), 1)

        # Location overlap check
        locations1 = set(event1.locations)
        locations2 = set(event2.locations)
        location_overlap = len(locations1 & locations2) / max(len(locations1 | locations2), 1)

        # Combine similarity measures
        overall_similarity = (
            semantic_similarity * 0.5 + entity_overlap * 0.3 + location_overlap * 0.2
        )

        return overall_similarity > 0.7

    async def _merge_event_group(self, events: list[TimelineEvent]) -> TimelineEvent:
        """Merge a group of similar events into one consolidated event"""
        if len(events) == 1:
            return events[0]

        # Use highest confidence event as base
        base_event = max(events, key=lambda e: e.confidence_score)

        # Merge information
        all_sources = []
        all_entities = set()
        all_locations = set()
        descriptions = []

        for event in events:
            all_sources.extend(event.sources)
            all_entities.update(event.entities_involved)
            all_locations.update(event.locations)
            descriptions.append(event.description)

        # Calculate weighted average timestamp
        total_weight = sum(e.confidence_score for e in events)
        weighted_timestamp = (
            sum(e.timestamp.timestamp() * e.confidence_score for e in events) / total_weight
        )

        merged_event = TimelineEvent(
            event_id=f"merged_{base_event.event_id}",
            timestamp=datetime.fromtimestamp(weighted_timestamp),
            title=base_event.title,
            description=await self._combine_descriptions(descriptions),
            sources=list(set(all_sources)),
            confidence_score=min(sum(e.confidence_score for e in events) / len(events) * 1.2, 1.0),
            event_type=base_event.event_type,
            entities_involved=list(all_entities),
            locations=list(all_locations),
            significance_score=max(e.significance_score for e in events),
        )

        return merged_event

    async def _detect_timeline_conflicts(
        self, events: list[TimelineEvent]
    ) -> list[TimelineConflict]:
        """Detect conflicts in the timeline"""
        conflicts = []

        # Group events by similar content/entities
        event_groups = await self._group_events_by_similarity(events)

        for group in event_groups:
            if len(group) > 1:
                # Check for temporal conflicts
                temporal_conflicts = await self._detect_temporal_conflicts(group)
                conflicts.extend(temporal_conflicts)

                # Check for factual conflicts
                factual_conflicts = await self._detect_factual_conflicts(group)
                conflicts.extend(factual_conflicts)

        return conflicts

    async def _build_final_timeline(self, events: list[TimelineEvent]) -> list[dict[str, Any]]:
        """Build the final timeline structure"""
        # Sort events chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        timeline = []
        for event in sorted_events:
            if event.confidence_score >= self.confidence_threshold:
                timeline_item = {
                    "timestamp": event.timestamp.isoformat(),
                    "title": event.title,
                    "description": event.description,
                    "type": event.event_type,
                    "confidence": event.confidence_score,
                    "significance": event.significance_score,
                    "sources": event.sources,
                    "entities": event.entities_involved,
                    "locations": event.locations,
                    "verification_status": event.verification_status,
                }
                timeline.append(timeline_item)

        return timeline

    # Helper methods for date parsing, similarity calculation, etc.
    async def _parse_flexible_date(self, date_str: str) -> datetime | None:
        """Parse various date formats flexibly"""
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except:
            # Try additional parsing methods
            try:
                # Handle relative dates
                if "ago" in date_str.lower():
                    # Extract number and unit
                    match = re.search(r"(\d+)\s*(day|week|month|year)s?\s*ago", date_str.lower())
                    if match:
                        amount = int(match.group(1))
                        unit = match.group(2)

                        now = datetime.now()
                        if unit == "day":
                            return now - timedelta(days=amount)
                        if unit == "week":
                            return now - timedelta(weeks=amount)
                        if unit == "month":
                            return now - timedelta(days=amount * 30)
                        if unit == "year":
                            return now - timedelta(days=amount * 365)

                # Handle decades
                decade_match = re.search(r"(\d{4})s?", date_str)
                if decade_match:
                    year = int(decade_match.group(1))
                    return datetime(year, 1, 1)

            except:
                pass

            return None

    async def health_check(self) -> bool:
        """Check timeline reconstructor health"""
        return True
