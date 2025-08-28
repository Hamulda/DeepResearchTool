#!/usr/bin/env python3
"""
Claim Graph System
Track claim relationships, contradictions, and conflict sets for evidence-based research

Author: Senior IT Specialist
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime
import networkx as nx
import numpy as np

import structlog

logger = structlog.get_logger(__name__)

class RelationshipType(Enum):
    """Types of relationships between claims"""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    QUALIFIES = "qualifies"
    NEUTRAL = "neutral"

@dataclass
class ClaimNode:
    """Individual claim in the graph"""
    id: str
    text: str
    confidence: float
    evidence: List[Dict[str, Any]]
    source_query: str
    timestamp: datetime
    verification_status: str = "unverified"  # unverified, verified, disputed
    dispute_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClaimRelationship:
    """Relationship between two claims"""
    id: str
    source_claim_id: str
    target_claim_id: str
    relationship_type: RelationshipType
    confidence: float
    evidence: List[Dict[str, Any]]
    explanation: str
    timestamp: datetime

@dataclass
class ConflictSet:
    """Set of conflicting claims with evidence"""
    id: str
    claim_ids: Set[str]
    conflict_type: str  # "direct_contradiction", "evidence_conflict", "temporal_inconsistency"
    severity: float  # 0.0 to 1.0
    resolution_strategy: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class ContradictionDetector:
    """Detect contradictions between claims using multiple strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contradiction_config = config.get("contradiction_detection", {})

        # Detection thresholds
        self.semantic_threshold = self.contradiction_config.get("semantic_threshold", 0.8)
        self.evidence_conflict_threshold = self.contradiction_config.get("evidence_threshold", 0.7)

        # Contradiction patterns
        self.contradiction_patterns = [
            (r"not|never|no|cannot|impossible", r"yes|always|can|possible"),
            (r"increase|rise|grow|more", r"decrease|fall|decline|less"),
            (r"effective|successful|works", r"ineffective|failed|doesn't work"),
            (r"safe|secure|protected", r"dangerous|risky|harmful"),
            (r"proven|confirmed|established", r"unproven|disputed|questionable")
        ]

        self.logger = structlog.get_logger(__name__)

    async def detect_contradictions(self, claims: List[ClaimNode]) -> List[ConflictSet]:
        """Detect contradictions between claims"""

        conflict_sets = []

        # Method 1: Direct semantic contradiction
        semantic_conflicts = await self._detect_semantic_contradictions(claims)
        conflict_sets.extend(semantic_conflicts)

        # Method 2: Evidence-based conflicts
        evidence_conflicts = await self._detect_evidence_conflicts(claims)
        conflict_sets.extend(evidence_conflicts)

        # Method 3: Temporal inconsistencies
        temporal_conflicts = await self._detect_temporal_inconsistencies(claims)
        conflict_sets.extend(temporal_conflicts)

        self.logger.info(f"Detected {len(conflict_sets)} conflict sets")
        return conflict_sets

    async def _detect_semantic_contradictions(self, claims: List[ClaimNode]) -> List[ConflictSet]:
        """Detect direct semantic contradictions between claim texts"""

        conflicts = []

        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], i+1):

                # Check for contradiction patterns
                contradiction_score = self._calculate_contradiction_score(claim1.text, claim2.text)

                if contradiction_score > self.semantic_threshold:
                    conflict_set = ConflictSet(
                        id=f"semantic_conflict_{uuid.uuid4().hex[:8]}",
                        claim_ids={claim1.id, claim2.id},
                        conflict_type="direct_contradiction",
                        severity=contradiction_score,
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict_set)

        return conflicts

    def _calculate_contradiction_score(self, text1: str, text2: str) -> float:
        """Calculate contradiction score between two texts"""

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        contradiction_score = 0.0
        pattern_count = 0

        for positive_pattern, negative_pattern in self.contradiction_patterns:
            # Check if one text has positive pattern and other has negative
            import re

            text1_positive = bool(re.search(positive_pattern, text1_lower))
            text1_negative = bool(re.search(negative_pattern, text1_lower))
            text2_positive = bool(re.search(positive_pattern, text2_lower))
            text2_negative = bool(re.search(negative_pattern, text2_lower))

            # Contradiction if opposite patterns found
            if (text1_positive and text2_negative) or (text1_negative and text2_positive):
                contradiction_score += 1.0

            pattern_count += 1

        return contradiction_score / pattern_count if pattern_count > 0 else 0.0

    async def _detect_evidence_conflicts(self, claims: List[ClaimNode]) -> List[ConflictSet]:
        """Detect conflicts in evidence between claims"""

        conflicts = []

        # Group claims by similar topics (simplified by shared keywords)
        topic_groups = self._group_claims_by_topic(claims)

        for topic, topic_claims in topic_groups.items():
            if len(topic_claims) < 2:
                continue

            # Check for conflicting evidence within topic group
            for i, claim1 in enumerate(topic_claims):
                for j, claim2 in enumerate(topic_claims[i+1:], i+1):

                    evidence_conflict = self._check_evidence_conflict(claim1, claim2)

                    if evidence_conflict:
                        conflict_set = ConflictSet(
                            id=f"evidence_conflict_{uuid.uuid4().hex[:8]}",
                            claim_ids={claim1.id, claim2.id},
                            conflict_type="evidence_conflict",
                            severity=evidence_conflict,
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict_set)

        return conflicts

    def _group_claims_by_topic(self, claims: List[ClaimNode]) -> Dict[str, List[ClaimNode]]:
        """Group claims by topic using keyword similarity"""

        from collections import defaultdict

        topic_groups = defaultdict(list)

        for claim in claims:
            # Extract key terms (simplified)
            import re
            words = re.findall(r'\b\w{4,}\b', claim.text.lower())

            # Use most frequent words as topic identifier
            if words:
                topic = "_".join(sorted(words)[:3])  # Top 3 words as topic
                topic_groups[topic].append(claim)

        return dict(topic_groups)

    def _check_evidence_conflict(self, claim1: ClaimNode, claim2: ClaimNode) -> Optional[float]:
        """Check if evidence between two claims conflicts"""

        # Check for conflicting sources
        claim1_sources = {ev.get("canonical_url", "") for ev in claim1.evidence}
        claim2_sources = {ev.get("canonical_url", "") for ev in claim2.evidence}

        # If same sources support different conclusions, potential conflict
        shared_sources = claim1_sources & claim2_sources

        if shared_sources and len(shared_sources) > 0:
            # Check if claims are semantically different
            contradiction_score = self._calculate_contradiction_score(claim1.text, claim2.text)

            if contradiction_score > 0.3:  # Lower threshold for evidence conflicts
                return contradiction_score * 0.8  # Scale down for evidence-based conflicts

        return None

    async def _detect_temporal_inconsistencies(self, claims: List[ClaimNode]) -> List[ConflictSet]:
        """Detect temporal inconsistencies in claims"""

        conflicts = []

        # Group claims by temporal expressions
        temporal_claims = []
        for claim in claims:
            temporal_info = self._extract_temporal_info(claim.text)
            if temporal_info:
                temporal_claims.append((claim, temporal_info))

        # Check for temporal conflicts
        for i, (claim1, time1) in enumerate(temporal_claims):
            for j, (claim2, time2) in enumerate(temporal_claims[i+1:], i+1):

                if self._check_temporal_conflict(time1, time2):
                    conflict_set = ConflictSet(
                        id=f"temporal_conflict_{uuid.uuid4().hex[:8]}",
                        claim_ids={claim1.id, claim2.id},
                        conflict_type="temporal_inconsistency",
                        severity=0.6,  # Medium severity for temporal conflicts
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict_set)

        return conflicts

    def _extract_temporal_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract temporal information from claim text"""

        import re

        # Simple temporal pattern matching
        year_pattern = r'\b(19|20)\d{2}\b'
        relative_pattern = r'\b(recent|latest|current|new|old|past|future)\b'

        years = re.findall(year_pattern, text)
        relative_terms = re.findall(relative_pattern, text.lower())

        if years or relative_terms:
            return {
                "years": years,
                "relative_terms": relative_terms,
                "has_temporal": True
            }

        return None

    def _check_temporal_conflict(self, time1: Dict[str, Any], time2: Dict[str, Any]) -> bool:
        """Check if two temporal expressions conflict"""

        # Simple heuristic: conflicting relative terms
        conflicting_pairs = [
            ("recent", "old"), ("new", "past"), ("current", "historical"),
            ("latest", "previous"), ("future", "past")
        ]

        terms1 = set(time1.get("relative_terms", []))
        terms2 = set(time2.get("relative_terms", []))

        for term1, term2 in conflicting_pairs:
            if term1 in terms1 and term2 in terms2:
                return True
            if term2 in terms1 and term1 in terms2:
                return True

        return False

class ClaimGraph:
    """Graph structure for managing claims and their relationships"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.DiGraph()

        # Storage
        self.claims: Dict[str, ClaimNode] = {}
        self.relationships: Dict[str, ClaimRelationship] = {}
        self.conflict_sets: Dict[str, ConflictSet] = {}

        # Components
        self.contradiction_detector = ContradictionDetector(config)

        self.logger = structlog.get_logger(__name__)

    def add_claim(self, claim: ClaimNode):
        """Add claim to graph"""

        self.claims[claim.id] = claim
        self.graph.add_node(claim.id, claim=claim)

        self.logger.debug(f"Added claim {claim.id} to graph")

    def add_relationship(self, relationship: ClaimRelationship):
        """Add relationship between claims"""

        self.relationships[relationship.id] = relationship

        # Add edge to graph
        self.graph.add_edge(
            relationship.source_claim_id,
            relationship.target_claim_id,
            relationship=relationship,
            weight=relationship.confidence
        )

        self.logger.debug(f"Added relationship {relationship.relationship_type.value} "
                         f"between {relationship.source_claim_id} and {relationship.target_claim_id}")

    def add_conflict_set(self, conflict_set: ConflictSet):
        """Add conflict set to graph"""

        self.conflict_sets[conflict_set.id] = conflict_set

        # Mark involved claims as disputed
        for claim_id in conflict_set.claim_ids:
            if claim_id in self.claims:
                claim = self.claims[claim_id]
                claim.verification_status = "disputed"
                claim.dispute_reason = f"Part of conflict set: {conflict_set.conflict_type}"

                # Reduce confidence
                claim.confidence *= 0.7  # 30% confidence penalty for disputed claims

        self.logger.info(f"Added conflict set {conflict_set.id} affecting {len(conflict_set.claim_ids)} claims")

    async def detect_and_add_contradictions(self):
        """Detect contradictions and add them to graph"""

        claims_list = list(self.claims.values())
        conflict_sets = await self.contradiction_detector.detect_contradictions(claims_list)

        for conflict_set in conflict_sets:
            self.add_conflict_set(conflict_set)

        return len(conflict_sets)

    def get_claim_relationships(self, claim_id: str) -> List[ClaimRelationship]:
        """Get all relationships for a claim"""

        relationships = []

        # Outgoing relationships
        for successor in self.graph.successors(claim_id):
            edge_data = self.graph[claim_id][successor]
            relationships.append(edge_data["relationship"])

        # Incoming relationships
        for predecessor in self.graph.predecessors(claim_id):
            edge_data = self.graph[predecessor][claim_id]
            relationships.append(edge_data["relationship"])

        return relationships

    def get_supporting_claims(self, claim_id: str) -> List[ClaimNode]:
        """Get claims that support the given claim"""

        supporting_claims = []

        for predecessor in self.graph.predecessors(claim_id):
            edge_data = self.graph[predecessor][claim_id]
            relationship = edge_data["relationship"]

            if relationship.relationship_type == RelationshipType.SUPPORTS:
                supporting_claims.append(self.claims[predecessor])

        return supporting_claims

    def get_contradicting_claims(self, claim_id: str) -> List[ClaimNode]:
        """Get claims that contradict the given claim"""

        contradicting_claims = []

        # Check direct contradictions
        for predecessor in self.graph.predecessors(claim_id):
            edge_data = self.graph[predecessor][claim_id]
            relationship = edge_data["relationship"]

            if relationship.relationship_type == RelationshipType.CONTRADICTS:
                contradicting_claims.append(self.claims[predecessor])

        # Check conflict sets
        for conflict_set in self.conflict_sets.values():
            if claim_id in conflict_set.claim_ids:
                for other_claim_id in conflict_set.claim_ids:
                    if other_claim_id != claim_id:
                        contradicting_claims.append(self.claims[other_claim_id])

        return contradicting_claims

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the claim graph"""

        disputed_claims = [c for c in self.claims.values() if c.verification_status == "disputed"]
        verified_claims = [c for c in self.claims.values() if c.verification_status == "verified"]

        return {
            "total_claims": len(self.claims),
            "total_relationships": len(self.relationships),
            "total_conflicts": len(self.conflict_sets),
            "disputed_claims": len(disputed_claims),
            "verified_claims": len(verified_claims),
            "average_confidence": np.mean([c.confidence for c in self.claims.values()]) if self.claims else 0,
            "graph_density": nx.density(self.graph),
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export graph for analysis or visualization"""

        return {
            "claims": [
                {
                    "id": claim.id,
                    "text": claim.text,
                    "confidence": claim.confidence,
                    "verification_status": claim.verification_status,
                    "evidence_count": len(claim.evidence),
                    "timestamp": claim.timestamp.isoformat()
                }
                for claim in self.claims.values()
            ],
            "relationships": [
                {
                    "id": rel.id,
                    "source": rel.source_claim_id,
                    "target": rel.target_claim_id,
                    "type": rel.relationship_type.value,
                    "confidence": rel.confidence,
                    "explanation": rel.explanation
                }
                for rel in self.relationships.values()
            ],
            "conflicts": [
                {
                    "id": conflict.id,
                    "claim_ids": list(conflict.claim_ids),
                    "type": conflict.conflict_type,
                    "severity": conflict.severity
                }
                for conflict in self.conflict_sets.values()
            ],
            "statistics": self.get_graph_statistics()
        }

def create_claim_graph(config: Dict[str, Any]) -> ClaimGraph:
    """Factory function for claim graph"""
    return ClaimGraph(config)
