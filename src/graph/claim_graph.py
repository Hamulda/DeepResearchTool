"""Claim Graph Engine s NetworkX
Lokální implementace pro M1 development místo Neo4j

Author: Senior Python/MLOps Agent
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Reprezentace tvrzení v grafu"""

    id: str
    text: str
    confidence: float
    source_ids: list[str]
    evidence_strength: float = 0.0
    verification_status: str = "unverified"  # unverified, supported, disputed, contradicted
    created_at: datetime = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Evidence:
    """Reprezentace důkazu"""

    id: str
    text: str
    source_id: str
    source_url: str
    credibility_score: float
    relevance_score: float
    extraction_method: str = "automatic"
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ClaimRelation:
    """Vztah mezi tvrzeními"""

    source_claim_id: str
    target_claim_id: str
    relation_type: str  # support, contradict, neutral, duplicate
    strength: float  # síla vztahu 0.0 - 1.0
    confidence: float
    evidence_ids: list[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.evidence_ids is None:
            self.evidence_ids = []
        if self.created_at is None:
            self.created_at = datetime.now()


class ClaimGraph:
    """NetworkX-based Claim Graph pro lokální development
    Nahrazuje Neo4j pro M1 prostředí
    """

    def __init__(self, graph_id: str = "default"):
        self.graph_id = graph_id
        self.graph = nx.DiGraph()  # Directed graph pro claim relations
        self.claims: dict[str, Claim] = {}
        self.evidence: dict[str, Evidence] = {}
        self.relations: dict[str, ClaimRelation] = {}
        self.logger = logging.getLogger(__name__)

    def add_claim(
        self,
        claim_id: str,
        text: str,
        confidence: float,
        source_ids: list[str],
        metadata: dict[str, Any] = None
    ) -> Claim:
        """Přidá nové tvrzení do grafu"""
        claim = Claim(
            id=claim_id,
            text=text,
            confidence=confidence,
            source_ids=source_ids,
            metadata=metadata or {}
        )

        self.claims[claim_id] = claim

        # Přidání do NetworkX grafu
        self.graph.add_node(
            claim_id,
            type="claim",
            text=text,
            confidence=confidence,
            created_at=claim.created_at.isoformat()
        )

        self.logger.debug(f"✅ Přidáno tvrzení: {claim_id[:20]}...")
        return claim

    def add_evidence(
        self,
        evidence_id: str,
        text: str,
        source_id: str,
        source_url: str,
        credibility_score: float,
        relevance_score: float,
        metadata: dict[str, Any] = None
    ) -> Evidence:
        """Přidá důkaz do grafu"""
        evidence = Evidence(
            id=evidence_id,
            text=text,
            source_id=source_id,
            source_url=source_url,
            credibility_score=credibility_score,
            relevance_score=relevance_score,
            metadata=metadata or {}
        )

        self.evidence[evidence_id] = evidence

        # Přidání do NetworkX grafu
        self.graph.add_node(
            evidence_id,
            type="evidence",
            text=text,
            source_id=source_id,
            credibility_score=credibility_score
        )

        self.logger.debug(f"✅ Přidán důkaz: {evidence_id[:20]}...")
        return evidence

    def add_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation_type: str,
        strength: float,
        confidence: float,
        evidence_ids: list[str] = None
    ) -> ClaimRelation:
        """Přidá vztah mezi tvrzeními"""
        if source_claim_id not in self.claims:
            raise ValueError(f"Source claim {source_claim_id} not found")
        if target_claim_id not in self.claims:
            raise ValueError(f"Target claim {target_claim_id} not found")

        relation_id = f"{source_claim_id}__{relation_type}__{target_claim_id}"

        relation = ClaimRelation(
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            evidence_ids=evidence_ids or []
        )

        self.relations[relation_id] = relation

        # Přidání edge do NetworkX grafu
        self.graph.add_edge(
            source_claim_id,
            target_claim_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            evidence_ids=evidence_ids or []
        )

        self.logger.debug(f"✅ Přidán vztah: {source_claim_id} -> {target_claim_id} ({relation_type})")
        return relation

    def link_evidence_to_claim(self, evidence_id: str, claim_id: str, support_type: str = "supports"):
        """Propojí důkaz s tvrzením"""
        if evidence_id not in self.evidence:
            raise ValueError(f"Evidence {evidence_id} not found")
        if claim_id not in self.claims:
            raise ValueError(f"Claim {claim_id} not found")

        # Přidání edge mezi důkazem a tvrzením
        self.graph.add_edge(
            evidence_id,
            claim_id,
            relation_type=support_type,
            edge_type="evidence_claim"
        )

        self.logger.debug(f"✅ Propojen důkaz {evidence_id} s tvrzením {claim_id}")

    def get_claim_support(self, claim_id: str) -> dict[str, Any]:
        """Získá podporu pro tvrzení"""
        if claim_id not in self.claims:
            return {}

        # Najdi všechny supporting relations
        supporting_claims = []
        contradicting_claims = []
        evidence_list = []

        # Incoming edges (co podporuje toto tvrzení)
        for predecessor in self.graph.predecessors(claim_id):
            edge_data = self.graph[predecessor][claim_id]

            if edge_data.get('edge_type') == 'evidence_claim':
                # Jedná se o důkaz
                if predecessor in self.evidence:
                    evidence_list.append(self.evidence[predecessor])
            else:
                # Jedná se o claim relation
                relation_type = edge_data.get('relation_type', '')
                if relation_type == 'support':
                    supporting_claims.append(self.claims[predecessor])
                elif relation_type == 'contradict':
                    contradicting_claims.append(self.claims[predecessor])

        return {
            "claim": self.claims[claim_id],
            "supporting_claims": supporting_claims,
            "contradicting_claims": contradicting_claims,
            "evidence": evidence_list,
            "support_score": self._calculate_support_score(claim_id),
            "verification_status": self._determine_verification_status(claim_id)
        }

    def _calculate_support_score(self, claim_id: str) -> float:
        """Vypočítá support score pro tvrzení"""
        support_sum = 0.0
        contradict_sum = 0.0

        for predecessor in self.graph.predecessors(claim_id):
            edge_data = self.graph[predecessor][claim_id]
            relation_type = edge_data.get('relation_type', '')
            strength = edge_data.get('strength', 0.0)

            if relation_type == 'support':
                support_sum += strength
            elif relation_type == 'contradict':
                contradict_sum += strength

        # Evidence contribution
        evidence_score = 0.0
        for predecessor in self.graph.predecessors(claim_id):
            if predecessor in self.evidence:
                evidence_obj = self.evidence[predecessor]
                evidence_score += evidence_obj.credibility_score * evidence_obj.relevance_score

        # Combined score
        total_score = support_sum + evidence_score - contradict_sum
        return max(0.0, min(1.0, total_score))

    def _determine_verification_status(self, claim_id: str) -> str:
        """Určí verification status tvrzení"""
        support_score = self._calculate_support_score(claim_id)

        if support_score >= 0.8:
            return "supported"
        if support_score >= 0.6:
            return "partially_supported"
        if support_score >= 0.4:
            return "neutral"
        if support_score >= 0.2:
            return "disputed"
        return "contradicted"

    def find_contradictions(self) -> list[dict[str, Any]]:
        """Najde rozpory v grafu"""
        contradictions = []

        for edge in self.graph.edges(data=True):
            source, target, data = edge
            if data.get('relation_type') == 'contradict':
                contradiction = {
                    "source_claim": self.claims.get(source),
                    "target_claim": self.claims.get(target),
                    "strength": data.get('strength', 0.0),
                    "confidence": data.get('confidence', 0.0),
                    "evidence_ids": data.get('evidence_ids', [])
                }
                contradictions.append(contradiction)

        return contradictions

    def get_claim_chains(self, claim_id: str, max_depth: int = 3) -> list[list[str]]:
        """Najde řetězce tvrzení od daného tvrzení"""
        chains = []

        def dfs_chains(current_id: str, path: list[str], depth: int):
            if depth >= max_depth:
                return

            for successor in self.graph.successors(current_id):
                if successor in self.claims:  # Pouze claim nodes
                    new_path = path + [successor]
                    chains.append(new_path.copy())
                    dfs_chains(successor, new_path, depth + 1)

        dfs_chains(claim_id, [claim_id], 0)
        return chains

    def export_to_json(self, filepath: str) -> None:
        """Export grafu do JSON formátu"""
        export_data = {
            "graph_id": self.graph_id,
            "claims": {k: asdict(v) for k, v in self.claims.items()},
            "evidence": {k: asdict(v) for k, v in self.evidence.items()},
            "relations": {k: asdict(v) for k, v in self.relations.items()},
            "graph_structure": nx.node_link_data(self.graph),
            "exported_at": datetime.now().isoformat()
        }

        # Handle datetime serialization
        for claim_data in export_data["claims"].values():
            if isinstance(claim_data.get("created_at"), datetime):
                claim_data["created_at"] = claim_data["created_at"].isoformat()

        for relation_data in export_data["relations"].values():
            if isinstance(relation_data.get("created_at"), datetime):
                relation_data["created_at"] = relation_data["created_at"].isoformat()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✅ Graf exportován do {filepath}")

    def import_from_json(self, filepath: str) -> None:
        """Import grafu z JSON formátu"""
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        self.graph_id = data.get("graph_id", "imported")

        # Import claims
        self.claims = {}
        for claim_id, claim_data in data.get("claims", {}).items():
            claim = Claim(**claim_data)
            if isinstance(claim.created_at, str):
                claim.created_at = datetime.fromisoformat(claim.created_at)
            self.claims[claim_id] = claim

        # Import evidence
        self.evidence = {}
        for evidence_id, evidence_data in data.get("evidence", {}).items():
            evidence = Evidence(**evidence_data)
            self.evidence[evidence_id] = evidence

        # Import relations
        self.relations = {}
        for relation_id, relation_data in data.get("relations", {}).items():
            relation = ClaimRelation(**relation_data)
            if isinstance(relation.created_at, str):
                relation.created_at = datetime.fromisoformat(relation.created_at)
            self.relations[relation_id] = relation

        # Reconstruct NetworkX graph
        graph_data = data.get("graph_structure", {})
        self.graph = nx.node_link_graph(graph_data, directed=True)

        self.logger.info(f"✅ Graf importován z {filepath}")

    def get_statistics(self) -> dict[str, Any]:
        """Získá statistiky grafu"""
        verification_counts = {}
        for claim in self.claims.values():
            status = self._determine_verification_status(claim.id)
            verification_counts[status] = verification_counts.get(status, 0) + 1

        relation_counts = {}
        for relation in self.relations.values():
            rel_type = relation.relation_type
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

        return {
            "total_claims": len(self.claims),
            "total_evidence": len(self.evidence),
            "total_relations": len(self.relations),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "verification_status_counts": verification_counts,
            "relation_type_counts": relation_counts,
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "graph_density": nx.density(self.graph)
        }

    def visualize_subgraph(self, center_claim_id: str, depth: int = 2) -> dict[str, Any]:
        """Vytvoří data pro vizualizaci podgrafu"""
        if center_claim_id not in self.claims:
            return {}

        # Najdi všechny nodes do dané hloubky
        nodes_to_include = set([center_claim_id])
        current_level = {center_claim_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Přidej predecessors a successors
                next_level.update(self.graph.predecessors(node))
                next_level.update(self.graph.successors(node))
            nodes_to_include.update(next_level)
            current_level = next_level

        # Vytvoř subgraf
        subgraph = self.graph.subgraph(nodes_to_include)

        # Připrav data pro vizualizaci
        viz_nodes = []
        viz_edges = []

        for node in subgraph.nodes():
            node_data = {"id": node}
            if node in self.claims:
                node_data.update({
                    "type": "claim",
                    "label": self.claims[node].text[:50] + "...",
                    "confidence": self.claims[node].confidence,
                    "verification_status": self._determine_verification_status(node)
                })
            elif node in self.evidence:
                node_data.update({
                    "type": "evidence",
                    "label": self.evidence[node].text[:30] + "...",
                    "credibility": self.evidence[node].credibility_score
                })
            viz_nodes.append(node_data)

        for edge in subgraph.edges(data=True):
            source, target, data = edge
            viz_edges.append({
                "source": source,
                "target": target,
                "relation_type": data.get("relation_type", "unknown"),
                "strength": data.get("strength", 0.5)
            })

        return {
            "nodes": viz_nodes,
            "edges": viz_edges,
            "center_node": center_claim_id,
            "subgraph_size": len(nodes_to_include)
        }


# Utility funkce
def create_example_claim_graph() -> ClaimGraph:
    """Vytvoří příklad claim grafu pro testování"""
    graph = ClaimGraph("example_graph")

    # Přidání claims
    claim1 = graph.add_claim(
        "claim_1",
        "Artificial intelligence improves healthcare outcomes",
        0.8,
        ["source_1", "source_2"]
    )

    claim2 = graph.add_claim(
        "claim_2",
        "Machine learning models can predict patient outcomes accurately",
        0.75,
        ["source_2", "source_3"]
    )

    claim3 = graph.add_claim(
        "claim_3",
        "AI diagnosis systems are unreliable in complex cases",
        0.6,
        ["source_4"]
    )

    # Přidání evidence
    evidence1 = graph.add_evidence(
        "evidence_1",
        "Study shows 23% improvement in diagnostic accuracy with AI assistance",
        "source_1",
        "https://example.com/study1",
        0.9,
        0.85
    )

    # Propojení evidence s claims
    graph.link_evidence_to_claim("evidence_1", "claim_1", "supports")

    # Přidání relations
    graph.add_relation("claim_1", "claim_2", "support", 0.7, 0.8)
    graph.add_relation("claim_3", "claim_1", "contradict", 0.6, 0.7)

    return graph


# Příklad použití
if __name__ == "__main__":
    # Vytvoření příkladu grafu
    graph = create_example_claim_graph()

    # Statistiky
    stats = graph.get_statistics()
    print(f"📊 Graf statistiky: {stats}")

    # Analýza podpory pro claim
    support = graph.get_claim_support("claim_1")
    print(f"🔍 Podpora pro claim_1: {support}")

    # Export/Import test
    graph.export_to_json("example_graph.json")
    print("✅ Graf exportován")
