#!/usr/bin/env python3
"""
Unit testy pro validační brány
Testy fail-hard chování a návrhy nápravy

Author: Senior Python/MLOps Agent
"""

import pytest
import asyncio
from unittest.mock import MagicMock
from src.utils.gates import (
    GateManager,
    GateConfig,
    GateType,
    EvidenceGateError,
    ComplianceGateError,
    MetricsGateError,
    create_default_gate_config,
    create_gate_manager,
)


class TestValidationGates:
    """Testy validačních bran"""

    def setup_method(self):
        """Setup pro každý test"""
        self.config = create_default_gate_config()
        self.gate_manager = GateManager(self.config)

    @pytest.mark.asyncio
    async def test_evidence_gate_success(self):
        """Test úspěšné evidence gate"""
        data = {
            "claims": [
                {
                    "text": "Test claim 1",
                    "citations": [
                        {"source_id": "source1", "passage": "Evidence 1"},
                        {"source_id": "source2", "passage": "Evidence 2"},
                    ],
                },
                {
                    "text": "Test claim 2",
                    "citations": [
                        {"source_id": "source3", "passage": "Evidence 3"},
                        {"source_id": "source4", "passage": "Evidence 4"},
                    ],
                },
            ]
        }

        # Nemělo by vyhodit výjimku
        report = await self.gate_manager.validate_single(GateType.EVIDENCE, data)
        assert report["passed"] is True
        assert report["total_claims"] == 2
        assert report["total_citations"] == 4

    @pytest.mark.asyncio
    async def test_evidence_gate_insufficient_citations(self):
        """Test selhání evidence gate - málo citací"""
        data = {
            "claims": [
                {
                    "text": "Test claim",
                    "citations": [{"source_id": "source1", "passage": "Only one citation"}],
                }
            ]
        }

        with pytest.raises(EvidenceGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.EVIDENCE, data)

        assert "má pouze 1 citací" in str(exc_info.value)
        assert "min_citations_per_claim" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_evidence_gate_insufficient_sources(self):
        """Test selhání evidence gate - málo nezávislých zdrojů"""
        data = {
            "claims": [
                {
                    "text": "Test claim",
                    "citations": [
                        {"source_id": "source1", "passage": "Citation 1"},
                        {"source_id": "source1", "passage": "Citation 2 from same source"},
                    ],
                }
            ]
        }

        with pytest.raises(EvidenceGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.EVIDENCE, data)

        assert "má pouze 1 nezávislých zdrojů" in str(exc_info.value)
        assert "source diversity" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_evidence_gate_no_claims(self):
        """Test selhání evidence gate - žádné claims"""
        data = {"claims": []}

        with pytest.raises(EvidenceGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.EVIDENCE, data)

        assert "Žádné claims nalezeny" in str(exc_info.value)
        assert "synthesis pipeline" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_compliance_gate_success(self):
        """Test úspěšné compliance gate"""
        data = {
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": [],
                "accessed_domains": ["example.com", "test.org"],
            }
        }

        report = await self.gate_manager.validate_single(GateType.COMPLIANCE, data)
        assert report["passed"] is True
        assert report["robots_violations"] == 0
        assert report["rate_limit_violations"] == 0

    @pytest.mark.asyncio
    async def test_compliance_gate_robots_violation(self):
        """Test selhání compliance gate - robots.txt porušení"""
        data = {
            "retrieval_metadata": {
                "robots_violations": ["example.com", "test.org"],
                "rate_limit_violations": [],
                "accessed_domains": [],
            }
        }

        with pytest.raises(ComplianceGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.COMPLIANCE, data)

        assert "Porušení robots.txt: 2 domén" in str(exc_info.value)
        assert "robots.txt checking" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_compliance_gate_rate_limit_violation(self):
        """Test selhání compliance gate - rate-limit porušení"""
        data = {
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": ["req1", "req2", "req3"],
                "accessed_domains": [],
            }
        }

        with pytest.raises(ComplianceGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.COMPLIANCE, data)

        assert "Porušení rate-limitů: 3 požadavků" in str(exc_info.value)
        assert "request rate" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_compliance_gate_blocked_domains(self):
        """Test selhání compliance gate - blokované domény"""
        self.config.blocked_domains = ["evil.com", "spam.org"]
        gate_manager = GateManager(self.config)

        data = {
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": [],
                "accessed_domains": ["example.com", "evil.com", "test.org"],
            }
        }

        with pytest.raises(ComplianceGateError) as exc_info:
            await gate_manager.validate_single(GateType.COMPLIANCE, data)

        assert "blokovaným doménám" in str(exc_info.value)
        assert "evil.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_metrics_gate_success(self):
        """Test úspěšné metrics gate"""
        data = {"metrics": {"recall_at_10": 0.8, "ndcg_at_10": 0.7, "citation_precision": 0.9}}

        report = await self.gate_manager.validate_single(GateType.METRICS, data)
        assert report["passed"] is True
        assert report["recall_at_10"] == 0.8
        assert report["ndcg_at_10"] == 0.7
        assert report["citation_precision"] == 0.9

    @pytest.mark.asyncio
    async def test_metrics_gate_low_recall(self):
        """Test selhání metrics gate - nízký recall"""
        data = {
            "metrics": {
                "recall_at_10": 0.5,  # Pod 0.7
                "ndcg_at_10": 0.8,
                "citation_precision": 0.9,
            }
        }

        with pytest.raises(MetricsGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.METRICS, data)

        assert "Recall@10 0.500 < 0.7" in str(exc_info.value)
        assert "ef_search_param" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_metrics_gate_low_ndcg(self):
        """Test selhání metrics gate - nízký nDCG"""
        data = {
            "metrics": {
                "recall_at_10": 0.8,
                "ndcg_at_10": 0.4,  # Pod 0.6
                "citation_precision": 0.9,
            }
        }

        with pytest.raises(MetricsGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.METRICS, data)

        assert "nDCG@10 0.400 < 0.6" in str(exc_info.value)
        assert "re-ranking parametry" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_metrics_gate_low_citation_precision(self):
        """Test selhání metrics gate - nízká citation precision"""
        data = {
            "metrics": {
                "recall_at_10": 0.8,
                "ndcg_at_10": 0.7,
                "citation_precision": 0.6,  # Pod 0.8
            }
        }

        with pytest.raises(MetricsGateError) as exc_info:
            await self.gate_manager.validate_single(GateType.METRICS, data)

        assert "Citation precision 0.600 < 0.8" in str(exc_info.value)
        assert "synthesis validation" in exc_info.value.remediation_hint

    @pytest.mark.asyncio
    async def test_validate_all_success(self):
        """Test úspěšné validace všech bran"""
        data = {
            "timestamp": "2025-08-27T10:00:00Z",
            "query": "Test query",
            "claims": [
                {
                    "text": "Test claim",
                    "citations": [
                        {"source_id": "source1", "passage": "Evidence 1"},
                        {"source_id": "source2", "passage": "Evidence 2"},
                    ],
                }
            ],
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": [],
                "accessed_domains": ["example.com"],
            },
            "metrics": {"recall_at_10": 0.8, "ndcg_at_10": 0.7, "citation_precision": 0.9},
        }

        report = await self.gate_manager.validate_all(data)
        assert report["overall_passed"] is True
        assert len(report["failed_gates"]) == 0
        assert "evidence" in report["gates"]
        assert "compliance" in report["gates"]
        assert "metrics" in report["gates"]

    @pytest.mark.asyncio
    async def test_validate_all_failure(self):
        """Test selhání validace - fail-hard"""
        data = {
            "claims": [],  # Prázdné claims -> selhání evidence gate
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": [],
                "accessed_domains": [],
            },
            "metrics": {"recall_at_10": 0.8, "ndcg_at_10": 0.7, "citation_precision": 0.9},
        }

        with pytest.raises(EvidenceGateError):
            await self.gate_manager.validate_all(data)

    @pytest.mark.asyncio
    async def test_skip_gates(self):
        """Test přeskočení bran"""
        data = {
            "claims": [],  # Toto by normálně selhalo
            "retrieval_metadata": {
                "robots_violations": [],
                "rate_limit_violations": [],
                "accessed_domains": [],
            },
            "metrics": {"recall_at_10": 0.8, "ndcg_at_10": 0.7, "citation_precision": 0.9},
        }

        # Přeskočíme evidence gate
        report = await self.gate_manager.validate_all(data, skip_gates=[GateType.EVIDENCE])
        assert report["overall_passed"] is True
        assert "evidence" not in report["gates"]
        assert "compliance" in report["gates"]
        assert "metrics" in report["gates"]

    def test_create_gate_manager_from_config(self):
        """Test vytvoření gate manageru z config"""
        config_dict = {"validation_gates": {"min_citations_per_claim": 3, "min_recall_at_10": 0.8}}

        manager = create_gate_manager(config_dict)
        assert manager.config.min_citations_per_claim == 3
        assert manager.config.min_recall_at_10 == 0.8

    def test_validation_history(self):
        """Test historie validací"""
        assert len(self.gate_manager.get_validation_history()) == 0

        # Historie se přidá po validate_all
        # (ne testováno async zde, jen struktura)

        self.gate_manager.clear_history()
        assert len(self.gate_manager.get_validation_history()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
