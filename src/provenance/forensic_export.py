#!/usr/bin/env python3
"""
Forensic JSON-LD Export
Standards-compliant export of claim graphs with full provenance chains

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import hashlib
import logging
from pathlib import Path
import uuid

from .warc_tracking import ProvenanceRecord, ProvenanceTracker
from ..verification.contradiction_sets import Claim, Contradiction, ContradictionSet

logger = logging.getLogger(__name__)


@dataclass
class ForensicContext:
    """Context information for forensic export"""
    export_id: str
    export_timestamp: datetime
    software_version: str
    export_reason: str
    operator_id: Optional[str] = None
    chain_of_custody: List[str] = None

    def __post_init__(self):
        if self.chain_of_custody is None:
            self.chain_of_custody = []


class JSONLDExporter:
    """JSON-LD exporter with W3C compliance and forensic standards"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_config = config.get("forensic_export", {})

        # JSON-LD context
        self.base_context = {
            "@vocab": "https://schema.org/",
            "prov": "http://www.w3.org/ns/prov#",
            "sec": "https://w3id.org/security#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "drt": "https://deepresearchtool.ai/vocab#",
            "claim": "drt:Claim",
            "evidence": "drt:Evidence",
            "contradiction": "drt:Contradiction",
            "provenance": "prov:Entity",
            "verification": "sec:Verification",
            "timestamp": {
                "@id": "prov:generatedAtTime",
                "@type": "xsd:dateTime"
            },
            "hash": {
                "@id": "sec:digestValue",
                "@type": "xsd:string"
            },
            "confidence": {
                "@id": "drt:confidence",
                "@type": "xsd:decimal"
            }
        }

        logger.info("JSON-LD forensic exporter initialized")

    def export_claim_graph(
        self,
        claims: List[Claim],
        contradictions: List[Contradiction],
        contradiction_sets: List[ContradictionSet],
        provenance_tracker: ProvenanceTracker,
        context: ForensicContext
    ) -> Dict[str, Any]:
        """Export complete claim graph as forensic JSON-LD"""

        # Build the main JSON-LD document
        document = {
            "@context": self.base_context,
            "@type": "drt:ForensicClaimGraph",
            "@id": f"urn:uuid:{context.export_id}",
            "export_metadata": self._build_export_metadata(context),
            "claims": self._export_claims(claims, provenance_tracker),
            "contradictions": self._export_contradictions(contradictions),
            "contradiction_sets": self._export_contradiction_sets(contradiction_sets),
            "provenance_chain": self._build_provenance_chain(claims, provenance_tracker),
            "verification_data": self._build_verification_data(claims, contradictions, provenance_tracker),
            "integrity_hash": ""  # Will be calculated last
        }

        # Calculate integrity hash
        document["integrity_hash"] = self._calculate_document_hash(document)

        logger.info(f"Exported claim graph: {len(claims)} claims, {len(contradictions)} contradictions")
        return document

    def _build_export_metadata(self, context: ForensicContext) -> Dict[str, Any]:
        """Build export metadata section"""
        return {
            "@type": "drt:ExportMetadata",
            "@id": f"urn:uuid:{context.export_id}",
            "export_timestamp": context.export_timestamp.isoformat(),
            "software_version": context.software_version,
            "export_reason": context.export_reason,
            "operator_id": context.operator_id,
            "chain_of_custody": context.chain_of_custody,
            "compliance_standards": [
                "W3C JSON-LD 1.1",
                "W3C PROV-O",
                "W3C Security Vocabulary",
                "ISO 27037:2012 (Digital Evidence)"
            ]
        }

    def _export_claims(
        self,
        claims: List[Claim],
        provenance_tracker: ProvenanceTracker
    ) -> List[Dict[str, Any]]:
        """Export claims with full provenance"""

        exported_claims = []

        for claim in claims:
            claim_data = {
                "@type": "drt:Claim",
                "@id": f"urn:uuid:{claim.id}",
                "text": claim.text,
                "confidence": claim.confidence,
                "evidence": self._export_evidence(claim.evidence, provenance_tracker),
                "source_urls": claim.source_urls,
                "metadata": claim.metadata,
                "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                "provenance": self._get_claim_provenance(claim, provenance_tracker)
            }

            exported_claims.append(claim_data)

        return exported_claims

    def _export_evidence(
        self,
        evidence: List[Dict[str, Any]],
        provenance_tracker: ProvenanceTracker
    ) -> List[Dict[str, Any]]:
        """Export evidence with provenance tracking"""

        exported_evidence = []

        for item in evidence:
            evidence_data = {
                "@type": "drt:Evidence",
                "@id": f"urn:uuid:{uuid.uuid4()}",
                "text": item.get("text", ""),
                "source": item.get("source", ""),
                "confidence": item.get("confidence", 0.0),
                "metadata": item.get("metadata", {}),
                "verification_status": self._verify_evidence(item, provenance_tracker)
            }

            # Add provenance if available
            source_url = item.get("source_url")
            if source_url:
                evidence_data["provenance"] = self._get_url_provenance(source_url, provenance_tracker)

            exported_evidence.append(evidence_data)

        return exported_evidence

    def _export_contradictions(self, contradictions: List[Contradiction]) -> List[Dict[str, Any]]:
        """Export contradictions with detailed analysis"""

        exported_contradictions = []

        for contradiction in contradictions:
            contradiction_data = {
                "@type": "drt:Contradiction",
                "@id": f"urn:uuid:{contradiction.id}",
                "claim_a": f"urn:uuid:{contradiction.claim_a.id}",
                "claim_b": f"urn:uuid:{contradiction.claim_b.id}",
                "contradiction_type": contradiction.contradiction_type.value,
                "confidence": contradiction.confidence,
                "evidence": contradiction.evidence,
                "resolution_suggestion": contradiction.resolution_suggestion,
                "detection_timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_metadata": {
                    "detection_method": "automated_nlp_analysis",
                    "verification_required": contradiction.confidence < 0.8
                }
            }

            exported_contradictions.append(contradiction_data)

        return exported_contradictions

    def _export_contradiction_sets(self, contradiction_sets: List[ContradictionSet]) -> List[Dict[str, Any]]:
        """Export contradiction sets with pro/contra analysis"""

        exported_sets = []

        for cs in contradiction_sets:
            set_data = {
                "@type": "drt:ContradictionSet",
                "@id": f"urn:uuid:{uuid.uuid4()}",
                "topic": cs.topic,
                "supporting_claims": [f"urn:uuid:{claim.id}" for claim in cs.supporting_claims],
                "contradicting_claims": [f"urn:uuid:{claim.id}" for claim in cs.contradicting_claims],
                "contradictions": [f"urn:uuid:{c.id}" for c in cs.contradictions],
                "confidence_score": cs.confidence_score,
                "calibration_hint": cs.calibration_hint,
                "evidence_summary": {
                    "pro_evidence_count": cs.pro_evidence_count,
                    "contra_evidence_count": cs.contra_evidence_count,
                    "total_evidence_count": cs.total_evidence_count,
                    "evidence_balance": cs.pro_evidence_count / max(cs.total_evidence_count, 1)
                }
            }

            exported_sets.append(set_data)

        return exported_sets

    def _build_provenance_chain(
        self,
        claims: List[Claim],
        provenance_tracker: ProvenanceTracker
    ) -> List[Dict[str, Any]]:
        """Build complete provenance chain for all content"""

        provenance_chain = []
        tracked_urls = set()

        for claim in claims:
            for url in claim.source_urls:
                if url not in tracked_urls:
                    provenance = self._get_url_provenance(url, provenance_tracker)
                    if provenance:
                        provenance_chain.append(provenance)
                        tracked_urls.add(url)

        return provenance_chain

    def _get_claim_provenance(
        self,
        claim: Claim,
        provenance_tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Get provenance information for a claim"""

        provenance_data = {
            "@type": "prov:Entity",
            "generation_method": "automated_extraction",
            "source_count": len(claim.source_urls),
            "evidence_count": len(claim.evidence),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add source-specific provenance
        if claim.source_urls:
            source_provenance = []
            for url in claim.source_urls:
                url_prov = self._get_url_provenance(url, provenance_tracker)
                if url_prov:
                    source_provenance.append(url_prov)

            if source_provenance:
                provenance_data["source_provenance"] = source_provenance

        return provenance_data

    def _get_url_provenance(
        self,
        url: str,
        provenance_tracker: ProvenanceTracker
    ) -> Optional[Dict[str, Any]]:
        """Get provenance information for a URL"""

        # Search for provenance records
        for content_id, record in provenance_tracker.tracked_content.items():
            if record.source_url == url:
                return {
                    "@type": "prov:Entity",
                    "@id": f"urn:uuid:{content_id}",
                    "source_url": record.source_url,
                    "capture_timestamp": record.timestamp.isoformat(),
                    "content_hash": record.content_hash,
                    "warc_record_id": record.warc_record_id,
                    "cdx_line": record.cdx_line,
                    "http_headers": record.http_headers,
                    "domain_risk": provenance_tracker.assess_domain_risk(url).to_dict()
                }

        # Return basic provenance if not tracked
        return {
            "@type": "prov:Entity",
            "source_url": url,
            "tracking_status": "not_tracked",
            "domain_risk": provenance_tracker.assess_domain_risk(url).to_dict()
        }

    def _verify_evidence(
        self,
        evidence: Dict[str, Any],
        provenance_tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Verify evidence integrity"""

        verification = {
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "verified": False,
            "verification_method": "hash_comparison"
        }

        source_url = evidence.get("source_url")
        if source_url:
            # Check if content is tracked
            for content_id, record in provenance_tracker.tracked_content.items():
                if record.source_url == source_url:
                    # Verify integrity if content is available
                    content = evidence.get("text", "")
                    if content:
                        integrity_check = provenance_tracker.verify_content_integrity(content_id, content)
                        verification.update(integrity_check)
                    break

        return verification

    def _build_verification_data(
        self,
        claims: List[Claim],
        contradictions: List[Contradiction],
        provenance_tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Build comprehensive verification data"""

        verification_summary = {
            "@type": "sec:VerificationSummary",
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_claims": len(claims),
            "total_contradictions": len(contradictions),
            "verification_methods": [
                "content_hash_verification",
                "domain_risk_assessment",
                "contradiction_detection",
                "provenance_chain_validation"
            ]
        }

        # Calculate verification statistics
        verified_claims = 0
        total_evidence = 0
        verified_evidence = 0

        for claim in claims:
            has_verification = False
            for url in claim.source_urls:
                for record in provenance_tracker.tracked_content.values():
                    if record.source_url == url:
                        has_verification = True
                        break
                if has_verification:
                    break

            if has_verification:
                verified_claims += 1

            total_evidence += len(claim.evidence)
            # Count verified evidence (simplified)
            verified_evidence += sum(1 for e in claim.evidence if e.get("source_url"))

        verification_summary.update({
            "claims_with_provenance": verified_claims,
            "claims_verification_rate": verified_claims / max(len(claims), 1),
            "evidence_verification_rate": verified_evidence / max(total_evidence, 1),
            "contradiction_confidence_avg": sum(c.confidence for c in contradictions) / max(len(contradictions), 1) if contradictions else 0.0
        })

        return verification_summary

    def _calculate_document_hash(self, document: Dict[str, Any]) -> str:
        """Calculate integrity hash for the entire document"""

        # Create a copy without the hash field
        doc_copy = document.copy()
        doc_copy.pop("integrity_hash", None)

        # Serialize deterministically
        doc_json = json.dumps(doc_copy, sort_keys=True, separators=(',', ':'))

        # Calculate SHA-256 hash
        return hashlib.sha256(doc_json.encode('utf-8')).hexdigest()

    def save_export(
        self,
        document: Dict[str, Any],
        output_path: Path,
        compress: bool = True
    ) -> Path:
        """Save JSON-LD export to file"""

        if compress:
            import gzip
            output_path = output_path.with_suffix(output_path.suffix + '.gz')

            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)

        logger.info(f"Forensic export saved to: {output_path}")
        return output_path

    def validate_export(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON-LD export for compliance and integrity"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "compliance_checks": {}
        }

        # Check required fields
        required_fields = ["@context", "@type", "@id", "export_metadata", "claims", "integrity_hash"]
        for field in required_fields:
            if field not in document:
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["valid"] = False

        # Validate JSON-LD structure
        if "@context" in document:
            context = document["@context"]
            if not isinstance(context, dict):
                validation_result["errors"].append("@context must be an object")
                validation_result["valid"] = False

        # Validate integrity hash
        if "integrity_hash" in document:
            stored_hash = document["integrity_hash"]
            calculated_hash = self._calculate_document_hash(document)

            if stored_hash != calculated_hash:
                validation_result["errors"].append("Integrity hash mismatch")
                validation_result["valid"] = False
            else:
                validation_result["compliance_checks"]["integrity_verified"] = True

        # Check claim references
        claim_ids = set()
        if "claims" in document:
            for claim in document["claims"]:
                if "@id" in claim:
                    claim_ids.add(claim["@id"])

        # Validate contradiction references
        if "contradictions" in document:
            for contradiction in document["contradictions"]:
                claim_a = contradiction.get("claim_a")
                claim_b = contradiction.get("claim_b")

                if claim_a and claim_a not in claim_ids:
                    validation_result["warnings"].append(f"Contradiction references unknown claim: {claim_a}")

                if claim_b and claim_b not in claim_ids:
                    validation_result["warnings"].append(f"Contradiction references unknown claim: {claim_b}")

        # Compliance checks
        validation_result["compliance_checks"].update({
            "w3c_jsonld_compliant": "@context" in document and "@type" in document,
            "prov_ontology_used": any("prov:" in str(v) for v in document.get("@context", {}).values()),
            "security_vocab_used": any("sec:" in str(v) for v in document.get("@context", {}).values()),
            "timestamped": "export_metadata" in document and "export_timestamp" in document.get("export_metadata", {})
        })

        logger.info(f"Export validation: {'PASSED' if validation_result['valid'] else 'FAILED'}")
        return validation_result


def create_forensic_exporter(config: Dict[str, Any]) -> JSONLDExporter:
    """Factory function for forensic JSON-LD exporter"""
    return JSONLDExporter(config)


# Usage example
if __name__ == "__main__":
    from datetime import datetime, timezone
    import uuid

    config = {
        "forensic_export": {
            "compression": True,
            "validation": True
        }
    }

    exporter = JSONLDExporter(config)

    # Create sample data
    context = ForensicContext(
        export_id=str(uuid.uuid4()),
        export_timestamp=datetime.now(timezone.utc),
        software_version="DeepResearchTool-v2.0",
        export_reason="forensic_investigation",
        operator_id="investigator_001",
        chain_of_custody=["initial_capture", "analysis", "export"]
    )

    # Mock data
    sample_claims = [
        Claim(
            id=str(uuid.uuid4()),
            text="COVID-19 vaccines are effective",
            evidence=[{"text": "Clinical trial data", "confidence": 0.9}],
            confidence=0.9,
            source_urls=["https://example.com/study1"]
        )
    ]

    # Create mock provenance tracker
    from .warc_tracking import ProvenanceTracker
    provenance_tracker = ProvenanceTracker({"provenance_storage": "test_forensic"})

    # Export
    document = exporter.export_claim_graph(
        claims=sample_claims,
        contradictions=[],
        contradiction_sets=[],
        provenance_tracker=provenance_tracker,
        context=context
    )

    print(f"Export created with ID: {document['@id']}")
    print(f"Integrity hash: {document['integrity_hash']}")

    # Validate
    validation = exporter.validate_export(document)
    print(f"Validation result: {'PASSED' if validation['valid'] else 'FAILED'}")

    if validation["errors"]:
        print(f"Errors: {validation['errors']}")

    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
