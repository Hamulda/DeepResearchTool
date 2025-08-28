#!/usr/bin/env python3
"""
WARC/CDX Provenance Tracking
Comprehensive tracking of data origins with timestamps, hashes, and forensic capabilities

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
import gzip
import base64

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a piece of content"""
    content_id: str
    source_url: str
    timestamp: datetime
    content_hash: str
    warc_record_id: Optional[str] = None
    cdx_line: Optional[str] = None
    http_headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "source_url": self.source_url,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            "warc_record_id": self.warc_record_id,
            "cdx_line": self.cdx_line,
            "http_headers": self.http_headers,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceRecord':
        return cls(
            content_id=data["content_id"],
            source_url=data["source_url"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content_hash=data["content_hash"],
            warc_record_id=data.get("warc_record_id"),
            cdx_line=data.get("cdx_line"),
            http_headers=data.get("http_headers", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class DomainRiskProfile:
    """Risk assessment for a domain"""
    domain: str
    risk_score: float  # 0.0 (low risk) to 1.0 (high risk)
    risk_factors: List[str]
    last_updated: datetime
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "last_updated": self.last_updated.isoformat(),
            "confidence": self.confidence
        }


class WARCRecordGenerator:
    """Generate WARC records for captured content"""

    def __init__(self):
        self.warc_version = "WARC/1.0"

    def create_warc_record(
        self,
        url: str,
        content: str,
        headers: Dict[str, str],
        timestamp: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """Create a WARC record and return (record_id, warc_content)"""

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Generate unique record ID
        record_id = self._generate_record_id(url, timestamp)

        # Create WARC headers
        warc_headers = {
            "WARC-Type": "response",
            "WARC-Target-URI": url,
            "WARC-Date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "WARC-Record-ID": f"<urn:uuid:{record_id}>",
            "Content-Type": "application/http; msgtype=response",
            "Content-Length": str(len(content))
        }

        # Build WARC record
        warc_content = f"{self.warc_version}\r\n"
        for key, value in warc_headers.items():
            warc_content += f"{key}: {value}\r\n"
        warc_content += "\r\n"

        # Add HTTP response simulation
        http_response = self._create_http_response(content, headers)
        warc_content += http_response
        warc_content += "\r\n\r\n"

        return record_id, warc_content

    def _generate_record_id(self, url: str, timestamp: datetime) -> str:
        """Generate unique record ID"""
        combined = f"{url}_{timestamp.isoformat()}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _create_http_response(self, content: str, headers: Dict[str, str]) -> str:
        """Create HTTP response portion of WARC record"""
        response = "HTTP/1.1 200 OK\r\n"

        # Add headers
        for key, value in headers.items():
            response += f"{key}: {value}\r\n"

        response += f"Content-Length: {len(content.encode('utf-8'))}\r\n"
        response += "\r\n"
        response += content

        return response


class CDXGenerator:
    """Generate CDX (Capture inDeX) entries for content"""

    def create_cdx_line(
        self,
        url: str,
        timestamp: datetime,
        mime_type: str,
        status_code: int,
        content_hash: str,
        content_length: int,
        warc_file: str,
        warc_offset: int
    ) -> str:
        """Create a CDX line in standard format"""

        # Parse URL components
        parsed = urlparse(url)

        # Canonicalize URL
        canonical_url = self._canonicalize_url(url)

        # Format timestamp
        timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")

        # CDX fields: urlkey timestamp original mimetype statuscode digest length offset filename
        cdx_fields = [
            canonical_url,
            timestamp_str,
            url,
            mime_type,
            str(status_code),
            content_hash,
            str(content_length),
            str(warc_offset),
            warc_file
        ]

        return " ".join(cdx_fields)

    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for CDX format"""
        parsed = urlparse(url.lower())

        # Remove common prefixes
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]

        # Reverse domain for sorting
        domain_parts = domain.split(".")
        reversed_domain = ",".join(reversed(domain_parts))

        # Combine with path
        path = parsed.path or "/"
        canonical = f"{reversed_domain}){path}"

        if parsed.query:
            canonical += f"?{parsed.query}"

        return canonical


class ProvenanceTracker:
    """Main provenance tracking system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provenance_config = config.get("provenance", {})

        # Storage paths
        self.storage_dir = Path(config.get("provenance_storage", "provenance_data"))
        self.storage_dir.mkdir(exist_ok=True)

        self.warc_dir = self.storage_dir / "warc"
        self.cdx_dir = self.storage_dir / "cdx"
        self.records_file = self.storage_dir / "provenance_records.jsonl"

        self.warc_dir.mkdir(exist_ok=True)
        self.cdx_dir.mkdir(exist_ok=True)

        # Generators
        self.warc_generator = WARCRecordGenerator()
        self.cdx_generator = CDXGenerator()

        # Domain risk assessment
        self.domain_risks: Dict[str, DomainRiskProfile] = {}
        self._load_domain_risks()

        # Content tracking
        self.tracked_content: Dict[str, ProvenanceRecord] = {}

        logger.info(f"Provenance tracker initialized (storage: {self.storage_dir})")

    def track_content(
        self,
        content: str,
        source_url: str,
        headers: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> ProvenanceRecord:
        """Track content with full provenance information"""

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if headers is None:
            headers = {}

        if metadata is None:
            metadata = {}

        # Generate content hash
        content_hash = self._generate_content_hash(content)
        content_id = self._generate_content_id(source_url, timestamp)

        # Create WARC record
        warc_record_id, warc_content = self.warc_generator.create_warc_record(
            source_url, content, headers, timestamp
        )

        # Save WARC file
        warc_filename = f"{content_id}.warc.gz"
        warc_path = self.warc_dir / warc_filename

        with gzip.open(warc_path, 'wt', encoding='utf-8') as f:
            f.write(warc_content)

        # Create CDX entry
        cdx_line = self.cdx_generator.create_cdx_line(
            url=source_url,
            timestamp=timestamp,
            mime_type=headers.get("Content-Type", "text/html"),
            status_code=200,
            content_hash=content_hash,
            content_length=len(content.encode('utf-8')),
            warc_file=warc_filename,
            warc_offset=0
        )

        # Save CDX entry
        cdx_filename = f"{datetime.now().strftime('%Y%m%d')}.cdx"
        cdx_path = self.cdx_dir / cdx_filename

        with open(cdx_path, 'a', encoding='utf-8') as f:
            f.write(cdx_line + "\n")

        # Create provenance record
        provenance_record = ProvenanceRecord(
            content_id=content_id,
            source_url=source_url,
            timestamp=timestamp,
            content_hash=content_hash,
            warc_record_id=warc_record_id,
            cdx_line=cdx_line,
            http_headers=headers,
            metadata=metadata
        )

        # Store record
        self._store_provenance_record(provenance_record)
        self.tracked_content[content_id] = provenance_record

        # Update domain risk assessment
        self._update_domain_risk(source_url)

        logger.info(f"Content tracked: {content_id} from {source_url}")
        return provenance_record

    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _generate_content_id(self, source_url: str, timestamp: datetime) -> str:
        """Generate unique content ID"""
        combined = f"{source_url}_{timestamp.isoformat()}"
        hash_part = hashlib.md5(combined.encode()).hexdigest()[:8]
        timestamp_part = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp_part}_{hash_part}"

    def _store_provenance_record(self, record: ProvenanceRecord):
        """Store provenance record to JSONL file"""
        with open(self.records_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def get_provenance_record(self, content_id: str) -> Optional[ProvenanceRecord]:
        """Retrieve provenance record by content ID"""
        if content_id in self.tracked_content:
            return self.tracked_content[content_id]

        # Search in stored records
        return self._search_stored_records(content_id)

    def _search_stored_records(self, content_id: str) -> Optional[ProvenanceRecord]:
        """Search for record in stored files"""
        try:
            with open(self.records_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record_data = json.loads(line.strip())
                    if record_data["content_id"] == content_id:
                        return ProvenanceRecord.from_dict(record_data)
        except FileNotFoundError:
            pass

        return None

    def verify_content_integrity(self, content_id: str, current_content: str) -> Dict[str, Any]:
        """Verify content integrity against stored hash"""

        record = self.get_provenance_record(content_id)
        if not record:
            return {
                "verified": False,
                "reason": "No provenance record found",
                "content_id": content_id
            }

        current_hash = self._generate_content_hash(current_content)

        return {
            "verified": current_hash == record.content_hash,
            "stored_hash": record.content_hash,
            "current_hash": current_hash,
            "timestamp": record.timestamp.isoformat(),
            "source_url": record.source_url,
            "content_id": content_id
        }

    def assess_domain_risk(self, url: str) -> DomainRiskProfile:
        """Assess risk profile for a domain"""

        domain = self._extract_domain(url)

        if domain in self.domain_risks:
            profile = self.domain_risks[domain]

            # Update if profile is old
            if (datetime.now() - profile.last_updated).days > 7:
                profile = self._calculate_domain_risk(domain)
                self.domain_risks[domain] = profile

            return profile
        else:
            profile = self._calculate_domain_risk(domain)
            self.domain_risks[domain] = profile
            return profile

    def _calculate_domain_risk(self, domain: str) -> DomainRiskProfile:
        """Calculate risk score for a domain"""

        risk_score = 0.0
        risk_factors = []

        # High-trust domains
        trusted_domains = {
            "arxiv.org": 0.1,
            "pubmed.ncbi.nlm.nih.gov": 0.1,
            "nature.com": 0.1,
            "science.org": 0.1,
            "who.int": 0.1,
            "cdc.gov": 0.1,
            "fda.gov": 0.1
        }

        if domain in trusted_domains:
            risk_score = trusted_domains[domain]
            risk_factors.append("verified_trusted_source")
        else:
            # TLD-based risk assessment
            if domain.endswith(('.gov', '.edu')):
                risk_score = 0.2
                risk_factors.append("government_education_domain")
            elif domain.endswith('.org'):
                risk_score = 0.3
                risk_factors.append("organization_domain")
            elif domain.endswith(('.com', '.net')):
                risk_score = 0.5
                risk_factors.append("commercial_domain")
            elif domain.endswith(('.tk', '.ml', '.ga', '.cf')):
                risk_score = 0.8
                risk_factors.append("free_domain_service")
            else:
                risk_score = 0.6
                risk_factors.append("unknown_tld")

            # Additional risk factors
            if len(domain.split('.')) > 3:
                risk_score += 0.1
                risk_factors.append("deep_subdomain")

            if any(suspicious in domain.lower() for suspicious in ['temp', 'test', 'fake', 'spam']):
                risk_score += 0.3
                risk_factors.append("suspicious_keywords")

            # Age and reputation (simplified)
            if domain.count('.') > 2:
                risk_score += 0.1
                risk_factors.append("complex_domain_structure")

        risk_score = min(risk_score, 1.0)
        confidence = 0.8 if domain in trusted_domains else 0.6

        return DomainRiskProfile(
            domain=domain,
            risk_score=risk_score,
            risk_factors=risk_factors,
            last_updated=datetime.now(),
            confidence=confidence
        )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""

    def _update_domain_risk(self, url: str):
        """Update domain risk assessment"""
        domain = self._extract_domain(url)
        if domain:
            self.assess_domain_risk(url)  # This will update the cache

    def _load_domain_risks(self):
        """Load previously calculated domain risks"""
        risk_file = self.storage_dir / "domain_risks.json"

        try:
            if risk_file.exists():
                with open(risk_file, 'r', encoding='utf-8') as f:
                    risk_data = json.load(f)

                for domain, data in risk_data.items():
                    self.domain_risks[domain] = DomainRiskProfile(
                        domain=domain,
                        risk_score=data["risk_score"],
                        risk_factors=data["risk_factors"],
                        last_updated=datetime.fromisoformat(data["last_updated"]),
                        confidence=data["confidence"]
                    )

                logger.info(f"Loaded {len(self.domain_risks)} domain risk profiles")
        except Exception as e:
            logger.warning(f"Failed to load domain risks: {e}")

    def save_domain_risks(self):
        """Save domain risk assessments to file"""
        risk_file = self.storage_dir / "domain_risks.json"

        risk_data = {}
        for domain, profile in self.domain_risks.items():
            risk_data[domain] = profile.to_dict()

        with open(risk_file, 'w', encoding='utf-8') as f:
            json.dump(risk_data, f, indent=2)

        logger.info(f"Saved {len(self.domain_risks)} domain risk profiles")

    def export_forensic_data(self, content_ids: List[str]) -> Dict[str, Any]:
        """Export forensic data for specified content IDs"""

        forensic_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_records": [],
            "domain_risks": {},
            "verification_chain": []
        }

        for content_id in content_ids:
            record = self.get_provenance_record(content_id)
            if record:
                # Add provenance record
                forensic_data["content_records"].append(record.to_dict())

                # Add domain risk assessment
                domain = self._extract_domain(record.source_url)
                if domain and domain not in forensic_data["domain_risks"]:
                    risk_profile = self.assess_domain_risk(record.source_url)
                    forensic_data["domain_risks"][domain] = risk_profile.to_dict()

                # Add verification chain
                verification = self.verify_content_integrity(content_id, "")  # Content not available here
                forensic_data["verification_chain"].append(verification)

        return forensic_data

    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked content"""

        total_records = len(self.tracked_content)

        # Analyze by domain
        domain_counts = {}
        for record in self.tracked_content.values():
            domain = self._extract_domain(record.source_url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Risk distribution
        risk_distribution = {"low": 0, "medium": 0, "high": 0}
        for domain in domain_counts:
            if domain in self.domain_risks:
                risk_score = self.domain_risks[domain].risk_score
                if risk_score < 0.3:
                    risk_distribution["low"] += domain_counts[domain]
                elif risk_score < 0.7:
                    risk_distribution["medium"] += domain_counts[domain]
                else:
                    risk_distribution["high"] += domain_counts[domain]

        return {
            "total_tracked_content": total_records,
            "unique_domains": len(domain_counts),
            "domain_distribution": dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "risk_distribution": risk_distribution,
            "storage_location": str(self.storage_dir),
            "warc_files": len(list(self.warc_dir.glob("*.warc.gz"))),
            "cdx_files": len(list(self.cdx_dir.glob("*.cdx")))
        }


def create_provenance_tracker(config: Dict[str, Any]) -> ProvenanceTracker:
    """Factory function for provenance tracker"""
    return ProvenanceTracker(config)


# Usage example
if __name__ == "__main__":
    config = {
        "provenance_storage": "test_provenance",
        "provenance": {
            "enabled": True,
            "track_all_content": True
        }
    }

    tracker = ProvenanceTracker(config)

    # Test content tracking
    test_content = "This is test content about COVID-19 vaccine effectiveness."
    test_url = "https://example.com/article123"
    test_headers = {"Content-Type": "text/html", "Server": "nginx"}

    record = tracker.track_content(test_content, test_url, test_headers)
    print(f"Tracked content: {record.content_id}")

    # Test verification
    verification = tracker.verify_content_integrity(record.content_id, test_content)
    print(f"Verification: {verification['verified']}")

    # Test domain risk assessment
    risk_profile = tracker.assess_domain_risk(test_url)
    print(f"Domain risk: {risk_profile.risk_score:.2f} ({risk_profile.risk_factors})")

    # Export forensic data
    forensic_data = tracker.export_forensic_data([record.content_id])
    print(f"Forensic export: {len(forensic_data['content_records'])} records")

    stats = tracker.get_tracking_stats()
    print(f"Tracking stats: {stats}")

    # Save domain risks
    tracker.save_domain_risks()
