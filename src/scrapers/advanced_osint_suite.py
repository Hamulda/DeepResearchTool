#!/usr/bin/env python3
"""Advanced OSINT Tools Suite for Deep Research Tool
Professional-grade open source intelligence gathering

Author: Advanced IT Specialist
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
import socket
import time
from typing import Any

import dns.resolver
import whois

logger = logging.getLogger(__name__)


@dataclass
class OSINTTarget:
    """Enhanced OSINT investigation target"""

    target_type: str  # 'person', 'organization', 'domain', 'email', 'phone', 'ip_address'
    target_value: str
    confidence: float = 0.5
    sources_found: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    risk_assessment: str = "unknown"
    verification_status: str = "unverified"


@dataclass
class OSINTIntelligence:
    """Comprehensive OSINT intelligence data"""

    target: OSINTTarget
    personal_data: dict[str, Any] = field(default_factory=dict)
    professional_data: dict[str, Any] = field(default_factory=dict)
    digital_footprint: dict[str, Any] = field(default_factory=dict)
    network_analysis: dict[str, Any] = field(default_factory=dict)
    threat_indicators: list[str] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    correlation_score: float = 0.0
    collection_timestamp: datetime = field(default_factory=datetime.now)


class AdvancedOSINTCollector:
    """Professional OSINT collector with multiple data sources"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Professional OSINT sources
        self.osint_sources = {
            "social_media": {
                "twitter_api": "https://api.twitter.com/2",
                "linkedin_search": "https://www.linkedin.com/search/results/people/",
                "facebook_graph": "https://graph.facebook.com",
                "instagram_basic": "https://graph.instagram.com",
                "reddit_api": "https://www.reddit.com/api",
            },
            "public_records": {
                "whitepages": "https://www.whitepages.com/name/",
                "spokeo": "https://www.spokeo.com/search",
                "pipl": "https://pipl.com/search",
                "intelius": "https://www.intelius.com/people-search",
                "truthfinder": "https://www.truthfinder.com/search",
            },
            "professional": {
                "crunchbase": "https://www.crunchbase.com/discover/organization.companies/",
                "bloomberg": "https://www.bloomberg.com/search",
                "sec_edgar": "https://www.sec.gov/edgar/search/",
                "zoominfo": "https://www.zoominfo.com/s/",
                "apollo": "https://www.apollo.io/search",
            },
            "domain_intelligence": {
                "whois_lookup": True,
                "dns_enumeration": True,
                "subdomain_discovery": True,
                "certificate_transparency": "https://crt.sh/?q=",
                "shodan_api": "https://api.shodan.io",
                "censys_api": "https://search.censys.io/api",
            },
            "threat_intelligence": {
                "virustotal": "https://www.virustotal.com/api/v3",
                "alienvault_otx": "https://otx.alienvault.com/api/v1",
                "urlvoid": "https://www.urlvoid.com/api1280",
                "hybrid_analysis": "https://www.hybrid-analysis.com/api/v2",
            },
        }

        # Entity recognition patterns
        self.entity_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "bitcoin_address": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            "social_security": r"\b\d{3}-\d{2}-\d{4}\b",
            "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
            "username": r"@[a-zA-Z0-9_]+",
            "url": r'https?://[^\s<>"]+',
            "mac_address": r"\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b",
        }

        # Correlation weights for intelligence assessment
        self.correlation_weights = {
            "exact_match": 1.0,
            "partial_match": 0.7,
            "related_entity": 0.5,
            "same_source": 0.3,
            "temporal_proximity": 0.4,
            "location_match": 0.6,
            "cross_platform": 0.8,
            "verified_account": 0.9,
        }

    async def conduct_comprehensive_osint(
        self, target: str, target_type: str = "auto"
    ) -> OSINTIntelligence:
        """Conduct comprehensive OSINT investigation"""
        await self._rate_limit()

        # Auto-detect target type if not specified
        if target_type == "auto":
            target_type = self._detect_target_type(target)

        osint_target = OSINTTarget(target_type=target_type, target_value=target, confidence=0.8)

        logger.info(f"Starting comprehensive OSINT on {target_type}: {target}")

        # Collect intelligence from multiple sources
        intelligence = OSINTIntelligence(target=osint_target)

        # Phase 1: Basic data collection
        await self._collect_basic_intelligence(intelligence)

        # Phase 2: Social media intelligence
        await self._collect_social_media_intelligence(intelligence)

        # Phase 3: Professional intelligence
        await self._collect_professional_intelligence(intelligence)

        # Phase 4: Technical intelligence (for domains/IPs)
        if target_type in ["domain", "ip_address"]:
            await self._collect_technical_intelligence(intelligence)

        # Phase 5: Threat intelligence assessment
        await self._assess_threat_indicators(intelligence)

        # Phase 6: Correlation analysis
        intelligence.correlation_score = self._calculate_intelligence_correlation(intelligence)

        # Phase 7: Risk assessment
        intelligence.target.risk_assessment = self._assess_target_risk(intelligence)

        logger.info(
            f"OSINT investigation completed. Correlation score: {intelligence.correlation_score:.2f}"
        )

        return intelligence

    def _detect_target_type(self, target: str) -> str:
        """Enhanced target type detection"""
        target_clean = target.lower().strip()

        # Check against patterns
        for entity_type, pattern in self.entity_patterns.items():
            if re.match(pattern, target):
                return entity_type

        # Heuristic detection
        if "@" in target and "." in target:
            return "email"
        if target.replace("-", "").replace(" ", "").replace("(", "").replace(")", "").isdigit():
            return "phone"
        if "." in target and " " not in target and len(target.split(".")) > 1:
            try:
                socket.gethostbyname(target)
                return "domain"
            except:
                pass
        elif len(target.split()) >= 2 and target.replace(" ", "").isalpha():
            return "person"
        elif target.isupper() or (
            "inc" in target.lower() or "llc" in target.lower() or "corp" in target.lower()
        ):
            return "organization"

        return "unknown"

    async def _collect_basic_intelligence(self, intelligence: OSINTIntelligence):
        """Collect basic intelligence about target"""
        target = intelligence.target

        try:
            if target.target_type == "person":
                await self._collect_person_basic_data(intelligence)
            elif target.target_type == "organization":
                await self._collect_organization_basic_data(intelligence)
            elif target.target_type == "email":
                await self._collect_email_basic_data(intelligence)
            elif target.target_type == "domain":
                await self._collect_domain_basic_data(intelligence)
            elif target.target_type == "ip_address":
                await self._collect_ip_basic_data(intelligence)

        except Exception as e:
            logger.error(f"Error collecting basic intelligence: {e!s}")

    async def _collect_person_basic_data(self, intelligence: OSINTIntelligence):
        """Collect basic data about a person"""
        person_name = intelligence.target.target_value

        # Search public records
        public_records = await self._search_public_records(person_name)
        intelligence.personal_data["public_records"] = public_records

        # Search professional networks
        professional_data = await self._search_professional_networks(person_name)
        intelligence.professional_data.update(professional_data)

        # Extract related entities
        intelligence.target.related_entities.extend(
            self._extract_related_entities_from_data(public_records)
        )

    async def _collect_organization_basic_data(self, intelligence: OSINTIntelligence):
        """Collect basic data about an organization"""
        org_name = intelligence.target.target_value

        # Search business databases
        business_data = await self._search_business_databases(org_name)
        intelligence.professional_data["business_records"] = business_data

        # Search regulatory filings
        regulatory_data = await self._search_regulatory_filings(org_name)
        intelligence.professional_data["regulatory_filings"] = regulatory_data

    async def _collect_email_basic_data(self, intelligence: OSINTIntelligence):
        """Collect basic data about an email address"""
        email = intelligence.target.target_value

        # Extract domain for analysis
        domain = email.split("@")[1] if "@" in email else ""

        # Email validation and analysis
        email_analysis = await self._analyze_email_address(email)
        intelligence.personal_data["email_analysis"] = email_analysis

        # Domain analysis
        if domain:
            domain_data = await self._analyze_domain(domain)
            intelligence.network_analysis["email_domain"] = domain_data

    async def _collect_domain_basic_data(self, intelligence: OSINTIntelligence):
        """Collect basic data about a domain"""
        domain = intelligence.target.target_value

        # WHOIS lookup
        whois_data = await self._perform_whois_lookup(domain)
        intelligence.network_analysis["whois"] = whois_data

        # DNS analysis
        dns_data = await self._perform_dns_analysis(domain)
        intelligence.network_analysis["dns"] = dns_data

        # Subdomain enumeration
        subdomains = await self._enumerate_subdomains(domain)
        intelligence.network_analysis["subdomains"] = subdomains

    async def _collect_ip_basic_data(self, intelligence: OSINTIntelligence):
        """Collect basic data about an IP address"""
        ip_address = intelligence.target.target_value

        # Geolocation lookup
        geolocation = await self._lookup_ip_geolocation(ip_address)
        intelligence.network_analysis["geolocation"] = geolocation

        # Port scanning (limited scope for security)
        open_ports = await self._scan_common_ports(ip_address)
        intelligence.network_analysis["open_ports"] = open_ports

        # Reverse DNS lookup
        reverse_dns = await self._reverse_dns_lookup(ip_address)
        intelligence.network_analysis["reverse_dns"] = reverse_dns

    async def _collect_social_media_intelligence(self, intelligence: OSINTIntelligence):
        """Collect social media intelligence"""
        target_value = intelligence.target.target_value

        try:
            # Search across multiple platforms
            social_data = {}

            # Twitter/X search
            twitter_data = await self._search_twitter(target_value)
            if twitter_data:
                social_data["twitter"] = twitter_data

            # LinkedIn search
            linkedin_data = await self._search_linkedin(target_value)
            if linkedin_data:
                social_data["linkedin"] = linkedin_data

            # Reddit search
            reddit_data = await self._search_reddit(target_value)
            if reddit_data:
                social_data["reddit"] = reddit_data

            # GitHub search (for technical targets)
            github_data = await self._search_github(target_value)
            if github_data:
                social_data["github"] = github_data

            intelligence.digital_footprint["social_media"] = social_data

            # Build timeline from social media activity
            self._build_social_media_timeline(intelligence, social_data)

        except Exception as e:
            logger.error(f"Error collecting social media intelligence: {e!s}")

    async def _collect_professional_intelligence(self, intelligence: OSINTIntelligence):
        """Collect professional intelligence"""
        target_value = intelligence.target.target_value

        try:
            professional_data = {}

            # Crunchbase search for business information
            crunchbase_data = await self._search_crunchbase(target_value)
            if crunchbase_data:
                professional_data["crunchbase"] = crunchbase_data

            # SEC filings search
            sec_data = await self._search_sec_filings(target_value)
            if sec_data:
                professional_data["sec_filings"] = sec_data

            # Professional licensing databases
            licensing_data = await self._search_professional_licenses(target_value)
            if licensing_data:
                professional_data["licenses"] = licensing_data

            intelligence.professional_data.update(professional_data)

        except Exception as e:
            logger.error(f"Error collecting professional intelligence: {e!s}")

    async def _collect_technical_intelligence(self, intelligence: OSINTIntelligence):
        """Collect technical intelligence for domains and IPs"""
        try:
            # Certificate transparency logs
            cert_data = await self._search_certificate_transparency(
                intelligence.target.target_value
            )
            intelligence.network_analysis["certificates"] = cert_data

            # Shodan search (if API available)
            shodan_data = await self._search_shodan(intelligence.target.target_value)
            if shodan_data:
                intelligence.network_analysis["shodan"] = shodan_data

            # Technology stack detection
            tech_stack = await self._detect_technology_stack(intelligence.target.target_value)
            intelligence.network_analysis["technology_stack"] = tech_stack

        except Exception as e:
            logger.error(f"Error collecting technical intelligence: {e!s}")

    async def _assess_threat_indicators(self, intelligence: OSINTIntelligence):
        """Assess threat indicators"""
        try:
            threat_indicators = []

            # Check against threat intelligence feeds
            if intelligence.target.target_type in ["domain", "ip_address", "email"]:

                # VirusTotal check
                vt_result = await self._check_virustotal(intelligence.target.target_value)
                if vt_result and vt_result.get("malicious_count", 0) > 0:
                    threat_indicators.append(
                        f"Flagged by {vt_result['malicious_count']} security vendors"
                    )

                # URLVoid check
                urlvoid_result = await self._check_urlvoid(intelligence.target.target_value)
                if urlvoid_result and urlvoid_result.get("suspicious"):
                    threat_indicators.append("Suspicious activity detected")

                # Check for known bad indicators
                bad_indicators = await self._check_bad_indicators(intelligence.target.target_value)
                threat_indicators.extend(bad_indicators)

            intelligence.threat_indicators = threat_indicators

        except Exception as e:
            logger.error(f"Error assessing threat indicators: {e!s}")

    # Specific search implementations
    async def _search_public_records(self, name: str) -> dict[str, Any]:
        """Search public records databases"""
        # Implementation would search multiple public record sources
        # For demo purposes, returning mock structure
        return {"found_records": 0, "sources_searched": ["whitepages", "spokeo"], "data": {}}

    async def _search_professional_networks(self, name: str) -> dict[str, Any]:
        """Search professional networks"""
        # Implementation would search LinkedIn, professional directories
        return {"linkedin_profiles": [], "professional_associations": [], "employment_history": []}

    async def _search_business_databases(self, org_name: str) -> dict[str, Any]:
        """Search business databases"""
        return {"company_records": [], "financial_data": {}, "key_personnel": []}

    async def _search_regulatory_filings(self, org_name: str) -> dict[str, Any]:
        """Search regulatory filings"""
        return {"sec_filings": [], "patents": [], "trademarks": []}

    async def _analyze_email_address(self, email: str) -> dict[str, Any]:
        """Analyze email address for patterns and indicators"""
        analysis = {
            "valid_format": bool(re.match(self.entity_patterns["email"], email)),
            "domain": email.split("@")[1] if "@" in email else "",
            "username": email.split("@")[0] if "@" in email else "",
            "disposable_email": False,  # Would check against disposable email lists
            "business_email": False,  # Would check against business domain patterns
        }

        # Additional analysis could include:
        # - Breach database checks
        # - Social media account discovery
        # - Professional network presence

        return analysis

    async def _analyze_domain(self, domain: str) -> dict[str, Any]:
        """Analyze domain characteristics"""
        analysis = {
            "registrar": "",
            "creation_date": None,
            "expiration_date": None,
            "name_servers": [],
            "reputation_score": 0.5,
        }

        # Implementation would include full domain analysis
        return analysis

    async def _perform_whois_lookup(self, domain: str) -> dict[str, Any]:
        """Perform WHOIS lookup"""
        try:
            w = whois.whois(domain)
            return {
                "registrar": w.registrar,
                "creation_date": w.creation_date,
                "expiration_date": w.expiration_date,
                "name_servers": w.name_servers,
                "organization": w.org,
                "country": w.country,
            }
        except Exception as e:
            logger.error(f"WHOIS lookup failed: {e!s}")
            return {}

    async def _perform_dns_analysis(self, domain: str) -> dict[str, Any]:
        """Perform DNS analysis"""
        dns_data = {}

        try:
            # A records
            a_records = [str(rdata) for rdata in dns.resolver.resolve(domain, "A")]
            dns_data["a_records"] = a_records

            # MX records
            try:
                mx_records = [str(rdata) for rdata in dns.resolver.resolve(domain, "MX")]
                dns_data["mx_records"] = mx_records
            except:
                dns_data["mx_records"] = []

            # NS records
            try:
                ns_records = [str(rdata) for rdata in dns.resolver.resolve(domain, "NS")]
                dns_data["ns_records"] = ns_records
            except:
                dns_data["ns_records"] = []

        except Exception as e:
            logger.error(f"DNS analysis failed: {e!s}")

        return dns_data

    async def _enumerate_subdomains(self, domain: str) -> list[str]:
        """Enumerate subdomains"""
        # Common subdomain list for basic enumeration
        common_subdomains = [
            "www",
            "mail",
            "ftp",
            "admin",
            "api",
            "dev",
            "test",
            "staging",
            "blog",
            "shop",
            "support",
            "help",
            "docs",
            "cdn",
            "app",
        ]

        found_subdomains = []

        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"
            try:
                socket.gethostbyname(full_domain)
                found_subdomains.append(full_domain)
            except:
                continue

        return found_subdomains

    async def _lookup_ip_geolocation(self, ip_address: str) -> dict[str, Any]:
        """Lookup IP geolocation"""
        # Implementation would use geolocation API
        return {
            "country": "Unknown",
            "city": "Unknown",
            "isp": "Unknown",
            "organization": "Unknown",
        }

    async def _scan_common_ports(self, ip_address: str) -> list[int]:
        """Scan common ports (limited for security)"""
        common_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        open_ports = []

        # Note: This is a very basic implementation
        # Production version would use proper tools like nmap-python
        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((ip_address, port))
                if result == 0:
                    open_ports.append(port)
                sock.close()
            except:
                continue

        return open_ports

    async def _reverse_dns_lookup(self, ip_address: str) -> str:
        """Perform reverse DNS lookup"""
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except:
            return ""

    # Social media search methods (mock implementations)
    async def _search_twitter(self, target: str) -> dict[str, Any]:
        """Search Twitter/X for target"""
        return {"profiles": [], "mentions": [], "activity_score": 0}

    async def _search_linkedin(self, target: str) -> dict[str, Any]:
        """Search LinkedIn for target"""
        return {"profiles": [], "companies": [], "connections": 0}

    async def _search_reddit(self, target: str) -> dict[str, Any]:
        """Search Reddit for target"""
        return {"posts": [], "comments": [], "subreddits": []}

    async def _search_github(self, target: str) -> dict[str, Any]:
        """Search GitHub for target"""
        return {"repositories": [], "commits": [], "activity": []}

    async def _search_crunchbase(self, target: str) -> dict[str, Any]:
        """Search Crunchbase for business information"""
        return {"companies": [], "people": [], "investments": []}

    async def _search_sec_filings(self, target: str) -> dict[str, Any]:
        """Search SEC filings"""
        return {"filings": [], "companies": []}

    async def _search_professional_licenses(self, target: str) -> dict[str, Any]:
        """Search professional licensing databases"""
        return {"licenses": [], "certifications": []}

    async def _search_certificate_transparency(self, domain: str) -> dict[str, Any]:
        """Search certificate transparency logs"""
        return {"certificates": [], "issuers": []}

    async def _search_shodan(self, target: str) -> dict[str, Any]:
        """Search Shodan for target"""
        return {"services": [], "vulnerabilities": [], "locations": []}

    async def _detect_technology_stack(self, domain: str) -> dict[str, Any]:
        """Detect technology stack"""
        return {"web_server": "", "cms": "", "frameworks": [], "languages": []}

    async def _check_virustotal(self, target: str) -> dict[str, Any]:
        """Check VirusTotal for threat indicators"""
        return {"malicious_count": 0, "suspicious_count": 0, "clean_count": 0}

    async def _check_urlvoid(self, target: str) -> dict[str, Any]:
        """Check URLVoid for suspicious activity"""
        return {"suspicious": False, "risk_score": 0}

    async def _check_bad_indicators(self, target: str) -> list[str]:
        """Check against known bad indicator lists"""
        return []

    def _extract_related_entities_from_data(self, data: dict[str, Any]) -> list[str]:
        """Extract related entities from collected data"""
        entities = []

        # Extract entities using regex patterns
        data_str = str(data)
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, data_str)
            entities.extend([f"{match}:{entity_type}" for match in matches])

        return list(set(entities))

    def _build_social_media_timeline(
        self, intelligence: OSINTIntelligence, social_data: dict[str, Any]
    ):
        """Build timeline from social media activity"""
        timeline_events = []

        for platform, data in social_data.items():
            # Extract timeline events from platform data
            # Implementation would parse actual social media data
            pass

        intelligence.timeline = sorted(
            timeline_events, key=lambda x: x.get("timestamp", datetime.min)
        )

    def _calculate_intelligence_correlation(self, intelligence: OSINTIntelligence) -> float:
        """Calculate correlation score for intelligence data"""
        score = 0.5  # Base score

        # Factor in data source diversity
        sources_count = len(intelligence.target.sources_found)
        score += min(0.3, sources_count * 0.05)

        # Factor in cross-platform matches
        if intelligence.digital_footprint.get("social_media"):
            platforms = len(intelligence.digital_footprint["social_media"])
            score += min(0.2, platforms * 0.04)

        # Factor in professional data
        if intelligence.professional_data:
            score += 0.1

        # Factor in threat indicators (negative correlation)
        if intelligence.threat_indicators:
            score -= min(0.2, len(intelligence.threat_indicators) * 0.05)

        return min(1.0, max(0.0, score))

    def _assess_target_risk(self, intelligence: OSINTIntelligence) -> str:
        """Assess risk level of target"""
        threat_count = len(intelligence.threat_indicators)
        correlation_score = intelligence.correlation_score

        if threat_count > 3 or correlation_score < 0.3:
            return "high"
        if threat_count > 1 or correlation_score < 0.6:
            return "medium"
        return "low"

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()


class OSINTAnalyzer:
    """Analyzer for OSINT intelligence data"""

    def __init__(self):
        self.risk_thresholds = {
            "high_risk_indicators": 5,
            "medium_risk_indicators": 2,
            "correlation_threshold": 0.7,
        }

    def analyze_osint_collection(
        self, intelligence_reports: list[OSINTIntelligence]
    ) -> dict[str, Any]:
        """Analyze collection of OSINT intelligence reports"""
        analysis = {
            "total_targets": len(intelligence_reports),
            "target_types": {},
            "risk_distribution": {"low": 0, "medium": 0, "high": 0},
            "correlation_analysis": {},
            "threat_landscape": {},
            "recommendations": [],
        }

        # Target type distribution
        for intel in intelligence_reports:
            target_type = intel.target.target_type
            analysis["target_types"][target_type] = analysis["target_types"].get(target_type, 0) + 1

        # Risk distribution
        for intel in intelligence_reports:
            risk = intel.target.risk_assessment
            analysis["risk_distribution"][risk] += 1

        # Correlation analysis
        high_correlation = [i for i in intelligence_reports if i.correlation_score > 0.8]
        analysis["correlation_analysis"] = {
            "high_correlation_targets": len(high_correlation),
            "average_correlation": sum(i.correlation_score for i in intelligence_reports)
            / len(intelligence_reports),
            "correlation_distribution": self._analyze_correlation_distribution(
                intelligence_reports
            ),
        }

        # Threat landscape
        all_threats = []
        for intel in intelligence_reports:
            all_threats.extend(intel.threat_indicators)

        threat_frequency = {}
        for threat in all_threats:
            threat_frequency[threat] = threat_frequency.get(threat, 0) + 1

        analysis["threat_landscape"] = {
            "total_threats": len(all_threats),
            "unique_threats": len(set(all_threats)),
            "most_common_threats": sorted(
                threat_frequency.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

        # Generate recommendations
        analysis["recommendations"] = self._generate_osint_recommendations(
            analysis, intelligence_reports
        )

        return analysis

    def _analyze_correlation_distribution(
        self, intelligence_reports: list[OSINTIntelligence]
    ) -> dict[str, int]:
        """Analyze correlation score distribution"""
        distribution = {
            "excellent (0.8-1.0)": 0,
            "good (0.6-0.8)": 0,
            "average (0.4-0.6)": 0,
            "poor (0.0-0.4)": 0,
        }

        for intel in intelligence_reports:
            score = intel.correlation_score
            if score >= 0.8:
                distribution["excellent (0.8-1.0)"] += 1
            elif score >= 0.6:
                distribution["good (0.6-0.8)"] += 1
            elif score >= 0.4:
                distribution["average (0.4-0.6)"] += 1
            else:
                distribution["poor (0.0-0.4)"] += 1

        return distribution

    def _generate_osint_recommendations(
        self, analysis: dict[str, Any], intelligence_reports: list[OSINTIntelligence]
    ) -> list[str]:
        """Generate OSINT recommendations"""
        recommendations = []

        # Risk-based recommendations
        high_risk_count = analysis["risk_distribution"]["high"]
        if high_risk_count > 0:
            recommendations.append(
                f"Immediate attention required: {high_risk_count} high-risk targets identified"
            )

        # Correlation recommendations
        avg_correlation = analysis["correlation_analysis"]["average_correlation"]
        if avg_correlation < 0.5:
            recommendations.append(
                "Low correlation scores indicate need for additional data sources"
            )

        # Threat recommendations
        unique_threats = analysis["threat_landscape"]["unique_threats"]
        if unique_threats > 10:
            recommendations.append(
                "Multiple threat indicators detected - conduct deeper security analysis"
            )

        # Data quality recommendations
        poor_correlation = analysis["correlation_analysis"]["correlation_distribution"][
            "poor (0.0-0.4)"
        ]
        if poor_correlation > len(intelligence_reports) * 0.3:
            recommendations.append(
                "High percentage of poor correlations - review data collection methods"
            )

        return recommendations
