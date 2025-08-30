"""
Historical Web Excavator pro DeepResearchTool
Pokročilý archeologický mining engine pro historická data a zapomenuté domény.
"""

import asyncio
import json
import logging
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import dns.resolver
from pydantic import BaseModel, Field

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class ArchaeologicalFind:
    """Reprezentace archeologického nálezu z webu"""
    url: str
    domain: str
    timestamp: datetime
    content_type: str
    content_snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    confidence_score: float = 1.0


@dataclass
class DomainExcavation:
    """Výsledky archeologické expedice pro doménu"""
    domain: str
    excavation_date: datetime
    finds: List[ArchaeologicalFind] = field(default_factory=list)
    subdomain_discoveries: Set[str] = field(default_factory=set)
    historical_timeline: List[Dict[str, Any]] = field(default_factory=list)
    certificate_history: List[Dict[str, Any]] = field(default_factory=list)
    dns_archaeology: Dict[str, Any] = field(default_factory=dict)


class WaybackMachineClient:
    """Pokročilý klient pro Wayback Machine API"""

    def __init__(self):
        self.base_url = "https://web.archive.org"
        self.cdx_api = "https://web.archive.org/cdx/search/cdx"
        self.availability_api = "https://archive.org/wayback/available"

    async def get_snapshots(
        self,
        url: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Získání snapshotů z Wayback Machine"""
        cache_key = f"wayback_snapshots:{url}:{from_date}:{to_date}:{limit}"
        cached_result = await cache_get(cache_key)

        if cached_result:
            return cached_result

        params = {
            'url': url,
            'output': 'json',
            'limit': limit,
            'collapse': 'timestamp:6'  # Seskupení podle hodin
        }

        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.cdx_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 1:  # První řádek jsou hlavičky
                            headers = data[0]
                            snapshots = []

                            for row in data[1:]:
                                snapshot = dict(zip(headers, row))
                                snapshots.append(snapshot)

                            await cache_set(cache_key, snapshots, ttl=3600)
                            return snapshots

            except Exception as e:
                logger.error(f"Chyba při získávání Wayback snapshotů: {e}")

        return []

    async def get_historical_content(self, url: str, timestamp: str) -> Optional[str]:
        """Získání historického obsahu"""
        wayback_url = f"{self.base_url}/web/{timestamp}/{url}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(wayback_url) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception as e:
                logger.error(f"Chyba při načítání historického obsahu: {e}")

        return None


class DNSArchaeologist:
    """DNS archeologické nástroje pro pasivní DNS analýzu"""

    def __init__(self, securitytrails_api_key: Optional[str] = None):
        self.securitytrails_api_key = securitytrails_api_key
        self.securitytrails_base = "https://api.securitytrails.com/v1"

    async def excavate_dns_history(self, domain: str) -> Dict[str, Any]:
        """Archeologické prohledávání DNS historie"""
        cache_key = f"dns_archaeology:{domain}"
        cached_result = await cache_get(cache_key)

        if cached_result:
            return cached_result

        dns_archaeology = {
            "domain": domain,
            "historical_records": [],
            "subdomain_discoveries": set(),
            "passive_dns_data": {},
            "mx_history": [],
            "ns_history": []
        }

        # Pasivní DNS přes SecurityTrails API
        if self.securitytrails_api_key:
            dns_archaeology["passive_dns_data"] = await self._query_securitytrails(domain)

        # Základní DNS dotazy pro současné záznamy
        current_records = await self._get_current_dns_records(domain)
        dns_archaeology["current_records"] = current_records

        # Heuristické objevování subdomén
        subdomain_discoveries = await self._discover_subdomains(domain)
        dns_archaeology["subdomain_discoveries"] = subdomain_discoveries

        await cache_set(cache_key, dns_archaeology, ttl=1800)
        return dns_archaeology

    async def _query_securitytrails(self, domain: str) -> Dict[str, Any]:
        """Dotaz na SecurityTrails API pro pasivní DNS"""
        if not self.securitytrails_api_key:
            return {}

        headers = {"APIKEY": self.securitytrails_api_key}

        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                # Historie A záznamů
                url = f"{self.securitytrails_base}/history/{domain}/dns/a"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()

            except Exception as e:
                logger.error(f"SecurityTrails API chyba: {e}")

        return {}

    async def _get_current_dns_records(self, domain: str) -> Dict[str, List[str]]:
        """Získání aktuálních DNS záznamů"""
        records = {
            "A": [],
            "AAAA": [],
            "MX": [],
            "NS": [],
            "TXT": [],
            "CNAME": []
        }

        for record_type in records.keys():
            try:
                resolver = dns.resolver.Resolver()
                answers = resolver.resolve(domain, record_type)
                records[record_type] = [str(rdata) for rdata in answers]
            except Exception:
                pass  # Záznam neexistuje

        return records

    async def _discover_subdomains(self, domain: str) -> Set[str]:
        """Heuristické objevování subdomén"""
        common_subdomains = [
            "www", "mail", "ftp", "blog", "shop", "admin", "api", "cdn",
            "staging", "dev", "test", "backup", "old", "legacy", "archive",
            "support", "help", "docs", "portal", "secure", "login"
        ]

        discovered = set()

        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"
            try:
                resolver = dns.resolver.Resolver()
                resolver.resolve(full_domain, "A")
                discovered.add(full_domain)
            except Exception:
                pass

        return discovered


class CertificateTransparencyMiner:
    """Miner pro Certificate Transparency logs"""

    def __init__(self):
        self.crt_sh_api = "https://crt.sh/"

    async def excavate_certificate_history(self, domain: str) -> List[Dict[str, Any]]:
        """Archeologické prohledávání CT logů"""
        cache_key = f"ct_archaeology:{domain}"
        cached_result = await cache_get(cache_key)

        if cached_result:
            return cached_result

        certificates = []

        # Dotaz na crt.sh
        params = {
            "q": f"%.{domain}",
            "output": "json",
            "exclude": "expired"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.crt_sh_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for cert in data:
                            certificate_info = {
                                "id": cert.get("id"),
                                "logged_at": cert.get("entry_timestamp"),
                                "not_before": cert.get("not_before"),
                                "not_after": cert.get("not_after"),
                                "common_name": cert.get("common_name"),
                                "name_value": cert.get("name_value"),
                                "issuer_name": cert.get("issuer_name")
                            }
                            certificates.append(certificate_info)

            except Exception as e:
                logger.error(f"Chyba při dotazu na crt.sh: {e}")

        await cache_set(cache_key, certificates, ttl=3600)
        return certificates


class HistoricalWebExcavator:
    """
    Pokročilý archeologický web mining engine pro hluboké prohledávání
    historických dat, zapomenutých domén a legacy infrastruktury.
    """

    def __init__(
        self,
        securitytrails_api_key: Optional[str] = None,
        max_concurrent_requests: int = 10,
        request_delay: float = 1.0
    ):
        self.wayback_client = WaybackMachineClient()
        self.dns_archaeologist = DNSArchaeologist(securitytrails_api_key)
        self.ct_miner = CertificateTransparencyMiner()

        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay

        # Semafór pro omezení souběžných požadavků
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info("HistoricalWebExcavator inicializován")

    async def excavate_forgotten_domains(
        self,
        domain: str,
        depth: int = 3,
        time_range_years: int = 10
    ) -> DomainExcavation:
        """
        Hlavní metoda pro archeologické prohledávání zapomenutých domén
        """
        logger.info(f"Zahájení archeologické expedice pro doménu: {domain}")

        excavation = DomainExcavation(
            domain=domain,
            excavation_date=datetime.now()
        )

        # Paralelní archeologické procesy
        tasks = [
            self._excavate_wayback_machine(domain, time_range_years),
            self._excavate_dns_archaeology(domain),
            self._excavate_certificate_transparency(domain),
            self._discover_related_domains(domain, depth)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Zpracování výsledků
        wayback_finds, dns_data, cert_history, related_domains = results

        if not isinstance(wayback_finds, Exception):
            excavation.finds.extend(wayback_finds)

        if not isinstance(dns_data, Exception):
            excavation.dns_archaeology = dns_data
            if "subdomain_discoveries" in dns_data:
                excavation.subdomain_discoveries.update(dns_data["subdomain_discoveries"])

        if not isinstance(cert_history, Exception):
            excavation.certificate_history = cert_history

        if not isinstance(related_domains, Exception):
            excavation.subdomain_discoveries.update(related_domains)

        # Sestavení historické timeline
        excavation.historical_timeline = self._build_historical_timeline(excavation)

        logger.info(f"Archeologická expedice dokončena: {len(excavation.finds)} nálezů")
        return excavation

    async def _excavate_wayback_machine(
        self,
        domain: str,
        time_range_years: int
    ) -> List[ArchaeologicalFind]:
        """Archeologické prohledávání Wayback Machine"""
        finds = []

        # Časové rozmezí
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_range_years * 365)

        from_date = start_date.strftime("%Y%m%d")
        to_date = end_date.strftime("%Y%m%d")

        # Získání snapshotů
        snapshots = await self.wayback_client.get_snapshots(
            f"{domain}/*", from_date, to_date, limit=500
        )

        # Omezení souběžných požadavků na historický obsah
        semaphore_tasks = []

        for snapshot in snapshots[:50]:  # Omezení na 50 snapshotů
            task = self._process_wayback_snapshot(snapshot, domain)
            semaphore_tasks.append(task)

        # Zpracování s omezením souběžnosti
        processed_finds = await asyncio.gather(*semaphore_tasks, return_exceptions=True)

        for find in processed_finds:
            if not isinstance(find, Exception) and find:
                finds.append(find)

        return finds

    async def _process_wayback_snapshot(
        self,
        snapshot: Dict[str, Any],
        domain: str
    ) -> Optional[ArchaeologicalFind]:
        """Zpracování jednotlivého Wayback snapshot"""
        async with self._request_semaphore:
            await asyncio.sleep(self.request_delay)  # Rate limiting

            try:
                timestamp = snapshot.get("timestamp", "")
                original_url = snapshot.get("original", "")

                if not timestamp or not original_url:
                    return None

                # Získání historického obsahu
                content = await self.wayback_client.get_historical_content(
                    original_url, timestamp
                )

                if content:
                    # Extrakce metadata z obsahu
                    content_snippet = content[:500] if len(content) > 500 else content

                    # Parsing timestamp
                    parsed_timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M%S")

                    find = ArchaeologicalFind(
                        url=original_url,
                        domain=domain,
                        timestamp=parsed_timestamp,
                        content_type=snapshot.get("mimetype", "text/html"),
                        content_snippet=content_snippet,
                        metadata={
                            "wayback_timestamp": timestamp,
                            "status_code": snapshot.get("statuscode"),
                            "digest": snapshot.get("digest"),
                            "length": snapshot.get("length")
                        },
                        source="wayback_machine",
                        confidence_score=0.9
                    )

                    return find

            except Exception as e:
                logger.warning(f"Chyba při zpracování snapshot: {e}")

        return None

    async def _excavate_dns_archaeology(self, domain: str) -> Dict[str, Any]:
        """DNS archeologické prohledávání"""
        return await self.dns_archaeologist.excavate_dns_history(domain)

    async def _excavate_certificate_transparency(self, domain: str) -> List[Dict[str, Any]]:
        """Prohledávání Certificate Transparency logů"""
        return await self.ct_miner.excavate_certificate_history(domain)

    async def _discover_related_domains(self, domain: str, depth: int) -> Set[str]:
        """Objevování souvisejících domén"""
        discovered = set()

        # Reverse DNS na známých IP adresách
        dns_records = await self.dns_archaeologist._get_current_dns_records(domain)

        for ip in dns_records.get("A", []):
            related = await self._reverse_dns_lookup(ip)
            discovered.update(related)

        # Whois data mining (základní implementace)
        whois_related = await self._whois_related_domains(domain)
        discovered.update(whois_related)

        return discovered

    async def _reverse_dns_lookup(self, ip: str) -> Set[str]:
        """Reverse DNS lookup pro objevování souvisejících domén"""
        try:
            import socket
            hostname = socket.gethostbyaddr(ip)[0]
            return {hostname}
        except Exception:
            return set()

    async def _whois_related_domains(self, domain: str) -> Set[str]:
        """Získání souvisejících domén z Whois dat"""
        # Základní implementace - v produkci by se napojila na Whois API
        return set()

    def _build_historical_timeline(self, excavation: DomainExcavation) -> List[Dict[str, Any]]:
        """Sestavení historické timeline z archeologických nálezů"""
        timeline = []

        # Události z Wayback Machine
        for find in excavation.finds:
            timeline.append({
                "timestamp": find.timestamp,
                "event_type": "wayback_snapshot",
                "description": f"Wayback snapshot: {find.url}",
                "source": find.source,
                "metadata": find.metadata
            })

        # Události z Certificate Transparency
        for cert in excavation.certificate_history:
            if cert.get("logged_at"):
                timeline.append({
                    "timestamp": cert["logged_at"],
                    "event_type": "certificate_issued",
                    "description": f"SSL certifikát vydán pro {cert.get('common_name')}",
                    "source": "certificate_transparency",
                    "metadata": cert
                })

        # Seřazení podle času
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    async def search_historical_content(
        self,
        domain: str,
        search_terms: List[str],
        time_range_years: int = 5
    ) -> List[ArchaeologicalFind]:
        """
        Prohledávání historického obsahu podle specifických termínů
        """
        excavation = await self.excavate_forgotten_domains(domain, time_range_years=time_range_years)

        matching_finds = []

        for find in excavation.finds:
            content_lower = find.content_snippet.lower()

            for term in search_terms:
                if term.lower() in content_lower:
                    find.confidence_score *= 1.2  # Boost pro relevantní obsah
                    matching_finds.append(find)
                    break

        return matching_finds

    async def generate_archaeological_report(self, excavation: DomainExcavation) -> Dict[str, Any]:
        """Generování komplexního archeologického reportu"""
        report = {
            "executive_summary": {
                "domain": excavation.domain,
                "excavation_date": excavation.excavation_date.isoformat(),
                "total_finds": len(excavation.finds),
                "subdomain_discoveries": len(excavation.subdomain_discoveries),
                "historical_timeline_events": len(excavation.historical_timeline),
                "certificate_entries": len(excavation.certificate_history)
            },
            "detailed_findings": {
                "archaeological_finds": [
                    {
                        "url": find.url,
                        "timestamp": find.timestamp.isoformat(),
                        "content_type": find.content_type,
                        "source": find.source,
                        "confidence_score": find.confidence_score,
                        "metadata": find.metadata
                    }
                    for find in excavation.finds
                ],
                "subdomain_discoveries": list(excavation.subdomain_discoveries),
                "dns_archaeology": excavation.dns_archaeology,
                "certificate_history": excavation.certificate_history
            },
            "historical_analysis": {
                "timeline": excavation.historical_timeline,
                "patterns_detected": self._analyze_patterns(excavation),
                "security_implications": self._assess_security_implications(excavation)
            },
            "recommendations": self._generate_recommendations(excavation)
        }

        return report

    def _analyze_patterns(self, excavation: DomainExcavation) -> List[str]:
        """Analýza vzorců v archeologických datech"""
        patterns = []

        # Analýza frekvence snapshotů
        if len(excavation.finds) > 10:
            patterns.append("Vysoká archivní aktivita - doména byla často crawlována")

        # Analýza subdomén
        if len(excavation.subdomain_discoveries) > 5:
            patterns.append("Rozsáhlá subdomain infrastruktura detekována")

        # Analýza změn v čase
        timeline_years = set()
        for event in excavation.historical_timeline:
            if isinstance(event["timestamp"], datetime):
                timeline_years.add(event["timestamp"].year)
            elif isinstance(event["timestamp"], str):
                try:
                    dt = datetime.fromisoformat(event["timestamp"])
                    timeline_years.add(dt.year)
                except:
                    pass

        if len(timeline_years) > 3:
            patterns.append(f"Dlouhodobá aktivita napříč {len(timeline_years)} lety")

        return patterns

    def _assess_security_implications(self, excavation: DomainExcavation) -> List[str]:
        """Hodnocení bezpečnostních implikací"""
        implications = []

        # Kontrola starých certifikátů
        if excavation.certificate_history:
            implications.append("Historické SSL certifikáty nalezeny - možné útoky na starší konfigurace")

        # Kontrola odhalených subdomén
        if excavation.subdomain_discoveries:
            implications.append("Objevené subdomény mohou obsahovat zapomenuté služby")

        # Analýza historického obsahu
        sensitive_patterns = ["admin", "login", "password", "api", "secret"]

        for find in excavation.finds:
            content_lower = find.content_snippet.lower()
            for pattern in sensitive_patterns:
                if pattern in content_lower:
                    implications.append(f"Citlivý obsah detekován v historických záznamech: {pattern}")
                    break

        return implications

    def _generate_recommendations(self, excavation: DomainExcavation) -> List[str]:
        """Generování doporučení na základě archeologických nálezů"""
        recommendations = []

        if excavation.subdomain_discoveries:
            recommendations.append(
                "Prověřte všechny objevené subdomény na aktivní služby a zabezpečení"
            )

        if excavation.certificate_history:
            recommendations.append(
                "Auditujte historické SSL certifikáty a ověřte aktuální konfiguraci"
            )

        if len(excavation.finds) > 20:
            recommendations.append(
                "Vysoký objem historických dat - zvažte implementaci robots.txt pro kontrolu crawlingu"
            )

        recommendations.append(
            "Pravidelně monitorujte Certificate Transparency logy pro vaše domény"
        )

        return recommendations
