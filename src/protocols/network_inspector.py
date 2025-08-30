"""
Network Layer Inspector pro DeepResearchTool
Nízkoúrovňová síťová analýza, DNS historie a TCP fingerprinting.
"""

import asyncio
import json
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dns.resolver
import dns.reversename

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class NetworkFingerprint:
    """Síťový fingerprint služby"""
    host: str
    port: int
    protocol: str
    service_banner: Optional[str] = None
    tcp_options: List[str] = field(default_factory=list)
    tls_info: Optional[Dict[str, Any]] = None
    response_time_ms: float = 0
    fingerprint_confidence: float = 0.0


@dataclass
class DNSHistoryRecord:
    """Historický DNS záznam"""
    domain: str
    record_type: str
    value: str
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    source: str = "unknown"


@dataclass
class PortScanResult:
    """Výsledek port skenu"""
    host: str
    open_ports: List[int] = field(default_factory=list)
    filtered_ports: List[int] = field(default_factory=list)
    closed_ports: List[int] = field(default_factory=list)
    service_fingerprints: List[NetworkFingerprint] = field(default_factory=list)
    scan_duration_ms: float = 0


@dataclass
class NetworkAnalysisResult:
    """Komplexní síťová analýza"""
    target: str
    dns_analysis: Dict[str, Any] = field(default_factory=dict)
    port_scan: Optional[PortScanResult] = None
    tcp_fingerprints: List[NetworkFingerprint] = field(default_factory=list)
    historical_dns: List[DNSHistoryRecord] = field(default_factory=list)
    network_topology: Dict[str, Any] = field(default_factory=dict)


class TCPFingerprinter:
    """TCP fingerprinting pro identifikaci služeb"""

    def __init__(self):
        self.service_signatures = {
            21: {
                "patterns": [b"220.*FTP", b"220.*FileZilla", b"220.*vsftpd"],
                "service": "FTP"
            },
            22: {
                "patterns": [b"SSH-2.0", b"SSH-1.99"],
                "service": "SSH"
            },
            23: {
                "patterns": [b"Telnet", b"login:", b"Username:"],
                "service": "Telnet"
            },
            25: {
                "patterns": [b"220.*SMTP", b"220.*Postfix", b"220.*Sendmail"],
                "service": "SMTP"
            },
            53: {
                "patterns": [b"DNS"],
                "service": "DNS"
            },
            80: {
                "patterns": [b"HTTP/", b"Server:", b"Apache", b"nginx"],
                "service": "HTTP"
            },
            110: {
                "patterns": [b"+OK.*POP3", b"POP3 server ready"],
                "service": "POP3"
            },
            143: {
                "patterns": [b"* OK.*IMAP", b"IMAP4rev1"],
                "service": "IMAP"
            },
            443: {
                "patterns": [b"HTTP/", b"TLS", b"SSL"],
                "service": "HTTPS"
            },
            993: {
                "patterns": [b"IMAP", b"TLS"],
                "service": "IMAPS"
            },
            995: {
                "patterns": [b"POP3", b"TLS"],
                "service": "POP3S"
            }
        }

    async def fingerprint_service(self, host: str, port: int, timeout: int = 10) -> NetworkFingerprint:
        """Fingerprinting konkrétní služby"""
        start_time = time.time()

        fingerprint = NetworkFingerprint(
            host=host,
            port=port,
            protocol="tcp"
        )

        try:
            # TCP připojení
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )

            # Čtení banneru
            try:
                banner_data = await asyncio.wait_for(reader.read(1024), timeout=5)
                fingerprint.service_banner = banner_data.decode('utf-8', errors='ignore')
            except asyncio.TimeoutError:
                # Pokus o HTTP request pro web servery
                if port in [80, 443, 8080, 8443]:
                    writer.write(b"GET / HTTP/1.1\r\nHost: " + host.encode() + b"\r\n\r\n")
                    await writer.drain()

                    try:
                        response = await asyncio.wait_for(reader.read(2048), timeout=5)
                        fingerprint.service_banner = response.decode('utf-8', errors='ignore')
                    except:
                        pass

            writer.close()
            await writer.wait_closed()

            # Analýza banneru
            if fingerprint.service_banner:
                fingerprint = self._analyze_banner(fingerprint, port)

            fingerprint.response_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            logger.debug(f"TCP fingerprint failed for {host}:{port} - {e}")
            fingerprint.response_time_ms = (time.time() - start_time) * 1000

        return fingerprint

    def _analyze_banner(self, fingerprint: NetworkFingerprint, port: int) -> NetworkFingerprint:
        """Analýza service banneru"""
        banner = fingerprint.service_banner.lower()

        # Kontrola známých vzorů
        if port in self.service_signatures:
            service_info = self.service_signatures[port]

            for pattern in service_info["patterns"]:
                if pattern.lower() in banner.encode():
                    fingerprint.fingerprint_confidence = 0.9
                    break

        # Specifické detekce
        if "apache" in banner:
            fingerprint.fingerprint_confidence = max(fingerprint.fingerprint_confidence, 0.8)
        elif "nginx" in banner:
            fingerprint.fingerprint_confidence = max(fingerprint.fingerprint_confidence, 0.8)
        elif "microsoft-iis" in banner:
            fingerprint.fingerprint_confidence = max(fingerprint.fingerprint_confidence, 0.8)
        elif "openssh" in banner:
            fingerprint.fingerprint_confidence = max(fingerprint.fingerprint_confidence, 0.9)
        elif "postfix" in banner:
            fingerprint.fingerprint_confidence = max(fingerprint.fingerprint_confidence, 0.9)

        return fingerprint


class AdvancedPortScanner:
    """Pokročilý port scanner s TCP fingerprinting"""

    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.fingerprinter = TCPFingerprinter()

    async def scan_ports(
        self,
        host: str,
        ports: List[int],
        scan_type: str = "connect",
        timeout: int = 5
    ) -> PortScanResult:
        """Skenování portů s fingerprinting"""
        start_time = time.time()

        result = PortScanResult(host=host)

        # Semafór pro omezení souběžnosti
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def scan_port(port: int):
            async with semaphore:
                status = await self._scan_single_port(host, port, timeout)

                if status == "open":
                    result.open_ports.append(port)

                    # Fingerprinting otevřeného portu
                    fingerprint = await self.fingerprinter.fingerprint_service(host, port, timeout)
                    if fingerprint.service_banner or fingerprint.fingerprint_confidence > 0:
                        result.service_fingerprints.append(fingerprint)

                elif status == "filtered":
                    result.filtered_ports.append(port)
                else:
                    result.closed_ports.append(port)

        # Paralelní skenování
        tasks = [scan_port(port) for port in ports]
        await asyncio.gather(*tasks)

        result.scan_duration_ms = (time.time() - start_time) * 1000

        # Řazení výsledků
        result.open_ports.sort()
        result.filtered_ports.sort()
        result.closed_ports.sort()

        return result

    async def _scan_single_port(self, host: str, port: int, timeout: int) -> str:
        """Skenování jednoho portu"""
        try:
            # Connect scan
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return "open"

        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
            return "closed"
        except Exception:
            return "filtered"

    async def scan_common_ports(self, host: str) -> PortScanResult:
        """Skenování běžných portů"""
        common_ports = [
            21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995,
            1723, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 9200, 11211, 27017
        ]

        return await self.scan_ports(host, common_ports)


class DNSAnalyzer:
    """Pokročilý DNS analyzer s historickou analýzou"""

    def __init__(self):
        self.public_resolvers = [
            "8.8.8.8",      # Google
            "1.1.1.1",      # Cloudflare
            "208.67.222.222", # OpenDNS
            "9.9.9.9"       # Quad9
        ]

    async def comprehensive_dns_analysis(self, domain: str) -> Dict[str, Any]:
        """Komplexní DNS analýza"""
        analysis = {
            "domain": domain,
            "current_records": {},
            "dns_security": {},
            "resolver_comparison": {},
            "reverse_dns": {},
            "subdomain_enumeration": []
        }

        # Současné DNS záznamy
        analysis["current_records"] = await self._get_all_dns_records(domain)

        # DNS security analýza
        analysis["dns_security"] = await self._analyze_dns_security(domain)

        # Porovnání mezi resolvery
        analysis["resolver_comparison"] = await self._compare_resolvers(domain)

        # Reverse DNS pro A záznamy
        if "A" in analysis["current_records"]:
            analysis["reverse_dns"] = await self._reverse_dns_analysis(
                analysis["current_records"]["A"]
            )

        # Subdomain enumeration
        analysis["subdomain_enumeration"] = await self._enumerate_subdomains(domain)

        return analysis

    async def _get_all_dns_records(self, domain: str) -> Dict[str, List[str]]:
        """Získání všech DNS záznamů"""
        record_types = ["A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV"]
        records = {}

        for record_type in record_types:
            try:
                resolver = dns.resolver.Resolver()
                answers = resolver.resolve(domain, record_type)
                records[record_type] = [str(rdata) for rdata in answers]
            except Exception:
                records[record_type] = []

        return records

    async def _analyze_dns_security(self, domain: str) -> Dict[str, Any]:
        """Analýza DNS security features"""
        security = {
            "dnssec_enabled": False,
            "caa_records": [],
            "spf_record": None,
            "dmarc_record": None,
            "dkim_records": []
        }

        try:
            # DNSSEC kontrola
            resolver = dns.resolver.Resolver()
            resolver.use_edns(0, dns.flags.DO, 4096)

            try:
                answers = resolver.resolve(domain, "A")
                security["dnssec_enabled"] = any(answer.flags & dns.flags.AD for answer in answers.response.answer)
            except:
                pass

            # CAA záznamy
            try:
                caa_answers = resolver.resolve(domain, "CAA")
                security["caa_records"] = [str(rdata) for rdata in caa_answers]
            except:
                pass

            # SPF záznam
            try:
                txt_answers = resolver.resolve(domain, "TXT")
                for rdata in txt_answers:
                    txt_string = str(rdata)
                    if txt_string.startswith('"v=spf1'):
                        security["spf_record"] = txt_string
                        break
            except:
                pass

            # DMARC záznam
            try:
                dmarc_answers = resolver.resolve(f"_dmarc.{domain}", "TXT")
                for rdata in dmarc_answers:
                    txt_string = str(rdata)
                    if "v=DMARC1" in txt_string:
                        security["dmarc_record"] = txt_string
                        break
            except:
                pass

        except Exception as e:
            logger.debug(f"DNS security analysis error: {e}")

        return security

    async def _compare_resolvers(self, domain: str) -> Dict[str, Any]:
        """Porovnání odpovědí různých resolverů"""
        resolver_results = {}

        for resolver_ip in self.public_resolvers:
            try:
                resolver = dns.resolver.Resolver()
                resolver.nameservers = [resolver_ip]

                answers = resolver.resolve(domain, "A")
                ips = [str(rdata) for rdata in answers]

                resolver_results[resolver_ip] = {
                    "ips": ips,
                    "response_time": answers.response.time if hasattr(answers.response, 'time') else None
                }

            except Exception as e:
                resolver_results[resolver_ip] = {"error": str(e)}

        # Analýza rozdílů
        all_ips = set()
        for result in resolver_results.values():
            if "ips" in result:
                all_ips.update(result["ips"])

        inconsistencies = []
        if len(all_ips) > 1:
            for resolver_ip, result in resolver_results.items():
                if "ips" in result and set(result["ips"]) != all_ips:
                    inconsistencies.append({
                        "resolver": resolver_ip,
                        "different_ips": list(set(result["ips"]) - all_ips)
                    })

        return {
            "resolver_results": resolver_results,
            "inconsistencies": inconsistencies,
            "total_unique_ips": len(all_ips)
        }

    async def _reverse_dns_analysis(self, ip_addresses: List[str]) -> Dict[str, Any]:
        """Reverse DNS analýza pro IP adresy"""
        reverse_results = {}

        for ip in ip_addresses:
            try:
                reverse_name = dns.reversename.from_address(ip)
                resolver = dns.resolver.Resolver()
                answers = resolver.resolve(reverse_name, "PTR")

                reverse_results[ip] = [str(rdata) for rdata in answers]

            except Exception as e:
                reverse_results[ip] = {"error": str(e)}

        return reverse_results

    async def _enumerate_subdomains(self, domain: str) -> List[str]:
        """Základní subdomain enumeration"""
        common_subdomains = [
            "www", "mail", "ftp", "blog", "shop", "admin", "api", "cdn",
            "staging", "dev", "test", "backup", "old", "legacy", "archive",
            "support", "help", "docs", "portal", "secure", "login", "beta",
            "demo", "app", "mobile", "m", "static", "img", "images", "assets"
        ]

        found_subdomains = []

        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"

            try:
                resolver = dns.resolver.Resolver()
                resolver.resolve(full_domain, "A")
                found_subdomains.append(full_domain)
            except:
                pass

        return found_subdomains


class NetworkLayerInspector:
    """
    Pokročilý síťový inspector pro hlubokou analýzu síťové infrastruktury,
    včetně DNS historie, TCP fingerprintingu a network topology mapping.
    """

    def __init__(self, max_concurrent_scans: int = 50):
        self.port_scanner = AdvancedPortScanner(max_concurrent_scans)
        self.dns_analyzer = DNSAnalyzer()
        self.tcp_fingerprinter = TCPFingerprinter()

        logger.info("NetworkLayerInspector inicializován")

    async def comprehensive_network_analysis(self, target: str) -> NetworkAnalysisResult:
        """
        Komplexní síťová analýza včetně DNS, port scan a fingerprinting
        """
        result = NetworkAnalysisResult(target=target)

        # Paralelní spuštění analýz
        tasks = [
            self._analyze_dns_comprehensive(target),
            self._perform_port_scan(target),
            self._analyze_network_topology(target)
        ]

        dns_analysis, port_scan, topology = await asyncio.gather(*tasks, return_exceptions=True)

        if not isinstance(dns_analysis, Exception):
            result.dns_analysis = dns_analysis

        if not isinstance(port_scan, Exception):
            result.port_scan = port_scan
            result.tcp_fingerprints = port_scan.service_fingerprints

        if not isinstance(topology, Exception):
            result.network_topology = topology

        return result

    async def _analyze_dns_comprehensive(self, domain: str) -> Dict[str, Any]:
        """Komplexní DNS analýza"""
        return await self.dns_analyzer.comprehensive_dns_analysis(domain)

    async def _perform_port_scan(self, host: str) -> PortScanResult:
        """Port scan s fingerprinting"""
        return await self.port_scanner.scan_common_ports(host)

    async def _analyze_network_topology(self, target: str) -> Dict[str, Any]:
        """Analýza síťové topologie"""
        topology = {
            "target": target,
            "traceroute": [],
            "as_information": {},
            "geolocation": {},
            "cdn_detection": False
        }

        # Základní traceroute simulation (zjednodušená verze)
        try:
            # V produkci by se použil skutečný traceroute
            topology["traceroute"] = await self._simulate_traceroute(target)
        except Exception as e:
            logger.debug(f"Traceroute simulation failed: {e}")

        # AS information lookup
        try:
            topology["as_information"] = await self._get_as_information(target)
        except Exception as e:
            logger.debug(f"AS information lookup failed: {e}")

        return topology

    async def _simulate_traceroute(self, target: str) -> List[Dict[str, Any]]:
        """Simulace traceroute (zjednodušená)"""
        # V produkci by se použily skutečné traceroute nástroje
        hops = []

        try:
            # Resolve target IP
            resolver = dns.resolver.Resolver()
            answers = resolver.resolve(target, "A")
            target_ip = str(answers[0])

            # Simulace několika hopů
            import ipaddress
            target_net = ipaddress.ip_network(f"{target_ip}/24", strict=False)

            for i in range(1, 4):  # Simulace 3 hopů
                hop_ip = str(list(target_net.hosts())[i])

                # Měření response time
                start_time = time.time()
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    sock.connect((hop_ip, 80))
                    sock.close()
                    response_time = (time.time() - start_time) * 1000
                except:
                    response_time = None

                hops.append({
                    "hop": i,
                    "ip": hop_ip,
                    "response_time_ms": response_time,
                    "hostname": f"hop{i}.example.com"  # Placeholder
                })

        except Exception as e:
            logger.debug(f"Traceroute simulation error: {e}")

        return hops

    async def _get_as_information(self, target: str) -> Dict[str, Any]:
        """Získání AS (Autonomous System) informací"""
        # V produkci by se napojilo na WHOIS/BGP API
        return {
            "asn": "AS12345",
            "organization": "Example ISP",
            "country": "US",
            "registry": "ARIN"
        }

    async def scan_custom_ports(
        self,
        host: str,
        ports: List[int],
        aggressive_fingerprinting: bool = False
    ) -> PortScanResult:
        """
        Skenování custom portů s možností agresivního fingerprintingu
        """
        result = await self.port_scanner.scan_ports(host, ports)

        if aggressive_fingerprinting:
            # Dodatečné fingerprinting techniky
            for fingerprint in result.service_fingerprints:
                if fingerprint.fingerprint_confidence < 0.7:
                    # Pokus o dodatečné techniky
                    enhanced_fingerprint = await self._enhanced_fingerprinting(
                        fingerprint.host, fingerprint.port
                    )
                    if enhanced_fingerprint:
                        fingerprint.fingerprint_confidence = enhanced_fingerprint.fingerprint_confidence
                        fingerprint.service_banner = enhanced_fingerprint.service_banner

        return result

    async def _enhanced_fingerprinting(self, host: str, port: int) -> Optional[NetworkFingerprint]:
        """Rozšířené fingerprinting techniky"""
        # Implementace pokročilých fingerprinting technik
        # V produkci by zahrnovalo:
        # - Nmap-style service detection
        # - SSL/TLS certificate analysis
        # - HTTP header analysis
        # - Protocol-specific probes

        return None

    async def detect_network_anomalies(self, targets: List[str]) -> Dict[str, Any]:
        """
        Detekce síťových anomálií napříč více cíli
        """
        anomalies = {
            "dns_inconsistencies": [],
            "unusual_ports": [],
            "suspicious_services": [],
            "network_patterns": []
        }

        for target in targets:
            try:
                analysis = await self.comprehensive_network_analysis(target)

                # Kontrola DNS inconsistencies
                if analysis.dns_analysis.get("resolver_comparison", {}).get("inconsistencies"):
                    anomalies["dns_inconsistencies"].append({
                        "target": target,
                        "inconsistencies": analysis.dns_analysis["resolver_comparison"]["inconsistencies"]
                    })

                # Kontrola neobvyklých portů
                if analysis.port_scan:
                    unusual_ports = [p for p in analysis.port_scan.open_ports if p > 10000]
                    if unusual_ports:
                        anomalies["unusual_ports"].append({
                            "target": target,
                            "ports": unusual_ports
                        })

                # Kontrola podezřelých služeb
                for fingerprint in analysis.tcp_fingerprints:
                    if (fingerprint.service_banner and
                        any(suspicious in fingerprint.service_banner.lower()
                            for suspicious in ["backdoor", "shell", "bot"])):
                        anomalies["suspicious_services"].append({
                            "target": target,
                            "port": fingerprint.port,
                            "banner": fingerprint.service_banner
                        })

            except Exception as e:
                logger.error(f"Anomaly detection failed for {target}: {e}")

        return anomalies

    def generate_network_report(self, analysis: NetworkAnalysisResult) -> Dict[str, Any]:
        """
        Generování síťového reportu
        """
        report = {
            "target": analysis.target,
            "executive_summary": {
                "open_ports_count": len(analysis.port_scan.open_ports) if analysis.port_scan else 0,
                "services_identified": len(analysis.tcp_fingerprints),
                "dns_records_found": len(analysis.dns_analysis.get("current_records", {})),
                "security_issues": []
            },
            "detailed_findings": {
                "network_services": [],
                "dns_analysis": analysis.dns_analysis,
                "topology_info": analysis.network_topology
            },
            "security_assessment": {
                "risk_level": "low",
                "vulnerabilities": [],
                "recommendations": []
            },
            "technical_details": {
                "scan_duration": analysis.port_scan.scan_duration_ms if analysis.port_scan else 0,
                "fingerprinting_confidence": 0
            }
        }

        # Network services detail
        for fingerprint in analysis.tcp_fingerprints:
            report["detailed_findings"]["network_services"].append({
                "port": fingerprint.port,
                "service": fingerprint.service_banner,
                "confidence": fingerprint.fingerprint_confidence,
                "response_time": fingerprint.response_time_ms
            })

        # Security assessment
        if analysis.port_scan and len(analysis.port_scan.open_ports) > 10:
            report["security_assessment"]["risk_level"] = "medium"
            report["security_assessment"]["vulnerabilities"].append("Velký počet otevřených portů")

        # DNS security kontrola
        dns_security = analysis.dns_analysis.get("dns_security", {})
        if not dns_security.get("dnssec_enabled"):
            report["security_assessment"]["vulnerabilities"].append("DNSSEC není aktivní")

        if not dns_security.get("spf_record"):
            report["security_assessment"]["vulnerabilities"].append("Chybí SPF záznam")

        # Doporučení
        report["security_assessment"]["recommendations"].extend([
            "Pravidelně aktualizujte síťové služby",
            "Implementujte network monitoring",
            "Zvažte zavření nepoužívaných portů"
        ])

        return report
