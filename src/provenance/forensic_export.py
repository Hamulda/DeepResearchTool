"""Forensic Export System
Generování forenzních reportů s kompletní proveniencí dat

Author: Senior Python/MLOps Agent
"""

import asyncio
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import zipfile

logger = logging.getLogger(__name__)


@dataclass
class ForensicReport:
    """Struktura forenzního reportu"""

    report_id: str
    query: str
    timestamp: datetime
    total_sources: int
    verified_sources: int
    claim_count: int
    contradiction_count: int
    confidence_score: float
    processing_time: float
    methodology: str = "LangGraph Research Agent"


class ForensicExporter:
    """Hlavní třída pro export forenzních reportů
    """

    def __init__(self, output_dir: str = "./forensic_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _generate_report_id(self, query: str) -> str:
        """Generování unique ID pro report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"report_{timestamp}_{query_hash}"

    def _calculate_confidence_score(self,
                                  validation_scores: dict[str, float],
                                  claim_graph_stats: dict[str, Any]) -> float:
        """Výpočet celkového confidence score"""
        # Váhy pro různé faktory
        weights = {
            "source_quality": 0.3,
            "citation_completeness": 0.25,
            "claim_support": 0.25,
            "verification_completeness": 0.2
        }

        # Získání skóre z validace
        source_quality = validation_scores.get("quality", 0.5)
        relevance = validation_scores.get("relevance", 0.5)

        # Skóre z claim graph
        total_claims = claim_graph_stats.get("total_claims", 0)
        supported_claims = claim_graph_stats.get("verification_status_counts", {}).get("supported", 0)

        # Výpočty
        citation_completeness = min(1.0, validation_scores.get("coverage", 0.5))
        claim_support = supported_claims / total_claims if total_claims > 0 else 0.5
        verification_completeness = min(1.0, claim_graph_stats.get("total_evidence", 0) / max(total_claims, 1))

        # Vážený průměr
        confidence = (
            weights["source_quality"] * source_quality +
            weights["citation_completeness"] * citation_completeness +
            weights["claim_support"] * claim_support +
            weights["verification_completeness"] * verification_completeness
        )

        return min(1.0, max(0.0, confidence))

    async def generate_markdown_report(
        self,
        query: str,
        synthesis: str,
        sources: list[dict[str, Any]],
        claim_graph: Any,
        validation_scores: dict[str, float],
        processing_time: float,
        metadata: dict[str, Any] = None
    ) -> str:
        """Generuje kompletní Markdown report s kompletní proveniencí

        Args:
            query: Původní výzkumný dotaz
            synthesis: Finální syntéza
            sources: Seznam použitých zdrojů
            claim_graph: ClaimGraph instance
            validation_scores: Skóre validace
            processing_time: Doba zpracování
            metadata: Dodatečná metadata

        Returns:
            Markdown formátovaný report

        """
        # Příprava základních informací
        report_id = self._generate_report_id(query)
        timestamp = datetime.now(UTC)

        # Statistiky claim graph
        if hasattr(claim_graph, 'get_statistics'):
            claim_stats = claim_graph.get_statistics()
        else:
            claim_stats = {}

        # Výpočet confidence score
        confidence = self._calculate_confidence_score(validation_scores, claim_stats)

        # Vytvoření forenzního reportu
        forensic_report = ForensicReport(
            report_id=report_id,
            query=query,
            timestamp=timestamp,
            total_sources=len(sources),
            verified_sources=len([s for s in sources if s.get('verified', False)]),
            claim_count=claim_stats.get('total_claims', 0),
            contradiction_count=claim_stats.get('relation_type_counts', {}).get('contradict', 0),
            confidence_score=confidence,
            processing_time=processing_time
        )

        # Generování Markdown reportu
        report_lines = []

        # Header
        report_lines.extend([
            "# Forenzní výzkumný report",
            "",
            f"**Report ID:** `{report_id}`  ",
            f"**Datum:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Metodologie:** {forensic_report.methodology}  ",
            f"**Confidence Score:** {confidence:.2f}/1.00  ",
            "",
            "---",
            ""
        ])

        # Původní dotaz
        report_lines.extend([
            "## 📋 Výzkumný dotaz",
            "",
            f"> {query}",
            "",
        ])

        # Exekutivní souhrn
        report_lines.extend([
            "## 📊 Exekutivní souhrn",
            "",
            f"- **Celkem zdrojů:** {forensic_report.total_sources}",
            f"- **Ověřené zdroje:** {forensic_report.verified_sources}",
            f"- **Extrahovaná tvrzení:** {forensic_report.claim_count}",
            f"- **Nalezené rozpory:** {forensic_report.contradiction_count}",
            f"- **Doba zpracování:** {forensic_report.processing_time:.2f} sekund",
            f"- **Celkové skóre důvěryhodnosti:** {confidence:.2f}/1.00",
            "",
        ])

        # Validační metriky
        if validation_scores:
            report_lines.extend([
                "## ✅ Validační metriky",
                "",
                "| Metrika | Skóre | Status |",
                "|---------|-------|--------|",
            ])

            for metric, score in validation_scores.items():
                status = "✅ Vyhovující" if score >= 0.7 else "⚠️ Slabé" if score >= 0.5 else "❌ Nevyhovující"
                report_lines.append(f"| {metric.capitalize()} | {score:.2f} | {status} |")

            report_lines.append("")

        # Hlavní syntéza
        report_lines.extend([
            "## 📄 Hlavní syntéza",
            "",
            synthesis,
            "",
        ])

        # Analýza tvrzení s proveniencí
        if hasattr(claim_graph, 'claims') and claim_graph.claims:
            report_lines.extend([
                "## 🧩 Analýza tvrzení s kompletní proveniencí",
                "",
            ])

            for claim_id, claim in claim_graph.claims.items():
                # Získání podpory pro tvrzení
                if hasattr(claim_graph, 'get_claim_support'):
                    support_data = claim_graph.get_claim_support(claim_id)
                else:
                    support_data = {}

                verification_status = support_data.get('verification_status', 'neověřeno')
                support_score = support_data.get('support_score', 0.0)

                # Status emoji
                status_emoji = {
                    'supported': '✅',
                    'partially_supported': '🟡',
                    'disputed': '⚠️',
                    'contradicted': '❌',
                    'neutral': '⚪'
                }.get(verification_status, '❓')

                report_lines.extend([
                    f"### {status_emoji} Tvrzení: {claim.text}",
                    "",
                    f"**Status:** {verification_status}  ",
                    f"**Podpora:** {support_score:.2f}/1.00  ",
                    f"**Confidence:** {claim.confidence:.2f}/1.00  ",
                    f"**Extrahováno:** {claim.created_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
                    "",
                ])

                # Evidence
                evidence_list = support_data.get('evidence', [])
                if evidence_list:
                    report_lines.extend([
                        "**Důkazy:**",
                        "",
                    ])

                    for i, evidence in enumerate(evidence_list, 1):
                        credibility = evidence.credibility_score if hasattr(evidence, 'credibility_score') else 0.7
                        relevance = evidence.relevance_score if hasattr(evidence, 'relevance_score') else 0.7
                        source_url = evidence.source_url if hasattr(evidence, 'source_url') else "N/A"

                        report_lines.extend([
                            f"{i}. **Důkaz ID:** `{evidence.id}`",
                            f"   - **Text:** {evidence.text[:200]}{'...' if len(evidence.text) > 200 else ''}",
                            f"   - **Zdroj:** {evidence.source_id}",
                            f"   - **URL:** {source_url}",
                            f"   - **Důvěryhodnost:** {credibility:.2f}/1.00",
                            f"   - **Relevance:** {relevance:.2f}/1.00",
                            "",
                        ])

                # Supporting/Contradicting claims
                supporting_claims = support_data.get('supporting_claims', [])
                contradicting_claims = support_data.get('contradicting_claims', [])

                if supporting_claims:
                    report_lines.extend([
                        "**Podporující tvrzení:**",
                        "",
                    ])
                    for supporting in supporting_claims:
                        report_lines.append(f"- {supporting.text}")
                    report_lines.append("")

                if contradicting_claims:
                    report_lines.extend([
                        "**Protiřečící tvrzení:**",
                        "",
                    ])
                    for contradicting in contradicting_claims:
                        report_lines.append(f"- {contradicting.text}")
                    report_lines.append("")

                report_lines.append("---")
                report_lines.append("")

        # Zdroje a citace
        report_lines.extend([
            "## 📚 Kompletní seznam zdrojů a citací",
            "",
            "| # | Zdroj | URL | Typ | Časový otisk | Status |",
            "|---|-------|-----|-----|--------------|--------|",
        ])

        for i, source in enumerate(sources, 1):
            source_type = source.get('metadata', {}).get('source_type', 'neznámý')
            url = source.get('source', 'N/A')
            timestamp_str = source.get('metadata', {}).get('timestamp', 'N/A')
            verified = "✅ Ověřeno" if source.get('verified', False) else "⚪ Neověřeno"

            # Zkrácení URL pro lepší čitelnost
            display_url = url[:50] + "..." if len(url) > 50 else url

            report_lines.append(f"| {i} | {source_type} | {display_url} | {source_type} | {timestamp_str} | {verified} |")

        report_lines.extend([
            "",
            "## 🔍 Metodologie a omezení",
            "",
            "### Metodologie",
            "- **Vyhledávání:** Hybridní přístup kombinující sémantické a lexikální vyhledávání",
            "- **Validace:** Automatická validace zdrojů s LLM-based hodnocením",
            "- **Extrakce tvrzení:** Automatická extrakce pomocí LLM s pattern matching",
            "- **Verification:** Cross-referencing mezi zdroji a detekce rozporů",
            "",
            "### Omezení",
            "- Automatická extrakce tvrzení může být neúplná",
            "- Validace zdrojů je založena na heuristikách",
            "- Některé nuance mohou být ztraceny při kompresi",
            "- Časové razítko odpovídá času stažení, ne publikování",
            "",
            "### Doporučení pro další výzkum",
            "- Manuální verifikace klíčových tvrzení",
            "- Konzultace s domain experty",
            "- Rozšíření zdrojové báze pro komplexnější témata",
            "",
            "---",
            "",
            f"*Report generován automaticky systémem Deep Research Tool v {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            f"*Report ID: {report_id}*"
        ])

        return "\n".join(report_lines)

    async def export_complete_investigation(
        self,
        query: str,
        results: dict[str, Any],
        output_dir: str = "./forensic_reports"
    ) -> dict[str, str]:
        """Exportuje kompletní forenzní investigaci

        Args:
            query: Výzkumný dotaz
            results: Kompletní výsledky z agenta
            output_dir: Výstupní adresář

        Returns:
            Dict s cestami k vytvořeným souborům

        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_id = self._generate_report_id(query)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        try:
            # 1. Markdown report
            markdown_report = await self.generate_markdown_report(
                query=query,
                synthesis=results.get("synthesis", ""),
                sources=results.get("retrieved_docs", []),
                claim_graph=results.get("claim_graph"),
                validation_scores=results.get("validation_scores", {}),
                processing_time=results.get("processing_time", 0),
                metadata=results.get("metadata", {})
            )

            markdown_path = output_path / f"{report_id}_report.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            exported_files["markdown_report"] = str(markdown_path)

            # 2. JSON dump všech dat
            json_path = output_path / f"{report_id}_raw_data.json"

            # Příprava dat pro JSON export
            json_data = {
                "report_metadata": {
                    "report_id": report_id,
                    "query": query,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "version": "1.0"
                },
                "results": self._serialize_results_for_json(results)
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            exported_files["raw_data"] = str(json_path)

            # 3. Claim graph export
            if results.get("claim_graph") and hasattr(results["claim_graph"], "export_to_json"):
                claim_graph_path = output_path / f"{report_id}_claim_graph.json"
                results["claim_graph"].export_to_json(str(claim_graph_path))
                exported_files["claim_graph"] = str(claim_graph_path)

            # 4. CSV export zdrojů
            sources_csv_path = output_path / f"{report_id}_sources.csv"
            await self._export_sources_csv(results.get("retrieved_docs", []), sources_csv_path)
            exported_files["sources_csv"] = str(sources_csv_path)

            # 5. Komprimovaný archiv
            zip_path = output_path / f"{report_id}_complete_investigation.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_type, file_path in exported_files.items():
                    zipf.write(file_path, Path(file_path).name)

            exported_files["archive"] = str(zip_path)

            logger.info(f"✅ Forenzní investigace exportována: {report_id}")

        except Exception as e:
            logger.error(f"Chyba při exportu forenzní investigace: {e}")
            exported_files["error"] = str(e)

        return exported_files

    def _serialize_results_for_json(self, results: dict[str, Any]) -> dict[str, Any]:
        """Serializuje výsledky pro JSON export"""
        serialized = {}

        for key, value in results.items():
            if key == "claim_graph":
                # ClaimGraph není přímo serializovatelný
                if hasattr(value, 'get_statistics'):
                    serialized[key] = {
                        "statistics": value.get_statistics(),
                        "type": "ClaimGraph"
                    }
                else:
                    serialized[key] = {"type": "ClaimGraph", "data": "not_serializable"}
            elif isinstance(value, (str, int, float, bool, list, dict)):
                serialized[key] = value
            else:
                serialized[key] = str(value)

        return serialized

    async def _export_sources_csv(self, sources: list[dict[str, Any]], csv_path: Path):
        """Exportuje zdroje do CSV formátu"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['source_id', 'url', 'source_type', 'timestamp', 'content_length', 'verified', 'metadata']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for i, source in enumerate(sources):
                writer.writerow({
                    'source_id': f"source_{i+1}",
                    'url': source.get('source', ''),
                    'source_type': source.get('metadata', {}).get('source_type', 'unknown'),
                    'timestamp': source.get('metadata', {}).get('timestamp', ''),
                    'content_length': len(source.get('content', '')),
                    'verified': source.get('verified', False),
                    'metadata': json.dumps(source.get('metadata', {}))
                })


# Utility funkce pro rychlé použití
async def export_investigation_report(
    query: str,
    results: dict[str, Any],
    output_dir: str = "./forensic_reports"
) -> dict[str, str]:
    """Convenience funkce pro rychlý export forenzní investigace

    Args:
        query: Výzkumný dotaz
        results: Výsledky z research agenta
        output_dir: Výstupní adresář

    Returns:
        Dict s cestami k vytvořeným souborům

    """
    exporter = ForensicExporter(output_dir)
    return await exporter.export_complete_investigation(query, results, output_dir)


# Příklad použití
async def example_forensic_export():
    """Příklad použití forenzního exportu"""
    # Simulovaná data
    mock_results = {
        "synthesis": "## Ukázková syntéza\n\nToto je příklad syntézy výzkumu...",
        "retrieved_docs": [
            {
                "content": "Obsah dokumentu 1...",
                "source": "https://example.com/doc1",
                "metadata": {
                    "source_type": "academic",
                    "timestamp": "2023-12-01T10:00:00Z"
                },
                "verified": True
            }
        ],
        "validation_scores": {
            "relevance": 0.85,
            "quality": 0.78,
            "coverage": 0.92
        },
        "processing_time": 45.2,
        "metadata": {
            "architecture": "langgraph",
            "total_documents": 1
        }
    }

    # Export
    exported = await export_investigation_report(
        "Jaké jsou trendy v AI?",
        mock_results
    )

    print("📁 Exportované soubory:")
    for file_type, path in exported.items():
        print(f"  {file_type}: {path}")


if __name__ == "__main__":
    asyncio.run(example_forensic_export())
