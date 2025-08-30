#!/usr/bin/env python3
"""
PII Leak Detection Script
Detekuje úniky osobních údajů (PII) v kódu a logách

Author: Senior Python/MLOps Agent
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIDetector:
    """Detektor úniků osobních údajů"""

    def __init__(self):
        self.pii_patterns = [
            # Email adresy
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email address detected"),

            # Telefonní čísla
            (r'\b\+?1?\d{9,15}\b', "Phone number detected"),
            (r'\b\d{3}-\d{3}-\d{4}\b', "US phone number detected"),

            # IP adresy (externí)
            (r'\b(?!127\.|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "External IP address detected"),

            # Kreditní karty (základní pattern)
            (r'\b4\d{15}\b', "Potential Visa credit card detected"),
            (r'\b5[1-5]\d{14}\b', "Potential MasterCard detected"),

            # Rodná čísla (české)
            (r'\b\d{6}/\d{3,4}\b', "Czech birth number detected"),

            # API klíče a tokeny
            (r'[A-Za-z0-9]{32,}', "Potential API key or token detected"),

            # Jména a příjmení (heuristika)
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', "Potential full name detected"),
        ]

        # Výjimky - běžné technické termíny
        self.exceptions = [
            "localhost",
            "example.com",
            "test@test.com",
            "admin@admin.com",
            "user@domain.com",
            "John Doe",
            "Jane Smith",
            "Test User",
            "Example Example"
        ]

    def is_exception(self, text: str) -> bool:
        """Kontrola, zda se jedná o výjimku"""
        return any(exc.lower() in text.lower() for exc in self.exceptions)

    def check_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Kontrola jednoho souboru na PII"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, message in self.pii_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        matched_text = match.group()

                        # Kontrola výjimek
                        if self.is_exception(matched_text):
                            continue

                        # Speciální kontroly
                        if "API key" in message and len(matched_text) < 20:
                            continue  # Příliš krátké pro API klíč

                        if "IP address" in message and matched_text.startswith("127."):
                            continue  # Localhost

                        issues.append({
                            "file": str(file_path),
                            "line": line_num,
                            "issue": message,
                            "content": line.strip()[:50] + "...",
                            "matched": matched_text[:20] + "..." if len(matched_text) > 20 else matched_text,
                            "severity": "HIGH"
                        })

        except Exception as e:
            logger.warning(f"Could not check file {file_path}: {e}")

        return issues

    def check_logs(self, log_dir: Path) -> List[Dict[str, Any]]:
        """Kontrola log souborů na PII"""
        issues = []

        if not log_dir.exists():
            return issues

        for log_file in log_dir.rglob("*.log"):
            file_issues = self.check_file(log_file)
            issues.extend(file_issues)

        return issues

    def run_check(self, root_path: Path = None) -> Dict[str, Any]:
        """Spuštění kompletní PII kontroly"""
        if root_path is None:
            root_path = Path.cwd()

        all_issues = []

        # Kontrola Python souborů
        for py_file in root_path.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            issues = self.check_file(py_file)
            all_issues.extend(issues)

        # Kontrola konfiguračních souborů
        for config_file in root_path.rglob("*.yaml"):
            issues = self.check_file(config_file)
            all_issues.extend(issues)

        for config_file in root_path.rglob("*.json"):
            issues = self.check_file(config_file)
            all_issues.extend(issues)

        # Kontrola logů
        log_dirs = [root_path / "logs", root_path / "data" / "logs"]
        for log_dir in log_dirs:
            issues = self.check_logs(log_dir)
            all_issues.extend(issues)

        # Sumarizace
        high_count = len([i for i in all_issues if i.get("severity") == "HIGH"])

        return {
            "total_issues": len(all_issues),
            "high_issues": high_count,
            "issues": all_issues,
            "passed": len(all_issues) == 0
        }


def main():
    """Main entry point"""
    detector = PIIDetector()
    results = detector.run_check()

    print(f"PII Leak Detection Results:")
    print(f"Total potential leaks: {results['total_issues']}")
    print(f"High severity: {results['high_issues']}")

    if results['issues']:
        print("\nPotential PII leaks found:")
        for issue in results['issues']:
            print(f"  {issue['severity']}: {issue['file']}:{issue['line']} - {issue['issue']}")
            print(f"    Matched: {issue['matched']}")

    if not results['passed']:
        print("\n❌ PII leak check failed!")
        sys.exit(1)
    else:
        print("\n✅ PII leak check passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
