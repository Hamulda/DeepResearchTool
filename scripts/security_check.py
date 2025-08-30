#!/usr/bin/env python3
"""
Security Check Script
Kontroluje bezpečnostní problémy v kódu a konfiguraci

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


class SecurityChecker:
    """Kontrola bezpečnostních problémů"""

    def __init__(self):
        self.security_patterns = [
            # Hardcoded secrets
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected"),

            # Dangerous functions
            (r'\beval\s*\(', "Use of eval() function detected"),
            (r'\bexec\s*\(', "Use of exec() function detected"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Unsafe subprocess call with shell=True"),

            # SQL injection patterns
            (r'execute\s*\(\s*["\'].*%.*["\']', "Potential SQL injection via string formatting"),
            (r'\.format\s*\(.*SELECT|INSERT|UPDATE|DELETE', "Potential SQL injection in query"),

            # Path traversal
            (r'\.\./', "Potential path traversal detected"),
            (r'open\s*\(\s*.*\+.*["\']', "Potential unsafe file operation"),

            # Network security
            (r'verify\s*=\s*False', "SSL verification disabled"),
            (r'check_hostname\s*=\s*False', "Hostname verification disabled"),
        ]

        self.critical_files = [
            "config.yaml",
            "config_m1_local.yaml",
            ".env",
            "secrets.json"
        ]

    def check_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Kontrola jednoho souboru"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, message in self.security_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            "file": str(file_path),
                            "line": line_num,
                            "issue": message,
                            "content": line.strip()[:100],
                            "severity": "HIGH"
                        })

        except Exception as e:
            logger.warning(f"Could not check file {file_path}: {e}")

        return issues

    def check_config_security(self, config_path: Path) -> List[Dict[str, Any]]:
        """Kontrola bezpečnosti konfigurace"""
        issues = []

        if not config_path.exists():
            return issues

        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Kontrola defaultních hesel
            def check_dict(d, path=""):
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key

                    if isinstance(value, dict):
                        check_dict(value, current_path)
                    elif isinstance(value, str):
                        if key.lower() in ['password', 'secret', 'token', 'key'] and value in ['', 'admin', 'password', '123456']:
                            issues.append({
                                "file": str(config_path),
                                "line": 0,
                                "issue": f"Weak or default {key} in config",
                                "content": f"{current_path}: {value}",
                                "severity": "CRITICAL"
                            })

            check_dict(config)

        except Exception as e:
            logger.warning(f"Could not check config {config_path}: {e}")

        return issues

    def run_check(self, root_path: Path = None) -> Dict[str, Any]:
        """Spuštění kompletní bezpečnostní kontroly"""
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
        for config_file in self.critical_files:
            config_path = root_path / config_file
            issues = self.check_config_security(config_path)
            all_issues.extend(issues)

        # Sumarizace
        critical_count = len([i for i in all_issues if i.get("severity") == "CRITICAL"])
        high_count = len([i for i in all_issues if i.get("severity") == "HIGH"])

        return {
            "total_issues": len(all_issues),
            "critical_issues": critical_count,
            "high_issues": high_count,
            "issues": all_issues,
            "passed": len(all_issues) == 0
        }


def main():
    """Main entry point"""
    checker = SecurityChecker()
    results = checker.run_check()

    print(f"Security Check Results:")
    print(f"Total issues: {results['total_issues']}")
    print(f"Critical: {results['critical_issues']}")
    print(f"High: {results['high_issues']}")

    if results['issues']:
        print("\nIssues found:")
        for issue in results['issues']:
            print(f"  {issue['severity']}: {issue['file']}:{issue['line']} - {issue['issue']}")

    if not results['passed']:
        print("\n❌ Security check failed!")
        sys.exit(1)
    else:
        print("\n✅ Security check passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
