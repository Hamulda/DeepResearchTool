#!/usr/bin/env python3
"""
Validační skript pro Fázi 1 - Profesionalizace kódu
Kontroluje, zda jsou splněny všechny kritéria přijetí
"""

import os
import sys
import importlib
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import json


class Phase1Validator:
    """Validátor pro kritéria přijetí Fáze 1"""
    
    def __init__(self):
        self.results = {
            "task_1_1_poetry": False,
            "task_1_2_secrets": False,
            "task_1_3_error_handling": False,
            "task_1_4_testing": False,
            "overall_score": 0
        }
        self.details = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Provede kompletní validaci Fáze 1"""
        print("🔍 Spouštím validaci Fáze 1: Refaktoring a Profesionalizace Kódu\n")
        
        self._validate_poetry_management()
        self._validate_secrets_management()
        self._validate_error_handling()
        self._validate_testing_framework()
        
        self._calculate_overall_score()
        self._print_summary()
        
        return {
            "results": self.results,
            "details": self.details,
            "passed": self.results["overall_score"] >= 80
        }
    
    def _validate_poetry_management(self):
        """Úkol 1.1: Správa závislostí pomocí Poetry"""
        print("📦 Validuji úkol 1.1: Správa závislostí pomocí Poetry")
        
        checks = []
        
        # Check pyproject.toml existence
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            checks.append("✅ pyproject.toml existuje")
            
            # Check tenacity dependency
            content = pyproject_path.read_text()
            if "tenacity" in content:
                checks.append("✅ tenacity knihovna je v dependencies")
            else:
                checks.append("❌ tenacity knihovna chybí v dependencies")
        else:
            checks.append("❌ pyproject.toml neexistuje")
        
        # Check requirements.txt absence
        requirements_path = Path("requirements.txt")
        if not requirements_path.exists():
            checks.append("✅ requirements.txt byl odstraněn")
        else:
            checks.append("❌ requirements.txt stále existuje")
        
        # Check Makefile updates (simplified check)
        makefile_path = Path("Makefile")
        if makefile_path.exists():
            content = makefile_path.read_text()
            if "poetry" in content:
                checks.append("✅ Makefile obsahuje poetry příkazy")
            else:
                checks.append("⚠️  Makefile neobsahuje poetry příkazy")
        
        success_rate = sum(1 for check in checks if check.startswith("✅")) / len(checks)
        self.results["task_1_1_poetry"] = success_rate >= 0.75
        
        self.details.extend(checks)
        print(f"   Úspěšnost: {success_rate:.1%}\n")
    
    def _validate_secrets_management(self):
        """Úkol 1.2: Správa konfigurací a tajemství"""
        print("🔐 Validuji úkol 1.2: Správa konfigurací a tajemství")
        
        checks = []
        
        # Check .env.example existence
        if Path(".env.example").exists():
            checks.append("✅ .env.example existuje")
        else:
            checks.append("❌ .env.example neexistuje")
        
        # Check .env in .gitignore
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            if ".env" in content:
                checks.append("✅ .env je v .gitignore")
            else:
                checks.append("❌ .env není v .gitignore")
        
        # Check new config system
        config_path = Path("src/core/config.py")
        if config_path.exists():
            checks.append("✅ Nový konfigurační systém existuje")
            
            content = config_path.read_text()
            if "pydantic" in content and "BaseSettings" in content:
                checks.append("✅ Používá Pydantic pro type-safe config")
            else:
                checks.append("❌ Nepoužívá Pydantic pro konfiguraci")
                
            if "os.getenv" in content or "env=" in content:
                checks.append("✅ Načítá z environment variables")
            else:
                checks.append("❌ Nenačítá z environment variables")
        else:
            checks.append("❌ Nový konfigurační systém neexistuje")
        
        # Check for hardcoded secrets in code
        hardcoded_found = self._scan_for_hardcoded_secrets()
        if not hardcoded_found:
            checks.append("✅ Žádné hardcoded secrets nalezeny")
        else:
            checks.append(f"❌ Nalezeny hardcoded secrets: {len(hardcoded_found)}")
        
        success_rate = sum(1 for check in checks if check.startswith("✅")) / len(checks)
        self.results["task_1_2_secrets"] = success_rate >= 0.8
        
        self.details.extend(checks)
        print(f"   Úspěšnost: {success_rate:.1%}\n")
    
    def _validate_error_handling(self):
        """Úkol 1.3: Robustní zpracování chyb"""
        print("🛡️  Validuji úkol 1.3: Robustní zpracování chyb")
        
        checks = []
        
        # Check error handling module
        error_handling_path = Path("src/core/error_handling.py")
        if error_handling_path.exists():
            checks.append("✅ Error handling modul existuje")
            
            content = error_handling_path.read_text()
            
            if "tenacity" in content:
                checks.append("✅ Používá tenacity pro retry")
            else:
                checks.append("❌ Nepoužívá tenacity pro retry")
                
            if "CircuitBreaker" in content:
                checks.append("✅ Implementuje Circuit Breaker pattern")
            else:
                checks.append("❌ Neimplementuje Circuit Breaker pattern")
                
            if "ErrorAggregator" in content:
                checks.append("✅ Má error aggregation")
            else:
                checks.append("❌ Nemá error aggregation")
        else:
            checks.append("❌ Error handling modul neexistuje")
        
        # Check updated scrapers
        web_scraper_path = Path("src/scrapers/web_scraper.py")
        if web_scraper_path.exists():
            content = web_scraper_path.read_text()
            if "error_handling" in content and "retry" in content:
                checks.append("✅ Web scraper používá robustní error handling")
            else:
                checks.append("❌ Web scraper nepoužívá robustní error handling")
        
        tor_scraper_path = Path("src/scrapers/tor_scraper.py")
        if tor_scraper_path.exists():
            content = tor_scraper_path.read_text()
            if "error_handling" in content and "circuit_breaker" in content:
                checks.append("✅ Tor scraper používá robustní error handling")
            else:
                checks.append("❌ Tor scraper nepoužívá robustní error handling")
        
        success_rate = sum(1 for check in checks if check.startswith("✅")) / len(checks)
        self.results["task_1_3_error_handling"] = success_rate >= 0.8
        
        self.details.extend(checks)
        print(f"   Úspěšnost: {success_rate:.1%}\n")
    
    def _validate_testing_framework(self):
        """Úkol 1.4: Zavedení testování a statické analýzy"""
        print("🧪 Validuji úkol 1.4: Testování a statická analýza")
        
        checks = []
        
        # Check pytest configuration
        pytest_ini_path = Path("pytest.ini")
        pyproject_path = Path("pyproject.toml")
        
        if pytest_ini_path.exists() or (pyproject_path.exists() and "pytest" in pyproject_path.read_text()):
            checks.append("✅ Pytest konfigurace existuje")
        else:
            checks.append("❌ Pytest konfigurace neexistuje")
        
        # Check mypy configuration
        mypy_ini_path = Path("mypy.ini")
        if mypy_ini_path.exists() or (pyproject_path.exists() and "mypy" in pyproject_path.read_text()):
            checks.append("✅ MyPy konfigurace existuje")
        else:
            checks.append("❌ MyPy konfigurace neexistuje")
        
        # Check test files
        test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
        if test_files:
            checks.append(f"✅ Nalezeno {len(test_files)} testovacích souborů")
        else:
            checks.append("❌ Žádné testovací soubory nenalezeny")
        
        # Check for dev dependencies
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            dev_deps = ["pytest", "mypy", "ruff", "bandit"]
            found_deps = sum(1 for dep in dev_deps if dep in content)
            if found_deps >= 3:
                checks.append(f"✅ Vývojové závislosti nalezeny ({found_deps}/4)")
            else:
                checks.append(f"❌ Chybí vývojové závislosti ({found_deps}/4)")
        
        success_rate = sum(1 for check in checks if check.startswith("✅")) / len(checks)
        self.results["task_1_4_testing"] = success_rate >= 0.75
        
        self.details.extend(checks)
        print(f"   Úspěšnost: {success_rate:.1%}\n")
    
    def _scan_for_hardcoded_secrets(self) -> List[str]:
        """Skenuje kód pro hardcoded secrets"""
        suspicious_patterns = [
            "api_key",
            "secret_key", 
            "password",
            "token",
            "sk-", # OpenAI API keys
            "xoxb-", # Slack tokens
        ]
        
        found_secrets = []
        
        # Scan Python files
        for py_file in Path("src").rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in suspicious_patterns:
                    if f'"{pattern}' in content or f"'{pattern}" in content:
                        # Check if it's not just a variable name
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line and '=' in line and not line.strip().startswith('#'):
                                # Skip if it's loading from env
                                if 'os.getenv' not in line and 'env=' not in line:
                                    found_secrets.append(f"{py_file}:{i+1}")
            except Exception:
                continue
        
        return found_secrets
    
    def _calculate_overall_score(self):
        """Vypočítá celkové skóre"""
        task_scores = [
            self.results["task_1_1_poetry"],
            self.results["task_1_2_secrets"], 
            self.results["task_1_3_error_handling"],
            self.results["task_1_4_testing"]
        ]
        
        self.results["overall_score"] = (sum(task_scores) / len(task_scores)) * 100
    
    def _print_summary(self):
        """Vytiskne shrnutí validace"""
        print("=" * 60)
        print("📊 SHRNUTÍ VALIDACE FÁZE 1")
        print("=" * 60)
        
        tasks = [
            ("1.1 Poetry správa závislostí", self.results["task_1_1_poetry"]),
            ("1.2 Správa tajemství", self.results["task_1_2_secrets"]),
            ("1.3 Robustní error handling", self.results["task_1_3_error_handling"]),
            ("1.4 Testování a analýza", self.results["task_1_4_testing"])
        ]
        
        for task_name, passed in tasks:
            status = "✅ SPLNĚNO" if passed else "❌ NESPLNĚNO"
            print(f"{task_name:.<40} {status}")
        
        print("-" * 60)
        overall_status = "🎉 ÚSPĚCH" if self.results["overall_score"] >= 80 else "⚠️  POTŘEBUJE PRÁCI"
        print(f"Celkové skóre: {self.results['overall_score']:.1f}% - {overall_status}")
        
        if self.results["overall_score"] >= 80:
            print("\n🚀 Fáze 1 je úspěšně dokončena! Můžete pokračovat na Fázi 2.")
        else:
            print("\n🔧 Fáze 1 potřebuje další práci před pokračováním na Fázi 2.")
    
    def save_results(self, filename: str = "phase1_validation_results.json"):
        """Uloží výsledky validace do JSON souboru"""
        with open(filename, 'w') as f:
            json.dump({
                "phase": 1,
                "timestamp": "2025-08-30",
                "results": self.results,
                "details": self.details
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Výsledky uloženy do {filename}")


def main():
    """Hlavní funkce"""
    validator = Phase1Validator()
    results = validator.validate_all()
    validator.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()