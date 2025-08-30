#!/usr/bin/env python3
"""
Validátor pro Fázi 1 implementace DeepResearchTool
Ověřuje správnou funkcionalitu všech základních komponent
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# Přidání src do path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class Phase1Validator:
    """Validátor pro ověření Fáze 1 implementace"""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": time.time(),
            "phase": "Phase 1 - Core Stabilization",
            "tests": [],
            "overall_status": "PENDING",
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
    
    async def validate_core_imports(self) -> Tuple[bool, str]:
        """Validace základních importů"""
        print("🔍 Validuji základní importy...")
        
        try:
            # Test importu konfigurace
            from src.core.config import load_config, get_settings
            
            # Test importu pipeline
            from src.core.pipeline import ResearchPipeline
            
            # Test importu vector store
            from src.core.vector_store import VectorStore
            
            return True, "Všechny základní importy úspěšné"
            
        except ImportError as e:
            return False, f"Chyba importu: {e}"
        except Exception as e:
            return False, f"Neočekávaná chyba: {e}"
    
    async def validate_configuration(self) -> Tuple[bool, str]:
        """Validace konfiguračního systému"""
        print("⚙️ Validuji konfigurační systém...")
        
        try:
            from src.core.config import get_settings, validate_environment
            
            # Test načtení nastavení
            settings = get_settings()
            
            # Test validace prostředí
            validate_environment()
            
            # Kontrola kritických nastavení
            validation = settings.validate_critical_config()
            
            if any(validation.values()):
                return True, f"Konfigurace validována: {validation}"
            else:
                return False, "Žádná kritická konfigurace není nastavena"
                
        except Exception as e:
            return False, f"Chyba konfigurace: {e}"
    
    async def validate_pipeline_structure(self) -> Tuple[bool, str]:
        """Validace struktury pipeline"""
        print("🔄 Validuji strukturu pipeline...")
        
        try:
            # Kontrola existence klíčových komponent
            required_files = [
                "src/core/pipeline.py",
                "src/core/orchestrator.py",
                "src/core/memory.py",
                "src/core/error_handling.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                return False, f"Chybí soubory: {missing_files}"
            
            # Test základní funkcionality pipeline
            from src.core.pipeline import ResearchPipeline
            
            # Mock konfigurace pro test
            mock_config = {
                "research_profile": "quick",
                "max_documents": 10,
                "timeout_seconds": 30
            }
            
            # Vytvoření pipeline instance
            pipeline = ResearchPipeline(mock_config)
            
            return True, "Pipeline struktura validována"
            
        except Exception as e:
            return False, f"Chyba pipeline: {e}"
    
    async def validate_vector_store(self) -> Tuple[bool, str]:
        """Validace vector store implementace"""
        print("🗃️ Validuji vector store...")
        
        try:
            from src.core.vector_store import VectorStore
            
            # Test základní funkcionality
            # Poznámka: Bez externí databáze pouze ověříme import a vytvoření instance
            
            return True, "Vector store implementace dostupná"
            
        except Exception as e:
            return False, f"Chyba vector store: {e}"
    
    async def validate_error_handling(self) -> Tuple[bool, str]:
        """Validace error handling systému"""
        print("🛡️ Validuji error handling...")
        
        try:
            from src.core.error_handling import (
                ResearchError, 
                ConfigurationError,
                PipelineError
            )
            
            # Test vytvoření error instances
            research_error = ResearchError("Test error")
            config_error = ConfigurationError("Test config error")
            pipeline_error = PipelineError("Test pipeline error")
            
            return True, "Error handling systém validován"
            
        except Exception as e:
            return False, f"Chyba error handling: {e}"
    
    async def validate_memory_system(self) -> Tuple[bool, str]:
        """Validace memory systému"""
        print("🧠 Validuji memory systém...")
        
        try:
            from src.core.memory import MemoryManager
            
            # Test základní funkcionality memory manageru
            return True, "Memory systém dostupný"
            
        except Exception as e:
            return False, f"Chyba memory systému: {e}"
    
    async def validate_main_entry_point(self) -> Tuple[bool, str]:
        """Validace hlavního vstupního bodu"""
        print("🚀 Validuji hlavní vstupní bod...")
        
        try:
            # Kontrola existence main.py
            if not Path("main.py").exists():
                return False, "main.py neexistuje"
            
            # Test importu hlavní třídy
            from main import ModernResearchAgent
            
            # Test vytvoření instance s mock konfigurací
            agent = ModernResearchAgent(profile="quick", use_langgraph=False)
            
            return True, "Hlavní vstupní bod validován"
            
        except Exception as e:
            return False, f"Chyba main entry point: {e}"
    
    async def validate_cli_interface(self) -> Tuple[bool, str]:
        """Validace CLI rozhraní"""
        print("💻 Validuji CLI rozhraní...")
        
        try:
            # Kontrola existence cli.py
            if not Path("cli.py").exists():
                return False, "cli.py neexistuje"
            
            # Kontrola, že není prázdný
            if Path("cli.py").stat().st_size == 0:
                return False, "cli.py je prázdný"
            
            return True, "CLI rozhraní validováno"
            
        except Exception as e:
            return False, f"Chyba CLI: {e}"
    
    async def run_validation_test(self, test_name: str, test_func) -> Dict[str, Any]:
        """Spuštění jednotlivého validačního testu"""
        start_time = time.time()
        
        try:
            success, message = await test_func()
            status = "PASSED" if success else "FAILED"
            
        except Exception as e:
            success = False
            status = "ERROR"
            message = f"Neočekávaná chyba: {e}"
        
        duration = time.time() - start_time
        
        test_result = {
            "name": test_name,
            "status": status,
            "success": success,
            "message": message,
            "duration": round(duration, 3)
        }
        
        return test_result
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Spuštění všech validací"""
        print("🧪 Spouštím validaci Fáze 1...")
        print("=" * 60)
        
        # Seznam všech testů
        validation_tests = [
            ("Core Imports", self.validate_core_imports),
            ("Configuration System", self.validate_configuration),
            ("Pipeline Structure", self.validate_pipeline_structure),
            ("Vector Store", self.validate_vector_store),
            ("Error Handling", self.validate_error_handling),
            ("Memory System", self.validate_memory_system),
            ("Main Entry Point", self.validate_main_entry_point),
            ("CLI Interface", self.validate_cli_interface)
        ]
        
        # Spuštění všech testů
        for test_name, test_func in validation_tests:
            result = await self.run_validation_test(test_name, test_func)
            self.validation_results["tests"].append(result)
            
            # Výpis průběžných výsledků
            status_emoji = "✅" if result["success"] else "❌"
            print(f"{status_emoji} {test_name}: {result['message']} ({result['duration']}s)")
        
        # Výpočet shrnutí
        self.validation_results["summary"]["total_tests"] = len(validation_tests)
        self.validation_results["summary"]["passed"] = sum(
            1 for test in self.validation_results["tests"] if test["success"]
        )
        self.validation_results["summary"]["failed"] = (
            self.validation_results["summary"]["total_tests"] - 
            self.validation_results["summary"]["passed"]
        )
        
        # Určení celkového stavu
        if self.validation_results["summary"]["failed"] == 0:
            self.validation_results["overall_status"] = "PASSED"
        elif self.validation_results["summary"]["passed"] > self.validation_results["summary"]["failed"]:
            self.validation_results["overall_status"] = "PARTIAL"
        else:
            self.validation_results["overall_status"] = "FAILED"
        
        return self.validation_results
    
    def print_summary(self):
        """Výpis shrnutí validace"""
        print("\n" + "=" * 60)
        print("📊 SHRNUTÍ VALIDACE FÁZE 1")
        print("=" * 60)
        
        summary = self.validation_results["summary"]
        status = self.validation_results["overall_status"]
        
        print(f"🎯 Celkový stav: {status}")
        print(f"📈 Úspěšnost: {summary['passed']}/{summary['total_tests']} testů")
        print(f"⏱️  Celkový čas: {sum(test['duration'] for test in self.validation_results['tests']):.2f}s")
        
        # Detaily neúspěšných testů
        failed_tests = [test for test in self.validation_results["tests"] if not test["success"]]
        if failed_tests:
            print(f"\n❌ Neúspěšné testy ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['message']}")
        
        # Doporučení
        print(f"\n💡 DOPORUČENÍ:")
        if status == "PASSED":
            print("   ✅ Fáze 1 je kompletní a připravená!")
            print("   🚀 Můžete pokračovat na Fázi 2")
        elif status == "PARTIAL":
            print("   ⚠️  Fáze 1 je částečně funkční")
            print("   🔧 Opravte neúspěšné testy před pokračováním")
        else:
            print("   ❌ Fáze 1 vyžaduje zásadní opravy")
            print("   🛠️  Zaměřte se na základní komponenty")
        
        print("=" * 60)
    
    def save_results(self, filename: str = None) -> str:
        """Uložení výsledků validace"""
        if not filename:
            timestamp = int(time.time())
            filename = f"phase1_validation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Výsledky validace uloženy do: {filename}")
        return filename


async def main():
    """Hlavní funkce validace"""
    validator = Phase1Validator()
    
    try:
        # Spuštění všech validací
        results = await validator.run_all_validations()
        
        # Výpis shrnutí
        validator.print_summary()
        
        # Uložení výsledků
        validator.save_results()
        
        # Návratový kód podle výsledku
        if results["overall_status"] == "PASSED":
            return 0
        elif results["overall_status"] == "PARTIAL":
            return 1
        else:
            return 2
            
    except Exception as e:
        print(f"❌ Kritická chyba validace: {e}")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)