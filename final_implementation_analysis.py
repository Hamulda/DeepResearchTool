#!/usr/bin/env python3
"""
Finální implementační analýza pro DeepResearchTool
Provádí kompletní audit a validaci všech komponent projektu
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime

# Přidání src do path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ProjectAnalyzer:
    """Analyzátor projektu pro kompletní audit"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0.0",
            "project_structure": {},
            "code_quality": {},
            "test_coverage": {},
            "configuration": {},
            "dependencies": {},
            "security": {},
            "performance": {},
            "recommendations": []
        }
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analýza struktury projektu"""
        print("🔍 Analyzuji strukturu projektu...")
        
        structure_analysis = {
            "core_modules": self._check_core_modules(),
            "test_structure": self._check_test_structure(),
            "documentation": self._check_documentation(),
            "configuration_files": self._check_configuration()
        }
        
        self.results["project_structure"] = structure_analysis
        return structure_analysis
    
    def _check_core_modules(self) -> Dict[str, Any]:
        """Kontrola jádrových modulů"""
        core_paths = [
            "src/core/config.py",
            "src/core/pipeline.py",
            "src/core/vector_store.py",
            "main.py",
            "cli.py",
            "dashboard.py"
        ]
        
        module_status = {}
        for path in core_paths:
            file_path = Path(path)
            if file_path.exists():
                size = file_path.stat().st_size
                module_status[path] = {
                    "exists": True,
                    "size": size,
                    "empty": size == 0
                }
            else:
                module_status[path] = {
                    "exists": False,
                    "size": 0,
                    "empty": True
                }
        
        return module_status
    
    def _check_test_structure(self) -> Dict[str, Any]:
        """Kontrola struktury testů"""
        test_dirs = ["tests/unit", "tests/integration"]
        test_analysis = {}
        
        for test_dir in test_dirs:
            test_path = Path(test_dir)
            if test_path.exists():
                test_files = list(test_path.glob("test_*.py"))
                test_analysis[test_dir] = {
                    "exists": True,
                    "test_count": len(test_files),
                    "files": [f.name for f in test_files]
                }
            else:
                test_analysis[test_dir] = {
                    "exists": False,
                    "test_count": 0,
                    "files": []
                }
        
        return test_analysis
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Kontrola dokumentace"""
        doc_files = [
            "README.md",
            "CHANGELOG.md",
            "IMPLEMENTATION_SUMMARY.md",
            "docs/"
        ]
        
        doc_status = {}
        for doc in doc_files:
            doc_path = Path(doc)
            doc_status[doc] = {
                "exists": doc_path.exists(),
                "is_directory": doc_path.is_dir() if doc_path.exists() else False
            }
        
        return doc_status
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Kontrola konfiguračních souborů"""
        config_files = [
            "pyproject.toml",
            "config.yaml",
            "config_m1_local.yaml",
            ".env.example",
            "docker-compose.yml"
        ]
        
        config_status = {}
        for config in config_files:
            config_path = Path(config)
            config_status[config] = {
                "exists": config_path.exists(),
                "size": config_path.stat().st_size if config_path.exists() else 0
            }
        
        return config_status
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analýza kvality kódu"""
        print("📊 Analyzuji kvalitu kódu...")
        
        quality_metrics = {
            "python_files_count": self._count_python_files(),
            "empty_files": self._find_empty_files(),
            "large_files": self._find_large_files(),
            "imports_analysis": self._analyze_imports()
        }
        
        self.results["code_quality"] = quality_metrics
        return quality_metrics
    
    def _count_python_files(self) -> int:
        """Počítání Python souborů"""
        return len(list(Path().rglob("*.py")))
    
    def _find_empty_files(self) -> List[str]:
        """Nalezení prázdných souborů"""
        empty_files = []
        for py_file in Path().rglob("*.py"):
            if py_file.stat().st_size == 0:
                empty_files.append(str(py_file))
        return empty_files
    
    def _find_large_files(self) -> List[Dict[str, Any]]:
        """Nalezení velkých souborů (>1MB)"""
        large_files = []
        for py_file in Path().rglob("*.py"):
            size = py_file.stat().st_size
            if size > 1024 * 1024:  # 1MB
                large_files.append({
                    "file": str(py_file),
                    "size_mb": round(size / (1024 * 1024), 2)
                })
        return large_files
    
    def _analyze_imports(self) -> Dict[str, Any]:
        """Analýza importů"""
        import_analysis = {
            "external_dependencies": set(),
            "internal_imports": set(),
            "problematic_imports": []
        }
        
        for py_file in Path("src").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Jednoduchá analýza importů
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        if 'src.' in line:
                            import_analysis["internal_imports"].add(line)
                        else:
                            import_analysis["external_dependencies"].add(line)
                            
            except Exception as e:
                import_analysis["problematic_imports"].append({
                    "file": str(py_file),
                    "error": str(e)
                })
        
        # Konverze setů na listy pro JSON serializaci
        import_analysis["external_dependencies"] = list(import_analysis["external_dependencies"])
        import_analysis["internal_imports"] = list(import_analysis["internal_imports"])
        
        return import_analysis
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analýza závislostí"""
        print("📦 Analyzuji závislosti...")
        
        deps_analysis = {
            "pyproject_toml": self._analyze_pyproject(),
            "requirements_files": self._find_requirements(),
            "lock_files": self._check_lock_files()
        }
        
        self.results["dependencies"] = deps_analysis
        return deps_analysis
    
    def _analyze_pyproject(self) -> Dict[str, Any]:
        """Analýza pyproject.toml"""
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            return {"exists": False}
        
        try:
            import tomli
            with open(pyproject_path, 'rb') as f:
                data = tomli.load(f)
            
            return {
                "exists": True,
                "has_dependencies": "dependencies" in data.get("project", {}),
                "has_dev_dependencies": "dev" in data.get("project", {}).get("optional-dependencies", {}),
                "dependency_count": len(data.get("project", {}).get("dependencies", [])),
                "build_system": data.get("build-system", {}).get("build-backend", "unknown")
            }
        except ImportError:
            return {
                "exists": True,
                "parse_error": "tomli not available for parsing"
            }
        except Exception as e:
            return {
                "exists": True,
                "parse_error": str(e)
            }
    
    def _find_requirements(self) -> List[str]:
        """Nalezení requirements souborů"""
        req_patterns = ["requirements*.txt", "requirements/*.txt"]
        req_files = []
        
        for pattern in req_patterns:
            req_files.extend([str(p) for p in Path().glob(pattern)])
        
        return req_files
    
    def _check_lock_files(self) -> Dict[str, bool]:
        """Kontrola lock souborů"""
        lock_files = ["uv.lock", "poetry.lock", "Pipfile.lock"]
        return {lock_file: Path(lock_file).exists() for lock_file in lock_files}
    
    def generate_recommendations(self) -> List[str]:
        """Generování doporučení"""
        print("💡 Generuji doporučení...")
        
        recommendations = []
        
        # Kontrola prázdných souborů
        empty_files = self.results.get("code_quality", {}).get("empty_files", [])
        if empty_files:
            recommendations.append(
                f"⚠️  Nalezeno {len(empty_files)} prázdných souborů - doporučuji implementovat nebo odstranit"
            )
        
        # Kontrola testů
        test_structure = self.results.get("project_structure", {}).get("test_structure", {})
        unit_tests = test_structure.get("tests/unit", {}).get("test_count", 0)
        integration_tests = test_structure.get("tests/integration", {}).get("test_count", 0)
        
        if unit_tests < 5:
            recommendations.append("📝 Doporučuji přidat více unit testů (méně než 5)")
        
        if integration_tests < 3:
            recommendations.append("🔗 Doporučuji přidat více integration testů (méně než 3)")
        
        # Kontrola dokumentace
        doc_status = self.results.get("project_structure", {}).get("documentation", {})
        if not doc_status.get("README.md", {}).get("exists", False):
            recommendations.append("📚 Chybí README.md dokumentace")
        
        # Kontrola závislostí
        deps = self.results.get("dependencies", {})
        if not deps.get("pyproject_toml", {}).get("exists", False):
            recommendations.append("📦 Doporučuji používat pyproject.toml pro správu závislostí")
        
        self.results["recommendations"] = recommendations
        return recommendations
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Spuštění kompletní analýzy"""
        print("🚀 Spouštím kompletní analýzu projektu...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Provedení všech analýz
        self.analyze_project_structure()
        self.analyze_code_quality()
        self.analyze_dependencies()
        self.generate_recommendations()
        
        # Finalizace výsledků
        self.results["analysis_duration"] = time.time() - start_time
        self.results["total_recommendations"] = len(self.results["recommendations"])
        
        print(f"✅ Analýza dokončena za {self.results['analysis_duration']:.2f}s")
        print(f"💡 Vygenerováno {self.results['total_recommendations']} doporučení")
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Uložení výsledků do souboru"""
        if not filename:
            timestamp = int(time.time())
            filename = f"final_implementation_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Výsledky uloženy do: {filename}")
        return filename
    
    def print_summary(self):
        """Výpis shrnutí analýzy"""
        print("\n" + "=" * 60)
        print("📊 SHRNUTÍ ANALÝZY PROJEKTU")
        print("=" * 60)
        
        # Struktura projektu
        structure = self.results.get("project_structure", {})
        print(f"\n🏗️  STRUKTURA PROJEKTU:")
        
        core_modules = structure.get("core_modules", {})
        existing_modules = sum(1 for module in core_modules.values() if module.get("exists", False))
        print(f"   • Jádrové moduly: {existing_modules}/{len(core_modules)} existují")
        
        # Kvalita kódu
        quality = self.results.get("code_quality", {})
        print(f"\n📊 KVALITA KÓDU:")
        print(f"   • Python soubory: {quality.get('python_files_count', 0)}")
        print(f"   • Prázdné soubory: {len(quality.get('empty_files', []))}")
        print(f"   • Velké soubory: {len(quality.get('large_files', []))}")
        
        # Závislosti
        deps = self.results.get("dependencies", {})
        print(f"\n📦 ZÁVISLOSTI:")
        pyproject = deps.get("pyproject_toml", {})
        if pyproject.get("exists", False):
            print(f"   • pyproject.toml: ✅ ({pyproject.get('dependency_count', 0)} závislostí)")
        else:
            print(f"   • pyproject.toml: ❌")
        
        # Doporučení
        recommendations = self.results.get("recommendations", [])
        print(f"\n💡 DOPORUČENÍ ({len(recommendations)}):")
        for rec in recommendations[:5]:  # Zobrazit prvních 5
            print(f"   {rec}")
        
        if len(recommendations) > 5:
            print(f"   ... a {len(recommendations) - 5} dalších")
        
        print("\n" + "=" * 60)


async def main():
    """Hlavní funkce"""
    analyzer = ProjectAnalyzer()
    
    try:
        # Spuštění analýzy
        results = await analyzer.run_full_analysis()
        
        # Výpis shrnutí
        analyzer.print_summary()
        
        # Uložení výsledků
        filename = analyzer.save_results()
        
        print(f"\n🎉 Kompletní analýza dokončena!")
        print(f"📄 Detailní výsledky: {filename}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Chyba během analýzy: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())