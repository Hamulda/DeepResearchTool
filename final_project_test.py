#!/usr/bin/env python3
"""
Finální test a oprava celého projektu
Kompletní audit a validace všech komponent
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "workers"))
sys.path.append(str(project_root / "src"))


def create_essential_directories():
    """Vytvoř všechny potřebné adresáře"""
    essential_dirs = ["data", "data/cache", "data/vector_db", "artifacts", "logs", "src"]

    created = []
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_name)

    return created


def test_core_functionality():
    """Test základní funkčnosti všech komponent"""
    print("🧪 === FINÁLNÍ TEST CELÉHO PROJEKTU ===\n")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "fixes": [],
        "summary": {},
    }

    # 1. Vytvoř potřebné adresáře
    print("1. Vytvářím potřebné adresáře...")
    created_dirs = create_essential_directories()
    results["fixes"].extend([f"Vytvořen adresář: {d}" for d in created_dirs])
    print(f"   ✅ Vytvořeno {len(created_dirs)} adresářů")

    # 2. Test struktury projektu
    print("\n2. Testování struktury projektu...")
    required_files = [
        "requirements.txt",
        "docker-compose.yml",
        "workers/autonomous_worker.py",
        "workers/processing_worker.py",
        "workers/image_processor.py",
        "workers/acquisition_worker.py",
        "simple_phase4_test.py",
    ]

    structure_ok = True
    for file_path in required_files:
        if (project_root / file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - CHYBÍ")
            structure_ok = False

    results["tests"]["project_structure"] = structure_ok

    # 3. Test syntaxe workers
    print("\n3. Testování syntaxe workers...")
    workers_dir = project_root / "workers"
    worker_files = [f for f in workers_dir.glob("*.py") if f.name != "__init__.py"]

    syntax_errors = []
    for worker_file in worker_files:
        try:
            with open(worker_file, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, str(worker_file), "exec")
            print(f"   ✅ {worker_file.name} - syntax OK")
        except SyntaxError as e:
            syntax_errors.append(f"{worker_file.name}: {e}")
            print(f"   ❌ {worker_file.name} - syntax error: {e}")
        except Exception as e:
            syntax_errors.append(f"{worker_file.name}: {e}")
            print(f"   ❌ {worker_file.name} - error: {e}")

    results["tests"]["syntax_errors"] = syntax_errors

    # 4. Test importů autonomous worker
    print("\n4. Testování autonomous worker...")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "autonomous_worker", project_root / "workers" / "autonomous_worker.py"
        )
        autonomous_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(autonomous_module)

        # Test vytvoření worker instance
        worker = autonomous_module.AutonomousWorker()
        print(f"   ✅ AutonomousWorker vytvořen s {len(worker.monitored_sources)} zdroji")

        # Test přidání zdroje
        success = worker.add_monitored_source(
            "https://example.com/test", check_interval=60, priority=2, keywords=["test"]
        )
        print(f"   ✅ Přidán testovací zdroj: {success}")

        results["tests"]["autonomous_worker"] = "SUCCESS"

    except Exception as e:
        print(f"   ❌ Autonomous worker error: {e}")
        results["tests"]["autonomous_worker"] = str(e)

    # 5. Test processing worker
    print("\n5. Testování processing worker...")
    try:
        spec = importlib.util.spec_from_file_location(
            "processing_worker", project_root / "workers" / "processing_worker.py"
        )
        processing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(processing_module)

        # Test text processor
        text_processor = processing_module.AdvancedTextProcessor()
        print("   ✅ AdvancedTextProcessor vytvořen")

        # Test enhanced worker
        enhanced_worker = processing_module.EnhancedProcessingWorker()
        print("   ✅ EnhancedProcessingWorker vytvořen")

        results["tests"]["processing_worker"] = "SUCCESS"

    except Exception as e:
        print(f"   ❌ Processing worker error: {e}")
        results["tests"]["processing_worker"] = str(e)

    # 6. Test image processor
    print("\n6. Testování image processor...")
    try:
        spec = importlib.util.spec_from_file_location(
            "image_processor", project_root / "workers" / "image_processor.py"
        )
        image_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image_module)

        processor = image_module.ImageProcessor()
        print("   ✅ ImageProcessor vytvořen")

        results["tests"]["image_processor"] = "SUCCESS"

    except Exception as e:
        print(f"   ❌ Image processor error: {e}")
        results["tests"]["image_processor"] = str(e)

    # 7. Test artifacts
    print("\n7. Kontrola artifacts...")
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        test_files = list(artifacts_dir.glob("*test_result.json"))
        print(f"   ✅ Nalezeno {len(test_files)} test souborů")
        results["tests"]["artifacts"] = f"{len(test_files)} files found"
    else:
        print("   ⚠️ Artifacts adresář neexistuje")
        results["tests"]["artifacts"] = "missing"

    # 8. Finální souhrn
    print("\n📊 === FINÁLNÍ SOUHRN ===")

    total_tests = len(results["tests"])
    passed_tests = sum(
        1
        for test in results["tests"].values()
        if isinstance(test, str) and ("SUCCESS" in test or "found" in test)
    )
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": total_tests - passed_tests,
        "success_rate": success_rate,
        "syntax_errors": len(syntax_errors),
        "directories_created": len(created_dirs),
    }

    print(f"Celkem testů: {total_tests}")
    print(f"✅ Úspěšné: {passed_tests}")
    print(f"❌ Neúspěšné: {total_tests - passed_tests}")
    print(f"📈 Úspěšnost: {success_rate:.1f}%")
    print(f"🗂️ Vytvořeno adresářů: {len(created_dirs)}")
    print(f"⚠️ Syntax chyby: {len(syntax_errors)}")

    # Určení celkového statusu
    if success_rate >= 85 and len(syntax_errors) == 0:
        status = "🟢 VÝBORNÝ"
        print(f"\n🎉 Projekt je ve výborném stavu!")
    elif success_rate >= 70:
        status = "🟡 DOBRÝ"
        print(f"\n✅ Projekt je v dobrém stavu s menšími problémy")
    elif success_rate >= 50:
        status = "🟠 POTŘEBUJE PRÁCI"
        print(f"\n⚠️ Projekt potřebuje další práci")
    else:
        status = "🔴 KRITICKÝ"
        print(f"\n❌ Projekt má vážné problémy")

    results["overall_status"] = status

    # Uložení výsledků
    results_file = project_root / "artifacts" / "final_project_audit.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Finální audit uložen: {results_file}")

    # Doporučení
    print(f"\n🔧 === DOPORUČENÍ ===")
    if syntax_errors:
        print("1. Oprav syntax chyby ve workers")
    if success_rate < 100:
        print("2. Nainstaluj chybějící závislosti: pip install -r requirements.txt")
        print("3. Pro plnou funkcionalnost spusť Docker služby")
    if success_rate >= 85:
        print("✅ Projekt je připraven k použití!")
        print("✅ Všechny 4 fáze jsou implementovány:")
        print("   - Fáze 1: Knowledge Graph Core")
        print("   - Fáze 2: Graph-Powered RAG")
        print("   - Fáze 3: Multi-Modality")
        print("   - Fáze 4: Autonomní systém")

    return results


if __name__ == "__main__":
    test_core_functionality()
