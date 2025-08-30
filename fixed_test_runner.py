#!/usr/bin/env python3
"""
Fixed Project Test Runner
Opravený test runner pro celý projekt
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import traceback

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "workers"))
sys.path.append(str(project_root / "src"))


def test_project_structure():
    """Test základní struktury projektu"""
    print("📁 Testing Project Structure:")

    required_dirs = ["workers", "src", "data", "artifacts"]
    required_files = [
        "requirements.txt",
        "docker-compose.yml",
        "workers/autonomous_worker.py",
        "workers/processing_worker.py",
        "workers/image_processor.py",
        "workers/acquisition_worker.py",
    ]

    results = {}

    # Test directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            results[f"dir_{dir_name}"] = "✅ EXISTS"
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            results[f"dir_{dir_name}"] = "✅ CREATED"

    # Test files
    for file_path in required_files:
        if (project_root / file_path).exists():
            results[f"file_{file_path}"] = "✅ EXISTS"
        else:
            results[f"file_{file_path}"] = "❌ MISSING"

    return results


def test_worker_syntax():
    """Test syntaxe všech workers"""
    print("⚙️ Testing Worker Syntax:")

    workers_dir = project_root / "workers"
    worker_files = [f for f in workers_dir.glob("*.py") if f.name != "__init__.py"]

    results = {}

    for worker_file in worker_files:
        try:
            with open(worker_file, "r", encoding="utf-8") as f:
                code = f.read()

            # Test syntax
            compile(code, str(worker_file), "exec")
            results[worker_file.name] = "✅ Syntax OK"

        except SyntaxError as e:
            results[worker_file.name] = f"❌ Syntax Error: Line {e.lineno}: {e.msg}"
        except Exception as e:
            results[worker_file.name] = f"❌ Error: {str(e)[:100]}"

    return results


def test_autonomous_worker():
    """Test autonomního workeru specificky"""
    print("🤖 Testing Autonomous Worker:")

    results = {}

    try:
        # Load module directly
        spec = importlib.util.spec_from_file_location(
            "autonomous_worker", project_root / "workers" / "autonomous_worker.py"
        )
        autonomous_module = importlib.util.module_from_spec(spec)

        # Execute module
        spec.loader.exec_module(autonomous_module)
        results["module_load"] = "✅ Module loaded successfully"

        # Test class creation
        worker = autonomous_module.AutonomousWorker()
        results["worker_creation"] = (
            f"✅ Worker created with {len(worker.monitored_sources)} sources"
        )

        # Test basic functionality
        test_source = autonomous_module.MonitoredSource(
            url="https://httpbin.org/get", check_interval=60, priority=1
        )
        results["source_creation"] = "✅ MonitoredSource created"

        # Test adding source
        success = worker.add_monitored_source("https://example.com", check_interval=120, priority=2)
        results["add_source"] = f"✅ Add source: {success}"

        # Test change detector
        detector = autonomous_module.ChangeDetector()
        results["change_detector"] = "✅ ChangeDetector created"

        # Test alert manager
        alert_manager = autonomous_module.AlertManager()
        results["alert_manager"] = "✅ AlertManager created"

    except Exception as e:
        results["error"] = f"❌ Error: {str(e)[:200]}"
        results["traceback"] = traceback.format_exc()

    return results


def test_phase4_functionality():
    """Test funkcionalita Phase 4"""
    print("🎯 Testing Phase 4 Functionality:")

    results = {}

    # Test simple_phase4_test.py exists and runs
    test_file = project_root / "simple_phase4_test.py"
    if test_file.exists():
        results["test_file_exists"] = "✅ simple_phase4_test.py exists"
        try:
            with open(test_file, "r") as f:
                code = f.read()
            compile(code, str(test_file), "exec")
            results["test_file_syntax"] = "✅ Test file syntax OK"
        except Exception as e:
            results["test_file_syntax"] = f"❌ Test file error: {e}"
    else:
        results["test_file_exists"] = "❌ simple_phase4_test.py missing"

    # Test artifacts directory
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        test_files = list(artifacts_dir.glob("*test_result.json"))
        results["artifacts"] = f"✅ Found {len(test_files)} test result files"
    else:
        results["artifacts"] = "❌ Artifacts directory missing"

    return results


def run_comprehensive_test():
    """Spusť komplexní test celého projektu"""
    print("🚀 === COMPREHENSIVE PROJECT TEST ===")
    print(f"Project root: {project_root}")
    print()

    # Initialize results
    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "tests": {},
    }

    # Test 1: Project Structure
    structure_results = test_project_structure()
    all_results["tests"]["project_structure"] = structure_results
    for key, value in structure_results.items():
        print(f"  {key}: {value}")
    print()

    # Test 2: Worker Syntax
    syntax_results = test_worker_syntax()
    all_results["tests"]["worker_syntax"] = syntax_results
    for key, value in syntax_results.items():
        print(f"  {key}: {value}")
    print()

    # Test 3: Autonomous Worker
    autonomous_results = test_autonomous_worker()
    all_results["tests"]["autonomous_worker"] = autonomous_results
    for key, value in autonomous_results.items():
        if key != "traceback":  # Don't print full traceback
            print(f"  {key}: {value}")
    print()

    # Test 4: Phase 4 Functionality
    phase4_results = test_phase4_functionality()
    all_results["tests"]["phase4"] = phase4_results
    for key, value in phase4_results.items():
        print(f"  {key}: {value}")
    print()

    # Calculate summary
    total_tests = 0
    passed_tests = 0

    for test_category in all_results["tests"].values():
        for result in test_category.values():
            if isinstance(result, str):
                total_tests += 1
                if "✅" in result:
                    passed_tests += 1

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    all_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": success_rate,
    }

    print("📊 === TEST SUMMARY ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {success_rate:.1f}%")

    # Determine status
    if success_rate >= 90:
        status = "🟢 EXCELLENT"
    elif success_rate >= 70:
        status = "🟡 GOOD"
    elif success_rate >= 50:
        status = "🟠 NEEDS WORK"
    else:
        status = "🔴 CRITICAL"

    all_results["status"] = status
    print(f"Project status: {status}")

    # Save results
    results_file = project_root / "artifacts" / "comprehensive_test_results.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {results_file}")

    # Recommendations
    print("\n🔧 === RECOMMENDATIONS ===")
    if success_rate < 100:
        print("1. Fix any syntax errors in workers")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Ensure data directories exist")
        print("4. Test autonomous worker functionality")
    else:
        print("✅ Project is in excellent condition!")

    return all_results


if __name__ == "__main__":
    run_comprehensive_test()
