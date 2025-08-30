#!/usr/bin/env python3
"""
Comprehensive Project Test and Fix Script
Systematické testování a oprava celého projektu
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import traceback

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "workers"))
sys.path.append(str(project_root / "src"))


def test_basic_imports():
    """Test základních importů a závislostí"""
    print("🔍 Testing Basic Imports...")

    basic_imports = {
        "pathlib": "Path",
        "json": "loads",
        "datetime": "datetime",
        "asyncio": "create_task",
        "logging": "getLogger",
        "hashlib": "sha256",
        "dataclasses": "dataclass",
    }

    results = {}
    for module, attr in basic_imports.items():
        try:
            mod = __import__(module)
            if hasattr(mod, attr):
                results[module] = "✅ SUCCESS"
            else:
                results[module] = f"❌ Missing {attr}"
        except ImportError as e:
            results[module] = f"❌ ImportError: {e}"

    return results


def test_external_dependencies():
    """Test externích závislostí"""
    print("📦 Testing External Dependencies...")

    dependencies = [
        "requests",
        "aiohttp",
        "polars",
        "dramatiq",
        "fastapi",
        "pydantic",
        "redis",
        "lancedb",
    ]

    results = {}
    for dep in dependencies:
        try:
            __import__(dep)
            results[dep] = "✅ Available"
        except ImportError as e:
            results[dep] = f"❌ Missing: {e}"

    return results


def test_workers():
    """Test všech workers"""
    print("⚙️ Testing Workers...")

    workers_dir = project_root / "workers"
    worker_files = [
        "acquisition_worker.py",
        "discovery_worker.py",
        "llm_worker.py",
        "processing_worker.py",
        "autonomous_worker.py",
        "image_processor.py",
    ]

    results = {}
    for worker_file in worker_files:
        worker_path = workers_dir / worker_file
        if worker_path.exists():
            try:
                # Test syntax by compiling
                with open(worker_path, "r", encoding="utf-8") as f:
                    code = f.read()

                compile(code, worker_path, "exec")
                results[worker_file] = "✅ Syntax OK"

                # Try basic import
                module_name = worker_file.replace(".py", "")
                try:
                    mod = __import__(module_name)
                    results[worker_file] += " + Import OK"
                except Exception as e:
                    results[worker_file] += f" - Import Failed: {str(e)[:50]}"

            except SyntaxError as e:
                results[worker_file] = f"❌ Syntax Error: {e}"
            except Exception as e:
                results[worker_file] = f"❌ Error: {e}"
        else:
            results[worker_file] = "❌ File Missing"

    return results


def test_phase_implementations():
    """Test implementace všech fází"""
    print("🎯 Testing Phase Implementations...")

    results = {}

    # Test Phase 1: Knowledge Graph
    try:
        from workers.processing_worker import EnhancedProcessingWorker

        worker = EnhancedProcessingWorker()
        results["Phase 1 - Knowledge Graph"] = "✅ Processing Worker OK"
    except Exception as e:
        results["Phase 1 - Knowledge Graph"] = f"❌ Error: {str(e)[:100]}"

    # Test Phase 2: Graph-RAG
    try:
        # Check if Graph-RAG functions exist
        from workers.processing_worker import search_rag_documents

        results["Phase 2 - Graph-RAG"] = "✅ RAG Functions OK"
    except Exception as e:
        results["Phase 2 - Graph-RAG"] = f"❌ Error: {str(e)[:100]}"

    # Test Phase 3: Multi-Modality
    try:
        from workers.image_processor import ImageProcessor

        processor = ImageProcessor()
        results["Phase 3 - Multi-Modality"] = "✅ Image Processor OK"
    except Exception as e:
        results["Phase 3 - Multi-Modality"] = f"❌ Error: {str(e)[:100]}"

    # Test Phase 4: Autonomous System
    try:
        from workers.autonomous_worker import AutonomousWorker

        worker = AutonomousWorker()
        results["Phase 4 - Autonomous"] = "✅ Autonomous Worker OK"
    except Exception as e:
        results["Phase 4 - Autonomous"] = f"❌ Error: {str(e)[:100]}"

    return results


def fix_common_issues():
    """Oprav běžné problémy v projektu"""
    print("🔧 Fixing Common Issues...")

    fixes_applied = []

    # Fix 1: Ensure data directories exist
    data_dirs = ["data", "data/cache", "artifacts", "logs"]
    for dir_name in data_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            fixes_applied.append(f"Created directory: {dir_name}")

    # Fix 2: Update import paths in autonomous_worker
    autonomous_worker_path = project_root / "workers" / "autonomous_worker.py"
    if autonomous_worker_path.exists():
        try:
            with open(autonomous_worker_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix Docker paths to local paths
            if "/app/data" in content:
                content = content.replace("/app/data", 'Path.cwd() / "data"')
                with open(autonomous_worker_path, "w", encoding="utf-8") as f:
                    f.write(content)
                fixes_applied.append("Fixed paths in autonomous_worker.py")
        except Exception as e:
            fixes_applied.append(f"Failed to fix autonomous_worker.py: {e}")

    # Fix 3: Update AlertManager paths
    if autonomous_worker_path.exists():
        try:
            with open(autonomous_worker_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix AlertManager paths
            if 'self.data_dir = Path("/app/data")' in content:
                content = content.replace(
                    'self.data_dir = Path("/app/data")', 'self.data_dir = Path.cwd() / "data"'
                )
                with open(autonomous_worker_path, "w", encoding="utf-8") as f:
                    f.write(content)
                fixes_applied.append("Fixed AlertManager paths")
        except Exception as e:
            fixes_applied.append(f"Failed to fix AlertManager paths: {e}")

    # Fix 4: Create mock modules for missing dependencies
    mock_modules_dir = project_root / "mock_modules"
    if not mock_modules_dir.exists():
        mock_modules_dir.mkdir()

        # Create mock dramatiq
        mock_dramatiq = mock_modules_dir / "mock_dramatiq.py"
        with open(mock_dramatiq, "w") as f:
            f.write(
                '''
"""Mock Dramatiq for local testing"""

class MockBroker:
    def __init__(self, url=None):
        pass

class MockDramatiq:
    def set_broker(self, broker):
        pass
    
    def actor(self, queue_name=None):
        def decorator(func):
            return func
        return decorator

dramatiq = MockDramatiq()

def set_broker(broker):
    pass

def actor(queue_name=None):
    def decorator(func):
        return func
    return decorator
'''
            )
        fixes_applied.append("Created mock Dramatiq module")

    return fixes_applied


def create_simplified_test_runner():
    """Vytvoř zjednodušený test runner"""
    test_file = project_root / "run_comprehensive_test.py"

    test_code = '''#!/usr/bin/env python3
"""
Simplified Test Runner
Spustí všechny testy bez závislostí na externích službách
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "workers"))

def test_basic_functionality():
    """Test základní funkcionality"""
    print("🧪 === COMPREHENSIVE PROJECT TEST ===\\n")
    
    # Test 1: File Structure
    print("1. Testing File Structure:")
    required_files = [
        "workers/acquisition_worker.py",
        "workers/processing_worker.py", 
        "workers/autonomous_worker.py",
        "workers/image_processor.py",
        "requirements.txt",
        "docker-compose.yml"
    ]
    
    for file_path in required_files:
        if (project_root / file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    # Test 2: Worker Syntax
    print("\\n2. Testing Worker Syntax:")
    workers = ["acquisition_worker", "processing_worker", "autonomous_worker", "image_processor"]
    
    for worker in workers:
        try:
            with open(project_root / "workers" / f"{worker}.py", 'r') as f:
                code = f.read()
            compile(code, f"{worker}.py", 'exec')
            print(f"   ✅ {worker}.py - Syntax OK")
        except Exception as e:
            print(f"   ❌ {worker}.py - Error: {str(e)[:60]}")
    
    # Test 3: Core Classes
    print("\\n3. Testing Core Classes:")
    try:
        import importlib.util
        
        # Test autonomous_worker
        spec = importlib.util.spec_from_file_location(
            "autonomous_worker", 
            project_root / "workers" / "autonomous_worker.py"
        )
        autonomous_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(autonomous_module)
        
        # Test basic class instantiation
        worker = autonomous_module.AutonomousWorker()
        print(f"   ✅ AutonomousWorker - Created with {len(worker.monitored_sources)} sources")
        
    except Exception as e:
        print(f"   ❌ AutonomousWorker - Error: {str(e)[:60]}")
    
    print("\\n🎉 Test completed!")

if __name__ == "__main__":
    test_basic_functionality()
'''

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_code)

    return test_file


def main():
    """Hlavní test a oprava funkce"""
    print("🚀 === COMPREHENSIVE PROJECT AUDIT & FIX ===")
    print(f"Project root: {project_root}")
    print()

    # Test základních importů
    basic_results = test_basic_imports()
    print("Basic Imports Results:")
    for module, result in basic_results.items():
        print(f"  {module}: {result}")
    print()

    # Test externích závislostí
    dep_results = test_external_dependencies()
    print("External Dependencies Results:")
    for dep, result in dep_results.items():
        print(f"  {dep}: {result}")
    print()

    # Test workers
    worker_results = test_workers()
    print("Workers Test Results:")
    for worker, result in worker_results.items():
        print(f"  {worker}: {result}")
    print()

    # Test phase implementations
    phase_results = test_phase_implementations()
    print("Phase Implementation Results:")
    for phase, result in phase_results.items():
        print(f"  {phase}: {result}")
    print()

    # Apply fixes
    fixes = fix_common_issues()
    print("Applied Fixes:")
    for fix in fixes:
        print(f"  ✅ {fix}")
    print()

    # Create simplified test runner
    test_file = create_simplified_test_runner()
    print(f"✅ Created simplified test runner: {test_file}")

    # Generate summary report
    summary = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "basic_imports": basic_results,
        "external_dependencies": dep_results,
        "workers": worker_results,
        "phases": phase_results,
        "fixes_applied": fixes,
        "recommendations": [
            "Install missing dependencies from requirements.txt",
            "Run Docker services for full functionality",
            "Use simplified test runner for local testing",
            "Check Phase 4 autonomous worker configuration",
        ],
    }

    # Save report
    report_file = project_root / "artifacts" / "comprehensive_audit_report.json"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📊 Comprehensive audit report saved: {report_file}")
    print()
    print("🎯 === AUDIT SUMMARY ===")

    # Count successes and failures
    total_tests = 0
    passed_tests = 0

    for results in [basic_results, dep_results, worker_results, phase_results]:
        for result in results.values():
            total_tests += 1
            if "✅" in result:
                passed_tests += 1

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Fixes applied: {len(fixes)}")

    if success_rate >= 80:
        print("🟢 Project status: GOOD")
    elif success_rate >= 60:
        print("🟡 Project status: NEEDS MINOR FIXES")
    else:
        print("🔴 Project status: NEEDS MAJOR FIXES")

    print("✅ Comprehensive audit completed!")


if __name__ == "__main__":
    main()
