#!/usr/bin/env python3
"""
Environment Validation Script
Validuje prostředí, závislosti a M1 optimalizace

Author: Senior Python/MLOps Agent
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


def check_python_version() -> Tuple[bool, str]:
    """Zkontroluj Python verzi"""
    version = sys.version_info
    if version >= (3, 9):
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (vyžaduje 3.9+)"


def check_dependencies() -> List[Tuple[bool, str]]:
    """Zkontroluj závislosti"""
    required_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "qdrant_client",
        "fastapi",
        "aiohttp",
        "pydantic",
        "numpy",
        "pandas",
        "yaml",
        "asyncio",
    ]

    results = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            results.append((True, f"✅ {package}"))
        except ImportError:
            results.append((False, f"❌ {package} (chybí)"))

    return results


def check_pytorch_backend() -> Tuple[bool, str]:
    """Zkontroluj PyTorch backend pro M1"""
    try:
        import torch

        info = []
        if torch.backends.mps.is_available():
            info.append("MPS (Metal)")
        if torch.cuda.is_available():
            info.append("CUDA")

        if info:
            return True, f"✅ PyTorch backends: {', '.join(info)}"
        else:
            return True, "⚠️ PyTorch: pouze CPU backend"

    except ImportError:
        return False, "❌ PyTorch není nainstalován"


def check_ollama_connection() -> Tuple[bool, str]:
    """Zkontroluj připojení k Ollama"""
    import asyncio
    import aiohttp

    async def test_connection():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        return True, f"✅ Ollama: {len(models)} modelů dostupných"
                    else:
                        return False, f"❌ Ollama: HTTP {response.status}"
        except Exception as e:
            return False, f"❌ Ollama: {str(e)}"

    try:
        return asyncio.run(test_connection())
    except Exception as e:
        return False, f"❌ Ollama test selhání: {str(e)}"


def check_directories() -> List[Tuple[bool, str]]:
    """Zkontroluj potřebné adresáře"""
    required_dirs = [
        "src",
        "tests",
        "configs",
        "scripts",
        "docs",
        "data",
        "logs",
        "artifacts",
        "research_cache",
    ]

    results = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            results.append((True, f"✅ {dir_name}/"))
        else:
            # Vytvoř adresář
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results.append((True, f"✅ {dir_name}/ (vytvořeno)"))
            except Exception as e:
                results.append((False, f"❌ {dir_name}/: {str(e)}"))

    return results


def check_config_files() -> List[Tuple[bool, str]]:
    """Zkontroluj konfigurační soubory"""
    config_files = ["config.yaml", "config_m1_local.yaml", "requirements.txt"]

    results = []
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            results.append((True, f"✅ {config_file}"))
        else:
            results.append((False, f"❌ {config_file} (chybí)"))

    return results


def check_git_setup() -> Tuple[bool, str]:
    """Zkontroluj Git setup"""
    try:
        # Check if .git exists
        if not Path(".git").exists():
            return False, "❌ Git repository není inicializován"

        # Check pre-commit
        if Path(".pre-commit-config.yaml").exists():
            return True, "✅ Git + pre-commit setup"
        else:
            return True, "⚠️ Git OK, pre-commit chybí"

    except Exception as e:
        return False, f"❌ Git check selhání: {str(e)}"


def generate_validation_report() -> Dict[str, Any]:
    """Vygeneruj kompletní validation report"""
    print("🔧 Validating DeepResearchTool Environment")
    print("=" * 50)

    report = {"timestamp": str(Path.cwd()), "validation_results": {}}

    # Python version
    success, msg = check_python_version()
    print(msg)
    report["validation_results"]["python_version"] = {"success": success, "message": msg}

    print("\n📦 Dependencies:")
    dep_results = check_dependencies()
    all_deps_ok = all(result[0] for result in dep_results)
    for success, msg in dep_results:
        print(f"  {msg}")
    report["validation_results"]["dependencies"] = {
        "success": all_deps_ok,
        "details": [{"success": s, "message": m} for s, m in dep_results],
    }

    print("\n🔥 PyTorch Backend:")
    success, msg = check_pytorch_backend()
    print(f"  {msg}")
    report["validation_results"]["pytorch_backend"] = {"success": success, "message": msg}

    print("\n🦙 Ollama Connection:")
    success, msg = check_ollama_connection()
    print(f"  {msg}")
    report["validation_results"]["ollama"] = {"success": success, "message": msg}

    print("\n📁 Directories:")
    dir_results = check_directories()
    all_dirs_ok = all(result[0] for result in dir_results)
    for success, msg in dir_results:
        print(f"  {msg}")
    report["validation_results"]["directories"] = {
        "success": all_dirs_ok,
        "details": [{"success": s, "message": m} for s, m in dir_results],
    }

    print("\n⚙️ Configuration Files:")
    config_results = check_config_files()
    all_configs_ok = all(result[0] for result in config_results)
    for success, msg in config_results:
        print(f"  {msg}")
    report["validation_results"]["config_files"] = {
        "success": all_configs_ok,
        "details": [{"success": s, "message": m} for s, m in config_results],
    }

    print("\n🔀 Git Setup:")
    success, msg = check_git_setup()
    print(f"  {msg}")
    report["validation_results"]["git_setup"] = {"success": success, "message": msg}

    # Overall status
    critical_checks = [
        report["validation_results"]["python_version"]["success"],
        all_deps_ok,
        all_dirs_ok,
    ]

    overall_success = all(critical_checks)
    report["overall_success"] = overall_success

    print("\n" + "=" * 50)
    if overall_success:
        print("✅ Environment validation PASSED")
        print("\nReady for:")
        print("  make smoke-test")
        print("  make eval")
    else:
        print("❌ Environment validation FAILED")
        print("\nFix issues above and rerun:")
        print("  make validate-env")

    return report


def main():
    """Main entry point"""
    report = generate_validation_report()

    # Save report
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with open(artifacts_dir / "env_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved: artifacts/env_validation_report.json")

    # Exit with appropriate code
    if report["overall_success"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
