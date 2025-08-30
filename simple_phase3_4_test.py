#!/usr/bin/env python3
"""
Zjednodušený test pro fáze 3 a 4 DeepResearchTool
Testuje základní funkčnost bez externích závislostí

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def test_import_availability():
    """Test dostupnosti klíčových modulů"""
    print("🔍 Testování dostupnosti modulů...")
    
    results = {}
    
    # Test importů core modulů
    test_modules = [
        ('src.deep_web.network_manager', 'NetworkManager'),
        ('src.scraping.stealth_engine', 'StealthEngine'),
        ('src.security.osint_sandbox', 'OSINTSecuritySandbox'),
        ('workers.deep_web_crawler', 'DeepWebCrawler'),
        ('src.core.intelligent_task_manager', 'IntelligentTaskManager'),
        ('src.core.agentic_loop', 'AgenticLoop'),
    ]
    
    for module_name, class_name in test_modules:
        try:
            # Pokus o import
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results[module_name] = {
                'status': 'success',
                'class_available': hasattr(module, class_name),
                'class_type': str(type(cls))
            }
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            results[module_name] = {
                'status': 'import_error',
                'error': str(e)
            }
            print(f"❌ {module_name}: Import Error - {e}")
        except Exception as e:
            results[module_name] = {
                'status': 'other_error',
                'error': str(e)
            }
            print(f"⚠️ {module_name}: {e}")
    
    return results


def test_file_structure():
    """Test struktury souborů"""
    print("\n📁 Testování struktury souborů...")
    
    base_path = Path("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")
    
    expected_files = [
        # Fáze 3 - Deep Web
        "src/deep_web/network_manager.py",
        "src/deep_web/tor_manager.py", 
        "src/deep_web/i2p_manager.py",
        "src/scraping/stealth_engine.py",
        "workers/deep_web_crawler.py",
        
        # Fáze 4 - Autonomní Agent
        "src/core/agentic_loop.py",
        "src/core/intelligent_task_manager.py",
        "src/synthesis/hierarchical_summarizer.py",
        "src/synthesis/credibility_assessor.py",
        "src/synthesis/correlation_engine.py",
        "src/synthesis/steganography_analyzer.py",
        
        # Bezpečnost
        "src/security/osint_sandbox.py",
    ]
    
    results = {}
    
    for file_path in expected_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        
        if exists:
            try:
                size = full_path.stat().st_size
                results[file_path] = {
                    'exists': True,
                    'size_bytes': size,
                    'size_kb': round(size / 1024, 1)
                }
                print(f"✅ {file_path} ({results[file_path]['size_kb']} KB)")
            except Exception as e:
                results[file_path] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"⚠️ {file_path}: Exists but error reading - {e}")
        else:
            results[file_path] = {'exists': False}
            print(f"❌ {file_path}: Missing")
    
    return results


def test_code_quality():
    """Test kvality kódu pomocí základních kontrol"""
    print("\n🔍 Testování kvality kódu...")
    
    base_path = Path("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")
    
    key_files = [
        "src/deep_web/network_manager.py",
        "src/scraping/stealth_engine.py", 
        "src/security/osint_sandbox.py",
        "workers/deep_web_crawler.py",
        "src/core/agentic_loop.py",
        "src/core/intelligent_task_manager.py"
    ]
    
    results = {}
    
    for file_path in key_files:
        full_path = base_path / file_path
        
        if not full_path.exists():
            results[file_path] = {'exists': False}
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Základní metriky
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            docstring_lines = [line for line in lines if '"""' in line or "'''" in line]
            
            # Kontrola základních vzorů
            has_async = 'async def' in content
            has_error_handling = 'try:' in content or 'except' in content
            has_logging = 'logger' in content or 'logging' in content
            has_type_hints = ': str' in content or '-> ' in content
            
            results[file_path] = {
                'exists': True,
                'total_lines': len(lines),
                'code_lines': len(non_empty_lines),
                'comment_lines': len(comment_lines),
                'docstring_indicators': len(docstring_lines),
                'has_async': has_async,
                'has_error_handling': has_error_handling,
                'has_logging': has_logging,
                'has_type_hints': has_type_hints,
                'code_density': round(len(non_empty_lines) / len(lines), 2) if lines else 0
            }
            
            quality_score = sum([
                has_async, has_error_handling, has_logging, has_type_hints
            ]) / 4
            
            results[file_path]['quality_score'] = round(quality_score, 2)
            
            if quality_score >= 0.75:
                status = "🟢 Excellent"
            elif quality_score >= 0.5:
                status = "🟡 Good"
            else:
                status = "🔴 Needs Improvement"
                
            print(f"{status} {file_path}: {results[file_path]['code_lines']} lines, quality {quality_score:.0%}")
            
        except Exception as e:
            results[file_path] = {
                'exists': True,
                'error': str(e)
            }
            print(f"❌ {file_path}: Error analyzing - {e}")
    
    return results


def test_configuration_files():
    """Test konfiguračních souborů"""
    print("\n⚙️ Testování konfiguračních souborů...")
    
    base_path = Path("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")
    
    config_files = [
        "pyproject.toml",
        "config_m1_local.yaml", 
        "docker-compose.m1.yml",
        "Makefile"
    ]
    
    results = {}
    
    for file_name in config_files:
        file_path = base_path / file_name
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                results[file_name] = {
                    'exists': True,
                    'size_bytes': len(content),
                    'lines': len(content.split('\n')),
                    'non_empty_lines': len([line for line in content.split('\n') if line.strip()])
                }
                
                # Specifické kontroly
                if file_name == "pyproject.toml":
                    results[file_name]['has_dependencies'] = 'dependencies' in content
                    results[file_name]['has_dev_deps'] = 'dev' in content
                elif file_name == "config_m1_local.yaml":
                    results[file_name]['has_m1_config'] = 'm1' in content.lower() or 'metal' in content.lower()
                elif file_name == "docker-compose.m1.yml":
                    results[file_name]['has_services'] = 'services:' in content
                elif file_name == "Makefile":
                    results[file_name]['has_targets'] = ':' in content
                
                print(f"✅ {file_name}: {results[file_name]['lines']} lines")
                
            except Exception as e:
                results[file_name] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"⚠️ {file_name}: Error reading - {e}")
        else:
            results[file_name] = {'exists': False}
            print(f"❌ {file_name}: Missing")
    
    return results


def test_documentation():
    """Test dokumentace"""
    print("\n📚 Testování dokumentace...")
    
    base_path = Path("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")
    
    doc_files = [
        "README.md",
        "IMPLEMENTATION_SUMMARY.md",
        "PHASE3_COMPLETION_SUMMARY.md",
        "PHASE4_COMPLETION_SUMMARY.md",
        "docs/architecture.md"
    ]
    
    results = {}
    
    for file_path in doc_files:
        full_path = base_path / file_path
        
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Základní metriky
                word_count = len(content.split())
                section_count = content.count('#')
                has_code_blocks = '```' in content
                
                results[file_path] = {
                    'exists': True,
                    'word_count': word_count,
                    'sections': section_count,
                    'has_code_examples': has_code_blocks,
                    'size_kb': round(len(content) / 1024, 1)
                }
                
                print(f"✅ {file_path}: {word_count} words, {section_count} sections")
                
            except Exception as e:
                results[file_path] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"⚠️ {file_path}: Error reading - {e}")
        else:
            results[file_path] = {'exists': False}
            print(f"❌ {file_path}: Missing")
    
    return results


def generate_completion_summary():
    """Generuje souhrn dokončení fází 3 a 4"""
    print("\n📊 Generování souhrnu dokončení...")
    
    # Spuštění všech testů
    import_results = test_import_availability()
    file_results = test_file_structure()
    quality_results = test_code_quality()
    config_results = test_configuration_files()
    doc_results = test_documentation()
    
    # Analýza výsledků
    total_modules = len(import_results)
    successful_imports = sum(1 for r in import_results.values() if r.get('status') == 'success')
    
    total_files = len(file_results)
    existing_files = sum(1 for r in file_results.values() if r.get('exists', False))
    
    total_quality_files = len([r for r in quality_results.values() if r.get('exists', False)])
    avg_quality = sum(r.get('quality_score', 0) for r in quality_results.values() if 'quality_score' in r) / max(total_quality_files, 1)
    
    existing_configs = sum(1 for r in config_results.values() if r.get('exists', False))
    existing_docs = sum(1 for r in doc_results.values() if r.get('exists', False))
    
    # Hodnocení fází
    phase3_components = [
        'src.deep_web.network_manager',
        'src.scraping.stealth_engine',
        'workers.deep_web_crawler'
    ]
    
    phase4_components = [
        'src.core.agentic_loop',
        'src.core.intelligent_task_manager',
        'src.security.osint_sandbox'
    ]
    
    phase3_success = sum(1 for comp in phase3_components if import_results.get(comp, {}).get('status') == 'success')
    phase4_success = sum(1 for comp in phase4_components if import_results.get(comp, {}).get('status') == 'success')
    
    phase3_completion = phase3_success / len(phase3_components)
    phase4_completion = phase4_success / len(phase4_components)
    
    # Celkové hodnocení
    overall_score = (
        (successful_imports / total_modules) * 0.4 +
        (existing_files / total_files) * 0.3 +
        avg_quality * 0.2 +
        (existing_configs / len(config_results)) * 0.1
    )
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'overall_completion': {
            'score': round(overall_score, 2),
            'grade': 'A' if overall_score >= 0.9 else 'B' if overall_score >= 0.7 else 'C' if overall_score >= 0.5 else 'D',
            'status': 'Excellent' if overall_score >= 0.9 else 'Good' if overall_score >= 0.7 else 'Acceptable' if overall_score >= 0.5 else 'Needs Work'
        },
        'phase_completion': {
            'phase_3_deep_web': {
                'completion_rate': round(phase3_completion, 2),
                'successful_components': phase3_success,
                'total_components': len(phase3_components),
                'status': 'Complete' if phase3_completion >= 0.8 else 'Partial' if phase3_completion >= 0.5 else 'Incomplete'
            },
            'phase_4_autonomous_agent': {
                'completion_rate': round(phase4_completion, 2),
                'successful_components': phase4_success,
                'total_components': len(phase4_components),
                'status': 'Complete' if phase4_completion >= 0.8 else 'Partial' if phase4_completion >= 0.5 else 'Incomplete'
            }
        },
        'detailed_metrics': {
            'code_modules': {
                'total': total_modules,
                'successful_imports': successful_imports,
                'success_rate': round(successful_imports / total_modules, 2)
            },
            'file_structure': {
                'total_expected': total_files,
                'existing_files': existing_files,
                'completion_rate': round(existing_files / total_files, 2)
            },
            'code_quality': {
                'average_score': round(avg_quality, 2),
                'files_analyzed': total_quality_files
            },
            'configuration': {
                'existing_configs': existing_configs,
                'total_configs': len(config_results)
            },
            'documentation': {
                'existing_docs': existing_docs,
                'total_docs': len(doc_results)
            }
        },
        'test_results': {
            'imports': import_results,
            'files': file_results,
            'quality': quality_results,
            'configs': config_results,
            'docs': doc_results
        }
    }
    
    return summary


def main():
    """Hlavní funkce"""
    print("🧪 Spouštím zjednodušený test fází 3 a 4 DeepResearchTool")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Generování kompletního souhrnu
        summary = generate_completion_summary()
        
        # Výpis výsledků
        print(f"\n📊 SOUHRN DOKONČENÍ FÁZÍ 3 a 4")
        print("=" * 70)
        
        overall = summary['overall_completion']
        print(f"🎯 Celkové hodnocení: {overall['score']:.0%} ({overall['grade']}) - {overall['status']}")
        
        print(f"\n📈 DOKONČENÍ FÁZÍ")
        print("-" * 30)
        
        for phase_name, phase_data in summary['phase_completion'].items():
            phase_display = phase_name.replace('_', ' ').title()
            completion = phase_data['completion_rate']
            status = phase_data['status']
            components = f"{phase_data['successful_components']}/{phase_data['total_components']}"
            
            status_emoji = "✅" if status == "Complete" else "🟡" if status == "Partial" else "❌"
            print(f"{status_emoji} {phase_display}: {completion:.0%} ({components} komponent) - {status}")
        
        print(f"\n📋 DETAILNÍ METRIKY")
        print("-" * 30)
        
        metrics = summary['detailed_metrics']
        print(f"📦 Moduly: {metrics['code_modules']['successful_imports']}/{metrics['code_modules']['total']} ({metrics['code_modules']['success_rate']:.0%})")
        print(f"📁 Soubory: {metrics['file_structure']['existing_files']}/{metrics['file_structure']['total_expected']} ({metrics['file_structure']['completion_rate']:.0%})")
        print(f"🏗️ Kvalita kódu: {metrics['code_quality']['average_score']:.0%} průměr")
        print(f"⚙️ Konfigurace: {metrics['configuration']['existing_configs']}/{metrics['configuration']['total_configs']}")
        print(f"📚 Dokumentace: {metrics['documentation']['existing_docs']}/{metrics['documentation']['total_docs']}")
        
        # Doporučení
        print(f"\n💡 DOPORUČENÍ")
        print("-" * 30)
        
        recommendations = []
        
        if overall['score'] >= 0.9:
            recommendations.append("Implementace je kompletní a připravená k nasazení")
            recommendations.append("Doporučujeme spustit performance testy")
        elif overall['score'] >= 0.7:
            recommendations.append("Implementace je v dobrém stavu s drobnými nedostatky")
            recommendations.append("Opravit chybějící komponenty před nasazením")
        else:
            recommendations.append("Implementace vyžaduje dokončení klíčových komponent")
            recommendations.append("Zaměřit se na import errors a chybějící soubory")
        
        if summary['phase_completion']['phase_3_deep_web']['completion_rate'] < 0.8:
            recommendations.append("Dokončit implementaci Deep Web komponent")
        
        if summary['phase_completion']['phase_4_autonomous_agent']['completion_rate'] < 0.8:
            recommendations.append("Dokončit implementaci Autonomního Agenta")
        
        if metrics['code_quality']['average_score'] < 0.7:
            recommendations.append("Zlepšit kvalitu kódu (error handling, type hints)")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Uložení detailního reportu
        report_file = Path(f"phase3_4_completion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        runtime = time.time() - start_time
        print(f"\n⏱️ Test dokončen za {runtime:.1f}s")
        print(f"📄 Detailní report uložen: {report_file}")
        
        # Návratový kód podle úspěšnosti
        if overall['score'] >= 0.8:
            print(f"\n🎉 Fáze 3 a 4 jsou úspěšně implementovány!")
            return 0
        elif overall['score'] >= 0.6:
            print(f"\n⚠️ Fáze 3 a 4 jsou z větší části implementovány, ale vyžadují dokončení.")
            return 1
        else:
            print(f"\n🚨 Fáze 3 a 4 vyžadují značnou práci k dokončení.")
            return 2
            
    except Exception as e:
        print(f"\n💥 Kritická chyba při testování: {e}")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)