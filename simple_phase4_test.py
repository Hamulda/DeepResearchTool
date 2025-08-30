"""
Simple Phase 4 Autonomous System Test
Tests základní funkčnost autonomního systému bez plných závislostí
"""

import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime, timezone
import asyncio

# Add paths
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool/src")
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool/workers")
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")


def test_phase4_autonomous_system():
    """Test základní funkčnosti Phase 4 autonomního systému"""

    print("🤖 === Phase 4 Autonomous System Test ===")
    print()

    results = {
        "test_name": "Phase 4 Autonomous System Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "summary": {},
    }

    # Test 1: Import Autonomous Components
    print("1. Testing autonomous worker imports...")
    try:
        from autonomous_worker import (
            AutonomousWorker,
            ChangeDetector,
            EventDrivenAnalyzer,
            AlertManager,
        )

        results["tests"]["autonomous_worker_import"] = {
            "status": "PASS",
            "message": "AutonomousWorker imported successfully",
        }
        print("   ✅ AutonomousWorker imported")
    except ImportError as e:
        results["tests"]["autonomous_worker_import"] = {
            "status": "FAIL",
            "message": f"Import failed: {e}",
        }
        print(f"   ❌ AutonomousWorker import failed: {e}")

    # Test 2: MonitoredSource dataclass
    print("2. Testing MonitoredSource dataclass...")
    try:
        from autonomous_worker import MonitoredSource, AlertRule, Alert

        # Vytvoř test source
        test_source = MonitoredSource(
            url="https://example.com", check_interval=60, priority=3, keywords=["test", "security"]
        )

        assert test_source.url == "https://example.com"
        assert test_source.check_interval == 60
        assert test_source.priority == 3
        assert test_source.enabled == True

        results["tests"]["dataclasses"] = {
            "status": "PASS",
            "message": "MonitoredSource dataclass works correctly",
        }
        print("   ✅ MonitoredSource dataclass functional")
    except Exception as e:
        results["tests"]["dataclasses"] = {
            "status": "FAIL",
            "message": f"Dataclass test failed: {e}",
        }
        print(f"   ❌ Dataclass test failed: {e}")

    # Test 3: ChangeDetector functionality
    print("3. Testing ChangeDetector...")
    try:
        from autonomous_worker import ChangeDetector, MonitoredSource

        detector = ChangeDetector()

        # Test source pro change detection
        test_source = MonitoredSource(
            url="https://httpbin.org/get", check_interval=60  # Reliable test endpoint
        )

        # Zkus jednoduchou detekci
        result = detector.check_page_changes(test_source)

        if result.get("changed") is not None:  # Buď True nebo False
            results["tests"]["change_detector"] = {
                "status": "PASS",
                "message": f"ChangeDetector functional. Result: {result.get('changed')}",
            }
            print(f"   ✅ ChangeDetector works. Changed: {result.get('changed')}")
        else:
            results["tests"]["change_detector"] = {
                "status": "WARN",
                "message": f"ChangeDetector returned unclear result: {result}",
            }
            print(f"   ⚠️ ChangeDetector unclear result: {result}")

    except Exception as e:
        results["tests"]["change_detector"] = {
            "status": "FAIL",
            "message": f"ChangeDetector test failed: {e}",
        }
        print(f"   ❌ ChangeDetector test failed: {e}")

    # Test 4: AlertManager
    print("4. Testing AlertManager...")
    try:
        from autonomous_worker import AlertManager, Alert

        alert_manager = AlertManager()

        # Vytvoř test alert
        test_alert = Alert(
            alert_id="test_001",
            rule_id="test_rule",
            title="Test Alert",
            message="This is a test alert",
            severity="low",
            source_url="https://example.com",
            triggered_at=datetime.now(timezone.utc),
        )

        # Test file alert (nejjednodušší)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            success = loop.run_until_complete(alert_manager._send_file_alert(test_alert))
            if success:
                results["tests"]["alert_manager"] = {
                    "status": "PASS",
                    "message": "AlertManager file alerts working",
                }
                print("   ✅ AlertManager file alerts functional")
            else:
                results["tests"]["alert_manager"] = {
                    "status": "WARN",
                    "message": "AlertManager file alerts returned False",
                }
                print("   ⚠️ AlertManager file alerts failed")
        finally:
            loop.close()

    except Exception as e:
        results["tests"]["alert_manager"] = {
            "status": "FAIL",
            "message": f"AlertManager test failed: {e}",
        }
        print(f"   ❌ AlertManager test failed: {e}")

    # Test 5: Scheduler Integration
    print("5. Testing scheduler integration...")
    try:
        import schedule

        # Test základního schedulingu
        schedule.clear()  # Vyčisti existující úlohy

        def test_job():
            return "test completed"

        schedule.every(10).seconds.do(test_job)

        # Zkontroluj, že úloha byla naplánována
        if len(schedule.jobs) > 0:
            results["tests"]["scheduler"] = {
                "status": "PASS",
                "message": f"Scheduler working. Jobs scheduled: {len(schedule.jobs)}",
            }
            print(f"   ✅ Scheduler functional. Jobs: {len(schedule.jobs)}")
        else:
            results["tests"]["scheduler"] = {"status": "FAIL", "message": "No jobs scheduled"}
            print("   ❌ Scheduler: No jobs scheduled")

        schedule.clear()  # Cleanup

    except Exception as e:
        results["tests"]["scheduler"] = {"status": "FAIL", "message": f"Scheduler test failed: {e}"}
        print(f"   ❌ Scheduler test failed: {e}")

    # Test 6: Autonomous Worker Integration
    print("6. Testing AutonomousWorker integration...")
    try:
        from autonomous_worker import AutonomousWorker

        # Vytvořit worker (bez spuštění)
        worker = AutonomousWorker()

        # Test přidání monitorovaného zdroje
        success = worker.add_monitored_source(
            url="https://example.com/test",
            check_interval=120,
            priority=2,
            keywords=["test", "automation"],
        )

        if success and len(worker.monitored_sources) > 0:
            results["tests"]["autonomous_worker"] = {
                "status": "PASS",
                "message": f"AutonomousWorker functional. Sources: {len(worker.monitored_sources)}",
            }
            print(f"   ✅ AutonomousWorker functional. Sources: {len(worker.monitored_sources)}")
        else:
            results["tests"]["autonomous_worker"] = {
                "status": "WARN",
                "message": f"AutonomousWorker add_source failed or no sources",
            }
            print(
                f"   ⚠️ AutonomousWorker issue: success={success}, sources={len(worker.monitored_sources)}"
            )

    except Exception as e:
        results["tests"]["autonomous_worker"] = {
            "status": "FAIL",
            "message": f"AutonomousWorker test failed: {e}",
        }
        print(f"   ❌ AutonomousWorker test failed: {e}")

    # Test 7: Dependencies Check
    print("7. Checking Phase 4 dependencies...")
    dependencies = ["schedule", "requests", "smtplib", "threading", "asyncio"]
    available = 0

    for dep in dependencies:
        try:
            if dep == "smtplib":
                import smtplib
            elif dep == "threading":
                import threading
            elif dep == "asyncio":
                import asyncio
            elif dep == "requests":
                import requests
            elif dep == "schedule":
                import schedule
            available += 1
        except ImportError:
            pass

    results["tests"]["dependencies"] = {
        "status": "PASS" if available == len(dependencies) else "WARN",
        "message": f"{available}/{len(dependencies)} dependencies available ({available/len(dependencies)*100:.1f}%)",
        "available": available,
        "total": len(dependencies),
    }
    print(
        f"   📦 Dependencies: {available}/{len(dependencies)} available ({available/len(dependencies)*100:.1f}%)"
    )

    # Summary
    print()
    print("📊 === Test Summary ===")

    total_tests = len(results["tests"])
    passed = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
    warned = sum(1 for test in results["tests"].values() if test["status"] == "WARN")
    failed = sum(1 for test in results["tests"].values() if test["status"] == "FAIL")

    results["summary"] = {
        "total_tests": total_tests,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
    }

    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️ Warned: {warned}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")

    # Determine overall status
    if failed == 0 and warned <= 1:
        overall_status = "EXCELLENT"
        print("🎉 Overall status: EXCELLENT - Phase 4 ready!")
    elif failed <= 1:
        overall_status = "GOOD"
        print("✅ Overall status: GOOD - Phase 4 mostly functional")
    elif failed <= 2:
        overall_status = "NEEDS_WORK"
        print("⚠️ Overall status: NEEDS_WORK - Some issues to resolve")
    else:
        overall_status = "POOR"
        print("❌ Overall status: POOR - Major issues found")

    results["overall_status"] = overall_status

    # Save results
    output_file = Path("artifacts/phase4_test_result.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    test_phase4_autonomous_system()
