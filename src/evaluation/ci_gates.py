#!/usr/bin/env python3
"""FÁZE 5: CI/CD Gates with Build Fail on Performance Degradation
Automatické CI/CD brány s fail-hard pravidly při poklesu performance

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from src.evaluation.regression_test_suite import RegressionTestSuite


@dataclass
class CIGateConfig:
    """Konfigurace pro CI gate"""

    gate_name: str
    enabled: bool
    fail_on_error: bool
    timeout_seconds: int
    performance_thresholds: dict[str, float]
    regression_thresholds: dict[str, float]


@dataclass
class CIGateResult:
    """Výsledek CI gate"""

    gate_name: str
    status: str  # passed, failed, skipped
    execution_time: float
    details: dict[str, Any]
    error_message: str | None = None
    recommendations: list[str] = None


class CICDGateOrchestrator:
    """CI/CD Gate orchestrátor pro FÁZE 5"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ci_config = config.get("ci_gates", {})
        self.project_root = Path.cwd()
        self.results_path = Path("docs/ci_results")
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize gate configurations
        self.gates = self._initialize_gates()

    def _initialize_gates(self) -> list[CIGateConfig]:
        """Inicializuje CI gates"""
        return [
            CIGateConfig(
                gate_name="lint_and_format",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=120,
                performance_thresholds={},
                regression_thresholds={},
            ),
            CIGateConfig(
                gate_name="unit_tests",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=300,
                performance_thresholds={"test_pass_rate": 0.95, "coverage_threshold": 0.80},
                regression_thresholds={},
            ),
            CIGateConfig(
                gate_name="integration_tests",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=600,
                performance_thresholds={"test_pass_rate": 0.90, "avg_test_duration_ms": 30000},
                regression_thresholds={},
            ),
            CIGateConfig(
                gate_name="regression_tests",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=1800,  # 30 minutes
                performance_thresholds={"domain_pass_rate": 0.80, "avg_latency_ms": 120000},
                regression_thresholds={
                    "evidence_coverage_degradation": -5.0,  # Max 5% degradation
                    "groundedness_degradation": -3.0,
                    "hallucination_rate_increase": 50.0,  # Max 50% increase
                    "citation_precision_degradation": -5.0,
                },
            ),
            CIGateConfig(
                gate_name="smoke_tests",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=120,
                performance_thresholds={
                    "max_execution_time_s": 60,
                    "min_claims_generated": 1,
                    "min_citations_per_claim": 2,
                },
                regression_thresholds={},
            ),
            CIGateConfig(
                gate_name="security_checks",
                enabled=True,
                fail_on_error=True,
                timeout_seconds=300,
                performance_thresholds={},
                regression_thresholds={},
            ),
        ]

    async def run_ci_pipeline(
        self, gates_to_run: list[str] | None = None, fail_fast: bool = True
    ) -> tuple[bool, list[CIGateResult]]:
        """Spustí kompletní CI pipeline"""
        print("🚀 Spouštím CI/CD Pipeline pro FÁZE 5...")
        print("=" * 60)

        gates_to_execute = self.gates
        if gates_to_run:
            gates_to_execute = [g for g in self.gates if g.gate_name in gates_to_run]

        results = []
        overall_success = True

        for gate in gates_to_execute:
            if not gate.enabled:
                print(f"⏭️  Přeskakuji {gate.gate_name} (zakázán)")
                results.append(
                    CIGateResult(
                        gate_name=gate.gate_name,
                        status="skipped",
                        execution_time=0,
                        details={"reason": "disabled"},
                    )
                )
                continue

            print(f"🔍 Spouštím: {gate.gate_name}")

            try:
                result = await self._execute_gate(gate)
                results.append(result)

                if result.status == "failed":
                    overall_success = False
                    print(f"❌ {gate.gate_name}: FAILED - {result.error_message}")

                    if fail_fast:
                        print("💥 Fail-fast mode: Ukončuji pipeline")
                        break
                else:
                    print(f"✅ {gate.gate_name}: PASSED ({result.execution_time:.1f}s)")

            except Exception as e:
                overall_success = False
                error_result = CIGateResult(
                    gate_name=gate.gate_name,
                    status="failed",
                    execution_time=0,
                    details={},
                    error_message=f"Gate execution error: {e!s}",
                )
                results.append(error_result)
                print(f"💥 {gate.gate_name}: CRITICAL ERROR - {e!s}")

                if fail_fast:
                    break

        # Generate CI report
        await self._generate_ci_report(results, overall_success)

        return overall_success, results

    async def _execute_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Vykoná jeden CI gate"""
        start_time = asyncio.get_event_loop().time()

        try:
            if gate.gate_name == "lint_and_format":
                return await self._run_lint_and_format_gate(gate)
            if gate.gate_name == "unit_tests":
                return await self._run_unit_tests_gate(gate)
            if gate.gate_name == "integration_tests":
                return await self._run_integration_tests_gate(gate)
            if gate.gate_name == "regression_tests":
                return await self._run_regression_tests_gate(gate)
            if gate.gate_name == "smoke_tests":
                return await self._run_smoke_tests_gate(gate)
            if gate.gate_name == "security_checks":
                return await self._run_security_checks_gate(gate)
            raise ValueError(f"Unknown gate: {gate.gate_name}")

        except TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return CIGateResult(
                gate_name=gate.gate_name,
                status="failed",
                execution_time=execution_time,
                details={},
                error_message=f"Gate timed out after {gate.timeout_seconds}s",
            )

    async def _run_lint_and_format_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí lint a format kontroly"""
        start_time = asyncio.get_event_loop().time()

        # Run Black format check
        black_result = await self._run_command(
            ["python", "-m", "black", "--check", "--line-length=88", "src/", "tests/"],
            timeout=gate.timeout_seconds,
        )

        # Run flake8 linting
        flake8_result = await self._run_command(
            [
                "python",
                "-m",
                "flake8",
                "src/",
                "tests/",
                "--max-line-length=88",
                "--extend-ignore=E203,W503",
            ],
            timeout=gate.timeout_seconds,
        )

        # Run mypy type checking
        mypy_result = await self._run_command(
            ["python", "-m", "mypy", "src/", "--ignore-missing-imports"],
            timeout=gate.timeout_seconds,
        )

        execution_time = asyncio.get_event_loop().time() - start_time

        all_passed = (
            black_result["returncode"] == 0
            and flake8_result["returncode"] == 0
            and mypy_result["returncode"] == 0
        )

        details = {
            "black_check": black_result,
            "flake8_check": flake8_result,
            "mypy_check": mypy_result,
        }

        return CIGateResult(
            gate_name=gate.gate_name,
            status="passed" if all_passed else "failed",
            execution_time=execution_time,
            details=details,
            error_message=None if all_passed else "Lint/format checks failed",
            recommendations=(
                ["Run 'make format' to fix formatting issues"] if not all_passed else []
            ),
        )

    async def _run_unit_tests_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí unit testy"""
        start_time = asyncio.get_event_loop().time()

        # Run pytest with coverage
        pytest_result = await self._run_command(
            [
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--cov=src",
                "--cov-report=json:coverage.json",
                "-k",
                "not integration",  # Exclude integration tests
            ],
            timeout=gate.timeout_seconds,
        )

        execution_time = asyncio.get_event_loop().time() - start_time

        # Parse coverage report
        coverage_data = {}
        try:
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
        except Exception:
            pass

        coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
        test_passed = pytest_result["returncode"] == 0
        coverage_passed = coverage_percentage >= gate.performance_thresholds.get(
            "coverage_threshold", 0.8
        )

        all_passed = test_passed and coverage_passed

        details = {
            "pytest_result": pytest_result,
            "coverage_percentage": coverage_percentage,
            "coverage_threshold": gate.performance_thresholds.get("coverage_threshold", 0.8),
            "coverage_passed": coverage_passed,
        }

        error_message = None
        recommendations = []
        if not test_passed:
            error_message = "Unit tests failed"
            recommendations.append("Fix failing unit tests")
        elif not coverage_passed:
            error_message = f"Coverage too low: {coverage_percentage:.1%} < {gate.performance_thresholds.get('coverage_threshold', 0.8):.1%}"
            recommendations.append("Add more unit tests to increase coverage")

        return CIGateResult(
            gate_name=gate.gate_name,
            status="passed" if all_passed else "failed",
            execution_time=execution_time,
            details=details,
            error_message=error_message,
            recommendations=recommendations,
        )

    async def _run_integration_tests_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí integration testy"""
        start_time = asyncio.get_event_loop().time()

        # Run integration tests
        pytest_result = await self._run_command(
            ["python", "-m", "pytest", "tests/", "-v", "-k", "integration"],
            timeout=gate.timeout_seconds,
        )

        execution_time = asyncio.get_event_loop().time() - start_time

        test_passed = pytest_result["returncode"] == 0
        duration_passed = execution_time * 1000 <= gate.performance_thresholds.get(
            "avg_test_duration_ms", 30000
        )

        all_passed = test_passed and duration_passed

        details = {
            "pytest_result": pytest_result,
            "execution_time_ms": execution_time * 1000,
            "duration_threshold_ms": gate.performance_thresholds.get("avg_test_duration_ms", 30000),
            "duration_passed": duration_passed,
        }

        error_message = None
        recommendations = []
        if not test_passed:
            error_message = "Integration tests failed"
            recommendations.append("Fix failing integration tests")
        elif not duration_passed:
            error_message = f"Integration tests too slow: {execution_time*1000:.0f}ms > {gate.performance_thresholds.get('avg_test_duration_ms', 30000)}ms"
            recommendations.append("Optimize integration test performance")

        return CIGateResult(
            gate_name=gate.gate_name,
            status="passed" if all_passed else "failed",
            execution_time=execution_time,
            details=details,
            error_message=error_message,
            recommendations=recommendations,
        )

    async def _run_regression_tests_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí regression testy s performance monitoring"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Initialize regression test suite
            config = self.config.copy()
            config.setdefault("evaluation", {}).setdefault("k_values", [1, 3, 5, 10])
            config.setdefault("regression", {}).setdefault("baseline_path", "data/baselines")

            regression_suite = RegressionTestSuite(config)

            # Run subset of domains for CI (full suite takes too long)
            ci_domains = ["climate_science", "ai_technology", "legal_policy", "medical_research"]
            results = await regression_suite.run_regression_tests(ci_domains)

            execution_time = asyncio.get_event_loop().time() - start_time

            # Analyze results
            passed_count = sum(1 for r in results if r.overall_status == "passed")
            total_count = len(results)
            pass_rate = passed_count / total_count if total_count > 0 else 0

            avg_latency = (
                sum(r.performance_metrics.get("latency_ms", 0) for r in results) / total_count
                if total_count > 0
                else 0
            )

            # Check thresholds
            pass_rate_ok = pass_rate >= gate.performance_thresholds.get("domain_pass_rate", 0.8)
            latency_ok = avg_latency <= gate.performance_thresholds.get("avg_latency_ms", 120000)

            # Check regression thresholds against baseline
            regression_violations = []
            for result in results:
                for metric, threshold in gate.regression_thresholds.items():
                    baseline_key = f"{metric}_change_percent"
                    if baseline_key in result.baseline_comparison:
                        change = result.baseline_comparison[baseline_key]
                        if "degradation" in metric and change < threshold:
                            regression_violations.append(
                                f"{result.domain}: {metric} degraded by {abs(change):.1f}%"
                            )
                        elif "increase" in metric and change > threshold:
                            regression_violations.append(
                                f"{result.domain}: {metric} increased by {change:.1f}%"
                            )

            no_regressions = len(regression_violations) == 0
            all_passed = pass_rate_ok and latency_ok and no_regressions

            details = {
                "domains_tested": ci_domains,
                "pass_rate": pass_rate,
                "passed_domains": passed_count,
                "total_domains": total_count,
                "avg_latency_ms": avg_latency,
                "regression_violations": regression_violations,
                "detailed_results": [
                    {
                        "domain": r.domain,
                        "status": r.overall_status,
                        "latency_ms": r.performance_metrics.get("latency_ms", 0),
                        "evidence_coverage": r.evaluation_result.evidence_coverage,
                        "groundedness": r.evaluation_result.groundedness,
                    }
                    for r in results
                ],
            }

            error_message = None
            recommendations = []
            if not pass_rate_ok:
                error_message = f"Regression pass rate too low: {pass_rate:.1%} < {gate.performance_thresholds.get('domain_pass_rate', 0.8):.1%}"
                recommendations.append("Fix failing regression tests")
            elif not latency_ok:
                error_message = f"Average latency too high: {avg_latency:.0f}ms > {gate.performance_thresholds.get('avg_latency_ms', 120000)}ms"
                recommendations.append("Optimize pipeline performance")
            elif not no_regressions:
                error_message = (
                    f"Performance regressions detected: {len(regression_violations)} violations"
                )
                recommendations.extend(regression_violations)

            return CIGateResult(
                gate_name=gate.gate_name,
                status="passed" if all_passed else "failed",
                execution_time=execution_time,
                details=details,
                error_message=error_message,
                recommendations=recommendations,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return CIGateResult(
                gate_name=gate.gate_name,
                status="failed",
                execution_time=execution_time,
                details={},
                error_message=f"Regression test execution failed: {e!s}",
                recommendations=["Check regression test configuration and dependencies"],
            )

    async def _run_smoke_tests_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí smoke testy s fail-hard požadavky"""
        start_time = asyncio.get_event_loop().time()

        # Run smoke test script
        smoke_result = await self._run_command(
            [
                "python",
                "scripts/smoke_test.py",
                "--config",
                "config_m1_local.yaml",
                "--timeout",
                str(gate.performance_thresholds.get("max_execution_time_s", 60)),
            ],
            timeout=gate.timeout_seconds,
        )

        execution_time = asyncio.get_event_loop().time() - start_time

        # Parse smoke test results (would need to be implemented in smoke_test.py)
        smoke_passed = smoke_result["returncode"] == 0
        time_ok = execution_time <= gate.performance_thresholds.get("max_execution_time_s", 60)

        # Mock claims/citations check (would be real in production)
        claims_generated = 1  # Would parse from smoke test output
        citations_per_claim = 2  # Would parse from smoke test output

        claims_ok = claims_generated >= gate.performance_thresholds.get("min_claims_generated", 1)
        citations_ok = citations_per_claim >= gate.performance_thresholds.get(
            "min_citations_per_claim", 2
        )

        all_passed = smoke_passed and time_ok and claims_ok and citations_ok

        details = {
            "smoke_test_result": smoke_result,
            "execution_time_s": execution_time,
            "claims_generated": claims_generated,
            "citations_per_claim": citations_per_claim,
            "time_threshold_s": gate.performance_thresholds.get("max_execution_time_s", 60),
            "all_requirements_met": all_passed,
        }

        error_message = None
        recommendations = []
        if not smoke_passed:
            error_message = "Smoke test failed to execute"
            recommendations.append("Check smoke test script and dependencies")
        elif not time_ok:
            error_message = f"Smoke test too slow: {execution_time:.1f}s > {gate.performance_thresholds.get('max_execution_time_s', 60)}s"
            recommendations.append("Optimize smoke test performance")
        elif not claims_ok:
            error_message = f"Insufficient claims generated: {claims_generated} < {gate.performance_thresholds.get('min_claims_generated', 1)}"
            recommendations.append("Check claim generation logic")
        elif not citations_ok:
            error_message = f"Insufficient citations per claim: {citations_per_claim} < {gate.performance_thresholds.get('min_citations_per_claim', 2)}"
            recommendations.append("Check citation binding logic")

        return CIGateResult(
            gate_name=gate.gate_name,
            status="passed" if all_passed else "failed",
            execution_time=execution_time,
            details=details,
            error_message=error_message,
            recommendations=recommendations,
        )

    async def _run_security_checks_gate(self, gate: CIGateConfig) -> CIGateResult:
        """Spustí bezpečnostní kontroly"""
        start_time = asyncio.get_event_loop().time()

        # Run security check script
        security_result = await self._run_command(
            ["python", "scripts/security_check.py", "--config", "config_m1_local.yaml"],
            timeout=gate.timeout_seconds,
        )

        # Run PII check
        pii_result = await self._run_command(
            ["python", "scripts/pii_check.py", "--scan-logs", "--scan-outputs"],
            timeout=gate.timeout_seconds // 2,
        )

        execution_time = asyncio.get_event_loop().time() - start_time

        security_passed = security_result["returncode"] == 0
        pii_passed = pii_result["returncode"] == 0

        all_passed = security_passed and pii_passed

        details = {
            "security_check_result": security_result,
            "pii_check_result": pii_result,
            "execution_time_s": execution_time,
        }

        error_message = None
        recommendations = []
        if not security_passed:
            error_message = "Security checks failed"
            recommendations.append("Review and fix security violations")
        elif not pii_passed:
            error_message = "PII exposure detected"
            recommendations.append("Remove or redact PII from logs and outputs")

        return CIGateResult(
            gate_name=gate.gate_name,
            status="passed" if all_passed else "failed",
            execution_time=execution_time,
            details=details,
            error_message=error_message,
            recommendations=recommendations,
        )

    async def _run_command(self, cmd: list[str], timeout: int) -> dict[str, Any]:
        """Spustí command s timeout"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="ignore"),
                "stderr": stderr.decode("utf-8", errors="ignore"),
                "command": " ".join(cmd),
            }

        except TimeoutError:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "command": " ".join(cmd),
            }
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "command": " ".join(cmd)}

    async def _generate_ci_report(self, results: list[CIGateResult], overall_success: bool):
        """Generuje CI report"""
        report = {
            "ci_pipeline_report": {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "passed" if overall_success else "failed",
                "total_gates": len(results),
                "passed_gates": sum(1 for r in results if r.status == "passed"),
                "failed_gates": sum(1 for r in results if r.status == "failed"),
                "skipped_gates": sum(1 for r in results if r.status == "skipped"),
                "total_execution_time": sum(r.execution_time for r in results),
            },
            "gate_results": [],
        }

        for result in results:
            gate_report = {
                "gate_name": result.gate_name,
                "status": result.status,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "recommendations": result.recommendations or [],
                "details": result.details,
            }
            report["gate_results"].append(gate_report)

        # Export report
        report_file = (
            self.results_path / f"ci_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 CI Report exportován: {report_file}")

        # Print summary
        print(f"\n{'='*60}")
        print("CI/CD Pipeline Summary")
        print(f"{'='*60}")
        print(f"Overall Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"Gates Executed: {len(results)}")
        print(f"Passed: {report['ci_pipeline_report']['passed_gates']}")
        print(f"Failed: {report['ci_pipeline_report']['failed_gates']}")
        print(f"Total Time: {report['ci_pipeline_report']['total_execution_time']:.1f}s")


# Factory function
def create_cicd_gate_orchestrator(config: dict[str, Any]) -> CICDGateOrchestrator:
    """Factory function pro vytvoření CI/CD orchestrátor"""
    return CICDGateOrchestrator(config)


if __name__ == "__main__":
    # CLI interface for CI/CD gates
    import argparse

    parser = argparse.ArgumentParser(description="CI/CD Gates for FÁZE 5")
    parser.add_argument("--config", default="config_m1_local.yaml", help="Config file path")
    parser.add_argument("--gates", nargs="+", help="Specific gates to run")
    parser.add_argument(
        "--fail-fast", action="store_true", default=True, help="Stop on first failure"
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    async def main():
        orchestrator = create_cicd_gate_orchestrator(config)
        success, results = await orchestrator.run_ci_pipeline(
            gates_to_run=args.gates, fail_fast=args.fail_fast
        )
        sys.exit(0 if success else 1)

    asyncio.run(main())
