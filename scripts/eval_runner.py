#!/usr/bin/env python3
"""
Evaluation Runner Script
Spouští kompletní evaluaci na regression test set

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import statistics
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evaluation.evaluation_system import EvaluationSystem


class EvaluationRunner:
    """Runner pro kompletní evaluaci"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_system = EvaluationSystem(config)

    async def run_full_evaluation(self, output_file: str = None) -> Dict[str, Any]:
        """Spuštění kompletní evaluace"""

        print("🚀 Starting full evaluation...")
        start_time = time.time()

        try:
            # Initialize evaluation system
            print("🔧 Initializing evaluation system...")
            await self.eval_system.initialize()

            # Run evaluation
            print("📊 Running evaluation on regression test set...")
            evaluation_report = await self.eval_system.run_full_evaluation()

            total_time = time.time() - start_time
            evaluation_report["meta"] = {
                "total_evaluation_time": total_time,
                "timestamp": time.time(),
                "config_hash": hash(str(self.config))
            }

            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(evaluation_report, f, indent=2, default=str)
                print(f"💾 Evaluation results saved to {output_file}")

            return evaluation_report

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                "error": str(e),
                "meta": {
                    "total_evaluation_time": time.time() - start_time,
                    "timestamp": time.time()
                }
            }

    def print_evaluation_summary(self, report: Dict[str, Any]):
        """Tisk shrnutí evaluace"""

        if "error" in report:
            print(f"❌ Evaluation failed: {report['error']}")
            return

        summary = report.get("summary", {})

        print(f"\n✅ Evaluation completed!")
        print(f"⏱️  Total time: {report.get('meta', {}).get('total_evaluation_time', 0):.1f}s")

        # Basic stats
        print(f"\n📊 Basic Statistics:")
        print(f"  Test cases: {summary.get('total_test_cases', 0)}")
        print(f"  Success rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"  Average time per case: {summary.get('performance_stats', {}).get('average_case_time', 0):.1f}s")

        # Quality metrics
        overall_metrics = summary.get("overall_metrics", {})
        if overall_metrics:
            print(f"\n📈 Quality Metrics:")
            for metric_name, metric_data in overall_metrics.items():
                mean_val = metric_data.get('overall_mean', 0)
                threshold_pass = metric_data.get('threshold_pass_rate', 0) * 100

                # Color coding based on performance
                if mean_val >= 0.8:
                    status = "🟢"
                elif mean_val >= 0.6:
                    status = "🟡"
                else:
                    status = "🔴"

                print(f"  {status} {metric_name}: {mean_val:.3f} (pass rate: {threshold_pass:.1f}%)")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:10], 1):  # Top 10
                if "KRITICKÉ" in rec or "CRITICAL" in rec:
                    print(f"  {i}. 🔴 {rec}")
                elif "VAROVÁNÍ" in rec or "WARNING" in rec:
                    print(f"  {i}. 🟡 {rec}")
                else:
                    print(f"  {i}. 🟢 {rec}")

        # CI gate status
        ci_status = self._check_ci_gates(report)
        if ci_status["passed"]:
            print(f"\n✅ CI Gates: PASSED")
        else:
            print(f"\n❌ CI Gates: FAILED")
            for failure in ci_status["failures"]:
                print(f"  • {failure}")

    def _check_ci_gates(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Kontrola CI gate thresholds"""

        summary = report.get("summary", {})
        overall_metrics = summary.get("overall_metrics", {})

        # Get CI thresholds from config
        ci_thresholds = self.config.get("evaluation", {}).get("ci_thresholds", {})
        profile = "thorough"  # Default profile for CI
        thresholds = ci_thresholds.get(profile, {})

        failures = []

        # Check each threshold
        for metric_name, threshold in thresholds.items():
            if metric_name in overall_metrics:
                actual_value = overall_metrics[metric_name].get('overall_mean', 0)

                if metric_name == "hallucination_rate":
                    # Lower is better for hallucination rate
                    if actual_value > threshold:
                        failures.append(f"{metric_name}: {actual_value:.3f} > {threshold}")
                else:
                    # Higher is better for other metrics
                    if actual_value < threshold:
                        failures.append(f"{metric_name}: {actual_value:.3f} < {threshold}")

        return {
            "passed": len(failures) == 0,
            "failures": failures
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluation Runner")
    parser.add_argument("--config", "-c", default="config_m1_local.yaml", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--profile", choices=["quick", "thorough"], default="thorough", help="Evaluation profile")
    parser.add_argument("--ci-mode", action="store_true", help="CI mode - exit with error code if gates fail")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file {args.config} not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Override evaluation profile if specified
    if args.profile:
        config.setdefault("evaluation", {})["profile"] = args.profile
        print(f"📊 Using evaluation profile: {args.profile}")

    # Create and run evaluation
    runner = EvaluationRunner(config)

    # Generate output filename if not specified
    output_file = args.output
    if not output_file and args.ci_mode:
        timestamp = int(time.time())
        output_file = f"evaluation_results/eval_{timestamp}.json"
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    report = await runner.run_full_evaluation(output_file)

    # Print summary
    runner.print_evaluation_summary(report)

    # CI mode handling
    if args.ci_mode:
        ci_status = runner._check_ci_gates(report)
        if not ci_status["passed"]:
            print(f"\n❌ CI gates failed! Exiting with error code 1")
            sys.exit(1)
        else:
            print(f"\n✅ All CI gates passed!")
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
