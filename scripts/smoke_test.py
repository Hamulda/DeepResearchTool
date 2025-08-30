#!/usr/bin/env python3
"""
Smoke Test - Rychlá validace (<60s, ≥1 claim, ≥2 citations)
Fail-hard test pro FÁZI 1 akceptační kritéria

Author: Senior Python/MLOps Agent
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.gates import create_gate_manager, GateType
from src.core.enhanced_orchestrator import EnhancedOrchestrator
import config  # Import our config module


async def run_smoke_test() -> bool:
    """
    Spustí smoke test s fail-hard kritérii:
    - Doba běhu < 60 sekund
    - ≥1 claim vygenerován
    - ≥2 citace per claim
    - Všechny validation gates prošly
    """
    print("🔥 SMOKE TEST - DeepResearchTool")
    print("=" * 50)
    print("Kritéria:")
    print("  ⏱️  Doba běhu: <60 sekund")
    print("  📋 Claims: ≥1")
    print("  📚 Citace: ≥2 per claim")
    print("  ✅ Validation gates: všechny")
    print()

    start_time = time.time()

    try:
        # Load config
        config_data = config.load_config("config_m1_local.yaml")

        # Initialize gate manager
        gate_manager = create_gate_manager(config_data)

        # Initialize orchestrator with quick profile
        config_data["research_profile"] = "quick"
        config_data["max_documents"] = 20  # Limit for speed
        config_data["synthesis"]["max_claims"] = 3  # Limit claims

        orchestrator = EnhancedOrchestrator(config_data)
        await orchestrator.initialize()

        # Test query
        test_query = "What are the latest developments in large language models?"

        print(f"🔍 Testing query: {test_query}")
        print()

        # Run research
        result = await orchestrator.process_query(test_query)

        # Validate timing
        elapsed_time = time.time() - start_time
        print(f"⏱️  Elapsed time: {elapsed_time:.1f}s")

        if elapsed_time >= 60:
            print(f"❌ FAIL: Smoke test exceeded 60s limit ({elapsed_time:.1f}s)")
            return False

        # Validate structure
        if "claims" not in result:
            print("❌ FAIL: No 'claims' in result")
            return False

        claims = result["claims"]
        if len(claims) < 1:
            print(f"❌ FAIL: Expected ≥1 claims, got {len(claims)}")
            return False

        print(f"📋 Claims generated: {len(claims)}")

        # Validate citations per claim
        for i, claim in enumerate(claims):
            citations = claim.get("citations", [])
            if len(citations) < 2:
                print(f"❌ FAIL: Claim {i+1} has only {len(citations)} citations (required ≥2)")
                return False

            print(f"  📚 Claim {i+1}: {len(citations)} citations")

        # Prepare validation data
        validation_data = {
            "timestamp": datetime.now().isoformat(),
            "query": test_query,
            "claims": claims,
            "retrieval_metadata": result.get("retrieval_metadata", {}),
            "metrics": result.get("metrics", {})
        }

        # Run validation gates
        print("\n🚪 Running validation gates...")

        try:
            validation_report = await gate_manager.validate_all(validation_data)
            print("✅ All validation gates passed")

        except Exception as e:
            print(f"❌ FAIL: Validation gate failed: {e}")
            return False

        # Save artifacts
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Save smoke test result
        smoke_result = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "success": True,
            "query": test_query,
            "claims_count": len(claims),
            "total_citations": sum(len(claim.get("citations", [])) for claim in claims),
            "validation_report": validation_report,
            "performance_metrics": {
                "under_60s": elapsed_time < 60,
                "min_claims": len(claims) >= 1,
                "min_citations_per_claim": all(len(claim.get("citations", [])) >= 2 for claim in claims)
            }
        }

        with open(artifacts_dir / "smoke_test_result.json", "w") as f:
            json.dump(smoke_result, f, indent=2)

        print(f"\n✅ SMOKE TEST PASSED")
        print(f"   Time: {elapsed_time:.1f}s / 60s")
        print(f"   Claims: {len(claims)}")
        print(f"   Citations: {sum(len(claim.get('citations', [])) for claim in claims)}")
        print(f"   Artifacts: artifacts/smoke_test_result.json")

        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ FAIL: Exception in smoke test: {e}")
        print(f"   Time: {elapsed_time:.1f}s")

        # Save failure report
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        failure_result = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

        with open(artifacts_dir / "smoke_test_failure.json", "w") as f:
            json.dump(failure_result, f, indent=2)

        return False


async def main():
    """Main entry point"""
    success = await run_smoke_test()

    if success:
        print("\n🎉 Smoke test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Smoke test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
