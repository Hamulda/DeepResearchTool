#!/usr/bin/env python3
"""
Pre-flight Environment Verification for Apple Silicon M1
Ensures native ARM64 execution without Rosetta 2 emulation

Author: Senior Python/MLOps Agent
"""

import platform
import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentVerificationError(Exception):
    """Critical environment verification failure"""
    pass


def verify_arm64_architecture() -> None:
    """
    Rigorously verify native ARM64 execution environment.
    
    Raises:
        EnvironmentVerificationError: If not running on native ARM64
    """
    try:
        # Check Python platform processor
        processor = platform.processor()
        if processor != 'arm':
            raise EnvironmentVerificationError(
                f"FATAL ERROR: Detected processor '{processor}' instead of 'arm'. "
                f"This indicates execution under Rosetta 2 emulation. "
                f"Performance will be critically degraded. Aborting."
            )
        
        # Double-check with system uname
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True, check=True)
        uname_arch = result.stdout.strip()
        
        if uname_arch != 'arm64':
            raise EnvironmentVerificationError(
                f"FATAL ERROR: System architecture '{uname_arch}' is not 'arm64'. "
                f"This indicates execution under Rosetta 2 emulation or incompatible system. "
                f"Performance will be critically degraded. Aborting."
            )
        
        # Check Python binary architecture
        python_arch = platform.machine()
        if python_arch != 'arm64':
            raise EnvironmentVerificationError(
                f"FATAL ERROR: Python binary architecture '{python_arch}' is not 'arm64'. "
                f"You may be using an x86_64 Python interpreter. "
                f"Install native ARM64 Python for optimal performance. Aborting."
            )
        
        logger.info("✅ Environment verification PASSED: Native ARM64 execution confirmed")
        logger.info(f"✅ Processor: {processor}")
        logger.info(f"✅ System architecture: {uname_arch}")
        logger.info(f"✅ Python architecture: {python_arch}")
        
    except subprocess.CalledProcessError as e:
        raise EnvironmentVerificationError(
            f"FATAL ERROR: Failed to execute system commands for architecture verification: {e}"
        ) from e
    except Exception as e:
        raise EnvironmentVerificationError(
            f"FATAL ERROR: Unexpected error during environment verification: {e}"
        ) from e


def verify_memory_availability() -> None:
    """
    Verify sufficient memory for M1 8GB configuration.
    """
    try:
        # Get available memory info
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True, check=True)
        total_memory_bytes = int(result.stdout.split()[-1])
        total_memory_gb = total_memory_bytes / (1024**3)
        
        if total_memory_gb < 7.5:  # Account for system overhead
            logger.warning(
                f"⚠️  WARNING: Available memory ({total_memory_gb:.1f}GB) is below optimal "
                f"threshold for M1 operations. Performance may be suboptimal."
            )
        else:
            logger.info(f"✅ Memory verification PASSED: {total_memory_gb:.1f}GB available")
            
    except Exception as e:
        logger.warning(f"⚠️  WARNING: Could not verify memory availability: {e}")


def main() -> None:
    """
    Execute complete environment verification.
    
    Exits with code 1 on critical failures.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("🔍 Starting pre-flight environment verification...")
        
        # Critical verifications (will abort on failure)
        verify_arm64_architecture()
        
        # Non-critical verifications (warnings only)
        verify_memory_availability()
        
        logger.info("🎉 All environment verifications completed successfully!")
        logger.info("✅ System is optimally configured for DeepResearchTool on Apple Silicon M1")
        
    except EnvironmentVerificationError as e:
        logger.error(str(e))
        logger.error("❌ ABORTING: Critical environment verification failed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ ABORTING: Unexpected error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()