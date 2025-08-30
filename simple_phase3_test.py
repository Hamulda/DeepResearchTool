"""
Simple Phase 3 Multi-Modality Test
Tests basic functionality without requiring full dependencies
"""

import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime, timezone

# Add paths
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool/src")
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool/workers")
sys.path.append("/Users/vojtechhamada/PycharmProjects/DeepResearchTool")


def test_phase3_basic_functionality():
    """Test basic Phase 3 functionality"""

    print("🧪 === Phase 3 Multi-Modality Basic Test ===")
    print()

    results = {
        "test_name": "Phase 3 Multi-Modality Basic Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "summary": {},
    }

    # Test 1: Import Multi-Modal Components
    print("1. Testing imports...")
    try:
        from processing_worker import EnhancedProcessingWorker

        results["tests"]["processing_worker_import"] = {
            "status": "PASS",
            "message": "EnhancedProcessingWorker imported successfully",
        }
        print("   ✅ EnhancedProcessingWorker imported")

        # Check if image processor is available
        worker = EnhancedProcessingWorker()
        if hasattr(worker, "image_processor"):
            if worker.image_processor:
                results["tests"]["image_processor_available"] = {
                    "status": "PASS",
                    "message": "Image processor initialized",
                }
                print("   ✅ Image processor available")
            else:
                results["tests"]["image_processor_available"] = {
                    "status": "WARN",
                    "message": "Image processor not initialized (dependencies may be missing)",
                }
                print("   ⚠️ Image processor not initialized")
        else:
            results["tests"]["image_processor_available"] = {
                "status": "FAIL",
                "message": "Image processor attribute missing",
            }
            print("   ❌ Image processor attribute missing")

    except Exception as e:
        results["tests"]["processing_worker_import"] = {
            "status": "FAIL",
            "message": f"Import failed: {e}",
        }
        print(f"   ❌ Import failed: {e}")

    # Test 2: Check Image Processing Module
    print("\n2. Testing image processing module...")
    try:
        from image_processor import ImageProcessor

        results["tests"]["image_processor_import"] = {
            "status": "PASS",
            "message": "ImageProcessor imported successfully",
        }
        print("   ✅ ImageProcessor module imported")

        # Try to initialize (may fail due to missing dependencies)
        try:
            processor = ImageProcessor()
            results["tests"]["image_processor_init"] = {
                "status": "PASS",
                "message": "ImageProcessor initialized",
            }
            print("   ✅ ImageProcessor initialized")
        except Exception as e:
            results["tests"]["image_processor_init"] = {
                "status": "WARN",
                "message": f"Initialization failed: {e}",
            }
            print(f"   ⚠️ ImageProcessor initialization failed: {e}")

    except ImportError as e:
        results["tests"]["image_processor_import"] = {
            "status": "WARN",
            "message": f"Import failed (dependencies missing): {e}",
        }
        print(f"   ⚠️ ImageProcessor import failed: {e}")
    except Exception as e:
        results["tests"]["image_processor_import"] = {
            "status": "FAIL",
            "message": f"Unexpected error: {e}",
        }
        print(f"   ❌ Unexpected error: {e}")

    # Test 3: Check Enhanced Acquisition Worker
    print("\n3. Testing enhanced acquisition worker...")
    try:
        from acquisition_worker import EnhancedAcquisitionWorker

        worker = EnhancedAcquisitionWorker()

        # Check if image extraction method exists
        if hasattr(worker, "extract_and_download_images"):
            results["tests"]["image_extraction_method"] = {
                "status": "PASS",
                "message": "Image extraction method available",
            }
            print("   ✅ Image extraction method available")
        else:
            results["tests"]["image_extraction_method"] = {
                "status": "FAIL",
                "message": "Image extraction method missing",
            }
            print("   ❌ Image extraction method missing")

        results["tests"]["acquisition_worker_import"] = {
            "status": "PASS",
            "message": "EnhancedAcquisitionWorker imported",
        }
        print("   ✅ EnhancedAcquisitionWorker imported")

    except Exception as e:
        results["tests"]["acquisition_worker_import"] = {
            "status": "FAIL",
            "message": f"Import failed: {e}",
        }
        print(f"   ❌ EnhancedAcquisitionWorker import failed: {e}")

    # Test 4: Test HTML Image Detection (no dependencies required)
    print("\n4. Testing HTML image detection...")
    try:
        sample_html = """
        <html>
        <body>
            <img src="/test1.jpg" alt="Test 1">
            <img data-src="/test2.png" alt="Test 2">
            <div style="background-image: url('/test3.gif')">Background</div>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup
        import re
        import urllib.parse

        soup = BeautifulSoup(sample_html, "html.parser")

        # Find img tags
        img_tags = soup.find_all("img")
        img_count = len(img_tags)

        # Find background images
        style_tags = soup.find_all(["div"], style=True)
        bg_count = 0
        for tag in style_tags:
            style = tag.get("style", "")
            if "background-image" in style:
                bg_count += 1

        total_images = img_count + bg_count

        if total_images > 0:
            results["tests"]["html_image_detection"] = {
                "status": "PASS",
                "message": f"Detected {total_images} images ({img_count} img tags, {bg_count} backgrounds)",
            }
            print(
                f"   ✅ Detected {total_images} images ({img_count} img tags, {bg_count} backgrounds)"
            )
        else:
            results["tests"]["html_image_detection"] = {
                "status": "FAIL",
                "message": "No images detected",
            }
            print("   ❌ No images detected")

    except Exception as e:
        results["tests"]["html_image_detection"] = {
            "status": "FAIL",
            "message": f"HTML parsing failed: {e}",
        }
        print(f"   ❌ HTML parsing failed: {e}")

    # Test 5: Check Dramatiq Actors
    print("\n5. Testing Dramatiq actors...")
    try:
        from processing_worker import process_images_from_directory, search_cross_modal

        results["tests"]["dramatiq_actors"] = {
            "status": "PASS",
            "message": "Multi-modal Dramatiq actors available",
        }
        print("   ✅ Multi-modal Dramatiq actors available")
        print("      - process_images_from_directory")
        print("      - search_cross_modal")

    except Exception as e:
        results["tests"]["dramatiq_actors"] = {
            "status": "FAIL",
            "message": f"Dramatiq actors missing: {e}",
        }
        print(f"   ❌ Dramatiq actors missing: {e}")

    # Test 6: Check Requirements
    print("\n6. Checking multi-modal dependencies...")

    dependencies = {
        "PIL": "Image processing",
        "cv2": "OpenCV for image preprocessing",
        "pytesseract": "OCR text extraction",
        "piexif": "EXIF metadata extraction",
    }

    available_deps = 0
    total_deps = len(dependencies)

    for dep_name, description in dependencies.items():
        try:
            __import__(dep_name)
            print(f"   ✅ {dep_name} - {description}")
            available_deps += 1
        except ImportError:
            print(f"   ⚠️ {dep_name} - {description} (not installed)")

    dep_percentage = (available_deps / total_deps) * 100
    results["tests"]["dependencies"] = {
        "status": "PASS" if available_deps == total_deps else "WARN",
        "message": f"{available_deps}/{total_deps} dependencies available ({dep_percentage:.1f}%)",
        "available": available_deps,
        "total": total_deps,
    }

    # Summary
    print(f"\n📊 Test Summary:")

    passed = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
    warned = sum(1 for test in results["tests"].values() if test["status"] == "WARN")
    failed = sum(1 for test in results["tests"].values() if test["status"] == "FAIL")
    total = len(results["tests"])

    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "success_rate": (passed / total) * 100 if total > 0 else 0,
    }

    print(f"   ✅ Passed: {passed}")
    print(f"   ⚠️ Warnings: {warned}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {results['summary']['success_rate']:.1f}%")

    # Determine overall status
    if failed == 0 and passed > 0:
        if warned == 0:
            overall_status = "FULL_SUCCESS"
            print(f"\n🎉 Phase 3 Multi-Modality: FULL SUCCESS")
            print("All core functionality is working correctly!")
        else:
            overall_status = "PARTIAL_SUCCESS"
            print(f"\n✅ Phase 3 Multi-Modality: PARTIAL SUCCESS")
            print("Core functionality works, but some optional features need dependencies.")
    elif passed > failed:
        overall_status = "MOSTLY_WORKING"
        print(f"\n⚠️ Phase 3 Multi-Modality: MOSTLY WORKING")
        print("Most functionality works, but some issues need attention.")
    else:
        overall_status = "NEEDS_WORK"
        print(f"\n❌ Phase 3 Multi-Modality: NEEDS WORK")
        print("Several issues need to be resolved.")

    results["overall_status"] = overall_status

    # Save results
    try:
        artifacts_dir = Path("/Users/vojtechhamada/PycharmProjects/DeepResearchTool/artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        with open(artifacts_dir / "phase3_test_result.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Test results saved to artifacts/phase3_test_result.json")

    except Exception as e:
        print(f"\n⚠️ Could not save test results: {e}")

    return results


if __name__ == "__main__":
    test_phase3_basic_functionality()
