"""
Image Processor - Phase 3 Multi-Modality
Zpracování obrázků s OCR, EXIF extrakce a multi-modální embeddings
"""

import os
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import requests
from urllib.parse import urlparse, urljoin

# Image processing imports
try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS, GPSTAGS
    import pytesseract
    from sentence_transformers import SentenceTransformer
    import torch

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"⚠️ Image processing dependencies not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Pokročilý procesor obrázků pro multi-modální analýzu
    """

    def __init__(self, download_dir: str = "/tmp/images"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # Initialize multi-modal model if available
        self.clip_model = None
        if DEPENDENCIES_AVAILABLE:
            try:
                self.clip_model = SentenceTransformer("clip-ViT-B-32")
                logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load CLIP model: {e}")

    def extract_images_from_html(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extrakce všech obrázků z HTML obsahu
        """
        import re

        images = []

        # Find img tags
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        img_matches = re.findall(img_pattern, html_content, re.IGNORECASE)

        for img_url in img_matches:
            full_url = urljoin(base_url, img_url)
            images.append(
                {"url": full_url, "type": "img_tag", "source_html": f'<img src="{img_url}">'}
            )

        # Find CSS background images
        bg_pattern = r'background-image:\s*url\(["\']?([^"\')\s]+)["\']?\)'
        bg_matches = re.findall(bg_pattern, html_content, re.IGNORECASE)

        for bg_url in bg_matches:
            full_url = urljoin(base_url, bg_url)
            images.append(
                {
                    "url": full_url,
                    "type": "background_image",
                    "source_html": f"background-image: url({bg_url})",
                }
            )

        logger.info(f"🖼️ Found {len(images)} images in HTML")
        return images

    def download_image(self, image_url: str, source_url: str) -> Optional[Path]:
        """
        Stažení obrázku s error handling
        """
        try:
            # Create safe filename
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:10]
            parsed = urlparse(image_url)
            extension = Path(parsed.path).suffix or ".jpg"
            filename = f"{url_hash}{extension}"
            filepath = self.download_dir / filename

            # Skip if already exists
            if filepath.exists():
                return filepath

            # Download with timeout
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Save image
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"📥 Downloaded image: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Failed to download {image_url}: {e}")
            return None

    def extract_exif_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Extrakce EXIF metadat z obrázku
        """
        if not DEPENDENCIES_AVAILABLE:
            return {}

        try:
            image = Image.open(image_path)
            exifdata = image.getexif()

            metadata = {
                "filename": image_path.name,
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "exif_available": len(exifdata) > 0,
            }

            if exifdata:
                # Extract basic EXIF
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[f"exif_{tag}"] = str(value)

                # Extract GPS data if available
                gps_info = exifdata.get_ifd(0x8825)  # GPS IFD
                if gps_info:
                    gps_data = {}
                    for key, value in gps_info.items():
                        name = GPSTAGS.get(key, key)
                        gps_data[name] = value
                    metadata["gps_data"] = gps_data

            logger.info(f"📊 Extracted EXIF from {image_path.name}")
            return metadata

        except Exception as e:
            logger.error(f"❌ EXIF extraction failed for {image_path}: {e}")
            return {"error": str(e)}

    def extract_text_ocr(self, image_path: Path) -> Dict[str, Any]:
        """
        OCR extrakce textu z obrázku
        """
        if not DEPENDENCIES_AVAILABLE:
            return {"text": "", "confidence": 0, "error": "OCR not available"}

        try:
            image = Image.open(image_path)

            # Run OCR with confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # Assume uniform block of text
            )

            # Extract text and calculate confidence
            words = []
            confidences = []

            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i])

                if word and conf > 0:
                    words.append(word)
                    confidences.append(conf)

            full_text = " ".join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            result = {
                "text": full_text,
                "word_count": len(words),
                "confidence": avg_confidence,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"🔤 OCR extracted {len(words)} words from {image_path.name}")
            return result

        except Exception as e:
            logger.error(f"❌ OCR failed for {image_path}: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}

    def generate_embedding(self, image_path: Path) -> Optional[List[float]]:
        """
        Generování multi-modálního embeddings pro obrázek
        """
        if not self.clip_model:
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            embedding = self.clip_model.encode(image)

            logger.info(f"🎯 Generated embedding for {image_path.name}")
            return embedding.tolist()

        except Exception as e:
            logger.error(f"❌ Embedding generation failed for {image_path}: {e}")
            return None

    def process_image_complete(self, image_url: str, source_url: str) -> Dict[str, Any]:
        """
        Kompletní zpracování obrázku - download, EXIF, OCR, embedding
        """
        result = {
            "image_url": image_url,
            "source_url": source_url,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
        }

        # Download image
        image_path = self.download_image(image_url, source_url)
        if not image_path:
            result["error"] = "Failed to download image"
            return result

        result["local_path"] = str(image_path)

        # Extract EXIF metadata
        result["metadata"] = self.extract_exif_metadata(image_path)

        # Extract text via OCR
        result["ocr"] = self.extract_text_ocr(image_path)

        # Generate embedding
        embedding = self.generate_embedding(image_path)
        if embedding:
            result["embedding"] = embedding

        result["success"] = True
        logger.info(f"✅ Complete image processing done for {image_path.name}")

        return result


# Global instance
image_processor = ImageProcessor()
