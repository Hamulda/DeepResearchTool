"""Image Processing Module - Fáze 3: Multi-Modality
Zpracování obrazových dat včetně OCR, EXIF metadat a cross-modálního vyhledávání
"""

import base64
from datetime import UTC, datetime
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

# Image processing imports
try:
    import cv2
    import numpy as np
    import piexif
    from PIL import ExifTags, Image
    import pytesseract
    from sentence_transformers import SentenceTransformer

    # CLIP model pro cross-modální embeddings
    try:
        import clip
        import torch

        CLIP_AVAILABLE = True
    except ImportError:
        CLIP_AVAILABLE = False

    PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Image processing knihovny nejsou dostupné: {e}")
    PROCESSING_AVAILABLE = False
    CLIP_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Procesor pro zpracování obrazových dat"""

    def __init__(self):
        if not PROCESSING_AVAILABLE:
            raise ImportError("Image processing knihovny nejsou dostupné")

        # Konfigurace Tesseract OCR
        self.tesseract_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()-_=+[]{}|;:\'\"<>/~`"

        # Sentence transformer pro text embeddings
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # CLIP model pro cross-modální embeddings
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_device = "cpu"

        if CLIP_AVAILABLE:
            try:
                self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load(
                    "ViT-B/32", device=self.clip_device
                )
                logger.info(f"✅ CLIP model načten na {self.clip_device}")
            except Exception as e:
                logger.warning(f"⚠️ CLIP model se nepodařilo načíst: {e}")
                CLIP_AVAILABLE = False

        logger.info("✅ Image Processor inicializován")

    def process_image(self, image_path: str, source_url: str = "") -> dict[str, Any]:
        """Kompletní zpracování obrázku

        Args:
            image_path: Cesta k obrázku
            source_url: URL zdroje obrázku

        Returns:
            Slovník s extrahovanými daty

        """
        try:
            logger.info(f"📸 Zpracovávám obrázek: {image_path}")

            # Základní info o souboru
            file_info = self._get_file_info(image_path)

            # Načti obrázek
            image = Image.open(image_path)

            # Extrakce EXIF metadat
            exif_data = self._extract_exif_metadata(image)

            # OCR extrakce textu
            ocr_result = self._extract_text_ocr(image)

            # Analýza obsahu obrázku
            content_analysis = self._analyze_image_content(image)

            # Generování embeddings
            embeddings = self._generate_image_embeddings(image, ocr_result.get("text", ""))

            # Vytvoř hash pro deduplikaci
            image_hash = self._calculate_image_hash(image)

            result = {
                "file_info": file_info,
                "exif_metadata": exif_data,
                "ocr_result": ocr_result,
                "content_analysis": content_analysis,
                "embeddings": embeddings,
                "image_hash": image_hash,
                "source_url": source_url,
                "processed_at": datetime.now(UTC).isoformat(),
                "processor_version": "1.0",
            }

            logger.info(f"✅ Obrázek zpracován: {len(ocr_result.get('text', ''))} znaků OCR")
            return result

        except Exception as e:
            logger.error(f"❌ Chyba při zpracování obrázku {image_path}: {e}")
            return {
                "error": str(e),
                "file_path": image_path,
                "source_url": source_url,
                "processed_at": datetime.now(UTC).isoformat(),
            }

    def _get_file_info(self, image_path: str) -> dict[str, Any]:
        """Získej základní informace o souboru"""
        try:
            path = Path(image_path)
            stat = path.stat()

            return {
                "filename": path.name,
                "file_size": stat.st_size,
                "file_extension": path.suffix.lower(),
                "created_at": datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Chyba při získávání file info: {e}")
            return {"error": str(e)}

    def _extract_exif_metadata(self, image: Image.Image) -> dict[str, Any]:
        """Extrahuj EXIF metadata z obrázku"""
        try:
            exif_dict = {}

            # Základní EXIF data z PIL
            if hasattr(image, "_getexif") and image._getexif() is not None:
                exif = image._getexif()

                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)

                    # Převeď hodnoty na JSON-serializable formát
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8")
                        except:
                            value = base64.b64encode(value).decode("utf-8")
                    elif isinstance(value, tuple):
                        value = list(value)

                    exif_dict[str(tag)] = value

            # Detailní EXIF pomocí piexif
            try:
                if image.format in ["JPEG", "TIFF"]:
                    # Uložit obrázek do bytes pro piexif
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format=image.format)
                    img_bytes.seek(0)

                    exif_detailed = piexif.load(img_bytes.getvalue())

                    # GPS data
                    if "GPS" in exif_detailed:
                        gps_data = self._parse_gps_data(exif_detailed["GPS"])
                        if gps_data:
                            exif_dict["GPS"] = gps_data

                    # Camera info
                    if "Exif" in exif_detailed:
                        camera_info = self._parse_camera_info(exif_detailed["Exif"])
                        exif_dict.update(camera_info)

            except Exception as e:
                logger.warning(f"⚠️ Piexif extrakce selhala: {e}")

            # Přidej základní vlastnosti obrázku
            exif_dict.update(
                {
                    "image_width": image.width,
                    "image_height": image.height,
                    "image_mode": image.mode,
                    "image_format": image.format,
                }
            )

            return exif_dict

        except Exception as e:
            logger.error(f"❌ EXIF extrakce selhala: {e}")
            return {"error": str(e)}

    def _parse_gps_data(self, gps_dict: dict) -> dict[str, Any] | None:
        """Parsuj GPS souřadnice z EXIF"""
        try:
            if not gps_dict:
                return None

            def convert_to_degrees(value):
                """Převeď GPS souřadnice na stupně"""
                if isinstance(value, tuple) and len(value) == 3:
                    degrees, minutes, seconds = value
                    return float(degrees) + float(minutes) / 60 + float(seconds) / 3600
                return value

            gps_info = {}

            # Latitude
            if piexif.GPSIFD.GPSLatitude in gps_dict:
                lat = convert_to_degrees(gps_dict[piexif.GPSIFD.GPSLatitude])
                lat_ref = gps_dict.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode("utf-8")
                if lat_ref == "S":
                    lat = -lat
                gps_info["latitude"] = lat

            # Longitude
            if piexif.GPSIFD.GPSLongitude in gps_dict:
                lon = convert_to_degrees(gps_dict[piexif.GPSIFD.GPSLongitude])
                lon_ref = gps_dict.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode("utf-8")
                if lon_ref == "W":
                    lon = -lon
                gps_info["longitude"] = lon

            # Altitude
            if piexif.GPSIFD.GPSAltitude in gps_dict:
                altitude = gps_dict[piexif.GPSIFD.GPSAltitude]
                if isinstance(altitude, tuple):
                    altitude = float(altitude[0]) / float(altitude[1])
                gps_info["altitude"] = altitude

            # Timestamp
            if piexif.GPSIFD.GPSTimeStamp in gps_dict:
                time_stamp = gps_dict[piexif.GPSIFD.GPSTimeStamp]
                if isinstance(time_stamp, tuple) and len(time_stamp) == 3:
                    hours, minutes, seconds = time_stamp
                    if isinstance(hours, tuple):
                        hours = float(hours[0]) / float(hours[1])
                    if isinstance(minutes, tuple):
                        minutes = float(minutes[0]) / float(minutes[1])
                    if isinstance(seconds, tuple):
                        seconds = float(seconds[0]) / float(seconds[1])

                    gps_info["gps_time"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            return gps_info if gps_info else None

        except Exception as e:
            logger.error(f"❌ GPS parsing error: {e}")
            return None

    def _parse_camera_info(self, exif_dict: dict) -> dict[str, Any]:
        """Parsuj informace o fotoaparátu"""
        try:
            camera_info = {}

            # Mapování EXIF tagů
            tag_mapping = {
                piexif.ExifIFD.Make: "camera_make",
                piexif.ExifIFD.Model: "camera_model",
                piexif.ExifIFD.Software: "software",
                piexif.ExifIFD.DateTime: "datetime_taken",
                piexif.ExifIFD.ExposureTime: "exposure_time",
                piexif.ExifIFD.FNumber: "f_number",
                piexif.ExifIFD.ISO: "iso",
                piexif.ExifIFD.Flash: "flash",
                piexif.ExifIFD.FocalLength: "focal_length",
            }

            for exif_tag, field_name in tag_mapping.items():
                if exif_tag in exif_dict:
                    value = exif_dict[exif_tag]

                    # Konverze hodnot
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8").strip("\x00")
                        except:
                            continue
                    elif isinstance(value, tuple) and len(value) == 2:
                        # Rational number
                        if value[1] != 0:
                            value = float(value[0]) / float(value[1])
                        else:
                            value = float(value[0])

                    camera_info[field_name] = value

            return camera_info

        except Exception as e:
            logger.error(f"❌ Camera info parsing error: {e}")
            return {}

    def _extract_text_ocr(self, image: Image.Image) -> dict[str, Any]:
        """Extrahuj text z obrázku pomocí OCR"""
        try:
            # Preprocess obrázku pro lepší OCR
            processed_image = self._preprocess_for_ocr(image)

            # OCR extrakce
            extracted_text = pytesseract.image_to_string(
                processed_image, config=self.tesseract_config
            ).strip()

            # OCR s detailními informacemi
            ocr_data = pytesseract.image_to_data(
                processed_image, output_type=pytesseract.Output.DICT, config=self.tesseract_config
            )

            # Filtruj kvalitní detekce
            high_confidence_words = []
            for i, conf in enumerate(ocr_data["conf"]):
                if int(conf) > 30:  # Confidence threshold
                    word = ocr_data["text"][i].strip()
                    if word:
                        high_confidence_words.append(
                            {
                                "text": word,
                                "confidence": int(conf),
                                "bbox": {
                                    "x": ocr_data["left"][i],
                                    "y": ocr_data["top"][i],
                                    "width": ocr_data["width"][i],
                                    "height": ocr_data["height"][i],
                                },
                            }
                        )

            return {
                "text": extracted_text,
                "text_length": len(extracted_text),
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "high_confidence_words": high_confidence_words,
                "total_words_detected": len([w for w in ocr_data["text"] if w.strip()]),
                "average_confidence": (
                    np.mean([c for c in ocr_data["conf"] if c > 0]) if ocr_data["conf"] else 0
                ),
            }

        except Exception as e:
            logger.error(f"❌ OCR extrakce selhala: {e}")
            return {"text": "", "text_length": 0, "word_count": 0, "error": str(e)}

    def _preprocess_for_ocr(self, image: Image.Image) -> np.ndarray:
        """Preprocess obrázku pro lepší OCR výsledky"""
        try:
            # Převeď na numpy array
            img_array = np.array(image.convert("RGB"))

            # Převeď na grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Zvětši obrázek pokud je malý
            height, width = gray.shape
            if height < 300 or width < 300:
                scale_factor = max(300 / height, 300 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Gaussian blur pro odstranění šumu
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Adaptive threshold pro lepší kontrast
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            return processed

        except Exception as e:
            logger.error(f"❌ Image preprocessing error: {e}")
            # Fallback na původní obrázek
            return np.array(image.convert("L"))

    def _analyze_image_content(self, image: Image.Image) -> dict[str, Any]:
        """Základní analýza obsahu obrázku"""
        try:
            analysis = {}

            # Barevná analýza
            if image.mode == "RGB":
                colors = image.getcolors(maxcolors=256 * 256 * 256)
                if colors:
                    # Dominantní barvy
                    dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                    analysis["dominant_colors"] = [
                        {"color": color, "count": count} for count, color in dominant_colors
                    ]

            # Histogram analýza
            histogram = image.histogram()
            if histogram:
                analysis["brightness_avg"] = (
                    sum(i * v for i, v in enumerate(histogram)) / sum(histogram)
                    if sum(histogram) > 0
                    else 0
                )

            # Detekce typu obsahu (heuristicky)
            content_type = self._detect_content_type(image)
            analysis["content_type"] = content_type

            return analysis

        except Exception as e:
            logger.error(f"❌ Content analysis error: {e}")
            return {"error": str(e)}

    def _detect_content_type(self, image: Image.Image) -> str:
        """Heuristická detekce typu obsahu"""
        try:
            width, height = image.size
            aspect_ratio = width / height

            # Screenshot detection (často 16:9 nebo 4:3)
            if 1.2 < aspect_ratio < 2.0 and min(width, height) > 400:
                return "screenshot"

            # Photo detection
            if image.mode == "RGB" and min(width, height) > 200:
                return "photo"

            # Document/text detection (často vysoký a úzký)
            if aspect_ratio < 0.8 or aspect_ratio > 2.5:
                return "document"

            # Icon/logo detection (malý a čtvercový)
            if max(width, height) < 200 and 0.5 < aspect_ratio < 2.0:
                return "icon"

            return "unknown"

        except Exception as e:
            logger.error(f"❌ Content type detection error: {e}")
            return "unknown"

    def _generate_image_embeddings(self, image: Image.Image, ocr_text: str) -> dict[str, Any]:
        """Generuj embeddings pro cross-modální vyhledávání"""
        try:
            embeddings = {}

            # Text embedding z OCR textu
            if ocr_text.strip():
                text_embedding = self.sentence_model.encode([ocr_text])[0]
                embeddings["text_embedding"] = text_embedding.tolist()
                embeddings["text_embedding_model"] = "all-MiniLM-L6-v2"
                embeddings["text_embedding_dim"] = len(text_embedding)

            # CLIP embedding pro cross-modální vyhledávání
            if CLIP_AVAILABLE and self.clip_model:
                try:
                    # Image embedding
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)

                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(image_input)
                        image_embedding = image_features.cpu().numpy()[0]

                    embeddings["clip_image_embedding"] = image_embedding.tolist()
                    embeddings["clip_embedding_model"] = "ViT-B/32"
                    embeddings["clip_embedding_dim"] = len(image_embedding)

                    # Text embedding z CLIP (pokud máme text)
                    if ocr_text.strip():
                        text_input = clip.tokenize([ocr_text[:77]]).to(
                            self.clip_device
                        )  # CLIP limit

                        with torch.no_grad():
                            text_features = self.clip_model.encode_text(text_input)
                            clip_text_embedding = text_features.cpu().numpy()[0]

                        embeddings["clip_text_embedding"] = clip_text_embedding.tolist()

                except Exception as e:
                    logger.warning(f"⚠️ CLIP embedding error: {e}")

            return embeddings

        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            return {"error": str(e)}

    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Vytvoř hash pro detekci duplicitních obrázků"""
        try:
            # Resize na malou velikost pro hash
            small_image = image.resize((8, 8), Image.Resampling.LANCZOS).convert("L")

            # Vytvoř hash z pixel hodnot
            pixels = list(small_image.getdata())

            # Average hash
            avg = sum(pixels) / len(pixels)
            hash_bits = "".join("1" if pixel > avg else "0" for pixel in pixels)

            # Převeď na hexadecimální
            hash_hex = hex(int(hash_bits, 2))[2:]

            return hash_hex

        except Exception as e:
            logger.error(f"❌ Image hash error: {e}")
            return hashlib.md5(str(image.size).encode()).hexdigest()

    def batch_process_images(
        self, image_paths: list[str], source_urls: list[str] = None
    ) -> list[dict[str, Any]]:
        """Zpracuj více obrázků současně"""
        try:
            if source_urls is None:
                source_urls = [""] * len(image_paths)

            results = []

            for i, image_path in enumerate(image_paths):
                source_url = source_urls[i] if i < len(source_urls) else ""
                result = self.process_image(image_path, source_url)
                results.append(result)

            logger.info(f"✅ Batch zpracování dokončeno: {len(results)} obrázků")
            return results

        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            return []

    def search_similar_images(
        self, query_embedding: list[float], image_embeddings: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Vyhledej podobné obrázky pomocí embedding similarity"""
        try:
            similarities = []

            for img_data in image_embeddings:
                embeddings = img_data.get("embeddings", {})

                # Porovnej s CLIP embeddings pokud jsou dostupné
                if "clip_image_embedding" in embeddings:
                    img_embedding = np.array(embeddings["clip_image_embedding"])
                    query_emb = np.array(query_embedding)

                    # Cosine similarity
                    similarity = np.dot(query_emb, img_embedding) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(img_embedding)
                    )

                    similarities.append({"image_data": img_data, "similarity": float(similarity)})

            # Seřaď podle podobnosti
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            return similarities[:top_k]

        except Exception as e:
            logger.error(f"❌ Image search error: {e}")
            return []
