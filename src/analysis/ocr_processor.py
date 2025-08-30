"""
OCR processor pro extrakci textu z obrázků
Podporuje multiple OCR engines s fallback mechanismy
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime
import io

import numpy as np
from PIL import Image
import requests

from ..core.error_handling import scraping_retry, ErrorAggregator, timeout_after
from ..core.config import get_settings

logger = logging.getLogger(__name__)

# OCR engine availability checks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available")


class OCREngine:
    """Base class for OCR engines"""
    
    def __init__(self, name: str):
        self.name = name
        self.confidence_threshold = 0.5
        self.supported_languages = ['en', 'cs', 'de', 'fr', 'es']
    
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """Extract text from image"""
        raise NotImplementedError


class TesseractEngine(OCREngine):
    """Tesseract OCR engine wrapper"""
    
    def __init__(self):
        super().__init__("tesseract")
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available")
        
        # Configure Tesseract
        self.config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúčďěňřšťůžČĎĚŇŘŠŤŮŽ.,!?;:()[]{}"\'-/\\@#$%^&*+=<>|`~'
    
    @timeout_after(30)
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """Extract text using Tesseract"""
        try:
            # Prepare image
            pil_image = await self._prepare_image(image)
            
            # Run OCR in thread pool
            loop = asyncio.get_event_loop()
            
            # Extract text with confidence
            text_data = await loop.run_in_executor(
                None, 
                lambda: pytesseract.image_to_data(
                    pil_image, 
                    config=self.config, 
                    output_type=pytesseract.Output.DICT
                )
            )
            
            # Process results
            extracted_text = []
            confidences = []
            
            for i in range(len(text_data['text'])):
                text = text_data['text'][i].strip()
                confidence = int(text_data['conf'][i])
                
                if text and confidence > self.confidence_threshold * 100:
                    extracted_text.append(text)
                    confidences.append(confidence)
            
            full_text = ' '.join(extracted_text)
            avg_confidence = np.mean(confidences) / 100 if confidences else 0
            
            return {
                'engine': self.name,
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(extracted_text),
                'success': bool(full_text.strip()),
                'raw_data': text_data if len(str(text_data)) < 10000 else None  # Limit size
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'engine': self.name,
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }
    
    async def _prepare_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Prepare image for OCR processing"""
        if isinstance(image, (str, Path)):
            return Image.open(str(image))
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


class EasyOCREngine(OCREngine):
    """EasyOCR engine wrapper"""
    
    def __init__(self, languages: List[str] = None):
        super().__init__("easyocr")
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available")
        
        self.languages = languages or ['en', 'cs']
        self.reader = None
    
    async def _initialize_reader(self):
        """Initialize EasyOCR reader lazily"""
        if self.reader is None:
            loop = asyncio.get_event_loop()
            self.reader = await loop.run_in_executor(
                None, 
                lambda: easyocr.Reader(self.languages, gpu=False)  # CPU only for M1 compatibility
            )
    
    @timeout_after(45)
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        try:
            await self._initialize_reader()
            
            # Prepare image
            if isinstance(image, Image.Image):
                # Convert PIL to numpy array
                image_array = np.array(image)
            elif isinstance(image, (str, Path)):
                image_array = str(image)
            else:
                image_array = image
            
            # Run OCR in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.reader.readtext(image_array, detail=1)
            )
            
            # Process results
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > self.confidence_threshold:
                    extracted_texts.append(text)
                    confidences.append(confidence)
            
            full_text = ' '.join(extracted_texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'engine': self.name,
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(extracted_texts),
                'success': bool(full_text.strip()),
                'bounding_boxes': len(results)
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {
                'engine': self.name,
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine wrapper"""
    
    def __init__(self, language: str = 'en'):
        super().__init__("paddleocr")
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not available")
        
        self.language = language
        self.ocr = None
    
    async def _initialize_ocr(self):
        """Initialize PaddleOCR lazily"""
        if self.ocr is None:
            loop = asyncio.get_event_loop()
            self.ocr = await loop.run_in_executor(
                None,
                lambda: PaddleOCR(
                    use_angle_cls=True, 
                    lang=self.language,
                    use_gpu=False  # CPU only for M1
                )
            )
    
    @timeout_after(60)
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """Extract text using PaddleOCR"""
        try:
            await self._initialize_ocr()
            
            # Prepare image path
            if isinstance(image, Image.Image):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name
            elif isinstance(image, (str, Path)):
                image_path = str(image)
            else:
                # Convert numpy array to PIL and save
                pil_image = Image.fromarray(image)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    pil_image.save(tmp.name)
                    image_path = tmp.name
            
            # Run OCR
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.ocr.ocr(image_path, cls=True)
            )
            
            # Process results
            extracted_texts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        if confidence > self.confidence_threshold:
                            extracted_texts.append(text)
                            confidences.append(confidence)
            
            full_text = ' '.join(extracted_texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Clean up temp file
            if isinstance(image, (Image.Image, np.ndarray)):
                Path(image_path).unlink(missing_ok=True)
            
            return {
                'engine': self.name,
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(extracted_texts),
                'success': bool(full_text.strip()),
                'lines_detected': len(results[0]) if results and results[0] else 0
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return {
                'engine': self.name,
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }


class MultiOCRProcessor:
    """Multi-engine OCR processor with fallback mechanisms"""
    
    def __init__(self, preferred_engines: List[str] = None):
        self.error_aggregator = ErrorAggregator()
        self.preferred_engines = preferred_engines or ['easyocr', 'tesseract', 'paddleocr']
        self.engines = {}
        self.cache = {}  # Simple in-memory cache
        self.max_cache_size = 100
        
        # Initialize available engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        for engine_name in self.preferred_engines:
            try:
                if engine_name == 'tesseract' and TESSERACT_AVAILABLE:
                    self.engines[engine_name] = TesseractEngine()
                elif engine_name == 'easyocr' and EASYOCR_AVAILABLE:
                    self.engines[engine_name] = EasyOCREngine()
                elif engine_name == 'paddleocr' and PADDLEOCR_AVAILABLE:
                    self.engines[engine_name] = PaddleOCREngine()
            except Exception as e:
                logger.warning(f"Failed to initialize {engine_name}: {e}")
        
        if not self.engines:
            logger.warning("No OCR engines available!")
        else:
            logger.info(f"Initialized OCR engines: {list(self.engines.keys())}")
    
    def _get_image_hash(self, image: Union[Image.Image, np.ndarray, str, Path]) -> str:
        """Generate hash for image caching"""
        try:
            if isinstance(image, (str, Path)):
                # Hash file path and modification time
                file_path = Path(image)
                if file_path.exists():
                    stat = file_path.stat()
                    content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
                else:
                    content = str(image)
            elif isinstance(image, Image.Image):
                # Hash image data
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                content = buffer.getvalue()
            else:
                # Hash numpy array
                content = image.tobytes()
            
            return hashlib.md5(str(content).encode()).hexdigest()
        except:
            return str(hash(str(image)))
    
    @scraping_retry
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str, Path], 
                          use_cache: bool = True) -> Dict[str, Any]:
        """Extract text using multiple OCR engines with fallback"""
        
        # Check cache
        image_hash = self._get_image_hash(image) if use_cache else None
        if image_hash and image_hash in self.cache:
            logger.debug("Using cached OCR result")
            return self.cache[image_hash]
        
        if not self.engines:
            return {
                'text': '',
                'confidence': 0,
                'success': False,
                'error': 'No OCR engines available',
                'engines_tried': []
            }
        
        best_result = None
        engines_tried = []
        
        # Try engines in order of preference
        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Trying OCR engine: {engine_name}")
                result = await engine.extract_text(image)
                result['timestamp'] = datetime.now().isoformat()
                engines_tried.append(engine_name)
                
                if result.get('success') and result.get('text', '').strip():
                    # Success - use this result
                    if not best_result or result.get('confidence', 0) > best_result.get('confidence', 0):
                        best_result = result
                    
                    # If confidence is high enough, stop trying other engines
                    if result.get('confidence', 0) > 0.8:
                        break
                
                self.error_aggregator.add_success()
                
            except Exception as e:
                self.error_aggregator.add_error(e, f"OCR engine {engine_name}")
                logger.warning(f"OCR engine {engine_name} failed: {e}")
                continue
        
        # Prepare final result
        if best_result:
            final_result = {
                **best_result,
                'engines_tried': engines_tried,
                'total_engines': len(self.engines)
            }
        else:
            final_result = {
                'text': '',
                'confidence': 0,
                'success': False,
                'error': 'All OCR engines failed',
                'engines_tried': engines_tried,
                'total_engines': len(self.engines)
            }
        
        # Cache result
        if use_cache and image_hash:
            # Limit cache size
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[image_hash] = final_result
        
        return final_result
    
    async def extract_text_from_url(self, image_url: str) -> Dict[str, Any]:
        """Extract text from image URL"""
        try:
            # Download image
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Open as PIL image
            image = Image.open(io.BytesIO(response.content))
            
            # Process with OCR
            result = await self.extract_text(image)
            result['source_url'] = image_url
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image from URL {image_url}: {e}")
            return {
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e),
                'source_url': image_url
            }
    
    async def extract_text_batch(self, images: List[Union[Image.Image, str, Path]]) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        results = []
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent OCR operations
        
        async def process_image(image):
            async with semaphore:
                return await self.extract_text(image)
        
        tasks = [process_image(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'text': '',
                    'confidence': 0,
                    'success': False,
                    'error': str(result),
                    'image_index': i
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get OCR processing statistics"""
        error_summary = self.error_aggregator.get_summary()
        
        return {
            'available_engines': list(self.engines.keys()),
            'preferred_engines': self.preferred_engines,
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'success_rate': error_summary['success_rate'],
            'total_operations': error_summary['total_operations'],
            'failed_operations': error_summary['failed_operations']
        }


# Utility functions
async def extract_text_from_image(image: Union[Image.Image, str, Path]) -> str:
    """Quick utility for single image OCR"""
    processor = MultiOCRProcessor()
    result = await processor.extract_text(image)
    return result.get('text', '')


async def extract_text_from_images(images: List[Union[Image.Image, str, Path]]) -> List[str]:
    """Quick utility for batch image OCR"""
    processor = MultiOCRProcessor()
    results = await processor.extract_text_batch(images)
    return [result.get('text', '') for result in results]