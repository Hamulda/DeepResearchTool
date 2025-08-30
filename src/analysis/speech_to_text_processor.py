"""
Speech-to-Text processor s podporou multiple engines
Implementuje Whisper, Azure Cognitive Services a Google Speech-to-Text s fallback
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime
import io
import json

import numpy as np

from ..core.error_handling import scraping_retry, ErrorAggregator, timeout_after
from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Speech engine availability checks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("OpenAI Whisper not available")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("SpeechRecognition library not available")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("Pydub not available - audio format conversion limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available - advanced audio processing limited")


class AudioProcessor:
    """Audio preprocessing utilities"""
    
    @staticmethod
    def convert_to_wav(audio_path: Union[str, Path], target_sample_rate: int = 16000) -> str:
        """Convert audio file to WAV format"""
        if not PYDUB_AVAILABLE:
            # Fallback: assume already in correct format
            return str(audio_path)
        
        try:
            audio = AudioSegment.from_file(str(audio_path))
            
            # Convert to mono and target sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(target_sample_rate)
            
            # Export to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio.export(tmp.name, format='wav')
                return tmp.name
                
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return str(audio_path)
    
    @staticmethod
    def split_audio(audio_path: Union[str, Path], chunk_duration: int = 60) -> List[str]:
        """Split long audio into chunks"""
        if not PYDUB_AVAILABLE:
            return [str(audio_path)]
        
        try:
            audio = AudioSegment.from_file(str(audio_path))
            chunks = []
            
            # Split into chunks of specified duration (in seconds)
            chunk_length_ms = chunk_duration * 1000
            
            for i in range(0, len(audio), chunk_length_ms):
                chunk = audio[i:i + chunk_length_ms]
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    chunk.export(tmp.name, format='wav')
                    chunks.append(tmp.name)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Audio splitting failed: {e}")
            return [str(audio_path)]
    
    @staticmethod
    def get_audio_info(audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = len(y) / sr
                return {
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': 1 if y.ndim == 1 else y.shape[0],
                    'samples': len(y)
                }
            elif PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(str(audio_path))
                return {
                    'duration': len(audio) / 1000.0,  # Convert to seconds
                    'sample_rate': audio.frame_rate,
                    'channels': audio.channels,
                    'samples': len(audio.raw_data)
                }
            else:
                return {'duration': 0, 'sample_rate': 16000, 'channels': 1, 'samples': 0}
                
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {'duration': 0, 'sample_rate': 16000, 'channels': 1, 'samples': 0}


class WhisperEngine:
    """OpenAI Whisper speech-to-text engine"""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.supported_languages = ['en', 'cs', 'de', 'fr', 'es', 'ru', 'ja', 'zh']
    
    async def _load_model(self):
        """Load Whisper model lazily"""
        if self.model is None:
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper not available")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: whisper.load_model(self.model_size)
            )
            logger.info(f"Loaded Whisper model: {self.model_size}")
    
    @timeout_after(300)  # 5 minute timeout for long audio
    async def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            await self._load_model()
            
            # Convert audio to WAV if needed
            wav_path = AudioProcessor.convert_to_wav(audio_path)
            
            # Get audio info
            audio_info = AudioProcessor.get_audio_info(wav_path)
            
            # Transcribe in thread pool
            loop = asyncio.get_event_loop()
            
            transcribe_options = {
                'fp16': False,  # Better compatibility with M1
                'language': language,
                'task': 'transcribe'
            }
            
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(str(wav_path), **transcribe_options)
            )
            
            # Clean up temporary file
            if wav_path != str(audio_path):
                Path(wav_path).unlink(missing_ok=True)
            
            # Process segments for detailed output
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'confidence': segment.get('avg_logprob', 0)  # Whisper uses log probability
                })
            
            return {
                'engine': 'whisper',
                'text': result.get('text', '').strip(),
                'language': result.get('language', language or 'unknown'),
                'segments': segments,
                'duration': audio_info.get('duration', 0),
                'success': bool(result.get('text', '').strip()),
                'model_size': self.model_size
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                'engine': 'whisper',
                'text': '',
                'success': False,
                'error': str(e)
            }


class GoogleSpeechEngine:
    """Google Speech-to-Text using SpeechRecognition library"""
    
    def __init__(self):
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("SpeechRecognition library not available")
        
        self.recognizer = sr.Recognizer()
        self.supported_languages = ['en-US', 'cs-CZ', 'de-DE', 'fr-FR', 'es-ES']
    
    @timeout_after(180)  # 3 minute timeout
    async def transcribe(self, audio_path: Union[str, Path], language: str = 'en-US') -> Dict[str, Any]:
        """Transcribe audio using Google Speech-to-Text"""
        try:
            # Convert to WAV
            wav_path = AudioProcessor.convert_to_wav(audio_path)
            
            # Load audio file
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Transcribe in thread pool
            loop = asyncio.get_event_loop()
            
            text = await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize_google(
                    audio_data, 
                    language=language,
                    show_all=False
                )
            )
            
            # Clean up temporary file
            if wav_path != str(audio_path):
                Path(wav_path).unlink(missing_ok=True)
            
            return {
                'engine': 'google',
                'text': text.strip() if text else '',
                'language': language,
                'success': bool(text and text.strip()),
            }
            
        except sr.UnknownValueError:
            return {
                'engine': 'google',
                'text': '',
                'success': False,
                'error': 'Could not understand audio'
            }
        except sr.RequestError as e:
            return {
                'engine': 'google',
                'text': '',
                'success': False,
                'error': f'Google API error: {e}'
            }
        except Exception as e:
            logger.error(f"Google Speech transcription failed: {e}")
            return {
                'engine': 'google',
                'text': '',
                'success': False,
                'error': str(e)
            }


class SphinxEngine:
    """CMU Sphinx (offline) speech recognition"""
    
    def __init__(self):
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("SpeechRecognition library not available")
        
        self.recognizer = sr.Recognizer()
    
    @timeout_after(120)
    async def transcribe(self, audio_path: Union[str, Path], language: str = 'en') -> Dict[str, Any]:
        """Transcribe audio using CMU Sphinx (offline)"""
        try:
            # Convert to WAV
            wav_path = AudioProcessor.convert_to_wav(audio_path)
            
            # Load audio file
            with sr.AudioFile(wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Transcribe in thread pool
            loop = asyncio.get_event_loop()
            
            text = await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize_sphinx(audio_data)
            )
            
            # Clean up temporary file
            if wav_path != str(audio_path):
                Path(wav_path).unlink(missing_ok=True)
            
            return {
                'engine': 'sphinx',
                'text': text.strip() if text else '',
                'language': language,
                'success': bool(text and text.strip()),
                'offline': True
            }
            
        except sr.UnknownValueError:
            return {
                'engine': 'sphinx',
                'text': '',
                'success': False,
                'error': 'Could not understand audio',
                'offline': True
            }
        except Exception as e:
            logger.error(f"Sphinx transcription failed: {e}")
            return {
                'engine': 'sphinx',
                'text': '',
                'success': False,
                'error': str(e),
                'offline': True
            }


class MultiSpeechProcessor:
    """Multi-engine speech-to-text processor with fallback mechanisms"""
    
    def __init__(self, preferred_engines: List[str] = None, whisper_model: str = "base"):
        self.error_aggregator = ErrorAggregator()
        self.preferred_engines = preferred_engines or ['whisper', 'google', 'sphinx']
        self.engines = {}
        self.cache = {}
        self.max_cache_size = 50
        
        # Initialize available engines
        self._initialize_engines(whisper_model)
    
    def _initialize_engines(self, whisper_model: str):
        """Initialize available speech engines"""
        for engine_name in self.preferred_engines:
            try:
                if engine_name == 'whisper' and WHISPER_AVAILABLE:
                    self.engines[engine_name] = WhisperEngine(whisper_model)
                elif engine_name == 'google' and SPEECH_RECOGNITION_AVAILABLE:
                    self.engines[engine_name] = GoogleSpeechEngine()
                elif engine_name == 'sphinx' and SPEECH_RECOGNITION_AVAILABLE:
                    self.engines[engine_name] = SphinxEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize {engine_name}: {e}")
        
        if not self.engines:
            logger.warning("No speech-to-text engines available!")
        else:
            logger.info(f"Initialized speech engines: {list(self.engines.keys())}")
    
    def _get_audio_hash(self, audio_path: Union[str, Path]) -> str:
        """Generate hash for audio caching"""
        try:
            file_path = Path(audio_path)
            if file_path.exists():
                stat = file_path.stat()
                content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
                return hashlib.md5(content.encode()).hexdigest()
            else:
                return hashlib.md5(str(audio_path).encode()).hexdigest()
        except:
            return str(hash(str(audio_path)))
    
    @scraping_retry
    async def transcribe_audio(self, 
                              audio_path: Union[str, Path], 
                              language: Optional[str] = None,
                              use_cache: bool = True,
                              max_duration: int = 3600) -> Dict[str, Any]:
        """Transcribe audio using multiple engines with fallback"""
        
        # Check audio file exists
        if not Path(audio_path).exists():
            return {
                'text': '',
                'success': False,
                'error': f'Audio file not found: {audio_path}',
                'engines_tried': []
            }
        
        # Get audio info and check duration
        audio_info = AudioProcessor.get_audio_info(audio_path)
        if audio_info.get('duration', 0) > max_duration:
            return await self._transcribe_long_audio(audio_path, language, max_duration)
        
        # Check cache
        audio_hash = self._get_audio_hash(audio_path) if use_cache else None
        cache_key = f"{audio_hash}_{language}" if audio_hash else None
        
        if cache_key and cache_key in self.cache:
            logger.debug("Using cached transcription result")
            return self.cache[cache_key]
        
        if not self.engines:
            return {
                'text': '',
                'success': False,
                'error': 'No speech engines available',
                'engines_tried': []
            }
        
        best_result = None
        engines_tried = []
        
        # Try engines in order of preference
        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Trying speech engine: {engine_name}")
                result = await engine.transcribe(audio_path, language)
                result['timestamp'] = datetime.now().isoformat()
                result['audio_info'] = audio_info
                engines_tried.append(engine_name)
                
                if result.get('success') and result.get('text', '').strip():
                    # Success - use this result
                    best_result = result
                    
                    # If this is Whisper or a high-confidence result, stop trying
                    if engine_name == 'whisper' or len(result.get('text', '')) > 50:
                        break
                
                self.error_aggregator.add_success()
                
            except Exception as e:
                self.error_aggregator.add_error(e, f"Speech engine {engine_name}")
                logger.warning(f"Speech engine {engine_name} failed: {e}")
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
                'success': False,
                'error': 'All speech engines failed',
                'engines_tried': engines_tried,
                'total_engines': len(self.engines),
                'audio_info': audio_info
            }
        
        # Cache result
        if use_cache and cache_key:
            # Limit cache size
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = final_result
        
        return final_result
    
    async def _transcribe_long_audio(self, 
                                   audio_path: Union[str, Path], 
                                   language: Optional[str],
                                   max_chunk_duration: int = 60) -> Dict[str, Any]:
        """Transcribe long audio by splitting into chunks"""
        try:
            logger.info(f"Splitting long audio file: {audio_path}")
            
            # Split audio into chunks
            chunk_paths = AudioProcessor.split_audio(audio_path, max_chunk_duration)
            
            if len(chunk_paths) == 1:
                # Splitting failed, try with whole file anyway
                return await self.transcribe_audio(audio_path, language, use_cache=False, max_duration=999999)
            
            # Transcribe each chunk
            chunk_results = []
            full_text_parts = []
            
            for i, chunk_path in enumerate(chunk_paths):
                logger.debug(f"Transcribing chunk {i+1}/{len(chunk_paths)}")
                
                result = await self.transcribe_audio(chunk_path, language, use_cache=False, max_duration=999999)
                chunk_results.append(result)
                
                if result.get('success') and result.get('text'):
                    full_text_parts.append(result['text'])
                
                # Clean up chunk file
                Path(chunk_path).unlink(missing_ok=True)
            
            # Combine results
            full_text = ' '.join(full_text_parts)
            successful_chunks = sum(1 for r in chunk_results if r.get('success'))
            
            return {
                'text': full_text,
                'success': bool(full_text.strip()),
                'total_chunks': len(chunk_paths),
                'successful_chunks': successful_chunks,
                'chunk_results': chunk_results[:5],  # Limit stored results
                'engines_tried': list(set().union(*[r.get('engines_tried', []) for r in chunk_results])),
                'processing_method': 'chunked'
            }
            
        except Exception as e:
            logger.error(f"Long audio transcription failed: {e}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'processing_method': 'chunked'
            }
    
    async def transcribe_batch(self, audio_paths: List[Union[str, Path]], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple audio files in batch"""
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent transcriptions
        
        async def process_audio(audio_path):
            async with semaphore:
                return await self.transcribe_audio(audio_path, language)
        
        tasks = [process_audio(path) for path in audio_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'text': '',
                    'success': False,
                    'error': str(result),
                    'audio_index': i
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get speech processing statistics"""
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
async def transcribe_audio_file(audio_path: Union[str, Path], language: Optional[str] = None) -> str:
    """Quick utility for single audio transcription"""
    processor = MultiSpeechProcessor()
    result = await processor.transcribe_audio(audio_path, language)
    return result.get('text', '')


async def transcribe_audio_files(audio_paths: List[Union[str, Path]], language: Optional[str] = None) -> List[str]:
    """Quick utility for batch audio transcription"""
    processor = MultiSpeechProcessor()
    results = await processor.transcribe_batch(audio_paths, language)
    return [result.get('text', '') for result in results]