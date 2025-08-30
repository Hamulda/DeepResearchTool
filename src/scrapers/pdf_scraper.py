#!/usr/bin/env python3
"""Enhanced PDF Scraper with async processing and job queue
Supports parallel PDF processing with memory management
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import logging
import mimetypes
from pathlib import Path
import time
from typing import Any

import PyPDF2

logger = logging.getLogger(__name__)


@dataclass
class PDFProcessingJob:
    """Represents a PDF processing job"""

    file_path: Path
    priority: int = 0
    metadata: dict[str, Any] = None


@dataclass
class PDFProcessingResult:
    """Result of PDF processing"""

    file_path: Path
    success: bool
    content: str = ""
    metadata: dict[str, Any] = None
    error: str | None = None
    processing_time: float = 0.0


class AsyncPDFProcessor:
    """Asynchronous PDF processor with job queue and memory management"""

    def __init__(self, max_concurrent: int = 5, max_file_size_mb: int = 50):
        self.max_concurrent = max_concurrent
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.processing_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)

    async def add_job(self, job: PDFProcessingJob) -> None:
        """Add job to processing queue"""
        await self.processing_queue.put(job)

    async def process_pdf_file(self, file_path: Path) -> PDFProcessingResult:
        """Process single PDF file with error handling and memory management"""
        start_time = time.time()

        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return PDFProcessingResult(
                    file_path=file_path,
                    success=False,
                    error=f"File too large: {file_path.stat().st_size / 1024 / 1024:.1f}MB",
                )

            # Verify it's actually a PDF
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type != "application/pdf":
                return PDFProcessingResult(
                    file_path=file_path, success=False, error=f"Not a PDF file: {mime_type}"
                )

            # Process PDF in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            content, metadata = await loop.run_in_executor(
                self.executor, self._extract_pdf_content, file_path
            )

            processing_time = time.time() - start_time

            return PDFProcessingResult(
                file_path=file_path,
                success=True,
                content=content,
                metadata=metadata,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return PDFProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _extract_pdf_content(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Extract content from PDF file (runs in thread pool)"""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract metadata
                metadata = {
                    "title": pdf_reader.metadata.title if pdf_reader.metadata else None,
                    "author": pdf_reader.metadata.author if pdf_reader.metadata else None,
                    "subject": pdf_reader.metadata.subject if pdf_reader.metadata else None,
                    "creator": pdf_reader.metadata.creator if pdf_reader.metadata else None,
                    "creation_date": (
                        str(pdf_reader.metadata.creation_date) if pdf_reader.metadata else None
                    ),
                    "modification_date": (
                        str(pdf_reader.metadata.modification_date) if pdf_reader.metadata else None
                    ),
                    "num_pages": len(pdf_reader.pages),
                    "file_size": file_path.stat().st_size,
                    "file_path": str(file_path),
                }

                # Extract text content
                content_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            content_parts.append(f"[Page {page_num + 1}]\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue

                content = "\n\n".join(content_parts)
                return content, metadata

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise

    async def process_queue_worker(self) -> None:
        """Worker that processes jobs from queue"""
        while True:
            try:
                # Get job from queue
                job = await self.processing_queue.get()
                if job is None:  # Shutdown signal
                    break

                # Acquire semaphore to limit concurrent processing
                async with self.processing_semaphore:
                    result = await self.process_pdf_file(job.file_path)
                    await self.results_queue.put(result)

                # Mark job as done
                self.processing_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue worker: {e}")

    async def process_directory_async(
        self, directory: Path, pattern: str = "*.pdf"
    ) -> list[PDFProcessingResult]:
        """Process all PDFs in directory asynchronously"""
        # Find all PDF files
        pdf_files = list(directory.glob(pattern))
        if not pdf_files:
            logger.info(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Start workers
        workers = [
            asyncio.create_task(self.process_queue_worker()) for _ in range(self.max_concurrent)
        ]

        # Add jobs to queue
        for pdf_file in pdf_files:
            job = PDFProcessingJob(file_path=pdf_file)
            await self.add_job(job)

        # Wait for all jobs to complete
        await self.processing_queue.join()

        # Stop workers
        for _ in workers:
            await self.processing_queue.put(None)
        await asyncio.gather(*workers)

        # Collect results
        results = []
        while not self.results_queue.empty():
            result = await self.results_queue.get()
            results.append(result)

        # Log statistics
        successful = len([r for r in results if r.success])
        total_time = sum(r.processing_time for r in results)
        logger.info(f"Processed {successful}/{len(results)} PDFs in {total_time:.2f}s")

        return results

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


class EnhancedPDFScraper:
    """Enhanced PDF scraper with async capabilities"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.processor = AsyncPDFProcessor()

    async def search_async(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Async search for PDF documents"""
        await self._rate_limit()

        # Define search directories
        search_dirs = [
            Path("./data/pdf_archives"),
            Path("./research_cache/pdfs"),
            Path("/tmp/pdf_downloads"),  # Temporary download location
        ]

        all_results = []

        for search_dir in search_dirs:
            if search_dir.exists():
                logger.info(f"Processing PDFs in {search_dir}")
                processing_results = await self.processor.process_directory_async(search_dir)

                # Convert to search results format
                for result in processing_results:
                    if result.success and result.content:
                        # Check if content matches topic
                        if self._content_matches_topic(result.content, topic):
                            search_result = {
                                "title": result.metadata.get("title", result.file_path.name),
                                "content": result.content[:2000],  # Limit content size
                                "url": f"file://{result.file_path}",
                                "source_type": "pdf_document",
                                "source_url": str(result.file_path),
                                "date": self._extract_date_from_metadata(result.metadata),
                                "metadata": result.metadata,
                                "content_type": "pdf",
                                "processing_time": result.processing_time,
                            }

                            # Apply time range filter if specified
                            if time_range and search_result["date"]:
                                doc_date = search_result["date"]
                                if isinstance(doc_date, str):
                                    try:
                                        doc_date = datetime.fromisoformat(doc_date)
                                    except:
                                        doc_date = None

                                if doc_date and (
                                    doc_date < time_range[0] or doc_date > time_range[1]
                                ):
                                    continue

                            all_results.append(search_result)

        logger.info(f"Found {len(all_results)} relevant PDF documents for topic: {topic}")
        return all_results

    def _content_matches_topic(self, content: str, topic: str) -> bool:
        """Check if PDF content matches the search topic"""
        topic_lower = topic.lower()
        content_lower = content.lower()

        # Simple keyword matching - could be enhanced with NLP
        topic_words = topic_lower.split()
        matches = sum(1 for word in topic_words if word in content_lower)

        # Require at least 50% of topic words to match
        return matches >= len(topic_words) * 0.5

    def _extract_date_from_metadata(self, metadata: dict[str, Any]) -> datetime | None:
        """Extract date from PDF metadata"""
        date_fields = ["creation_date", "modification_date"]

        for field in date_fields:
            date_str = metadata.get(field)
            if date_str:
                try:
                    # Handle various date formats
                    if isinstance(date_str, str):
                        # Remove timezone info for simplicity
                        date_str = (
                            date_str.split("+")[0].split("-")[0]
                            if "+" in date_str or "-" in date_str
                            else date_str
                        )
                        return datetime.fromisoformat(date_str)
                except:
                    continue

        return None

    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    def search(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Synchronous search wrapper"""
        return asyncio.run(self.search_async(topic, time_range))

    async def cleanup(self):
        """Cleanup resources"""
        await self.processor.cleanup()


# Backward compatibility
PDFScraper = EnhancedPDFScraper
