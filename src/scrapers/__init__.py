"""
Scrapers module initialization
"""
from .wayback_scraper import WaybackMachineScraper
from .arxiv_scraper import ArxivScraper
from .rss_monitor import RSSMonitor
from .pdf_scraper import PDFScraper

__all__ = [
    'WaybackMachineScraper',
    'ArxivScraper',
    'RSSMonitor',
    'PDFScraper'
]
