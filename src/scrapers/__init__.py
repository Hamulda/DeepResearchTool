"""Scrapers module initialization
"""

from .arxiv_scraper import ArxivScraper
from .pdf_scraper import PDFScraper
from .rss_monitor import RSSMonitor
from .wayback_scraper import WaybackMachineScraper
from .web_scraper import WebScraper

__all__ = ["ArxivScraper", "PDFScraper", "RSSMonitor", "WaybackMachineScraper", "WebScraper"]
