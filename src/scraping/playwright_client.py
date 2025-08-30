"""Playwright Client with API Interception
Discovery mode captures XHR/fetch, Extraction mode replays via httpx over Tor
Minimizes headless sessions for M1 resource efficiency
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import Browser, Page, Request, Response, Route, async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - advanced scraping disabled")


@dataclass
class ApiEndpoint:
    """Captured API endpoint information"""

    url: str
    method: str
    headers: dict[str, str]
    payload: str | None
    response_headers: dict[str, str]
    response_body: str
    status_code: int
    timestamp: datetime
    content_type: str
    size_bytes: int


@dataclass
class DiscoveryResult:
    """Result from discovery mode"""

    page_url: str
    page_title: str
    api_endpoints: list[ApiEndpoint]
    static_content: dict[str, Any]
    discovered_at: datetime
    session_id: str


class DiscoveryMode:
    """Discovery mode to capture API endpoints and interactions"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.captured_endpoints: list[ApiEndpoint] = []
        self.static_content: dict[str, Any] = {}

    async def capture_api_calls(self, page: Page) -> None:
        """Setup route handlers to capture API calls"""

        async def handle_route(route: Route, request: Request):
            """Handle and capture API requests"""
            try:
                # Continue with the request
                response = await route.fetch()

                # Check if this is an API call (JSON, XHR, fetch)
                if self._is_api_request(request):
                    # Capture request details
                    payload = None
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            payload = request.post_data
                        except:
                            payload = None

                    # Get response body
                    response_body = ""
                    try:
                        response_body = await response.text()
                    except:
                        response_body = ""

                    # Create endpoint record
                    endpoint = ApiEndpoint(
                        url=request.url,
                        method=request.method,
                        headers=dict(request.headers),
                        payload=payload,
                        response_headers=dict(response.headers),
                        response_body=response_body,
                        status_code=response.status,
                        timestamp=datetime.now(),
                        content_type=response.headers.get("content-type", ""),
                        size_bytes=len(response_body.encode("utf-8")),
                    )

                    self.captured_endpoints.append(endpoint)
                    logger.info(f"Captured API call: {request.method} {request.url}")

                # Continue with response
                await route.fulfill(response=response)

            except Exception as e:
                logger.warning(f"Error capturing route {request.url}: {e}")
                await route.continue_()

        # Setup route interception for API calls
        await page.route("**/*", handle_route)

    def _is_api_request(self, request: Request) -> bool:
        """Determine if request is an API call"""
        url = request.url.lower()

        # Check for API patterns
        api_patterns = [
            "/api/",
            "/rest/",
            "/graphql",
            "/v1/",
            "/v2/",
            "/v3/",
            ".json",
            "/search",
            "/query",
            "/data",
        ]

        if any(pattern in url for pattern in api_patterns):
            return True

        # Check content type
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            return True

        # Check for XHR/fetch
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return True

        return False

    async def extract_static_content(self, page: Page) -> None:
        """Extract static page content"""
        try:
            self.static_content = {
                "title": await page.title(),
                "url": page.url,
                "html": await page.content(),
                "text": await page.evaluate("document.body.innerText"),
                "links": await page.evaluate(
                    """
                    Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        text: a.innerText.trim(),
                        href: a.href,
                        title: a.title
                    }))
                """
                ),
                "forms": await page.evaluate(
                    """
                    Array.from(document.querySelectorAll('form')).map(form => ({
                        action: form.action,
                        method: form.method,
                        inputs: Array.from(form.querySelectorAll('input')).map(input => ({
                            name: input.name,
                            type: input.type,
                            placeholder: input.placeholder
                        }))
                    }))
                """
                ),
            }
        except Exception as e:
            logger.warning(f"Error extracting static content: {e}")


class ExtractionMode:
    """Extraction mode replays captured APIs via httpx over Tor"""

    def __init__(self, network_client=None, rate_limit: float = 1.0):
        self.network_client = network_client
        self.rate_limit = rate_limit
        self.last_request_time = 0

    async def replay_api_calls(
        self,
        endpoints: list[ApiEndpoint],
        modify_params: Callable[[ApiEndpoint], ApiEndpoint] | None = None,
    ) -> list[dict[str, Any]]:
        """Replay captured API calls through network client"""
        results = []

        for endpoint in endpoints:
            try:
                # Rate limiting
                await self._enforce_rate_limit()

                # Modify endpoint if needed (pagination, parameters, etc.)
                if modify_params:
                    endpoint = modify_params(endpoint)

                # Replay the API call
                result = await self._replay_single_endpoint(endpoint)
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to replay {endpoint.url}: {e}")
                results.append({"url": endpoint.url, "error": str(e), "success": False})

        return results

    async def _replay_single_endpoint(self, endpoint: ApiEndpoint) -> dict[str, Any]:
        """Replay single API endpoint"""
        # Prepare request
        headers = dict(endpoint.headers)

        # Clean headers that shouldn't be replayed
        headers_to_remove = [
            "host",
            "content-length",
            "connection",
            "accept-encoding",
            "cookie",
            "referer",  # Remove for anonymity
        ]
        for header in headers_to_remove:
            headers.pop(header, None)

        # Add anonymity headers
        headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
        )

        # Make request through network client
        if self.network_client:
            # Use network manager (Tor/I2P)
            if endpoint.method.upper() == "GET":
                response = await self.network_client.get(endpoint.url, headers=headers)
            elif endpoint.method.upper() == "POST":
                response = await self.network_client.post(
                    endpoint.url, headers=headers, data=endpoint.payload
                )
            else:
                response = await self.network_client.request(
                    endpoint.method, endpoint.url, headers=headers, data=endpoint.payload
                )

            response_data = await response.text()

        else:
            # Direct httpx client
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    endpoint.method, endpoint.url, headers=headers, content=endpoint.payload
                )
                response_data = response.text

        return {
            "url": endpoint.url,
            "method": endpoint.method,
            "status_code": response.status_code,
            "data": response_data,
            "success": True,
            "replayed_at": datetime.now().isoformat(),
        }

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        import time

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()


class PlaywrightClient:
    """Main Playwright client with discovery and extraction modes"""

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright not available")

        self.headless = headless
        self.browser_type = browser_type
        self.browser: Browser | None = None
        self.playwright = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.playwright = await async_playwright().start()

        # Launch browser based on type
        if self.browser_type == "chromium":
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self.browser = await self.playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self.browser = await self.playwright.webkit.launch(headless=self.headless)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def discover_page_apis(
        self, url: str, wait_time: int = 5, scroll_pages: int = 3, click_elements: bool = True
    ) -> DiscoveryResult:
        """Discover API endpoints by interacting with page"""
        discovery = DiscoveryMode()

        # Create new page
        page = await self.browser.new_page()

        try:
            # Setup API capture
            await discovery.capture_api_calls(page)

            # Navigate to page
            await page.goto(url, wait_until="networkidle")

            # Wait for initial load
            await asyncio.sleep(wait_time)

            # Interact with page to trigger API calls
            if scroll_pages > 0:
                await self._scroll_page(page, scroll_pages)

            if click_elements:
                await self._click_interactive_elements(page)

            # Extract static content
            await discovery.extract_static_content(page)

            # Create result
            result = DiscoveryResult(
                page_url=url,
                page_title=discovery.static_content.get("title", ""),
                api_endpoints=discovery.captured_endpoints,
                static_content=discovery.static_content,
                discovered_at=datetime.now(),
                session_id=discovery.session_id,
            )

            logger.info(
                f"Discovery completed: {len(discovery.captured_endpoints)} API endpoints found"
            )
            return result

        finally:
            await page.close()

    async def _scroll_page(self, page: Page, scroll_pages: int):
        """Scroll page to trigger lazy loading"""
        for i in range(scroll_pages):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            # Scroll back up occasionally
            if i % 2 == 1:
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(1)

    async def _click_interactive_elements(self, page: Page):
        """Click buttons and interactive elements to trigger API calls"""
        try:
            # Find clickable elements
            buttons = await page.query_selector_all('button, [role="button"], .btn, .load-more')

            for button in buttons[:5]:  # Limit to first 5 buttons
                try:
                    # Check if button is visible and enabled
                    is_visible = await button.is_visible()
                    is_enabled = await button.is_enabled()

                    if is_visible and is_enabled:
                        await button.click()
                        await asyncio.sleep(2)  # Wait for potential API calls

                except Exception as e:
                    logger.debug(f"Failed to click element: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error clicking interactive elements: {e}")

    async def extract_with_replay(
        self,
        discovery_result: DiscoveryResult,
        network_client=None,
        modify_params: Callable | None = None,
    ) -> dict[str, Any]:
        """Extract data using API replay without browser"""
        extraction = ExtractionMode(network_client)

        # Replay captured API calls
        api_results = await extraction.replay_api_calls(
            discovery_result.api_endpoints, modify_params
        )

        return {
            "session_id": discovery_result.session_id,
            "original_url": discovery_result.page_url,
            "api_results": api_results,
            "static_content": discovery_result.static_content,
            "extracted_at": datetime.now().isoformat(),
            "total_apis_replayed": len(api_results),
            "successful_replays": sum(1 for r in api_results if r.get("success", False)),
        }

    def save_discovery_session(self, result: DiscoveryResult, output_path: Path) -> Path:
        """Save discovery session for later replay"""
        output_path.mkdir(parents=True, exist_ok=True)

        session_file = output_path / f"discovery_{result.session_id}.json"

        # Convert to serializable format
        data = {
            "page_url": result.page_url,
            "page_title": result.page_title,
            "discovered_at": result.discovered_at.isoformat(),
            "session_id": result.session_id,
            "static_content": result.static_content,
            "api_endpoints": [
                {**asdict(endpoint), "timestamp": endpoint.timestamp.isoformat()}
                for endpoint in result.api_endpoints
            ],
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Discovery session saved: {session_file}")
        return session_file

    @classmethod
    def load_discovery_session(cls, session_file: Path) -> DiscoveryResult:
        """Load saved discovery session"""
        with open(session_file) as f:
            data = json.load(f)

        # Reconstruct endpoints
        endpoints = []
        for ep_data in data["api_endpoints"]:
            ep_data["timestamp"] = datetime.fromisoformat(ep_data["timestamp"])
            endpoints.append(ApiEndpoint(**ep_data))

        return DiscoveryResult(
            page_url=data["page_url"],
            page_title=data["page_title"],
            api_endpoints=endpoints,
            static_content=data["static_content"],
            discovered_at=datetime.fromisoformat(data["discovered_at"]),
            session_id=data["session_id"],
        )


# Utility functions
async def discover_and_extract(
    url: str, network_client=None, headless: bool = True, output_path: Path | None = None
) -> dict[str, Any]:
    """Convenience function for full discover and extract workflow"""
    async with PlaywrightClient(headless=headless) as client:
        # Discovery phase
        logger.info(f"Starting discovery for {url}")
        discovery_result = await client.discover_page_apis(url)

        # Save session if output path provided
        if output_path:
            client.save_discovery_session(discovery_result, output_path)

        # Extraction phase
        logger.info("Starting extraction phase")
        extraction_result = await client.extract_with_replay(discovery_result, network_client)

        return {
            "discovery": discovery_result,
            "extraction": extraction_result,
            "summary": {
                "url": url,
                "apis_discovered": len(discovery_result.api_endpoints),
                "apis_replayed": extraction_result["total_apis_replayed"],
                "success_rate": extraction_result["successful_replays"]
                / max(1, extraction_result["total_apis_replayed"]),
            },
        }


__all__ = [
    "ApiEndpoint",
    "DiscoveryMode",
    "DiscoveryResult",
    "ExtractionMode",
    "PlaywrightClient",
    "discover_and_extract",
]
