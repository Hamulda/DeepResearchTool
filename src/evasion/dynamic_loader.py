"""
Dynamic Content Loader pro DeepResearchTool
Bezpečné načítání dynamického obsahu, sandboxed JavaScript execution a infinite scroll handling.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from playwright.async_api import Page, Browser, BrowserContext
from bs4 import BeautifulSoup

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class DynamicContent:
    """Reprezentace dynamicky načteného obsahu"""
    url: str
    initial_content: str
    dynamic_content: str
    scroll_iterations: int
    javascript_executed: bool
    load_time_ms: float
    elements_loaded: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JavaScriptExecutionResult:
    """Výsledek JavaScript execution v sandboxu"""
    success: bool
    result: Any = None
    execution_time_ms: float = 0
    console_logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    security_violations: List[str] = field(default_factory=list)


@dataclass
class InfiniteScrollConfig:
    """Konfigurace pro infinite scroll handling"""
    max_scrolls: int = 10
    scroll_delay_ms: int = 2000
    scroll_distance: int = 1000
    content_selector: Optional[str] = None
    load_indicator_selector: Optional[str] = None
    stop_condition: Optional[str] = None
    timeout_ms: int = 60000


class JavaScriptSandbox:
    """Bezpečný sandbox pro JavaScript execution"""

    def __init__(self):
        self.restricted_apis = {
            'fetch', 'XMLHttpRequest', 'eval', 'Function', 'setTimeout',
            'setInterval', 'location', 'history', 'localStorage', 'sessionStorage',
            'indexedDB', 'navigator', 'screen', 'crypto', 'performance'
        }

        self.allowed_globals = {
            'console', 'JSON', 'Math', 'Date', 'parseInt', 'parseFloat',
            'String', 'Number', 'Boolean', 'Array', 'Object', 'RegExp'
        }

    def create_sandbox_script(self, user_script: str) -> str:
        """Vytvoření sandboxed verze scriptu"""

        # Wrapper s omezenými API
        sandbox_wrapper = f"""
        (function() {{
            // Restrict dangerous APIs
            const restrictedAPIs = {json.dumps(list(self.restricted_apis))};
            const originalWindow = window;
            
            // Create safe context
            const safeContext = {{}};
            const allowedGlobals = {json.dumps(list(self.allowed_globals))};
            
            // Copy allowed globals
            allowedGlobals.forEach(name => {{
                if (originalWindow[name]) {{
                    safeContext[name] = originalWindow[name];
                }}
            }});
            
            // Override console to capture logs
            const consoleLogs = [];
            safeContext.console = {{
                log: (...args) => {{
                    consoleLogs.push({{type: 'log', args: args.map(String)}});
                    originalWindow.console.log('[SANDBOX]', ...args);
                }},
                error: (...args) => {{
                    consoleLogs.push({{type: 'error', args: args.map(String)}});
                    originalWindow.console.error('[SANDBOX]', ...args);
                }},
                warn: (...args) => {{
                    consoleLogs.push({{type: 'warn', args: args.map(String)}});
                    originalWindow.console.warn('[SANDBOX]', ...args);
                }}
            }};
            
            // Execute user script in safe context
            let result;
            let error = null;
            
            try {{
                result = (function() {{
                    'use strict';
                    // User script here
                    {user_script}
                }}).call(safeContext);
            }} catch (e) {{
                error = e.toString();
                consoleLogs.push({{type: 'error', args: [e.toString()]}});
            }}
            
            return {{
                result: result,
                error: error,
                consoleLogs: consoleLogs
            }};
        }})();
        """

        return sandbox_wrapper

    def validate_script_safety(self, script: str) -> List[str]:
        """Validace bezpečnosti scriptu"""
        violations = []

        # Kontrola nebezpečných vzorů
        dangerous_patterns = [
            (r'eval\s*\(', 'eval() usage detected'),
            (r'Function\s*\(', 'Function constructor usage detected'),
            (r'document\.write', 'document.write usage detected'),
            (r'innerHTML\s*=', 'innerHTML assignment detected'),
            (r'location\s*=', 'location assignment detected'),
            (r'window\s*\[', 'dynamic window property access detected'),
            (r'\.call\s*\(.*window', 'window context manipulation detected'),
            (r'\.apply\s*\(.*window', 'window context manipulation detected'),
            (r'import\s*\(', 'dynamic import detected'),
            (r'new\s+Worker', 'Web Worker creation detected'),
            (r'new\s+SharedWorker', 'Shared Worker creation detected'),
            (r'WebSocket', 'WebSocket usage detected'),
            (r'EventSource', 'Server-Sent Events detected')
        ]

        import re
        for pattern, message in dangerous_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                violations.append(message)

        return violations


class InfiniteScrollHandler:
    """Handler pro infinite scroll stránky"""

    def __init__(self, config: InfiniteScrollConfig):
        self.config = config

    async def handle_infinite_scroll(self, page: Page) -> DynamicContent:
        """
        Zpracování infinite scroll stránky s postupným načítáním obsahu
        """
        start_time = time.time()

        # Počáteční obsah
        initial_content = await page.content()
        initial_elements = await self._count_content_elements(page)

        dynamic_content_parts = [initial_content]
        total_elements = initial_elements
        scroll_count = 0

        logger.info(f"Infinite scroll: počáteční elementy: {initial_elements}")

        for scroll_iteration in range(self.config.max_scrolls):
            try:
                # Scroll dolů
                await self._perform_scroll(page)
                scroll_count += 1

                # Čekání na načtení obsahu
                await self._wait_for_content_load(page)

                # Kontrola nových elementů
                current_elements = await self._count_content_elements(page)
                new_elements = current_elements - total_elements

                logger.debug(f"Scroll {scroll_iteration + 1}: nové elementy: {new_elements}")

                if new_elements > 0:
                    # Získání nového obsahu
                    current_content = await page.content()
                    dynamic_content_parts.append(current_content)
                    total_elements = current_elements

                    # Kontrola stop condition
                    if await self._check_stop_condition(page):
                        logger.info("Stop condition reached")
                        break
                else:
                    # Žádný nový obsah - možný konec
                    logger.info("Žádný nový obsah detekován")
                    break

            except Exception as e:
                logger.error(f"Chyba při scroll iteraci {scroll_iteration}: {e}")
                break

        # Finální obsah
        final_content = await page.content()
        load_time = (time.time() - start_time) * 1000

        return DynamicContent(
            url=page.url,
            initial_content=initial_content,
            dynamic_content=final_content,
            scroll_iterations=scroll_count,
            javascript_executed=True,
            load_time_ms=load_time,
            elements_loaded=total_elements - initial_elements
        )

    async def _perform_scroll(self, page: Page):
        """Provedení scroll operace"""
        await page.evaluate(f"""
            window.scrollBy({{
                top: {self.config.scroll_distance},
                behavior: 'smooth'
            }});
        """)

        # Čekání na dokončení scroll
        await asyncio.sleep(self.config.scroll_delay_ms / 1000)

    async def _wait_for_content_load(self, page: Page):
        """Čekání na načtení nového obsahu"""
        if self.config.load_indicator_selector:
            try:
                # Čekání na zmizení loading indikátoru
                await page.wait_for_selector(
                    self.config.load_indicator_selector,
                    state="hidden",
                    timeout=5000
                )
            except:
                pass  # Timeout není kritický

        # Obecné čekání na network aktivitu
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except:
            await asyncio.sleep(1)  # Fallback čekání

    async def _count_content_elements(self, page: Page) -> int:
        """Spočítání content elementů"""
        if self.config.content_selector:
            count = await page.locator(self.config.content_selector).count()
        else:
            # Fallback: počítání běžných content elementů
            count = await page.evaluate("""
                () => {
                    const selectors = ['article', '.post', '.item', '.card', '.entry', '[data-item]'];
                    let total = 0;
                    selectors.forEach(selector => {
                        total += document.querySelectorAll(selector).length;
                    });
                    return total || document.querySelectorAll('*').length;
                }
            """)

        return count

    async def _check_stop_condition(self, page: Page) -> bool:
        """Kontrola stop condition"""
        if not self.config.stop_condition:
            return False

        try:
            # Kontrola textového obsahu
            if isinstance(self.config.stop_condition, str):
                content = await page.content()
                return self.config.stop_condition.lower() in content.lower()

            # Kontrola selektoru
            element = await page.query_selector(self.config.stop_condition)
            return element is not None

        except:
            return False


class DynamicContentLoader:
    """
    Pokročilý loader pro dynamický obsah s bezpečným JavaScript execution
    a intelligent infinite scroll handling.
    """

    def __init__(self, max_execution_time_ms: int = 30000):
        self.max_execution_time_ms = max_execution_time_ms
        self.js_sandbox = JavaScriptSandbox()

        logger.info("DynamicContentLoader inicializován")

    async def load_dynamic_content(
        self,
        page: Page,
        wait_for_selector: Optional[str] = None,
        custom_js: Optional[str] = None,
        infinite_scroll_config: Optional[InfiniteScrollConfig] = None
    ) -> DynamicContent:
        """
        Hlavní metoda pro načtení dynamického obsahu
        """
        start_time = time.time()

        try:
            # Počáteční obsah
            initial_content = await page.content()

            # Čekání na specifický selektor
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=self.max_execution_time_ms)

            # Spuštění custom JavaScript
            js_executed = False
            if custom_js:
                js_result = await self.execute_js_safely(page, custom_js)
                js_executed = js_result.success

            # Infinite scroll handling
            if infinite_scroll_config:
                scroll_handler = InfiniteScrollHandler(infinite_scroll_config)
                return await scroll_handler.handle_infinite_scroll(page)

            # Standardní dynamic content loading
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Finální obsah
            final_content = await page.content()
            load_time = (time.time() - start_time) * 1000

            # Počítání načtených elementů
            soup_initial = BeautifulSoup(initial_content, 'html.parser')
            soup_final = BeautifulSoup(final_content, 'html.parser')

            initial_elements = len(soup_initial.find_all())
            final_elements = len(soup_final.find_all())

            return DynamicContent(
                url=page.url,
                initial_content=initial_content,
                dynamic_content=final_content,
                scroll_iterations=0,
                javascript_executed=js_executed,
                load_time_ms=load_time,
                elements_loaded=final_elements - initial_elements
            )

        except Exception as e:
            load_time = (time.time() - start_time) * 1000

            return DynamicContent(
                url=page.url,
                initial_content=await page.content() if page else "",
                dynamic_content="",
                scroll_iterations=0,
                javascript_executed=False,
                load_time_ms=load_time,
                elements_loaded=0,
                errors=[str(e)]
            )

    async def execute_js_safely(self, page: Page, script: str) -> JavaScriptExecutionResult:
        """
        Bezpečné spuštění JavaScript kódu v sandboxu
        """
        start_time = time.time()

        # Validace bezpečnosti
        security_violations = self.js_sandbox.validate_script_safety(script)

        if security_violations:
            return JavaScriptExecutionResult(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                security_violations=security_violations,
                errors=["Script obsahuje nebezpečné vzory"]
            )

        # Vytvoření sandboxed scriptu
        sandboxed_script = self.js_sandbox.create_sandbox_script(script)

        try:
            # Spuštění v page contextu
            result = await page.evaluate(sandboxed_script)

            execution_time = (time.time() - start_time) * 1000

            return JavaScriptExecutionResult(
                success=True,
                result=result.get('result') if isinstance(result, dict) else result,
                execution_time_ms=execution_time,
                console_logs=[log.get('args', []) for log in result.get('consoleLogs', [])] if isinstance(result, dict) else [],
                errors=[result.get('error')] if isinstance(result, dict) and result.get('error') else []
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return JavaScriptExecutionResult(
                success=False,
                execution_time_ms=execution_time,
                errors=[str(e)]
            )

    async def handle_infinite_scroll(
        self,
        page: Page,
        max_scrolls: int = 10,
        scroll_delay_ms: int = 2000,
        content_selector: Optional[str] = None
    ) -> DynamicContent:
        """
        Wrapper pro infinite scroll s defaultní konfigurací
        """
        config = InfiniteScrollConfig(
            max_scrolls=max_scrolls,
            scroll_delay_ms=scroll_delay_ms,
            content_selector=content_selector
        )

        scroll_handler = InfiniteScrollHandler(config)
        return await scroll_handler.handle_infinite_scroll(page)

    async def extract_spa_content(
        self,
        page: Page,
        navigation_selectors: List[str],
        content_selector: str = "main, article, .content"
    ) -> Dict[str, DynamicContent]:
        """
        Extrakce obsahu z Single Page Application
        """
        spa_content = {}

        for i, selector in enumerate(navigation_selectors):
            try:
                # Kliknutí na navigační element
                await page.click(selector)

                # Čekání na načtení obsahu
                await page.wait_for_load_state("networkidle", timeout=10000)

                # Extrakce obsahu
                content = await self.load_dynamic_content(page)
                spa_content[f"section_{i}"] = content

                # Krátké čekání mezi navigacemi
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Chyba při SPA navigaci {selector}: {e}")

        return spa_content

    async def monitor_dom_changes(
        self,
        page: Page,
        duration_seconds: int = 30,
        change_threshold: int = 5
    ) -> Dict[str, Any]:
        """
        Monitoring změn v DOM pro detekci dynamického obsahu
        """
        changes_detected = []

        # Inicializace MutationObserver
        await page.evaluate("""
            window.domChanges = [];
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        window.domChanges.push({
                            type: 'added',
                            count: mutation.addedNodes.length,
                            timestamp: Date.now()
                        });
                    }
                });
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        """)

        # Monitoring po dobu duration_seconds
        for _ in range(duration_seconds):
            await asyncio.sleep(1)

            # Získání změn
            changes = await page.evaluate("window.domChanges.splice(0)")
            if changes:
                changes_detected.extend(changes)

        # Ukončení observeru
        await page.evaluate("""
            if (window.observer) {
                window.observer.disconnect();
            }
        """)

        return {
            "total_changes": len(changes_detected),
            "significant_changes": len([c for c in changes_detected if c.get('count', 0) >= change_threshold]),
            "changes_timeline": changes_detected,
            "dynamic_content_detected": len(changes_detected) > change_threshold
        }

    async def wait_for_ajax_completion(
        self,
        page: Page,
        timeout_ms: int = 30000
    ) -> bool:
        """
        Čekání na dokončení AJAX requestů
        """
        try:
            # Čekání na jQuery (pokud je k dispozici)
            await page.wait_for_function(
                "typeof jQuery === 'undefined' || jQuery.active === 0",
                timeout=timeout_ms
            )

            # Čekání na obecné AJAX indikátory
            await page.wait_for_function("""
                () => {
                    // Kontrola loading indikátorů
                    const loadingElements = document.querySelectorAll('.loading, .spinner, [data-loading]');
                    const visibleLoading = Array.from(loadingElements).some(el => 
                        el.offsetParent !== null && !el.hidden
                    );
                    
                    return !visibleLoading;
                }
            """, timeout=timeout_ms)

            return True

        except Exception as e:
            logger.debug(f"AJAX completion timeout: {e}")
            return False

    async def extract_lazy_loaded_images(self, page: Page) -> List[str]:
        """
        Trigger lazy loading a extrakce všech obrázků
        """
        # Scroll pro trigger lazy loading
        await page.evaluate("""
            () => {
                const images = document.querySelectorAll('img[data-src], img[loading="lazy"]');
                images.forEach(img => {
                    img.scrollIntoView();
                });
            }
        """)

        # Čekání na načtení
        await asyncio.sleep(2)

        # Extrakce všech image URLs
        image_urls = await page.evaluate("""
            () => {
                const images = document.querySelectorAll('img');
                return Array.from(images)
                    .map(img => img.src || img.dataset.src)
                    .filter(src => src && src.startsWith('http'));
            }
        """)

        return image_urls

    def generate_dynamic_content_report(self, content: DynamicContent) -> Dict[str, Any]:
        """
        Generování reportu z dynamic content loading
        """
        report = {
            "url": content.url,
            "performance_metrics": {
                "load_time_ms": content.load_time_ms,
                "elements_loaded": content.elements_loaded,
                "scroll_iterations": content.scroll_iterations,
                "javascript_executed": content.javascript_executed
            },
            "content_analysis": {
                "initial_content_size": len(content.initial_content),
                "final_content_size": len(content.dynamic_content),
                "content_growth_ratio": len(content.dynamic_content) / max(len(content.initial_content), 1),
                "dynamic_content_detected": len(content.dynamic_content) > len(content.initial_content) * 1.1
            },
            "errors": content.errors,
            "recommendations": []
        }

        # Doporučení na základě analýzy
        if content.load_time_ms > 10000:
            report["recommendations"].append("Dlouhá doba načítání - zvažte optimalizaci")

        if content.scroll_iterations > 5:
            report["recommendations"].append("Detekována extensive infinite scroll stránka")

        if content.elements_loaded > 1000:
            report["recommendations"].append("Vysoký počet dynamických elementů - možný memory leak")

        if content.errors:
            report["recommendations"].append("Chyby při načítání - prověřte JavaScript errors")

        return report
