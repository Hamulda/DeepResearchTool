"""
FÁZE 7: Rate Limiting Engine
Per-domain rate limiting s exponential backoff a inteligentním throttling
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Konfigurace rate limitingu pro doménu"""
    domain: str
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    burst_allowance: int = 5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 300.0
    reset_threshold_hours: float = 24.0


@dataclass
class DomainState:
    """Stav rate limitingu pro doménu"""
    request_times: deque = field(default_factory=deque)
    hourly_request_times: deque = field(default_factory=deque)
    consecutive_failures: int = 0
    backoff_until: float = 0.0
    last_request_time: float = 0.0
    total_requests: int = 0
    blocked_requests: int = 0


@dataclass
class RateLimitResult:
    """Výsledek rate limit kontroly"""
    allowed: bool
    wait_time: float = 0.0
    reason: str = ""
    requests_remaining: int = 0
    reset_time: float = 0.0


class RateLimitEngine:
    """
    Advanced rate limiting engine s per-domain tracking

    Features:
    - Per-domain rate limiting (minutové a hodinové limity)
    - Exponential backoff při selhání
    - Burst allowance pro kratké špičky
    - Automatické reset po období nečinnosti
    - Thread-safe async operace
    """

    def __init__(
        self,
        default_requests_per_minute: int = 30,
        default_requests_per_hour: int = 1000,
        default_burst_allowance: int = 5,
        cleanup_interval_minutes: int = 60
    ):
        self.default_requests_per_minute = default_requests_per_minute
        self.default_requests_per_hour = default_requests_per_hour
        self.default_burst_allowance = default_burst_allowance
        self.cleanup_interval_minutes = cleanup_interval_minutes

        # Per-domain konfigurace
        self.domain_configs: Dict[str, RateLimitConfig] = {}

        # Per-domain stav
        self.domain_states: Dict[str, DomainState] = defaultdict(DomainState)

        # Global statistiky
        self.global_stats = {
            "total_requests": 0,
            "total_blocked": 0,
            "total_domains": 0,
            "backoff_events": 0
        }

        # Lock pro thread safety
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(f"RateLimitEngine initialized: {default_requests_per_minute}/min, {default_requests_per_hour}/hour")

    def add_domain_config(self, config: RateLimitConfig) -> None:
        """Přidání specifické konfigurace pro doménu"""
        self.domain_configs[config.domain] = config
        logger.info(f"Added rate limit config for {config.domain}: {config.requests_per_minute}/min")

    def _extract_domain(self, url: str) -> str:
        """Extrakce domény z URL"""
        return urlparse(url).netloc.lower()

    def _get_domain_config(self, domain: str) -> RateLimitConfig:
        """Získání konfigurace pro doménu (s fallback na default)"""
        if domain in self.domain_configs:
            return self.domain_configs[domain]

        return RateLimitConfig(
            domain=domain,
            requests_per_minute=self.default_requests_per_minute,
            requests_per_hour=self.default_requests_per_hour,
            burst_allowance=self.default_burst_allowance
        )

    def _cleanup_old_requests(self, domain_state: DomainState, current_time: float) -> None:
        """Vyčištění starých requestů z historie"""
        # Minutové requesty (starší než 60 sekund)
        while (domain_state.request_times and
               current_time - domain_state.request_times[0] > 60):
            domain_state.request_times.popleft()

        # Hodinové requesty (starší než 3600 sekund)
        while (domain_state.hourly_request_times and
               current_time - domain_state.hourly_request_times[0] > 3600):
            domain_state.hourly_request_times.popleft()

    def _calculate_backoff_time(self, config: RateLimitConfig, failures: int) -> float:
        """Výpočet exponential backoff času"""
        backoff_time = config.backoff_factor ** failures
        return min(backoff_time, config.max_backoff_seconds)

    async def check_rate_limit(self, url: str) -> RateLimitResult:
        """
        Kontrola rate limitu pro URL

        Returns:
            RateLimitResult s informacemi o povolení/blokování
        """
        async with self._lock:
            domain = self._extract_domain(url)
            current_time = time.time()

            config = self._get_domain_config(domain)
            state = self.domain_states[domain]

            # Cleanup starých requestů
            self._cleanup_old_requests(state, current_time)

            # Kontrola backoff období
            if current_time < state.backoff_until:
                wait_time = state.backoff_until - current_time
                self.global_stats["total_blocked"] += 1
                state.blocked_requests += 1

                return RateLimitResult(
                    allowed=False,
                    wait_time=wait_time,
                    reason=f"Domain in backoff until {time.ctime(state.backoff_until)}",
                    requests_remaining=0,
                    reset_time=state.backoff_until
                )

            # Kontrola minutového limitu
            minute_requests = len(state.request_times)
            if minute_requests >= config.requests_per_minute:
                # Kontrola burst allowance
                time_since_last = current_time - state.last_request_time
                if time_since_last < 1.0 and minute_requests >= config.requests_per_minute + config.burst_allowance:
                    wait_time = 60 - (current_time - state.request_times[0])
                    self.global_stats["total_blocked"] += 1
                    state.blocked_requests += 1

                    return RateLimitResult(
                        allowed=False,
                        wait_time=max(wait_time, 0),
                        reason="Minute rate limit exceeded",
                        requests_remaining=0,
                        reset_time=current_time + wait_time
                    )

            # Kontrola hodinového limitu
            hour_requests = len(state.hourly_request_times)
            if hour_requests >= config.requests_per_hour:
                wait_time = 3600 - (current_time - state.hourly_request_times[0])
                self.global_stats["total_blocked"] += 1
                state.blocked_requests += 1

                return RateLimitResult(
                    allowed=False,
                    wait_time=max(wait_time, 0),
                    reason="Hour rate limit exceeded",
                    requests_remaining=0,
                    reset_time=current_time + wait_time
                )

            # Request povolen - aktualizace stavu
            state.request_times.append(current_time)
            state.hourly_request_times.append(current_time)
            state.last_request_time = current_time
            state.total_requests += 1
            state.consecutive_failures = 0  # Reset při úspěšném requestu

            # Global statistiky
            self.global_stats["total_requests"] += 1
            if domain not in [s.domain for s in self.domain_configs.values()]:
                self.global_stats["total_domains"] += 1

            return RateLimitResult(
                allowed=True,
                wait_time=0.0,
                reason="Request allowed",
                requests_remaining=config.requests_per_minute - len(state.request_times),
                reset_time=current_time + 60
            )

    async def record_failure(self, url: str, apply_backoff: bool = True) -> None:
        """
        Zaznamenání selhání requestu (pro exponential backoff)
        """
        async with self._lock:
            domain = self._extract_domain(url)
            config = self._get_domain_config(domain)
            state = self.domain_states[domain]

            state.consecutive_failures += 1

            if apply_backoff:
                backoff_time = self._calculate_backoff_time(config, state.consecutive_failures)
                state.backoff_until = time.time() + backoff_time

                self.global_stats["backoff_events"] += 1

                logger.warning(
                    f"Applied backoff to {domain}: {backoff_time:.1f}s "
                    f"(failure #{state.consecutive_failures})"
                )

    async def record_success(self, url: str) -> None:
        """Zaznamenání úspěšného requestu"""
        async with self._lock:
            domain = self._extract_domain(url)
            state = self.domain_states[domain]

            # Reset consecutive failures při úspěchu
            state.consecutive_failures = 0

            # Reset backoff pokud byl aktivní
            if time.time() < state.backoff_until:
                state.backoff_until = 0.0
                logger.info(f"Reset backoff for {domain} after successful request")

    async def get_wait_time(self, url: str) -> float:
        """Získání času, který je třeba počkat před dalším requestem"""
        result = await self.check_rate_limit(url)
        return result.wait_time if not result.allowed else 0.0

    async def wait_if_needed(self, url: str) -> bool:
        """
        Počkání pokud je to potřeba kvůli rate limitu

        Returns:
            bool: True pokud byl request povolen (po případném čekání)
        """
        result = await self.check_rate_limit(url)

        if not result.allowed and result.wait_time > 0:
            logger.info(f"Rate limit hit for {url}, waiting {result.wait_time:.1f}s")
            await asyncio.sleep(result.wait_time)

            # Zkus znovu po čekání
            result = await self.check_rate_limit(url)

        return result.allowed

    def get_domain_stats(self, domain: str) -> Dict[str, any]:
        """Statistiky pro konkrétní doménu"""
        if domain not in self.domain_states:
            return {"error": "Domain not found"}

        state = self.domain_states[domain]
        config = self._get_domain_config(domain)
        current_time = time.time()

        return {
            "domain": domain,
            "total_requests": state.total_requests,
            "blocked_requests": state.blocked_requests,
            "success_rate": (state.total_requests - state.blocked_requests) / max(state.total_requests, 1),
            "consecutive_failures": state.consecutive_failures,
            "in_backoff": current_time < state.backoff_until,
            "backoff_remaining": max(0, state.backoff_until - current_time),
            "requests_last_minute": len(state.request_times),
            "requests_last_hour": len(state.hourly_request_times),
            "minute_limit": config.requests_per_minute,
            "hour_limit": config.requests_per_hour,
            "minute_utilization": len(state.request_times) / max(config.requests_per_minute, 1),
            "hour_utilization": len(state.hourly_request_times) / max(config.requests_per_hour, 1)
        }

    def get_global_stats(self) -> Dict[str, any]:
        """Globální statistiky rate limiteru"""
        active_domains = len([d for d in self.domain_states.values() if d.total_requests > 0])

        return {
            **self.global_stats,
            "active_domains": active_domains,
            "average_success_rate": (
                (self.global_stats["total_requests"] - self.global_stats["total_blocked"]) /
                max(self.global_stats["total_requests"], 1)
            ),
            "configured_domains": len(self.domain_configs)
        }

    async def start_cleanup_task(self) -> None:
        """Spuštění background cleanup tasku"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started rate limit cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Zastavení background cleanup tasku"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped rate limit cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background loop pro čištění starých dat"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self._periodic_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodické čištění starých dat"""
        async with self._lock:
            current_time = time.time()
            domains_to_remove = []

            for domain, state in self.domain_states.items():
                # Cleanup starých requestů
                self._cleanup_old_requests(state, current_time)

                # Odstranění neaktivních domén
                if (state.total_requests == 0 or
                    (current_time - state.last_request_time > 86400)):  # 24 hodin
                    domains_to_remove.append(domain)

            # Odstranění neaktivních domén
            for domain in domains_to_remove:
                del self.domain_states[domain]

            if domains_to_remove:
                logger.info(f"Cleaned up {len(domains_to_remove)} inactive domains")


# Factory funkce
def create_rate_limit_engine(**kwargs) -> RateLimitEngine:
    """Factory pro vytvoření rate limit engine"""
    return RateLimitEngine(**kwargs)


# Demo použití
if __name__ == "__main__":
    async def demo():
        engine = RateLimitEngine(default_requests_per_minute=5)

        # Přidání specifické konfigurace
        engine.add_domain_config(RateLimitConfig(
            domain="example.com",
            requests_per_minute=10,
            burst_allowance=3
        ))

        await engine.start_cleanup_task()

        try:
            # Test několika requestů
            test_urls = [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://test.com/data"
            ]

            for i in range(12):
                for url in test_urls:
                    result = await engine.check_rate_limit(url)
                    print(f"Request {i+1} to {url}: {result.allowed} ({result.reason})")

                    if not result.allowed:
                        print(f"  Wait time: {result.wait_time:.1f}s")

                    # Simulace úspěchu/selhání
                    if result.allowed:
                        if i % 3 == 0:  # Každý třetí request selže
                            await engine.record_failure(url)
                        else:
                            await engine.record_success(url)

                await asyncio.sleep(0.1)

            # Statistiky
            print("\nGlobal stats:", engine.get_global_stats())
            print("Domain stats example.com:", engine.get_domain_stats("example.com"))

        finally:
            await engine.stop_cleanup_task()

    asyncio.run(demo())
