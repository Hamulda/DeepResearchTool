"""
Robustní zpracování chyb s retry mechanismy pro DeepResearchTool
Implementuje exponenciální backoff a specifické retry strategie
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union, Type, Tuple
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
    after_log,
)
import aiohttp
import requests
from requests.exceptions import (
    ConnectionError,
    Timeout,
    RequestException,
    HTTPError,
    TooManyRedirects,
    ProxyError,
)


logger = logging.getLogger(__name__)


# Definice výjimek pro retry
NETWORK_EXCEPTIONS = (
    ConnectionError,
    Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ProxyError,
    aiohttp.ClientError,
    aiohttp.ClientConnectionError,
    aiohttp.ClientTimeout,
    aiohttp.ServerTimeoutError,
    OSError,  # Pro nízkoúrovňové síťové chyby
)

TEMPORARY_HTTP_ERRORS = (429, 500, 502, 503, 504)


def should_retry_http_error(exception: Exception) -> bool:
    """Určuje, zda se má HTTP chyba opakovat"""
    if isinstance(exception, HTTPError):
        return exception.response.status_code in TEMPORARY_HTTP_ERRORS
    elif isinstance(exception, aiohttp.ClientResponseError):
        return exception.status in TEMPORARY_HTTP_ERRORS
    return False


def should_retry_result(result: Any) -> bool:
    """Určuje, zda se má výsledek považovat za neúspěšný a opakovat"""
    if result is None:
        return True
    if isinstance(result, dict) and result.get('error'):
        return True
    if isinstance(result, (list, str)) and len(result) == 0:
        return True
    return False


# Předkonfigurované retry dekorátory
network_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)

scraping_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(NETWORK_EXCEPTIONS) | retry_if_result(should_retry_result),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)

api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(NETWORK_EXCEPTIONS) | retry_if_exception_type((HTTPError,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)

llm_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type((
        *NETWORK_EXCEPTIONS,
        Exception  # LLM API může vyhodit různé chyby
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)


class CircuitBreaker:
    """Implementace Circuit Breaker patternu pro kritické služby"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Volá funkci s circuit breaker logikou"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset při úspěšném volání"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Increment failure count a možná otevření circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# Global circuit breakers pro kritické služby
tor_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
llm_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
database_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Dekorátor pro aplikaci circuit breaker patternu"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


@with_circuit_breaker(tor_circuit_breaker)
@network_retry
def safe_tor_request(url: str, **kwargs) -> requests.Response:
    """Bezpečný Tor request s retry a circuit breaker"""
    try:
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"Tor request failed for {url}: {e}")
        raise


@with_circuit_breaker(llm_circuit_breaker)
@llm_retry
async def safe_llm_call(llm_func: Callable, *args, **kwargs) -> Any:
    """Bezpečný LLM call s retry a circuit breaker"""
    try:
        result = await llm_func(*args, **kwargs)
        if not result:
            raise Exception("Empty LLM response")
        return result
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


@scraping_retry
async def safe_aiohttp_get(session: aiohttp.ClientSession, url: str, **kwargs) -> aiohttp.ClientResponse:
    """Bezpečný aiohttp GET s retry"""
    try:
        async with session.get(url, **kwargs) as response:
            if response.status in TEMPORARY_HTTP_ERRORS:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )
            response.raise_for_status()
            return response
    except Exception as e:
        logger.error(f"aiohttp request failed for {url}: {e}")
        raise


@network_retry
def safe_requests_get(url: str, **kwargs) -> requests.Response:
    """Bezpečný requests GET s retry"""
    try:
        response = requests.get(url, timeout=30, **kwargs)
        
        if response.status_code in TEMPORARY_HTTP_ERRORS:
            raise HTTPError(f"HTTP {response.status_code}", response=response)
        
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"requests GET failed for {url}: {e}")
        raise


def log_and_ignore_errors(func: Callable):
    """Dekorátor pro logování chyb bez přerušení běhu"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper


def timeout_after(seconds: int):
    """Dekorátor pro timeout u dlouhotrvajících operací"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
        return wrapper
    return decorator


class ErrorAggregator:
    """Agreguje chyby pro batch operace"""
    
    def __init__(self):
        self.errors = []
        self.success_count = 0
        self.total_count = 0
    
    def add_error(self, error: Exception, context: str = ""):
        """Přidá chybu do agregátoru"""
        self.errors.append({
            'error': str(error),
            'type': type(error).__name__,
            'context': context,
            'timestamp': time.time()
        })
        self.total_count += 1
    
    def add_success(self):
        """Zaznamenává úspěšnou operaci"""
        self.success_count += 1
        self.total_count += 1
    
    def get_summary(self) -> dict:
        """Vrací shrnutí chyb a úspěchů"""
        return {
            'total_operations': self.total_count,
            'successful_operations': self.success_count,
            'failed_operations': len(self.errors),
            'success_rate': self.success_count / self.total_count if self.total_count > 0 else 0,
            'errors': self.errors
        }
    
    def log_summary(self):
        """Loguje shrnutí chyb"""
        summary = self.get_summary()
        logger.info(f"Batch operation summary: {summary['success_rate']:.2%} success rate "
                   f"({summary['successful_operations']}/{summary['total_operations']})")
        
        if self.errors:
            logger.warning(f"Encountered {len(self.errors)} errors:")
            for error in self.errors[-5:]:  # Log only last 5 errors
                logger.warning(f"  {error['type']}: {error['error']} ({error['context']})")


# Utility functions pro časté chyby
def is_rate_limited(response: Union[requests.Response, aiohttp.ClientResponse]) -> bool:
    """Zkontroluje, zda response indikuje rate limiting"""
    if hasattr(response, 'status_code'):
        return response.status_code == 429
    elif hasattr(response, 'status'):
        return response.status == 429
    return False


def get_retry_after(response: Union[requests.Response, aiohttp.ClientResponse]) -> Optional[int]:
    """Extrahuje Retry-After header z response"""
    retry_after = response.headers.get('Retry-After')
    if retry_after:
        try:
            return int(retry_after)
        except ValueError:
            pass
    return None


async def respect_rate_limit(response: Union[requests.Response, aiohttp.ClientResponse]):
    """Respektuje rate limit s Retry-After header"""
    if is_rate_limited(response):
        retry_after = get_retry_after(response) or 60  # Default 60 seconds
        logger.warning(f"Rate limited, waiting {retry_after} seconds")
        await asyncio.sleep(retry_after)


if __name__ == "__main__":
    # Test retry mechanismů
    @network_retry
    def test_network_call():
        raise ConnectionError("Test error")
    
    try:
        test_network_call()
    except Exception as e:
        print(f"Final error after retries: {e}")
    
    print("Error handling module loaded successfully!")