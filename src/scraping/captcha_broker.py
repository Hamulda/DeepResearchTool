"""CAPTCHA Broker System
Pluggable interface for CAPTCHA solving services (2Captcha, Anti-Captcha)
Async job submission with cost tracking and retry logic
Stubbed by default - requires user API keys for activation
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Supported CAPTCHA types"""

    IMAGE = "image"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    FUNCAPTCHA = "funcaptcha"
    TEXT = "text"


@dataclass
class CaptchaConfig:
    """CAPTCHA broker configuration"""

    service_provider: str = "stub"  # stub, 2captcha, anticaptcha
    api_key: str | None = None
    timeout_seconds: int = 300
    polling_interval: int = 5
    max_retries: int = 3
    cost_limit_usd: float = 10.0
    enable_captcha_solving: bool = False  # Opt-in

    # Service-specific settings
    service_url: str | None = None
    soft_mode: bool = True  # Fail gracefully if service unavailable


@dataclass
class CaptchaTask:
    """CAPTCHA solving task"""

    task_id: str
    captcha_type: CaptchaType
    site_url: str
    site_key: str | None = None
    image_data: str | None = None  # Base64 encoded
    additional_params: dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CaptchaResult:
    """CAPTCHA solving result"""

    task_id: str
    success: bool
    solution: str | None = None
    error_message: str | None = None
    cost_usd: float = 0.0
    solve_time_seconds: float = 0.0
    service_provider: str = "unknown"
    solved_at: datetime | None = None


class BaseCaptchaSolver:
    """Base class for CAPTCHA solving services"""

    def __init__(self, config: CaptchaConfig):
        self.config = config
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def solve_captcha(self, task: CaptchaTask) -> CaptchaResult:
        """Solve CAPTCHA task - to be implemented by subclasses"""
        raise NotImplementedError

    async def get_balance(self) -> float:
        """Get account balance - to be implemented by subclasses"""
        raise NotImplementedError


class StubCaptchaSolver(BaseCaptchaSolver):
    """Stub CAPTCHA solver for testing and safe defaults"""

    async def solve_captcha(self, task: CaptchaTask) -> CaptchaResult:
        """Return stubbed result"""
        logger.info(f"STUB: Would solve {task.captcha_type.value} CAPTCHA for {task.site_url}")

        # Simulate processing time
        await asyncio.sleep(1)

        return CaptchaResult(
            task_id=task.task_id,
            success=False,
            error_message="CAPTCHA solving disabled (stub mode)",
            service_provider="stub",
            solved_at=datetime.now(),
        )

    async def get_balance(self) -> float:
        """Return stub balance"""
        return 0.0


class TwoCaptchaSolver(BaseCaptchaSolver):
    """2Captcha service integration"""

    def __init__(self, config: CaptchaConfig):
        super().__init__(config)
        self.base_url = "http://2captcha.com"
        self.submit_url = f"{self.base_url}/in.php"
        self.result_url = f"{self.base_url}/res.php"

    async def solve_captcha(self, task: CaptchaTask) -> CaptchaResult:
        """Solve CAPTCHA using 2Captcha API"""
        start_time = datetime.now()

        try:
            # Submit task
            captcha_id = await self._submit_task(task)
            if not captcha_id:
                return CaptchaResult(
                    task_id=task.task_id,
                    success=False,
                    error_message="Failed to submit task to 2Captcha",
                    service_provider="2captcha",
                )

            # Poll for result
            solution = await self._poll_result(captcha_id)

            solve_time = (datetime.now() - start_time).total_seconds()

            if solution:
                return CaptchaResult(
                    task_id=task.task_id,
                    success=True,
                    solution=solution,
                    cost_usd=self._get_task_cost(task.captcha_type),
                    solve_time_seconds=solve_time,
                    service_provider="2captcha",
                    solved_at=datetime.now(),
                )
            return CaptchaResult(
                task_id=task.task_id,
                success=False,
                error_message="2Captcha failed to solve",
                service_provider="2captcha",
            )

        except Exception as e:
            logger.error(f"2Captcha error: {e}")
            return CaptchaResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_provider="2captcha",
            )

    async def _submit_task(self, task: CaptchaTask) -> str | None:
        """Submit CAPTCHA task to 2Captcha"""
        data = {
            "key": self.config.api_key,
            "method": self._get_method_for_type(task.captcha_type),
            "soft_mode": 1 if self.config.soft_mode else 0,
        }

        if task.captcha_type == CaptchaType.RECAPTCHA_V2:
            data.update({"googlekey": task.site_key, "pageurl": task.site_url})
        elif task.captcha_type == CaptchaType.IMAGE:
            data.update({"body": task.image_data})
        elif task.captcha_type == CaptchaType.HCAPTCHA:
            data.update({"sitekey": task.site_key, "pageurl": task.site_url})

        async with self.session.post(self.submit_url, data=data) as response:
            result = await response.text()

            if result.startswith("OK|"):
                return result.split("|")[1]
            logger.error(f"2Captcha submit error: {result}")
            return None

    async def _poll_result(self, captcha_id: str) -> str | None:
        """Poll for CAPTCHA result"""
        data = {"key": self.config.api_key, "action": "get", "id": captcha_id}

        end_time = datetime.now() + timedelta(seconds=self.config.timeout_seconds)

        while datetime.now() < end_time:
            async with self.session.get(self.result_url, params=data) as response:
                result = await response.text()

                if result.startswith("OK|"):
                    return result.split("|")[1]
                if result == "CAPCHA_NOT_READY":
                    await asyncio.sleep(self.config.polling_interval)
                    continue
                logger.error(f"2Captcha result error: {result}")
                return None

        logger.warning("2Captcha timeout")
        return None

    def _get_method_for_type(self, captcha_type: CaptchaType) -> str:
        """Get 2Captcha method for CAPTCHA type"""
        mapping = {
            CaptchaType.IMAGE: "base64",
            CaptchaType.RECAPTCHA_V2: "userrecaptcha",
            CaptchaType.RECAPTCHA_V3: "userrecaptcha",
            CaptchaType.HCAPTCHA: "hcaptcha",
            CaptchaType.TEXT: "textcaptcha",
        }
        return mapping.get(captcha_type, "base64")

    def _get_task_cost(self, captcha_type: CaptchaType) -> float:
        """Get estimated cost for CAPTCHA type"""
        costs = {
            CaptchaType.IMAGE: 0.001,
            CaptchaType.RECAPTCHA_V2: 0.002,
            CaptchaType.RECAPTCHA_V3: 0.002,
            CaptchaType.HCAPTCHA: 0.002,
            CaptchaType.TEXT: 0.001,
        }
        return costs.get(captcha_type, 0.002)

    async def get_balance(self) -> float:
        """Get 2Captcha account balance"""
        try:
            data = {"key": self.config.api_key, "action": "getbalance"}

            async with self.session.get(self.result_url, params=data) as response:
                result = await response.text()
                return float(result)

        except Exception as e:
            logger.error(f"Error getting 2Captcha balance: {e}")
            return 0.0


class AntiCaptchaSolver(BaseCaptchaSolver):
    """Anti-Captcha service integration"""

    def __init__(self, config: CaptchaConfig):
        super().__init__(config)
        self.base_url = "https://api.anti-captcha.com"
        self.create_task_url = f"{self.base_url}/createTask"
        self.get_result_url = f"{self.base_url}/getTaskResult"

    async def solve_captcha(self, task: CaptchaTask) -> CaptchaResult:
        """Solve CAPTCHA using Anti-Captcha API"""
        start_time = datetime.now()

        try:
            # Create task
            task_id = await self._create_task(task)
            if not task_id:
                return CaptchaResult(
                    task_id=task.task_id,
                    success=False,
                    error_message="Failed to create task in Anti-Captcha",
                    service_provider="anticaptcha",
                )

            # Get result
            solution = await self._get_task_result(task_id)

            solve_time = (datetime.now() - start_time).total_seconds()

            if solution:
                return CaptchaResult(
                    task_id=task.task_id,
                    success=True,
                    solution=solution,
                    cost_usd=self._get_task_cost(task.captcha_type),
                    solve_time_seconds=solve_time,
                    service_provider="anticaptcha",
                    solved_at=datetime.now(),
                )
            return CaptchaResult(
                task_id=task.task_id,
                success=False,
                error_message="Anti-Captcha failed to solve",
                service_provider="anticaptcha",
            )

        except Exception as e:
            logger.error(f"Anti-Captcha error: {e}")
            return CaptchaResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_provider="anticaptcha",
            )

    async def _create_task(self, task: CaptchaTask) -> int | None:
        """Create task in Anti-Captcha"""
        task_data = self._build_task_data(task)

        payload = {
            "clientKey": self.config.api_key,
            "task": task_data,
            "softMode": self.config.soft_mode,
        }

        async with self.session.post(self.create_task_url, json=payload) as response:
            result = await response.json()

            if result.get("errorId") == 0:
                return result.get("taskId")
            logger.error(f"Anti-Captcha create task error: {result.get('errorDescription')}")
            return None

    async def _get_task_result(self, task_id: int) -> str | None:
        """Get task result from Anti-Captcha"""
        payload = {"clientKey": self.config.api_key, "taskId": task_id}

        end_time = datetime.now() + timedelta(seconds=self.config.timeout_seconds)

        while datetime.now() < end_time:
            async with self.session.post(self.get_result_url, json=payload) as response:
                result = await response.json()

                if result.get("status") == "ready":
                    solution_data = result.get("solution", {})
                    return solution_data.get("gRecaptchaResponse") or solution_data.get("text")
                if result.get("status") == "processing":
                    await asyncio.sleep(self.config.polling_interval)
                    continue
                logger.error(f"Anti-Captcha error: {result.get('errorDescription')}")
                return None

        logger.warning("Anti-Captcha timeout")
        return None

    def _build_task_data(self, task: CaptchaTask) -> dict[str, Any]:
        """Build task data for Anti-Captcha"""
        if task.captcha_type == CaptchaType.RECAPTCHA_V2:
            return {
                "type": "NoCaptchaTaskProxyless",
                "websiteURL": task.site_url,
                "websiteKey": task.site_key,
            }
        if task.captcha_type == CaptchaType.HCAPTCHA:
            return {
                "type": "HCaptchaTaskProxyless",
                "websiteURL": task.site_url,
                "websiteKey": task.site_key,
            }
        if task.captcha_type == CaptchaType.IMAGE:
            return {"type": "ImageToTextTask", "body": task.image_data}
        raise ValueError(f"Unsupported CAPTCHA type: {task.captcha_type}")

    def _get_task_cost(self, captcha_type: CaptchaType) -> float:
        """Get estimated cost for CAPTCHA type"""
        costs = {
            CaptchaType.IMAGE: 0.0015,
            CaptchaType.RECAPTCHA_V2: 0.002,
            CaptchaType.RECAPTCHA_V3: 0.002,
            CaptchaType.HCAPTCHA: 0.002,
        }
        return costs.get(captcha_type, 0.002)

    async def get_balance(self) -> float:
        """Get Anti-Captcha account balance"""
        try:
            payload = {"clientKey": self.config.api_key}
            balance_url = f"{self.base_url}/getBalance"

            async with self.session.post(balance_url, json=payload) as response:
                result = await response.json()
                return float(result.get("balance", 0))

        except Exception as e:
            logger.error(f"Error getting Anti-Captcha balance: {e}")
            return 0.0


class CaptchaBroker:
    """Main CAPTCHA broker with cost tracking and provider management"""

    def __init__(self, config: CaptchaConfig):
        self.config = config
        self.solver = self._create_solver()
        self.total_cost_usd = 0.0
        self.solved_count = 0
        self.failed_count = 0

    def _create_solver(self) -> BaseCaptchaSolver:
        """Create appropriate solver based on configuration"""
        if not self.config.enable_captcha_solving:
            return StubCaptchaSolver(self.config)

        if self.config.service_provider == "2captcha":
            if not self.config.api_key:
                logger.warning("2Captcha API key not provided, using stub")
                return StubCaptchaSolver(self.config)
            return TwoCaptchaSolver(self.config)

        if self.config.service_provider == "anticaptcha":
            if not self.config.api_key:
                logger.warning("Anti-Captcha API key not provided, using stub")
                return StubCaptchaSolver(self.config)
            return AntiCaptchaSolver(self.config)

        logger.info(f"Using stub solver for provider: {self.config.service_provider}")
        return StubCaptchaSolver(self.config)

    async def solve_captcha(
        self,
        captcha_type: CaptchaType,
        site_url: str,
        site_key: str | None = None,
        image_data: str | None = None,
        task_id: str | None = None,
    ) -> CaptchaResult:
        """Solve CAPTCHA with cost tracking"""
        # Check cost limit
        if self.total_cost_usd >= self.config.cost_limit_usd:
            return CaptchaResult(
                task_id=task_id or "cost_limit_exceeded",
                success=False,
                error_message=f"Cost limit exceeded (${self.config.cost_limit_usd})",
                service_provider=self.config.service_provider,
            )

        # Create task
        task = CaptchaTask(
            task_id=task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            captcha_type=captcha_type,
            site_url=site_url,
            site_key=site_key,
            image_data=image_data,
        )

        # Solve with retries
        last_result = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.solver as solver:
                    result = await solver.solve_captcha(task)

                    if result.success:
                        self.total_cost_usd += result.cost_usd
                        self.solved_count += 1
                        logger.info(f"CAPTCHA solved: {task.task_id} (${result.cost_usd:.4f})")
                        return result
                    last_result = result
                    if attempt < self.config.max_retries:
                        logger.warning(
                            f"CAPTCHA solve failed (attempt {attempt + 1}), retrying..."
                        )
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"CAPTCHA solve error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries:
                    last_result = CaptchaResult(
                        task_id=task.task_id,
                        success=False,
                        error_message=str(e),
                        service_provider=self.config.service_provider,
                    )

        # All attempts failed
        self.failed_count += 1
        return last_result or CaptchaResult(
            task_id=task.task_id,
            success=False,
            error_message="All solve attempts failed",
            service_provider=self.config.service_provider,
        )

    async def get_balance(self) -> float:
        """Get current account balance"""
        try:
            async with self.solver as solver:
                return await solver.get_balance()
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics"""
        total_attempts = self.solved_count + self.failed_count
        success_rate = self.solved_count / max(1, total_attempts)

        return {
            "total_cost_usd": self.total_cost_usd,
            "solved_count": self.solved_count,
            "failed_count": self.failed_count,
            "success_rate": success_rate,
            "cost_limit_usd": self.config.cost_limit_usd,
            "cost_remaining_usd": max(0, self.config.cost_limit_usd - self.total_cost_usd),
            "service_provider": self.config.service_provider,
            "enabled": self.config.enable_captcha_solving,
        }


# Utility functions
def create_captcha_config(
    provider: str = "stub",
    api_key: str | None = None,
    enable: bool = False,
    cost_limit: float = 10.0,
) -> CaptchaConfig:
    """Create CAPTCHA configuration"""
    return CaptchaConfig(
        service_provider=provider,
        api_key=api_key,
        enable_captcha_solving=enable,
        cost_limit_usd=cost_limit,
        soft_mode=True,
    )


async def solve_recaptcha_v2(broker: CaptchaBroker, site_url: str, site_key: str) -> str | None:
    """Convenience function to solve reCAPTCHA v2"""
    result = await broker.solve_captcha(CaptchaType.RECAPTCHA_V2, site_url, site_key=site_key)

    return result.solution if result.success else None


__all__ = [
    "CaptchaBroker",
    "CaptchaConfig",
    "CaptchaResult",
    "CaptchaTask",
    "CaptchaType",
    "create_captcha_config",
    "solve_recaptcha_v2",
]
