"""Evasion Profiles for Anti-Bot Circumvention
Randomized browser fingerprints, headers, and behavioral patterns
Per-site profiles with stealth capabilities for M1 optimization
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import Browser, BrowserContext, Page

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class BrowserProfile:
    """Browser fingerprint profile"""

    user_agent: str
    viewport_width: int
    viewport_height: int
    device_scale_factor: float
    locale: str
    timezone: str
    platform: str
    languages: list[str]
    color_depth: int
    hardware_concurrency: int
    memory_gb: int
    webgl_vendor: str
    webgl_renderer: str
    extra_headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_agent": self.user_agent,
            "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            "device_scale_factor": self.device_scale_factor,
            "locale": self.locale,
            "timezone": self.timezone,
            "platform": self.platform,
            "languages": self.languages,
            "color_depth": self.color_depth,
            "hardware_concurrency": self.hardware_concurrency,
            "memory_gb": self.memory_gb,
            "webgl_vendor": self.webgl_vendor,
            "webgl_renderer": self.webgl_renderer,
            "extra_headers": self.extra_headers,
        }


@dataclass
class EvasionProfile:
    """Complete evasion profile for a site or session"""

    profile_id: str
    site_domain: str
    browser_profile: BrowserProfile
    behavioral_settings: dict[str, Any]
    proxy_settings: dict[str, str] | None
    created_at: datetime
    last_used: datetime | None = None
    success_rate: float = 1.0
    usage_count: int = 0

    def update_usage(self, success: bool):
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.now()

        # Update success rate with weighted average
        weight = 0.1  # Give more weight to recent results
        if success:
            self.success_rate = (1 - weight) * self.success_rate + weight * 1.0
        else:
            self.success_rate = (1 - weight) * self.success_rate + weight * 0.0


class BrowserProfileGenerator:
    """Generate realistic browser profiles"""

    # Common user agents (macOS focused for M1 compatibility)
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    ]

    # Common viewport sizes
    VIEWPORTS = [
        (1920, 1080),
        (1366, 768),
        (1440, 900),
        (1536, 864),
        (1280, 720),
        (1600, 900),
        (1024, 768),
        (1680, 1050),
        (2560, 1440),
    ]

    # Timezone options
    TIMEZONES = [
        "America/New_York",
        "America/Los_Angeles",
        "America/Chicago",
        "Europe/London",
        "Europe/Berlin",
        "Europe/Paris",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Australia/Sydney",
    ]

    # Language preferences
    LANGUAGES = [
        ["en-US", "en"],
        ["en-GB", "en"],
        ["de-DE", "de"],
        ["fr-FR", "fr"],
        ["es-ES", "es"],
        ["it-IT", "it"],
        ["pt-BR", "pt"],
        ["ja-JP", "ja"],
    ]

    # WebGL vendors/renderers
    WEBGL_PROFILES = [
        ("Google Inc. (Intel)", "ANGLE (Intel, Intel Iris Pro OpenGL Engine, OpenGL 4.1)"),
        ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon Pro 555X OpenGL Engine, OpenGL 4.1)"),
        (
            "Google Inc. (NVIDIA)",
            "ANGLE (NVIDIA, NVIDIA GeForce GTX 1050 Ti OpenGL Engine, OpenGL 4.1)",
        ),
        ("Intel Inc.", "Intel Iris Pro OpenGL Engine"),
        ("AMD", "AMD Radeon Pro 560X OpenGL Engine"),
    ]

    @classmethod
    def generate_random_profile(cls) -> BrowserProfile:
        """Generate a random but realistic browser profile"""
        # Select random components
        user_agent = random.choice(cls.USER_AGENTS)
        viewport = random.choice(cls.VIEWPORTS)
        timezone = random.choice(cls.TIMEZONES)
        languages = random.choice(cls.LANGUAGES)
        webgl_vendor, webgl_renderer = random.choice(cls.WEBGL_PROFILES)

        # Determine platform from user agent
        if "Mac OS X" in user_agent:
            platform = "MacIntel"
        elif "Windows" in user_agent:
            platform = "Win32"
        else:
            platform = "Linux x86_64"

        # Hardware specs (M1 MacBook Air oriented)
        hardware_concurrency = random.choice([4, 8, 12, 16])  # M1 has 8 cores
        memory_gb = random.choice([8, 16, 32])  # Common M1 configs

        return BrowserProfile(
            user_agent=user_agent,
            viewport_width=viewport[0],
            viewport_height=viewport[1],
            device_scale_factor=random.choice([1.0, 1.25, 1.5, 2.0]),
            locale=languages[0][:2] + "_" + languages[0][-2:].upper(),
            timezone=timezone,
            platform=platform,
            languages=languages,
            color_depth=random.choice([24, 32]),
            hardware_concurrency=hardware_concurrency,
            memory_gb=memory_gb,
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            extra_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": f"{languages[0]},{languages[1]};q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": random.choice(["1", "0"]),
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )

    @classmethod
    def generate_site_specific_profile(
        cls, domain: str, base_profile: BrowserProfile = None
    ) -> BrowserProfile:
        """Generate profile optimized for specific site"""
        if base_profile is None:
            base_profile = cls.generate_random_profile()

        # Site-specific optimizations
        domain_lower = domain.lower()

        if "google" in domain_lower:
            # Google prefers Chrome
            base_profile.user_agent = random.choice(
                [ua for ua in cls.USER_AGENTS if "Chrome" in ua]
            )
        elif "facebook" in domain_lower or "instagram" in domain_lower:
            # Meta properties work well with mainstream browsers
            base_profile.extra_headers["sec-fetch-site"] = "same-origin"
            base_profile.extra_headers["sec-fetch-mode"] = "navigate"
        elif "linkedin" in domain_lower:
            # LinkedIn is strict about headers
            base_profile.extra_headers["sec-ch-ua"] = '"Chromium";v="120", "Not A(Brand";v="99"'
            base_profile.extra_headers["sec-ch-ua-mobile"] = "?0"
            base_profile.extra_headers["sec-ch-ua-platform"] = '"macOS"'

        return base_profile


class EvasionProfileManager:
    """Manage and rotate evasion profiles"""

    def __init__(self, profiles_dir: Path = None):
        self.profiles_dir = profiles_dir or Path("evasion_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.active_profiles: dict[str, EvasionProfile] = {}
        self.profile_rotation_interval = timedelta(hours=2)

        # Load existing profiles
        self._load_profiles()

    def get_profile_for_site(self, domain: str, force_new: bool = False) -> EvasionProfile:
        """Get or create evasion profile for site"""
        domain_key = self._normalize_domain(domain)

        # Check if we have an existing profile
        if not force_new and domain_key in self.active_profiles:
            profile = self.active_profiles[domain_key]

            # Check if profile needs rotation
            if self._should_rotate_profile(profile):
                logger.info(
                    f"Rotating profile for {domain} (success rate: {profile.success_rate:.2f})"
                )
                profile = self._create_new_profile(domain)

            return profile

        # Create new profile
        return self._create_new_profile(domain)

    def _create_new_profile(self, domain: str) -> EvasionProfile:
        """Create new evasion profile"""
        domain_key = self._normalize_domain(domain)

        # Generate browser profile
        browser_profile = BrowserProfileGenerator.generate_site_specific_profile(domain)

        # Behavioral settings
        behavioral_settings = {
            "mouse_movement": True,
            "random_delays": True,
            "typing_delays": True,
            "scroll_behavior": "human",
            "click_offset_variance": 5,
            "page_load_wait": random.uniform(2.0, 5.0),
            "interaction_delay_range": [0.5, 2.0],
            "scroll_pause_range": [1.0, 3.0],
        }

        # Create profile
        profile_id = self._generate_profile_id(domain)
        profile = EvasionProfile(
            profile_id=profile_id,
            site_domain=domain,
            browser_profile=browser_profile,
            behavioral_settings=behavioral_settings,
            proxy_settings=None,  # Will be set by network manager
            created_at=datetime.now(),
        )

        self.active_profiles[domain_key] = profile
        self._save_profile(profile)

        logger.info(f"Created new evasion profile for {domain}: {profile_id}")
        return profile

    def _should_rotate_profile(self, profile: EvasionProfile) -> bool:
        """Determine if profile should be rotated"""
        # Rotate if success rate is low
        if profile.success_rate < 0.5:
            return True

        # Rotate if profile is old
        if (
            profile.last_used
            and datetime.now() - profile.last_used > self.profile_rotation_interval
        ):
            return True

        # Rotate if used too many times
        if profile.usage_count > 50:
            return True

        return False

    def update_profile_success(self, profile: EvasionProfile, success: bool):
        """Update profile success rate"""
        profile.update_usage(success)
        self._save_profile(profile)

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain for consistent keying"""
        return domain.lower().replace("www.", "").split("/")[0]

    def _generate_profile_id(self, domain: str) -> str:
        """Generate unique profile ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_hash = hashlib.md5(domain.encode()).hexdigest()[:8]
        return f"profile_{domain_hash}_{timestamp}"

    def _load_profiles(self):
        """Load existing profiles from disk"""
        try:
            for profile_file in self.profiles_dir.glob("profile_*.json"):
                with open(profile_file) as f:
                    data = json.load(f)

                # Reconstruct profile
                browser_profile = BrowserProfile(**data["browser_profile"])

                profile = EvasionProfile(
                    profile_id=data["profile_id"],
                    site_domain=data["site_domain"],
                    browser_profile=browser_profile,
                    behavioral_settings=data["behavioral_settings"],
                    proxy_settings=data.get("proxy_settings"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_used=(
                        datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
                    ),
                    success_rate=data.get("success_rate", 1.0),
                    usage_count=data.get("usage_count", 0),
                )

                domain_key = self._normalize_domain(profile.site_domain)
                self.active_profiles[domain_key] = profile

        except Exception as e:
            logger.warning(f"Error loading profiles: {e}")

    def _save_profile(self, profile: EvasionProfile):
        """Save profile to disk"""
        try:
            profile_file = self.profiles_dir / f"{profile.profile_id}.json"

            data = {
                "profile_id": profile.profile_id,
                "site_domain": profile.site_domain,
                "browser_profile": profile.browser_profile.to_dict(),
                "behavioral_settings": profile.behavioral_settings,
                "proxy_settings": profile.proxy_settings,
                "created_at": profile.created_at.isoformat(),
                "last_used": profile.last_used.isoformat() if profile.last_used else None,
                "success_rate": profile.success_rate,
                "usage_count": profile.usage_count,
            }

            with open(profile_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Error saving profile {profile.profile_id}: {e}")

    def cleanup_old_profiles(self, max_age_days: int = 30):
        """Clean up old unused profiles"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        profiles_to_remove = []
        for domain_key, profile in self.active_profiles.items():
            if profile.created_at < cutoff_date and (
                profile.last_used is None or profile.last_used < cutoff_date
            ):
                profiles_to_remove.append(domain_key)

        for domain_key in profiles_to_remove:
            profile = self.active_profiles.pop(domain_key)
            profile_file = self.profiles_dir / f"{profile.profile_id}.json"
            if profile_file.exists():
                profile_file.unlink()
            logger.info(f"Cleaned up old profile: {profile.profile_id}")

    def get_profile_stats(self) -> dict[str, Any]:
        """Get statistics about managed profiles"""
        if not self.active_profiles:
            return {"total_profiles": 0}

        success_rates = [p.success_rate for p in self.active_profiles.values()]
        usage_counts = [p.usage_count for p in self.active_profiles.values()]

        return {
            "total_profiles": len(self.active_profiles),
            "average_success_rate": sum(success_rates) / len(success_rates),
            "total_usage": sum(usage_counts),
            "domains_covered": list(self.active_profiles.keys()),
            "profiles_needing_rotation": sum(
                1 for p in self.active_profiles.values() if self._should_rotate_profile(p)
            ),
        }


# Utility functions for profile application
async def apply_evasion_profile(page: Page, profile: EvasionProfile):
    """Apply evasion profile to Playwright page"""
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available - cannot apply evasion profile")
        return

    browser_profile = profile.browser_profile

    try:
        # Set viewport
        await page.set_viewport_size(
            width=browser_profile.viewport_width, height=browser_profile.viewport_height
        )

        # Set extra headers
        await page.set_extra_http_headers(browser_profile.extra_headers)

        # Override navigator properties
        await page.add_init_script(
            f"""
            Object.defineProperty(navigator, 'platform', {{
                get: () => '{browser_profile.platform}'
            }});
            
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => {browser_profile.hardware_concurrency}
            }});
            
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: () => {browser_profile.memory_gb}
            }});
            
            Object.defineProperty(navigator, 'languages', {{
                get: () => {json.dumps(browser_profile.languages)}
            }});
            
            Object.defineProperty(screen, 'colorDepth', {{
                get: () => {browser_profile.color_depth}
            }});
            
            // Override WebGL
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) {{
                    return '{browser_profile.webgl_vendor}';
                }}
                if (parameter === 37446) {{
                    return '{browser_profile.webgl_renderer}';
                }}
                return getParameter.call(this, parameter);
            }};
        """
        )

        logger.debug(f"Applied evasion profile {profile.profile_id} to page")

    except Exception as e:
        logger.warning(f"Error applying evasion profile: {e}")


__all__ = [
    "BrowserProfile",
    "BrowserProfileGenerator",
    "EvasionProfile",
    "EvasionProfileManager",
    "apply_evasion_profile",
]
