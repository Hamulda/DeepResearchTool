#!/usr/bin/env python3
"""
Behavior Camouflage Module for Phase 2 Implementation
Emulates human-like behavior patterns to avoid detection

Author: Advanced AI Research Assistant
Date: August 2025
"""

import asyncio
import random
import time
import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)


@dataclass
class HumanBehaviorProfile:
    """Profile defining human-like behavior characteristics"""

    reading_speed_wpm: int = 200  # Words per minute
    scroll_speed_variance: float = 0.3  # 30% variance
    click_delay_range: Tuple[float, float] = (0.1, 0.5)
    typing_speed_cps: int = 4  # Characters per second
    mouse_movement_style: str = "natural"  # natural, direct, cautious
    attention_span_seconds: int = 180  # 3 minutes average
    break_frequency: float = 0.15  # 15% chance of taking a break
    break_duration_range: Tuple[int, int] = (5, 30)  # seconds


class MousePathGenerator:
    """Generates natural mouse movement paths"""

    @staticmethod
    def bezier_curve(
        start: Tuple[int, int],
        end: Tuple[int, int],
        control_points: Optional[List[Tuple[int, int]]] = None,
        num_points: int = 20,
    ) -> List[Tuple[int, int]]:
        """Generate smooth Bezier curve between points"""
        if control_points is None:
            # Generate random control points for natural movement
            mid_x = (start[0] + end[0]) / 2 + random.randint(-50, 50)
            mid_y = (start[1] + end[1]) / 2 + random.randint(-20, 20)
            control_points = [(int(mid_x), int(mid_y))]

        points = []
        for i in range(num_points + 1):
            t = i / num_points
            if len(control_points) == 1:
                # Quadratic Bezier
                x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_points[0][0] + t**2 * end[0]
                y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_points[0][1] + t**2 * end[1]
            else:
                # Linear interpolation fallback
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])

            points.append((int(x), int(y)))

        return points

    @staticmethod
    def add_micro_movements(
        path: List[Tuple[int, int]], intensity: float = 0.5
    ) -> List[Tuple[int, int]]:
        """Add small random movements to simulate hand tremor"""
        modified_path = []
        for x, y in path:
            # Add small random offset
            offset_x = random.gauss(0, intensity)
            offset_y = random.gauss(0, intensity)
            modified_path.append((int(x + offset_x), int(y + offset_y)))

        return modified_path


class BehaviorCamouflage:
    """Main behavior camouflage system"""

    def __init__(self, profile: HumanBehaviorProfile = None):
        self.profile = profile or HumanBehaviorProfile()
        self.mouse_generator = MousePathGenerator()
        self.session_start = datetime.now()
        self.actions_performed = 0
        self.last_break = datetime.now()

        # Behavior tracking
        self.page_visit_times = []
        self.scroll_patterns = []
        self.click_timings = []

    async def random_delay(self, base_delay: float = 1.0, variance: float = 0.5) -> float:
        """Generate human-like random delay"""
        # Use log-normal distribution for more realistic timing
        delay = random.lognormvariate(math.log(base_delay), variance)
        delay = max(0.1, min(delay, base_delay * 3))  # Clamp to reasonable range

        await asyncio.sleep(delay)
        return delay

    async def simulate_reading_time(self, text_length: int) -> float:
        """Calculate realistic reading time based on text length"""
        words = text_length / 5  # Average 5 chars per word
        base_time = (words / self.profile.reading_speed_wpm) * 60

        # Add variance and minimum time
        variance_factor = random.uniform(0.7, 1.3)
        reading_time = max(2.0, base_time * variance_factor)

        await self.random_delay(reading_time, 0.2)
        return reading_time

    async def simulate_scroll_behavior(self, driver: webdriver, page_height: int):
        """Simulate natural scrolling behavior"""
        current_position = 0
        scroll_direction = 1  # 1 for down, -1 for up

        while current_position < page_height * 0.8:  # Don't scroll to bottom immediately
            # Random scroll amount
            scroll_amount = random.randint(100, 400) * scroll_direction

            # Occasionally scroll up to re-read
            if random.random() < 0.1 and current_position > 300:
                scroll_direction = -1
                scroll_amount = random.randint(50, 200) * scroll_direction
            else:
                scroll_direction = 1

            # Execute scroll
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            current_position += scroll_amount if scroll_direction > 0 else 0

            # Pause between scrolls
            await self.random_delay(random.uniform(0.5, 2.0), self.profile.scroll_speed_variance)

            # Record scroll pattern
            self.scroll_patterns.append(
                {"timestamp": datetime.now(), "amount": scroll_amount, "position": current_position}
            )

    async def human_like_click(self, driver: webdriver, element, use_natural_movement: bool = True):
        """Perform human-like click with natural mouse movement"""
        try:
            # Get element location
            location = element.location
            size = element.size

            # Calculate click point (slightly random within element)
            click_x = location["x"] + random.randint(5, size["width"] - 5)
            click_y = location["y"] + random.randint(5, size["height"] - 5)

            if use_natural_movement:
                # Get current mouse position (simulate from random starting point)
                start_x = random.randint(0, 1200)
                start_y = random.randint(0, 800)

                # Generate natural mouse path
                path = self.mouse_generator.bezier_curve((start_x, start_y), (click_x, click_y))
                path = self.mouse_generator.add_micro_movements(path)

                # Simulate mouse movement
                actions = ActionChains(driver)
                for x, y in path[::2]:  # Use every other point for performance
                    actions.move_by_offset(x - start_x, y - start_y)
                    start_x, start_y = x, y
                    await asyncio.sleep(0.01)  # Small delay between movements

            # Pre-click pause
            await self.random_delay(*self.profile.click_delay_range)

            # Perform click
            element.click()

            # Record click timing
            self.click_timings.append(datetime.now())
            self.actions_performed += 1

            # Post-click pause
            await self.random_delay(0.2, 0.1)

        except Exception as e:
            logger.error(f"Error in human-like click: {e}")
            # Fallback to simple click
            element.click()

    async def human_like_typing(self, element, text: str):
        """Type text with human-like patterns"""
        element.clear()

        for i, char in enumerate(text):
            element.send_keys(char)

            # Calculate typing delay based on character
            base_delay = 1.0 / self.profile.typing_speed_cps

            # Add variance for different characters
            if char in " \t\n":
                delay = base_delay * random.uniform(1.5, 3.0)  # Longer pause for spaces
            elif char in ".,!?;:":
                delay = base_delay * random.uniform(1.2, 2.0)  # Pause for punctuation
            elif i > 0 and text[i - 1 : i + 1] in ["th", "er", "in", "on"]:
                delay = base_delay * random.uniform(0.7, 1.0)  # Faster for common bigrams
            else:
                delay = base_delay * random.uniform(0.8, 1.5)

            # Occasionally pause longer (thinking)
            if random.random() < 0.05:
                delay *= random.uniform(2.0, 5.0)

            # Simulate typos occasionally
            if random.random() < 0.02 and len(text) > 5:
                # Type wrong character then correct it
                wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                element.send_keys(wrong_char)
                await asyncio.sleep(delay)
                element.send_keys("\b")  # Backspace
                await asyncio.sleep(delay * 0.5)

            await asyncio.sleep(delay)

    async def simulate_page_exploration(self, driver: webdriver):
        """Simulate realistic page exploration behavior"""
        try:
            # Get page dimensions
            page_height = driver.execute_script("return document.body.scrollHeight")
            page_width = driver.execute_script("return document.body.scrollWidth")

            # Initial pause to "load" and "orient"
            await self.random_delay(1.0, 0.3)

            # Simulate reading title/header
            try:
                title_elements = driver.find_elements(By.TAG_NAME, "h1")
                if title_elements:
                    title_text = title_elements[0].text
                    await self.simulate_reading_time(len(title_text))
            except:
                pass

            # Scroll and explore content
            await self.simulate_scroll_behavior(driver, page_height)

            # Randomly interact with some elements
            await self._random_element_interactions(driver)

            # Record page visit
            visit_time = (datetime.now() - self.session_start).seconds
            self.page_visit_times.append(visit_time)

        except Exception as e:
            logger.error(f"Error in page exploration: {e}")

    async def _random_element_interactions(self, driver: webdriver):
        """Randomly interact with page elements"""
        try:
            # Find interactive elements
            interactive_elements = []

            # Look for links (but don't click them)
            links = driver.find_elements(By.TAG_NAME, "a")
            interactive_elements.extend(links[:5])  # Limit to avoid too many

            # Look for buttons
            buttons = driver.find_elements(By.TAG_NAME, "button")
            interactive_elements.extend(buttons[:3])

            # Randomly hover over some elements
            for element in random.sample(interactive_elements, min(3, len(interactive_elements))):
                try:
                    # Check if element is visible
                    if element.is_displayed():
                        # Hover over element
                        ActionChains(driver).move_to_element(element).perform()
                        await self.random_delay(0.5, 0.2)

                        # Small chance to actually click (for non-navigation elements)
                        if (
                            random.random() < 0.1
                            and element.tag_name != "a"
                            and "submit" not in element.get_attribute("type", "")
                        ):
                            await self.human_like_click(driver, element)

                except Exception:
                    continue  # Skip problematic elements

        except Exception as e:
            logger.error(f"Error in element interactions: {e}")

    async def should_take_break(self) -> bool:
        """Determine if a break should be taken"""
        time_since_break = (datetime.now() - self.last_break).seconds

        # Increase break probability based on time and actions
        break_probability = self.profile.break_frequency

        if time_since_break > self.profile.attention_span_seconds:
            break_probability *= 2

        if self.actions_performed > 50:
            break_probability *= 1.5

        return random.random() < break_probability

    async def take_break(self):
        """Simulate taking a break"""
        if await self.should_take_break():
            break_duration = random.randint(*self.profile.break_duration_range)
            logger.info(f"Taking {break_duration}s break to simulate human behavior")

            await asyncio.sleep(break_duration)

            self.last_break = datetime.now()
            self.actions_performed = 0

            return True
        return False

    def get_behavior_report(self) -> Dict[str, Any]:
        """Generate behavior analysis report"""
        now = datetime.now()
        session_duration = (now - self.session_start).seconds

        return {
            "session_duration_seconds": session_duration,
            "actions_performed": self.actions_performed,
            "pages_visited": len(self.page_visit_times),
            "avg_page_time": (
                sum(self.page_visit_times) / len(self.page_visit_times)
                if self.page_visit_times
                else 0
            ),
            "scroll_actions": len(self.scroll_patterns),
            "clicks_performed": len(self.click_timings),
            "last_break_ago_seconds": (now - self.last_break).seconds,
            "profile": {
                "reading_speed_wpm": self.profile.reading_speed_wpm,
                "mouse_movement_style": self.profile.mouse_movement_style,
                "typing_speed_cps": self.profile.typing_speed_cps,
            },
        }


class AntiDetectionBrowser:
    """Browser wrapper with comprehensive anti-detection measures"""

    def __init__(self, browser_type: str = "firefox", headless: bool = False):
        self.browser_type = browser_type
        self.headless = headless
        self.driver = None
        self.behavior_camouflage = BehaviorCamouflage()

    def _get_firefox_options(self) -> webdriver.FirefoxOptions:
        """Get Firefox options with anti-detection settings"""
        options = webdriver.FirefoxOptions()

        if self.headless:
            options.add_argument("--headless")

        # Anti-fingerprinting settings
        options.set_preference("privacy.resistFingerprinting", True)
        options.set_preference("privacy.trackingprotection.enabled", True)
        options.set_preference("privacy.trackingprotection.socialtracking.enabled", True)
        options.set_preference("privacy.partition.network_state", True)

        # Disable WebRTC
        options.set_preference("media.peerconnection.enabled", False)

        # Disable geolocation
        options.set_preference("geo.enabled", False)

        # Randomize canvas fingerprint
        options.set_preference("privacy.resistFingerprinting.randomization.enabled", True)

        # Disable telemetry
        options.set_preference("toolkit.telemetry.enabled", False)
        options.set_preference("toolkit.telemetry.unified", False)

        # Custom user agent rotation
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
        ]
        options.set_preference("general.useragent.override", random.choice(user_agents))

        return options

    async def start_browser(self):
        """Start browser with anti-detection configuration"""
        try:
            if self.browser_type == "firefox":
                options = self._get_firefox_options()
                self.driver = webdriver.Firefox(options=options)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")

            # Set window size to common resolution
            self.driver.set_window_size(1366, 768)

            logger.info(f"Started {self.browser_type} browser with anti-detection measures")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False

    async def navigate_to_url(self, url: str, simulate_behavior: bool = True):
        """Navigate to URL with human-like behavior"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        try:
            # Navigate to URL
            self.driver.get(url)

            # Wait for page load
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            if simulate_behavior:
                # Simulate human-like page exploration
                await self.behavior_camouflage.simulate_page_exploration(self.driver)

                # Possibly take a break
                await self.behavior_camouflage.take_break()

            return True

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False

    async def close_browser(self):
        """Close browser and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
            finally:
                self.driver = None
