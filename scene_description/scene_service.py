"""Multi-provider scene description service with graceful fallback hierarchy."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from scene_description.providers.base import BaseSceneProvider
from scene_description.providers.gemini_provider import GeminiSceneProvider
from scene_description.providers.groq_provider import GroqSceneProvider
from scene_description.providers.local_provider import LocalSceneProvider

logger = logging.getLogger(__name__)


class SceneService:
    """Orchestrates scene description requests across primary, fallback, and local providers."""

    def __init__(
        self,
        gemini_provider: Optional[BaseSceneProvider] = None,
        groq_provider: Optional[BaseSceneProvider] = None,
        local_provider: Optional[BaseSceneProvider] = None,
        primary_name: str = "gemini",
        fallback_name: str = "groq",
        cooldown_seconds: float = 2.5,
        max_retries: int = 1,
    ):
        self.gemini = gemini_provider or GeminiSceneProvider()
        self.groq = groq_provider or GroqSceneProvider()
        self.local = local_provider or LocalSceneProvider()
        self.primary_name = str(primary_name).lower()
        self.fallback_name = str(fallback_name).lower()
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_retries = max(0, int(max_retries))
        self.last_described_at = 0.0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> SceneService:
        """Create SceneService instance from configuration dictionary."""
        # Try loading .env if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        sd_cfg = cfg.get("scene_description", {})
        timeout = float(sd_cfg.get("timeout_seconds", 5.0))
        cooldown = float(sd_cfg.get("cooldown_seconds", 2.5))
        max_retries = int(sd_cfg.get("max_retries", 1))

        gemini_model = str(sd_cfg.get("gemini_model", "gemini-2.5-flash"))
        groq_model = str(sd_cfg.get("groq_model", "llama-3.2-11b-vision-preview"))

        gemini = GeminiSceneProvider(model_name=gemini_model, timeout_seconds=timeout)
        groq = GroqSceneProvider(model_name=groq_model, timeout_seconds=timeout)
        local = LocalSceneProvider()

        primary_name = str(sd_cfg.get("primary_provider", "gemini")).lower()
        fallback_name = str(sd_cfg.get("fallback_provider", "groq")).lower()

        return cls(
            gemini_provider=gemini,
            groq_provider=groq,
            local_provider=local,
            primary_name=primary_name,
            fallback_name=fallback_name,
            cooldown_seconds=cooldown,
            max_retries=max_retries,
        )

    def describe_scene(
        self,
        image_bytes: bytes,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Describe the given camera frame using the provider fallback chain.

        Returns:
            Dict containing:
                - text: The descriptive text to be spoken.
                - provider: Name of the provider that produced the description.
                - latency: Total latency in seconds.
                - success: Boolean indicating whether description succeeded.
        """
        now = time.monotonic()
        if not force and (now - self.last_described_at < self.cooldown_seconds):
            remaining = self.cooldown_seconds - (now - self.last_described_at)
            logger.info("Scene description rate-limited (cooldown active: %.1fs remaining)", remaining)
            return {
                "text": "Please wait a moment before requesting another description.",
                "provider": "rate_limit",
                "latency": 0.0,
                "success": False,
            }

        start_time = time.perf_counter()

        # Determine provider order
        if self.primary_name == "local":
            primary_provider = self.local
            fallback_provider = None
        elif self.primary_name == "groq":
            primary_provider = self.groq
            fallback_provider = self.gemini
        else:
            primary_provider = self.gemini
            fallback_provider = self.groq

        # 1. Try Primary Cloud Provider
        if primary_provider and primary_provider.is_available():
            for attempt in range(self.max_retries + 1):
                try:
                    text = primary_provider.describe(image_bytes, detections, context)
                    if text:
                        self.last_described_at = time.monotonic()
                        latency = time.perf_counter() - start_time
                        logger.info("Scene description provided by %s in %.2fs", primary_provider.provider_name, latency)
                        return {
                            "text": text,
                            "provider": primary_provider.provider_name,
                            "latency": latency,
                            "success": True,
                        }
                except Exception as exc:
                    logger.warning(
                        "Primary provider %s failed (attempt %d/%d): %s",
                        primary_provider.provider_name,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    if attempt < self.max_retries:
                        time.sleep(0.5)

        # 2. Try Fallback Cloud Provider
        if fallback_provider and fallback_provider.is_available():
            for attempt in range(self.max_retries + 1):
                try:
                    text = fallback_provider.describe(image_bytes, detections, context)
                    if text:
                        self.last_described_at = time.monotonic()
                        latency = time.perf_counter() - start_time
                        logger.info(
                            "Scene description provided by fallback %s in %.2fs",
                            fallback_provider.provider_name,
                            latency,
                        )
                        return {
                            "text": text,
                            "provider": fallback_provider.provider_name,
                            "latency": latency,
                            "success": True,
                        }
                except Exception as exc:
                    logger.warning(
                        "Fallback provider %s failed (attempt %d/%d): %s",
                        fallback_provider.provider_name,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    if attempt < self.max_retries:
                        time.sleep(0.5)

        # 3. Offline Local Synthesizer Fallback
        try:
            logger.info("Generating offline local scene description from current detections")
            text = self.local.describe(image_bytes, detections, context)
            self.last_described_at = time.monotonic()
            latency = time.perf_counter() - start_time
            return {
                "text": text,
                "provider": "local",
                "latency": latency,
                "success": True,
            }
        except Exception as exc:
            logger.error("Local scene synthesizer failed: %s", exc)

        # 4. Ultimate Safety Fallback
        latency = time.perf_counter() - start_time
        return {
            "text": "Scene description is temporarily unavailable.",
            "provider": "none",
            "latency": latency,
            "success": False,
        }
