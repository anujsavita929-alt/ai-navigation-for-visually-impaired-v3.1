"""Scene description service with Groq cloud provider and local fallback hierarchy."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from scene_description.providers.base import BaseSceneProvider
from scene_description.providers.groq_provider import GroqSceneProvider
from scene_description.providers.local_provider import LocalSceneProvider

logger = logging.getLogger(__name__)

# Error classifications for fast-failover to local synthesizer
PERMANENT_ERROR_MARKERS = (
    "api_key_invalid",
    "invalid api key",
    "api key not valid",
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "not_found",
    "404",
    "decommissioned",
    "model_decommissioned",
    "model not found",
    "environment variable is not set",
    "missing api key",
)

# Upstream outages / server overloads / timeouts that should immediately fail over to local
TRANSIENT_OUTAGE_MARKERS = (
    "503",
    "unavailable",
    "high demand",
    "504",
    "deadline_exceeded",
    "deadline expired",
    "timed out",
    "timeout",
    "read operation timed out",
    "429",
    "resource_exhausted",
    "quota exceeded",
    "connection error",
)


def is_permanent_error(exc: Exception) -> bool:
    """Determine if an exception represents an unrecoverable credential, model, or configuration error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in PERMANENT_ERROR_MARKERS)


def should_failover_without_retry(exc: Exception) -> bool:
    """Determine if an exception should immediately trigger fallback to local without retrying."""
    if is_permanent_error(exc):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_OUTAGE_MARKERS)


class SceneService:
    """Orchestrates scene description requests across Groq and local rule-based synthesizer."""

    def __init__(
        self,
        groq_provider: Optional[BaseSceneProvider] = None,
        local_provider: Optional[BaseSceneProvider] = None,
        primary_name: str = "groq",
        fallback_name: str = "local",
        cooldown_seconds: float = 2.5,
        max_retries: int = 1,
        **kwargs: Any,
    ):
        self.groq = groq_provider or kwargs.get("primary_provider") or GroqSceneProvider()
        self.local = local_provider or LocalSceneProvider()
        self.primary_name = str(primary_name).lower()
        self.fallback_name = str(fallback_name).lower()
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_retries = max(0, int(max_retries))
        self.last_described_at = 0.0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> SceneService:
        """Create SceneService instance from configuration dictionary."""
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=False)
            else:
                load_dotenv(override=False)
        except ImportError:
            pass

        sd_cfg = cfg.get("scene_description", {})
        timeout = float(sd_cfg.get("timeout_seconds", 12.0))
        groq_timeout = float(sd_cfg.get("groq_timeout_seconds", timeout))
        cooldown = float(sd_cfg.get("cooldown_seconds", 2.5))
        max_retries = int(sd_cfg.get("max_retries", 1))

        groq_model = str(sd_cfg.get("groq_model", "qwen/qwen3.8-27b"))
        groq = GroqSceneProvider(model_name=groq_model, timeout_seconds=groq_timeout)
        local = LocalSceneProvider()

        primary_name = str(sd_cfg.get("primary_provider", "groq")).lower()
        fallback_name = str(sd_cfg.get("fallback_provider", "local")).lower()

        return cls(
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
        """Describe the given camera frame using Groq with automatic local fallback.

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

        # 1. Try Groq Cloud Provider (if primary is groq or cloud)
        if self.primary_name != "local" and self.groq.is_available():
            for attempt in range(self.max_retries + 1):
                try:
                    text = self.groq.describe(image_bytes, detections, context)
                    if text:
                        self.last_described_at = time.monotonic()
                        latency = time.perf_counter() - start_time
                        logger.info("Scene description provided by %s in %.2fs", self.groq.provider_name, latency)
                        return {
                            "text": text,
                            "provider": self.groq.provider_name,
                            "latency": latency,
                            "success": True,
                        }
                except Exception as exc:
                    fast_failover = should_failover_without_retry(exc)
                    logger.warning(
                        "Groq provider failed (attempt %d/%d, fast_failover=%s): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        fast_failover,
                        exc,
                    )
                    if fast_failover:
                        break
                    if attempt < self.max_retries:
                        time.sleep(0.4)

        # 2. Offline Local Synthesizer Fallback
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

        # 3. Ultimate Safety Fallback
        latency = time.perf_counter() - start_time
        return {
            "text": "Scene description is temporarily unavailable.",
            "provider": "none",
            "latency": latency,
            "success": False,
        }
