"""Google Gemini scene description provider using the official google-genai SDK."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from scene_description.providers.base import BaseSceneProvider

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistive navigation companion for a visually impaired user. "
    "Describe what is directly visible in the image concisely in 1 to 2 clear, natural sentences. "
    "Focus on immediate obstacles, pathways, people, doors, stairs, and spatial positioning "
    "(e.g., 'ahead', 'to your left', 'to your right'). "
    "Do not speculate or guess exact metric distances. "
    "Keep language direct and easy to comprehend when spoken aloud by a screen reader."
)


class GeminiSceneProvider(BaseSceneProvider):
    """Generates scene descriptions using Google Gemini Vision models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = 6.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model_name = model_name
        self.timeout_seconds = float(timeout_seconds)
        self.system_prompt = system_prompt
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        # Check if API key is present
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        return bool(key)

    def _get_client(self):
        if self._client is None:
            key = self.api_key or os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            try:
                from google import genai
                from google.genai import types

                timeout_ms = int(self.timeout_seconds * 1000)
                self._client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai SDK is not installed. Install via: pip install google-genai"
                ) from exc
        return self._client

    def describe(
        self,
        image_bytes: bytes,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.is_available():
            raise ValueError("Gemini provider is unavailable (missing API key).")

        client = self._get_client()
        from google.genai import types

        # Build prompt
        prompt = "Describe the scene ahead for safe navigation."
        if detections:
            detected_labels = [getattr(d, "label", "") for d in detections if getattr(d, "label", "")]
            if detected_labels:
                unique_labels = list(dict.fromkeys(detected_labels))[:6]
                prompt += f" Object detector currently flags: {', '.join(unique_labels)}."

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,
            max_output_tokens=150,
        )

        try:
            logger.info("Requesting scene description from Gemini (%s)", self.model_name)
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            text = getattr(response, "text", "") or ""
            text = text.strip()
            if not text:
                raise RuntimeError("Gemini returned an empty description.")

            return text

        except Exception as exc:
            logger.warning("Gemini scene description failed: %s", exc)
            raise
