"""GroqCloud scene description provider using the official groq SDK."""
from __future__ import annotations

import base64
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


def build_scene_prompt(detections: Optional[List[Any]] = None, context: Optional[Dict[str, Any]] = None) -> str:
    """Build a rich, grounded prompt incorporating detector and corridor evidence."""
    parts = ["Describe what is directly visible in the scene concisely for a visually impaired user navigating this space."]
    if detections:
        det_descs = []
        for d in detections[:6]:
            lbl = getattr(d, "label", "")
            if not lbl:
                continue
            direction = getattr(d, "direction", "ahead")
            dist_m = getattr(d, "distance_m", None)
            if dist_m is not None:
                det_descs.append(f"{lbl} ({dist_m:.1f}m {direction})")
            else:
                det_descs.append(f"{lbl} ({direction})")
        if det_descs:
            parts.append(f"Visual detector identified: {', '.join(det_descs)}.")

    if context:
        fs = context.get("free_space", {}).get("regions", {})
        if fs:
            centre_stat = fs.get("centre", "unknown")
            if centre_stat == "blocked":
                parts.append("Caution: Walking corridor ahead is obstructed.")
            elif centre_stat == "caution":
                parts.append("Note: Potential obstacle in the walking corridor.")

    return " ".join(parts)


class GroqSceneProvider(BaseSceneProvider):
    """Generates scene descriptions using GroqCloud Vision models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "qwen/qwen3.8-27b",
        timeout_seconds: float = 6.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self.model_name = model_name
        self.timeout_seconds = float(timeout_seconds)
        self.system_prompt = system_prompt
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        key = self.api_key or os.environ.get("GROQ_API_KEY", "").strip()
        return bool(key)

    def _get_client(self):
        if self._client is None:
            key = self.api_key or os.environ.get("GROQ_API_KEY", "").strip()
            if not key:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            try:
                from groq import Groq

                self._client = Groq(
                    api_key=key,
                    timeout=self.timeout_seconds,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "groq SDK is not installed. Install via: pip install groq"
                ) from exc
        return self._client

    def describe(
        self,
        image_bytes: bytes,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.is_available():
            raise ValueError("Groq provider is unavailable (missing API key).")

        client = self._get_client()

        prompt = build_scene_prompt(detections, context)

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]

        try:
            logger.info("Requesting scene description from Groq (%s)", self.model_name)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.2,
            )

            text = response.choices[0].message.content or ""
            text = text.strip()
            if not text:
                raise RuntimeError("Groq returned an empty description.")

            return text

        except Exception as exc:
            err_msg = str(exc)
            if "401" in err_msg or "Invalid API Key" in err_msg:
                logger.warning("Groq authentication failed: invalid API key.")
            elif "429" in err_msg or "rate_limit" in err_msg.lower():
                logger.warning("Groq rate limit or quota exceeded.")
            elif "404" in err_msg or "model_decommissioned" in err_msg.lower() or "not found" in err_msg.lower() or "decommissioned" in err_msg.lower():
                logger.warning("Groq model %s not found or decommissioned.", self.model_name)
            elif "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                logger.warning("Groq request timed out after %.1fs.", self.timeout_seconds)
            else:
                logger.warning("Groq scene description failed: %s", exc)
            raise
