"""Abstract base class for scene description providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSceneProvider(ABC):
    """Abstract interface for all scene description providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'groq', 'local')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider prerequisites (e.g. API keys, dependencies) are satisfied."""
        pass

    @abstractmethod
    def describe(
        self,
        image_bytes: bytes,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a concise, accessibility-focused description of the image.

        Args:
            image_bytes: JPEG-encoded image bytes of the current camera frame.
            detections: List of current Detection objects from the object detector.
            context: Additional contextual data (e.g. free_space status, corridor obstruction).

        Returns:
            A concise string describing the surroundings for a visually impaired user.
        """
        pass
