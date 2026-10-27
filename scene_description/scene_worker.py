"""Asynchronous single-slot background worker for scene description."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2

from scene_description.scene_service import SceneService

logger = logging.getLogger(__name__)


class SceneWorker:
    """Runs scene description asynchronously so the camera and navigation loops never stall."""

    def __init__(
        self,
        service: SceneService,
        on_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
        jpeg_quality: int = 80,
    ):
        self.service = service
        self.on_completed = on_completed
        self.jpeg_quality = int(jpeg_quality)

        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._is_busy = False
        self._latest_result: Optional[Dict[str, Any]] = None

        self._thread = threading.Thread(
            target=self._run,
            name="scene-describer",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._is_busy or self._pending is not None

    def request(
        self,
        frame,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Submit a new frame for scene description if not already busy.

        Returns:
            True if request was accepted, False if busy or closed.
        """
        with self._condition:
            if self._closed or self._is_busy or self._pending is not None:
                return False

            # Copy frame to isolate memory from video stream
            self._pending = (frame.copy(), detections or [], context or {})
            self._condition.notify()
            return True

    def poll_result(self) -> Optional[Dict[str, Any]]:
        """Return and clear the latest completed scene description result, if any."""
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            return result

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return

                frame, detections, context = self._pending
                self._pending = None
                self._is_busy = True

            try:
                # Encode frame to JPEG in background thread
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                ok, enc = cv2.imencode(".jpg", frame, encode_param)
                if not ok:
                    raise RuntimeError("Failed to encode camera frame to JPEG.")
                image_bytes = enc.tobytes()

                # Generate description via service
                result = self.service.describe_scene(
                    image_bytes=image_bytes,
                    detections=detections,
                    context=context,
                )

                with self._condition:
                    self._latest_result = result
                    self._is_busy = False

                if self.on_completed is not None:
                    try:
                        self.on_completed(result)
                    except Exception as exc:
                        logger.warning("Scene description callback raised an error: %s", exc)

            except Exception as exc:
                logger.error("Background scene description error: %s", exc)
                with self._condition:
                    self._latest_result = {
                        "text": "Scene description encountered an internal error.",
                        "provider": "error",
                        "latency": 0.0,
                        "success": False,
                    }
                    self._is_busy = False

    def close(self, timeout: float = 2.0):
        """Cleanly shut down the background thread."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
