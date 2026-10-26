"""Offline rule-based scene synthesizer using YOLO detections and spatial analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from scene_description.providers.base import BaseSceneProvider


class LocalSceneProvider(BaseSceneProvider):
    """Generates deterministic, concise scene descriptions completely offline."""

    @property
    def provider_name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        return True

    def describe(
        self,
        image_bytes: bytes,
        detections: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        detections = detections or []
        context = context or {}

        if not detections:
            free_space_status = context.get("free_space", {}).get("regions", {})
            centre = free_space_status.get("centre", "unknown")
            if centre == "unknown":
                return "No major objects detected. Path safety cannot be verified."
            return "No prominent objects detected in front of you."

        # Group detections by spatial orientation
        by_direction: Dict[str, List[str]] = {"left": [], "centre": [], "right": []}
        corridor_objects: List[str] = []

        # Sort detections by confidence / size
        sorted_dets = sorted(
            detections,
            key=lambda d: (
                getattr(d, "confidence", 0.0)
                * (
                    max(0.0, d.xyxy[2] - d.xyxy[0]) * max(0.0, d.xyxy[3] - d.xyxy[1])
                    if hasattr(d, "xyxy")
                    else 1.0
                )
            ),
            reverse=True,
        )

        for det in sorted_dets[:8]:
            label = getattr(det, "label", "object")
            direction = getattr(det, "direction", "centre")
            if direction not in by_direction:
                direction = "centre"
            by_direction[direction].append(label)

            # Check if this object is in the walking corridor
            if hasattr(det, "in_walking_corridor") and det.in_walking_corridor:
                corridor_objects.append(label)

        clauses: List[str] = []

        # Centre / Immediate path
        if by_direction["centre"]:
            unique_centre = list(dict.fromkeys(by_direction["centre"]))[:2]
            centre_str = " and ".join(f"a {lbl}" for lbl in unique_centre)
            clauses.append(f"{centre_str} is directly ahead")
        elif corridor_objects:
            unique_corridor = list(dict.fromkeys(corridor_objects))[:2]
            corr_str = " and ".join(f"a {lbl}" for lbl in unique_corridor)
            clauses.append(f"{corr_str} is in your walking path")

        # Left
        if by_direction["left"]:
            unique_left = list(dict.fromkeys(by_direction["left"]))[:2]
            left_str = " and ".join(f"a {lbl}" for lbl in unique_left)
            clauses.append(f"{left_str} on your left")

        # Right
        if by_direction["right"]:
            unique_right = list(dict.fromkeys(by_direction["right"]))[:2]
            right_str = " and ".join(f"a {lbl}" for lbl in unique_right)
            clauses.append(f"{right_str} on your right")

        if not clauses:
            return "No prominent objects identified in the current view."

        description = ", ".join(clauses).capitalize() + "."
        return description
