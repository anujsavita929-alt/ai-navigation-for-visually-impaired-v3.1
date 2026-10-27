"""Lightweight scene representation derived only from current detections."""
from dataclasses import dataclass


@dataclass
class SceneObject:
    label: str
    confidence: float
    bbox: tuple
    center_x: float
    center_y: float
    direction: str
    relative_depth: float | None
    track_id: int | None
    distance_m: float | None = None


class WorldModel:
    def build(self, detections, frame_shape):
        height, width = frame_shape[:2]
        objects = []
        for detection in detections:
            center_x, center_y = detection.center
            ratio = center_x / max(width, 1)
            if ratio < 0.38:
                direction = "left"
            elif ratio > 0.62:
                direction = "right"
            else:
                direction = "centre"
            detection.direction = direction
            objects.append(
                SceneObject(
                    detection.label,
                    detection.confidence,
                    detection.xyxy,
                    center_x / max(width, 1),
                    center_y / max(height, 1),
                    direction,
                    detection.relative_depth,
                    detection.track_id,
                    getattr(detection, "distance_m", None),
                )
            )
        return {"width": width, "height": height, "objects": objects}

    @staticmethod
    def find_target(scene, target):
        normalized_target = target.lower().strip()
        return [
            obj for obj in scene["objects"]
            if normalized_target in obj.label.lower()
        ]
