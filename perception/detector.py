"""Ultralytics object detector with optional YOLO-World open-vocabulary mode."""
from dataclasses import dataclass
import logging
import time


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple
    track_id: int | None = None
    relative_depth: float | None = None
    distance_m: float | None = None
    direction: str = "unknown"

    @property
    def center(self):
        return ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.45, image_size=640,
                 device="auto", backend="yolo", world_model_path="yolov8s-worldv2.pt",
                 world_classes=None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install requirements.txt first.") from exc
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device_arg = None if device == "auto" else device
        self.backend = str(backend).lower()
        self.model_path = model_path
        self.model = None
        if self.backend in {"world", "yolo-world", "yolo_world"}:
            try:
                from ultralytics import YOLOWorld
                self.model = YOLOWorld(world_model_path)
                classes = [str(x).strip() for x in (world_classes or []) if str(x).strip()]
                if not classes:
                    raise ValueError("YOLO-World requires perception.world_classes in settings.yaml")
                self.model.set_classes(classes)
                self.world_classes = classes
                self.backend = "yolo-world"
                logging.info("Loaded YOLO-World model %s with %d classes", world_model_path, len(classes))
            except Exception as exc:
                logging.warning("YOLO-World unavailable (%s); falling back to %s", exc, model_path)
                self.model = YOLO(model_path)
                self.backend = "yolo"
        else:
            self.model = YOLO(model_path)
        logging.info("Loaded detector backend=%s model=%s", self.backend, model_path)

    def predict(self, frame):
        started = time.perf_counter()
        kwargs = {"conf": self.confidence, "imgsz": self.image_size, "verbose": False}
        if self.device_arg is not None:
            kwargs["device"] = self.device_arg
        results = self.model.predict(frame, **kwargs)
        if not results:
            return [], time.perf_counter() - started
        result = results[0]
        output = []
        if result.boxes is not None:
            names = result.names
            for box in result.boxes:
                coords = tuple(float(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                output.append(Detection(str(names.get(class_id, class_id)), confidence, coords))
        return output, time.perf_counter() - started


    def ensure_target_class(self, target):
        """Add a requested target to active YOLO-World classes, or report unsupported."""
        target = str(target).strip()
        if not target:
            return False
        if self.backend == "yolo-world":
            current = getattr(self, "world_classes", None)
            if current is None:
                names = getattr(self.model, "names", {})
                current = list(names.values()) if isinstance(names, dict) else list(names)
            if not any(target.casefold() == str(name).casefold() for name in current):
                current = list(current) + [target]
                try:
                    self.model.set_classes(current)
                except Exception as exc:
                    logging.warning("Could not add YOLO-World target %r: %s", target, exc)
                    return False
            self.world_classes = current
            return True
        names = getattr(self.model, "names", {})
        values = list(names.values()) if isinstance(names, dict) else list(names)
        return any(target.casefold() == str(name).casefold() for name in values)

    @staticmethod
    def annotate(frame, detections):
        import cv2
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)
            # Build label: "label | xx.xx m | rel yy.yy"
            label_parts = [detection.label]
            if detection.distance_m is not None:
                label_parts.append(f"{detection.distance_m:.2f} m")
            if detection.relative_depth is not None:
                label_parts.append(f"rel {detection.relative_depth:.2f}")
            label = " | ".join(label_parts)
            if detection.track_id is not None:
                label = f"#{detection.track_id} " + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 2)
        return frame
