"""ZoeDepth metric depth estimation module."""
import logging
import time
import numpy as np


class ZoeDepthEstimator:
    """ZoeDepth metric depth estimator.

    Provides a compatible interface with the existing :class:`DepthEstimator`
    used elsewhere in the application. The estimator loads the HuggingFace
    ZoeDepth model, runs inference on a frame, and produces a
    normalised depth map for relative depth and visualisation as well as
    a calibrated metric depth map in metres.

    The correction parameters are configurable via *distance_scale* and
    *distance_offset* (default ``distance_scale=0.5``, ``distance_offset=0.0``).
    """

    def __init__(
        self,
        model: str = "Intel/zoedepth-nyu",
        device: str = "auto",
        distance_scale: float = 0.5,
        distance_offset: float = 0.0,
        ema_alpha: float = 0.12,
        **kwargs,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch

        if str(device).startswith("cuda"):
            self.device = str(device)
        elif device == "auto" and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        model_id = kwargs.get("model_id", model)
        scale = kwargs.get("scale", distance_scale)
        offset = kwargs.get("offset", distance_offset)

        self.model_id = str(model_id)
        self.scale = float(scale)
        self.offset = float(offset)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.calibrated = True

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        try:
            self.model = (
                AutoModelForDepthEstimation.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
        except Exception:
            from transformers import ZoeDepthForDepthEstimation
            self.model = (
                ZoeDepthForDepthEstimation.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )

        self._bounds = None
        self.last_raw_map = None
        self.last_map = None
        self.last_metric_map = None

        logging.info(
            "Loaded ZoeDepth model %s on %s (distance_scale=%.2f, distance_offset=%.2f)",
            self.model_id,
            self.device,
            self.scale,
            self.offset,
        )

    def estimate(self, frame):
        """Run inference on *frame* and return a tuple (norm_map, elapsed).

        Populates:
          - ``self.last_raw_map``: unscaled predicted metric depth in metres
          - ``self.last_metric_map``: scaled metric depth in metres (raw * scale + offset)
          - ``self.last_map``: normalised relative depth map (0.0 to 1.0; nearer is higher)
        """
        import cv2
        from PIL import Image

        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            if output.dim() == 2:
                output = output.unsqueeze(0).unsqueeze(0)
            elif output.dim() == 3:
                output = output.unsqueeze(1)
            raw = self.torch.nn.functional.interpolate(
                output,
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().detach().float().cpu().numpy()

        self.last_raw_map = raw.copy()

        # Apply configurable distance_scale and distance_offset (in metres)
        metric = raw * self.scale + self.offset
        metric = np.where(np.isfinite(metric), metric, np.nan)
        metric = np.where(metric < 0, np.nan, metric)
        self.last_metric_map = metric.astype(np.float32)

        # Normalise for relative depth and visualisation (nearer is higher/brighter)
        finite = metric[np.isfinite(metric)]
        if finite.size < 16:
            raise RuntimeError("Depth model returned an invalid depth map")

        lo, hi = np.percentile(finite, [5, 95]).astype(float)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        if self._bounds is None:
            self._bounds = (lo, hi)
        else:
            alpha = self.ema_alpha
            old_lo, old_hi = self._bounds
            self._bounds = (
                alpha * lo + (1 - alpha) * old_lo,
                alpha * hi + (1 - alpha) * old_hi,
            )
        stable_lo, stable_hi = self._bounds

        # Inverted normalisation: closer objects (smaller metric distance) -> higher value (near 1.0)
        norm = np.clip((stable_hi - metric) / max(stable_hi - stable_lo, 1e-6), 0.0, 1.0)
        norm[~np.isfinite(norm)] = 0.0
        self.last_map = norm.astype(np.float32)

        return self.last_map, time.perf_counter() - started

    def attach_depth(self, detections, depth_map=None, metric_map=None):
        """Attach relative depth and metric distance to YOLO detection objects.

        Calculates ``distance_m`` from the central 50% ROI of each detection bounding
        box to avoid boundary contamination.
        """
        if not isinstance(self, ZoeDepthEstimator):
            # Invoked as ZoeDepthEstimator.attach_depth(detections, depth_map, metric_map)
            actual_detections = self
            actual_depth_map = detections
            actual_metric_map = depth_map
        else:
            actual_detections = detections
            actual_depth_map = depth_map if depth_map is not None else getattr(self, "last_map", None)
            actual_metric_map = metric_map if metric_map is not None else getattr(self, "last_metric_map", None)

        if actual_depth_map is None and actual_metric_map is None:
            return actual_detections

        ref_map = actual_depth_map if actual_depth_map is not None else actual_metric_map
        h, w = ref_map.shape[:2]

        for detection in actual_detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # 1. Relative depth
            if actual_depth_map is not None:
                region = actual_depth_map[y1:y2, x1:x2]
                rel_vals = region[np.isfinite(region)]
                if rel_vals.size:
                    detection.relative_depth = float(np.median(rel_vals))

            # 2. Metric distance in metres from central 50% bounding-box ROI
            if actual_metric_map is not None:
                bw = x2 - x1
                bh = y2 - y1
                cx1 = x1 + int(0.25 * bw)
                cx2 = x2 - int(0.25 * bw)
                cy1 = y1 + int(0.25 * bh)
                cy2 = y2 - int(0.25 * bh)

                if cx2 > cx1 and cy2 > cy1:
                    metric_roi = actual_metric_map[cy1:cy2, cx1:cx2]
                    met_vals = metric_roi[np.isfinite(metric_roi)]
                    if met_vals.size:
                        detection.distance_m = float(np.median(met_vals))
                    else:
                        detection.distance_m = None
                else:
                    metric_region = actual_metric_map[y1:y2, x1:x2]
                    met_vals = metric_region[np.isfinite(metric_region)]
                    if met_vals.size:
                        detection.distance_m = float(np.median(met_vals))
                    else:
                        detection.distance_m = None
            else:
                detection.distance_m = None

        return actual_detections

    @staticmethod
    def colorize(depth_map):
        import cv2

        gray = np.uint8(np.clip(depth_map, 0, 1) * 255)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
