"""Monocular depth estimation with relative and calibrated metric depth."""
from __future__ import annotations

import logging
import os
import time
import numpy as np


class DepthEstimator:
    def __init__(
        self,
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        device="auto",
        ema_alpha=0.12,
        depth_mode="relative",
        metric_scale=1.0,
        metric_offset=0.0,
        calibration_path: str | os.PathLike | None = None,
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "Depth dependencies missing. Install requirements.txt and enable depth."
            ) from exc

        self.torch = torch

        if str(device).startswith("cuda"):
            self.device = str(device)
        elif device == "auto" and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        self.model_id = model_id
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))

        self.depth_mode = str(depth_mode).lower()
        # Load calibration if available; overrides metric_scale/offset when calibrated
        from calibration import load_calibration
        cal_scale, cal_offset, calibrated = load_calibration(calibration_path)
        if calibrated:
            self.metric_scale = float(cal_scale)
            self.metric_offset = float(cal_offset)
        else:
            self.metric_scale = float(metric_scale)
            self.metric_offset = float(metric_offset)
        self.calibrated = calibrated
        self._bounds = None
        self.last_map = None
        self.last_metric_map = None

        logging.info(
            "Loaded depth model %s on %s (mode=%s)",
            model_id,
            self.device,
            self.depth_mode,
        )

    def estimate(self, frame):
        import cv2
        from PIL import Image

        started = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            raw = self.torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().detach().float().cpu().numpy()
            self.last_raw_map = raw.copy()

        finite = raw[np.isfinite(raw)]

        if finite.size < 16:
            raise RuntimeError("Depth model returned an invalid depth map")

        # Depth Anything produces an inverse-depth-like signal.
        # We retain the existing relative representation for visualization
        # and backward compatibility.
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

        normalized = np.clip(
            (raw - stable_lo) / max(stable_hi - stable_lo, 1e-6),
            0.0,
            1.0,
        )

        normalized[~np.isfinite(normalized)] = 0.0

        self.last_map = normalized.astype(np.float32)

        # Convert the model's inverse‑depth signal into a calibrated distance map.
        # Depth Anything outputs an inverse‑depth‑like map where larger values mean nearer.
        # We treat the raw output directly as inverse depth.
        # Convert raw inverse‑depth to calibrated metric depth.
        # The original calibration uses a negative scale (distance = scale / raw + offset).
        # To obtain a monotonic increase of distance with increasing physical distance we flip the sign of the scale.
        inverse_depth = np.maximum(raw, 1e-6)
        # Negate the scale term so that farther objects (smaller raw depth) produce larger distances.
        metric_map = -self.metric_scale / inverse_depth + self.metric_offset
        # Preserve invalid values as NaN so callers can detect missing data.
        metric_map = np.where(np.isfinite(metric_map), metric_map, np.nan)
        # Enforce non‑negative distances (negative after offset is treated as unavailable).
        metric_map = np.where(metric_map < 0, np.nan, metric_map)

        self.last_metric_map = metric_map.astype(np.float32)

        return self.last_map, time.perf_counter() - started

    @staticmethod
    def colorize(depth_map):
        import cv2

        gray = np.uint8(
            np.clip(depth_map, 0, 1) * 255
        )

        return cv2.applyColorMap(
            gray,
            cv2.COLORMAP_INFERNO,
        )

    def attach_depth(self, detections, depth_map):
        """Attach relative depth and calibrated metric distance to detections."""

        if depth_map is None or getattr(depth_map, "size", 0) == 0:
            return detections

        h, w = depth_map.shape[:2]
        metric_map = self.last_metric_map

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)
            # Clamp coordinates to frame bounds
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            # Relative depth (already normalized 0‑1)
            region = depth_map[y1:y2, x1:x2]
            rel_vals = region[np.isfinite(region)]
            if rel_vals.size:
                detection.relative_depth = float(np.median(rel_vals))
            # Metric depth – only if calibrated and metric map present
            if getattr(self, "calibrated", False) and metric_map is not None:
                metric_region = metric_map[y1:y2, x1:x2]
                met_vals = metric_region[np.isfinite(metric_region)]
                if met_vals.size:
                    detection.distance_m = float(np.median(met_vals))
                else:
                    detection.distance_m = None
            else:
                detection.distance_m = None
        return detections