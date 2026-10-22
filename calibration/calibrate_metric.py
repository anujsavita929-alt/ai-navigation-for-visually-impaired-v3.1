import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Lazy import to avoid heavy dependencies unless CLI is used.

def _load_depth_estimator(model_id: str | None, device: str | None):
    """Create a DepthEstimator instance.

    The DepthEstimator constructor expects ``model_id`` (the model repo identifier).
    """
    from perception.depth_estimator import DepthEstimator

    return DepthEstimator(model_id=model_id, device=device)


def _median_raw_depth(depth_map: np.ndarray, roi_fraction: float = 0.1) -> float:
    """Return the median raw depth value from the central ROI.

    Args:
        depth_map: 2‑D numpy array of raw depth values (inverse depth).
        roi_fraction: Fraction of the image size to use for the central ROI.
    """
    h, w = depth_map.shape
    cx, cy = w // 2, h // 2
    rw, rh = int(w * roi_fraction // 2), int(h * roi_fraction // 2)
    roi = depth_map[cy - rh : cy + rh, cx - rw : cx + rw]
    if roi.size == 0:
        raise ValueError("ROI has zero size – cannot compute median depth.")
    return float(np.median(roi))


def _fit_calibration(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Fit ``scale`` and ``offset`` for ``distance = scale / raw + offset``.

    Uses a linear least‑squares fit on ``x = 1/raw`` and ``y = distance``.
    """
    if len(samples) < 2:
        raise ValueError("At least two calibration samples are required.")
    xs, ys = [], []
    for raw, dist in samples:
        if raw <= 0 or not np.isfinite(raw):
            raise ValueError(f"Invalid raw depth value {raw!r} in calibration sample.")
        if dist <= 0 or not np.isfinite(dist):
            raise ValueError(f"Invalid measured distance {dist!r} in calibration sample.")
        xs.append(1.0 / raw)
        ys.append(dist)
    A = np.column_stack([xs, np.ones_like(xs)])
    y = np.array(ys)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, offset = coeffs.tolist()
    return scale, offset


def _save_calibration(calib_path: Path, scale: float, offset: float) -> None:
    """Write the calibration JSON file matching the loader expectations."""
    data = {"scale": scale, "offset": offset, "valid": True}
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logging.info("Calibration saved to %s", calib_path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect depth calibration samples and fit a metric conversion. "
            "Run this with the SAME camera that will be used by the application."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a directory containing calibration images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Depth Anything model identifier (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of calibration samples to collect (minimum 2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "calibration.json"),
        help="Path where the calibration JSON will be written.",
    )
    parser.add_argument(
        "--roi-fraction",
        type=float,
        default=0.1,
        help="Fraction of the image (center) used for median depth extraction.",
    )
    args = parser.parse_args(argv)

    if args.samples < 2:
        parser.error("At least two samples are required for calibration.")

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        logging.error("Source path %s is not a directory.", source_dir)
        return 1
    # Gather image files deterministically.
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if len(image_paths) < args.samples:
        logging.error(
            "Not enough images in %s: need %d, found %d.", source_dir, args.samples, len(image_paths)
        )
        return 1

    estimator = _load_depth_estimator(args.model, args.device)
    samples: List[Tuple[float, float]] = []

    for i in range(args.samples):
        img_path = image_paths[i]
        print(f"Sample {i+1}/{args.samples}: using image {img_path.name}")
        try:
            dist_str = input(f"Enter known distance in metres for {img_path.name}: ")
        except EOFError:
            logging.error("No input received – aborting calibration.")
            return 1
        try:
            distance_m = float(dist_str)
        except ValueError:
            logging.error("Invalid distance value: %s", dist_str)
            return 1
        if distance_m <= 0:
            logging.error("Distance must be positive.")
            return 1
        import cv2
        frame = cv2.imread(str(img_path))
        if frame is None:
            logging.error("Failed to read image %s", img_path)
            return 1
        # Run estimation – depth estimator stores the raw map internally.
        _ = estimator.estimate(frame)
        if not hasattr(estimator, "last_raw_map"):
            logging.error("DepthEstimator did not expose raw depth map.")
            return 1
        try:
            median_raw = _median_raw_depth(estimator.last_raw_map, roi_fraction=args.roi_fraction)
        except Exception as exc:
            logging.error("Failed to compute median raw depth: %s", exc)
            return 1
        logging.info(
            "Sample %d: known distance=%.3f m, median raw depth=%.6f",
            i + 1,
            distance_m,
            median_raw,
        )
        samples.append((median_raw, distance_m))

    try:
        scale, offset = _fit_calibration(samples)
    except Exception as exc:
        logging.error("Calibration fitting failed: %s", exc)
        return 1
    logging.info("Fitted calibration: scale=%.6f, offset=%.6f", scale, offset)
    _save_calibration(Path(args.output), scale, offset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
