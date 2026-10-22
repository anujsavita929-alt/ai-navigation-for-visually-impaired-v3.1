import json
import os
import logging
from pathlib import Path
from typing import Tuple

DEFAULT_CALIBRATION_PATH = Path(__file__).parent.parent / "config" / "calibration.json"

def load_calibration(calibration_path: str | os.PathLike | None = None) -> Tuple[float, float, bool]:
    """Load metric calibration parameters.

    Returns a tuple ``(scale, offset, calibrated)``.
    If the calibration file does not exist, is malformed, or does not contain a
    ``"valid": true`` flag, the function returns the default ``scale=1.0`` and
    ``offset=0.0`` with ``calibrated=False``.
    """
    path = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale = float(data.get("scale", 1.0))
        offset = float(data.get("offset", 0.0))
        calibrated = bool(data.get("valid", False))
        if calibrated:
            logging.info("Loaded depth calibration from %s (scale=%s, offset=%s)", path, scale, offset)
        else:
            logging.warning("Calibration file %s is present but marked invalid; using defaults.", path)
        return scale, offset, calibrated
    except FileNotFoundError:
        logging.info("Calibration file %s not found; using uncalibrated defaults.", path)
        return 1.0, 0.0, False
    except Exception as exc:
        logging.warning("Failed to load calibration from %s: %s; using defaults.", path, exc)
        return 1.0, 0.0, False
