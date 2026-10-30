"""Load YAML configuration over validated built-in defaults."""
from pathlib import Path
import logging

DEFAULTS = {
    "camera": {"source": "0", "width": 640, "height": 480,
               "reconnect_attempts": 3, "reconnect_delay_seconds": 1.0},
    "perception": {"detector_model": "yolov8n.pt", "detector_backend": "yolo",
                   "world_model": "yolov8s-worldv2.pt", "world_classes": ["person", "chair", "bottle", "door"],
                   "confidence": 0.45, "image_size": 640, "device": "auto", "enable_depth": True,
                   "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
                   "depth_bounds_ema_alpha": 0.12, "depth_every_n_frames": 3, "enable_tracking": True},
    "zoedepth": {
        "enabled": True,
        "model": "Intel/zoedepth-nyu",
        "distance_scale": 0.5,
        "distance_offset": 0.0,
        "device": "auto",
    },
    "navigation": {"near_relative_depth_threshold": 0.72, "large_box_area_threshold": 0.12,
                   "corridor_box_area_threshold": 0.025, "corridor_y_start": 0.48,
                   "speech_cooldown_seconds": 3.0, "target_guidance_interval_seconds": 2.0,
                   "require_confident_free_space": True},
    "voice": {"enabled": True, "speech_recognition": "vosk",
              "vosk_model_path": "models/vosk-model-small-en-us-0.15", "tts_enabled": True, "tts_rate": 175},
    "scene_description": {
        "enabled": True, "primary_provider": "gemini", "fallback_provider": "groq",
        "gemini_model": "gemini-2.5-flash", "groq_model": "llama-3.2-11b-vision-preview",
        "timeout_seconds": 5.0, "cooldown_seconds": 2.5, "max_retries": 1, "jpeg_quality": 80,
    },
    "app": {"show_depth_window": True, "show_dashboard": True, "log_level": "INFO"},
}


def load_config(path="config/settings.yaml"):
    config = {section: values.copy() for section, values in DEFAULTS.items()}
    config_path = Path(path)
    if not config_path.exists():
        logging.warning("Config file %s not found; using defaults", config_path)
        return config
    try:
        import yaml
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("top-level YAML value must be a mapping")
        for section, values in loaded.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            elif section not in config:
                logging.warning("Ignoring unknown config section %r", section)
    except ImportError:
        logging.warning("PyYAML not installed; using built-in defaults")
    except Exception as exc:
        logging.warning("Could not read config %s (%s); using defaults", config_path, exc)
    return config
