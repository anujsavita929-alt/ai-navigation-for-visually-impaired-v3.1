"""Conservative image-region obstruction heuristic; not floor or occupancy mapping."""
class FreeSpaceAnalyzer:
    def __init__(self, near_threshold=0.72, large_box_area=0.12, corridor_area=0.025,
                 corridor_y_start=0.48, metric_warning_distance_m=1.5, metric_critical_distance_m=0.8):
        self.near_threshold = float(near_threshold)
        self.large_box_area = float(large_box_area)
        self.corridor_area = float(corridor_area)
        self.corridor_y_start = float(corridor_y_start)
        self.metric_warning_distance_m = float(metric_warning_distance_m)
        self.metric_critical_distance_m = float(metric_critical_distance_m)

    def analyze(self, detections, frame_shape):
        h, w = frame_shape[:2]
        regions = {"left": [], "centre": [], "right": []}
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            box_w, box_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0 or box_w <= 0 or box_h <= 0:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            key = "left" if cx < w * 0.38 else ("right" if cx > w * 0.62 else "centre")
            area = min(1.0, (box_w * box_h) / (w * h))
            lower_corridor = (cy >= h * self.corridor_y_start and
                              w * 0.38 <= cx <= w * 0.62 and
                              x1 < w * 0.72 and x2 > w * 0.28)
            depth = detection.relative_depth
            # Prefer metric distance when available, fallback to relative depth.
            near_by_depth = lower_corridor and (
                (detection.distance_m is not None and detection.distance_m <= self.metric_warning_distance_m) or
                (detection.distance_m is None and depth is not None and depth >= self.near_threshold)
            ) and area >= self.corridor_area
            large_in_corridor = lower_corridor and area >= self.large_box_area
            close = near_by_depth or large_in_corridor
            regions[key].append({"label": detection.label, "close": close, "area": area,
                                 "confidence": detection.confidence,
                                 "relative_depth": depth, "in_walking_corridor": lower_corridor})
        status = {}
        for key, items in regions.items():
            if not items:
                status[key] = "unknown"
            elif any(item["close"] for item in items):
                status[key] = "blocked"
            else:
                status[key] = "caution"
        return {"regions": status, "evidence": regions,
                "warning": "Heuristic only: camera regions are not verified traversable space."}
