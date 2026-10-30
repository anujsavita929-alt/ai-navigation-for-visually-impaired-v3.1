#!/usr/bin/env python
"""
Diagnostic script: YOLOv8n object detection + ZoeDepth metric depth estimation.

- Loads YOLOv8n model (ultralytics) and ZoeDepth model (Intel/zoedepth-nyu).
- Uses CUDA if available, otherwise CPU.
- Opens webcam (source 0) and processes each frame.
- For each detection with confidence >= 0.45, computes median depth inside the central 50% of the bounding box.
- Draws bounding boxes and annotates with class name and distance (m).
- Shows device info, inference time, and FPS.
- Press 'q' to quit.

This script is self‑contained and does not modify any production code.
"""

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation


def get_device() -> torch.device:
    """Return ``cuda:0`` if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_models(device: torch.device):
    """Load YOLOv8n and ZoeDepth models onto *device*.

    Returns:
        yolo_model (YOLO): Ultralytics YOLOv8n model.
        zoe_processor (AutoImageProcessor): Processor for ZoeDepth.
        zoe_model (ZoeDepthForDepthEstimation): Depth estimation model.
    """
    # YOLOv8n – weight file expected in repo root (yolov8n.pt)
    yolo_model = YOLO("yolov8n.pt")

    # ZoeDepth – HuggingFace Transformers
    zoe_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu")
    zoe_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu").to(device)
    zoe_model.eval()
    return yolo_model, zoe_processor, zoe_model


def compute_depth(frame: np.ndarray, processor, model, device: torch.device) -> np.ndarray:
    """Run ZoeDepth on *frame* and return a depth map sized to the original frame.

    Args:
        frame: BGR image from OpenCV.
        processor: AutoImageProcessor for ZoeDepth.
        model: ZoeDepth model.
        device: torch device.
    Returns:
        depth_map (np.ndarray): 2‑D array (H, W) with metric depth in metres.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_resized


def median_depth_in_roi(depth_map: np.ndarray, box: np.ndarray) -> float:
    """Calculate median depth inside the central 50 % of *box*.

    *box* is an array-like ``[x1, y1, x2, y2]`` in pixel coordinates.
    """
    x1, y1, x2, y2 = box.astype(int)
    w = x2 - x1
    h = y2 - y1
    cx1 = x1 + int(0.25 * w)
    cx2 = x2 - int(0.25 * w)
    cy1 = y1 + int(0.25 * h)
    cy2 = y2 - int(0.25 * h)
    roi = depth_map[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return float('nan')
    return float(np.median(roi))


def main():
    print("[Diagnostic] YOLOv8n + ZoeDepth metric depth estimation (stand‑alone script)")
    device = get_device()
    print(f"ZoeDepth device: {device}")

    yolo_model, zoe_processor, zoe_model = load_models(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam (source 0). Exiting.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Frame capture failed, exiting loop.")
            break

        start_inference = time.time()
        results = yolo_model.predict(frame, device=device, conf=0.45, iou=0.45, verbose=False)
        result = results[0]
        detections = result.boxes
        depth_map = compute_depth(frame, zoe_processor, zoe_model, device)
        inference_time = (time.time() - start_inference) * 1000

        for box in detections:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if confidence < 0.45:
                continue
            xyxy = box.xyxy.squeeze().cpu().numpy()
            raw_dist = median_depth_in_roi(depth_map, xyxy)
            corrected_dist = raw_dist / 2.0
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = result.names.get(cls_id, "obj") if hasattr(result, "names") else str(cls_id)
            # display corrected distance on the bounding box
            label = f"{class_name} | {corrected_dist:.2f} m"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 4), (x1 + label_size[0] + 4, label_y + 4), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # store latest raw and corrected values for debugging overlay
            debug_raw = raw_dist
            debug_corrected = corrected_dist

        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur_time - prev_time))
        prev_time = cur_time

        info_text = f"Device: {device} | Inference: {inference_time:.1f} ms | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Debug overlay for raw and corrected ZoeDepth values (show last detection)
        if 'debug_raw' in locals() and debug_raw is not None:
            debug_text = f"Raw ZoeDepth: {debug_raw:.2f} m | Corrected: {debug_corrected:.2f} m"
            cv2.putText(frame, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("YOLOv8n + ZoeDepth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
