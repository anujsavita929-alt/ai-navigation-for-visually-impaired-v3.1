import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

def main():
    # ------------------------------------------------------------
    # Load model and processor (once)
    # ------------------------------------------------------------
    print("Loading processor and ZoeDepth model...")
    processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu")
    model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu")

    # Model / config information
    print(f"Model class: {model.__class__.__name__}")
    print(f"Config class: {model.config.__class__.__name__}")
    for field in ["depth_estimation_type", "depth_range", "max_depth"]:
        if hasattr(model.config, field):
            print(f"model.config.{field}: {getattr(model.config, field)}")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA available – using device cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA not available – falling back to CPU")
    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Open webcam (source 0)
    # ------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (source 0)")
        return

    # FPS measurement helpers
    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    print("\nPress 'q' to quit. Place an object at a measured distance from the camera.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break

        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Preprocess for ZoeDepth
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(**inputs)
            predicted = output.predicted_depth  # raw depth tensor
            # Resize to original frame size (H, W)
            if predicted.dim() == 4:
                resized = torch.nn.functional.interpolate(
                    predicted, size=pil_img.size[::-1], mode="bicubic", align_corners=False
                )
                depth = resized.squeeze(0).squeeze(0)
            else:
                resized = torch.nn.functional.interpolate(
                    predicted.unsqueeze(1), size=pil_img.size[::-1], mode="bicubic", align_corners=False
                )
                depth = resized.squeeze(0).squeeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.time() - start

        # Compute central 20% ROI median (depth is a torch Tensor on device)
        depth_cpu = depth.detach().cpu().numpy()
        h, w = depth_cpu.shape
        roi_h = int(0.20 * h)
        roi_w = int(0.20 * w)
        y_center, x_center = h // 2, w // 2
        y0 = max(y_center - roi_h // 2, 0)
        y1 = min(y_center + roi_h // 2, h)
        x0 = max(x_center - roi_w // 2, 0)
        x1 = min(x_center + roi_w // 2, w)
        roi = depth_cpu[y0:y1, x0:x1]
        roi_median = np.median(roi)

        # FPS update (average over last second)
        frame_count += 1
        cur_time = time.time()
        if cur_time - prev_time >= 1.0:
            fps = frame_count / (cur_time - prev_time)
            prev_time = cur_time
            frame_count = 0

        # Overlay information on the original BGR frame
        overlay_texts = [
            f"ZoeDepth center: {roi_median:.2f} m",
            f"Inference time: {inference_time*1000:.1f} ms",
            f"FPS: {fps:.1f}",
            "Place an object at a measured distance from the camera",
        ]
        y_text = 30
        for txt in overlay_texts:
            cv2.putText(
                frame_bgr,
                txt,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y_text += 30

        cv2.imshow("ZoeDepth Realtime Diagnostic", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
