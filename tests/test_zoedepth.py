import unittest
from unittest.mock import patch
import numpy as np

from config import load_config
from perception.depth_estimator import DepthEstimator, create_depth_estimator
from perception.detector import Detection
from perception.zoedepth_estimator import ZoeDepthEstimator


class TestZoeDepthIntegration(unittest.TestCase):
    def test_cpu_zoedepth_when_cuda_unavailable(self):
        cfg = load_config()
        self.assertTrue(cfg.get("zoedepth", {}).get("enabled", False))

        with patch("torch.cuda.is_available", return_value=False):
            with patch("transformers.AutoImageProcessor.from_pretrained"), \
                 patch("transformers.AutoModelForDepthEstimation.from_pretrained") as mock_model:
                # Mock model loading and evaluation
                mock_model.return_value.to.return_value.eval.return_value = mock_model.return_value
                estimator = create_depth_estimator(cfg)

        # Should use ZoeDepth on CPU
        self.assertIsInstance(estimator, ZoeDepthEstimator)
        self.assertEqual(estimator.device, "cpu")
        self.assertEqual(estimator.scale, 0.5)
        self.assertEqual(estimator.offset, 0.0)

    def test_gpu_zoedepth_path_when_cuda_available(self):
        cfg = load_config()
        cfg["zoedepth"]["distance_scale"] = 0.5
        cfg["zoedepth"]["distance_offset"] = 0.1

        with patch("torch.cuda.is_available", return_value=True):
            with patch("transformers.AutoImageProcessor.from_pretrained"), \
                 patch("transformers.AutoModelForDepthEstimation.from_pretrained") as mock_model:
                mock_model.return_value.to.return_value.eval.return_value = mock_model.return_value
                estimator = create_depth_estimator(cfg)

        self.assertIsInstance(estimator, ZoeDepthEstimator)
        self.assertEqual(estimator.scale, 0.5)
        self.assertEqual(estimator.offset, 0.1)
        self.assertEqual(estimator.device, "cuda:0")

    def test_zoedepth_disabled_uses_depth_anything(self):
        cfg = load_config()
        cfg["zoedepth"]["enabled"] = False

        with patch("torch.cuda.is_available", return_value=True):
            with patch("transformers.AutoImageProcessor.from_pretrained"), \
                 patch("transformers.AutoModelForDepthEstimation.from_pretrained") as mock_model:
                mock_model.return_value.to.return_value.eval.return_value = mock_model.return_value
                estimator = create_depth_estimator(cfg)

        self.assertIsInstance(estimator, DepthEstimator)
        self.assertNotIsInstance(estimator, ZoeDepthEstimator)
        self.assertEqual(estimator.device, "cuda:0")
        self.assertEqual(estimator.metric_scale, -0.39859968679274704)
        self.assertEqual(estimator.metric_offset, 1.18342519023626)

    def test_zoedepth_load_failure_falls_back_to_depth_anything(self):
        cfg = load_config()

        with patch("torch.cuda.is_available", return_value=True):
            # Simulate ZoeDepth failure during initialization
            with patch("perception.zoedepth_estimator.ZoeDepthEstimator", side_effect=RuntimeError("CUDA out of memory")):
                with patch("transformers.AutoImageProcessor.from_pretrained"), \
                     patch("transformers.AutoModelForDepthEstimation.from_pretrained") as mock_model:
                    mock_model.return_value.to.return_value.eval.return_value = mock_model.return_value
                    estimator = create_depth_estimator(cfg)

        self.assertIsInstance(estimator, DepthEstimator)
        self.assertNotIsInstance(estimator, ZoeDepthEstimator)
        self.assertEqual(estimator.device, "cuda:0")
        self.assertEqual(estimator.metric_scale, -0.39859968679274704)
        self.assertEqual(estimator.metric_offset, 1.18342519023626)

    def test_central_50_percent_roi_depth(self):
        # Create an uninitialized ZoeDepthEstimator instance via __new__
        estimator = ZoeDepthEstimator.__new__(ZoeDepthEstimator)
        estimator.scale = 0.5
        estimator.offset = 0.0

        # Construct a 100x100 metric depth map:
        # Everywhere is 10.0 m except the central 50% ROI [25:75, 25:75] which is 2.5 m
        metric_map = np.full((100, 100), 10.0, dtype=np.float32)
        metric_map[25:75, 25:75] = 2.5
        norm_map = np.full((100, 100), 0.8, dtype=np.float32)

        estimator.last_metric_map = metric_map
        estimator.last_map = norm_map

        det = Detection(label="person", confidence=0.9, xyxy=(0, 0, 100, 100))
        estimator.attach_depth([det], norm_map)

        # The central 50% ROI of [0, 0, 100, 100] is [25, 25, 75, 75]
        # Median inside that ROI should be exactly 2.5 m (not 10.0 m)
        self.assertIsNotNone(det.distance_m)
        self.assertAlmostEqual(det.distance_m, 2.5, places=2)
        self.assertAlmostEqual(det.relative_depth, 0.8, places=2)


if __name__ == "__main__":
    unittest.main()
