"""Comprehensive tests for AI Scene Description feature (Groq cloud and Local rule synthesizer)."""
import os
import time
import unittest
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from perception.detector import Detection
from scene_description.providers.base import BaseSceneProvider
from scene_description.providers.groq_provider import GroqSceneProvider, build_scene_prompt
from scene_description.providers.local_provider import LocalSceneProvider
from scene_description.scene_service import SceneService
from scene_description.scene_worker import SceneWorker
from spatial.world_model import WorldModel
from voice.command_parser import CommandParser


class DummyProvider(BaseSceneProvider):
    def __init__(self, name="dummy", available=True, response="dummy response", raise_error=False):
        self._name = name
        self._available = available
        self._response = response
        self._raise_error = raise_error
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def describe(self, image_bytes, detections=None, context=None) -> str:
        self.call_count += 1
        if self._raise_error:
            raise RuntimeError(f"{self._name} test error")
        return self._response


class TestSceneDescription(unittest.TestCase):
    def setUp(self):
        # 10x10 synthetic test JPEG image
        import cv2
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", frame)
        self.test_jpeg = enc.tobytes()

    def test_local_provider_with_detections(self):
        provider = LocalSceneProvider()
        d1 = Detection("chair", 0.9, (200, 200, 300, 300), direction="centre")
        d2 = Detection("person", 0.85, (50, 100, 150, 400), direction="left")
        d3 = Detection("bottle", 0.7, (500, 200, 550, 300), direction="right")

        text = provider.describe(self.test_jpeg, detections=[d1, d2, d3])
        self.assertIn("chair", text.lower())
        self.assertIn("directly ahead", text.lower())
        self.assertIn("person", text.lower())
        self.assertIn("left", text.lower())
        self.assertIn("bottle", text.lower())
        self.assertIn("right", text.lower())

    def test_local_provider_empty_detections(self):
        provider = LocalSceneProvider()
        text = provider.describe(self.test_jpeg, detections=[])
        self.assertIn("no major objects", text.lower())

    def test_groq_provider_availability(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            groq = GroqSceneProvider()
            self.assertFalse(groq.is_available())

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake_test_key"}):
            groq = GroqSceneProvider()
            self.assertTrue(groq.is_available())

    def test_groq_provider_mock_describe(self):
        groq = GroqSceneProvider(api_key="fake_key")
        mock_choice = MagicMock()
        mock_choice.message.content = "A clear hallway with a person walking on your right."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        groq._client = mock_client

        text = groq.describe(self.test_jpeg)
        self.assertEqual(text, "A clear hallway with a person walking on your right.")
        self.assertTrue(mock_client.chat.completions.create.called)

    def test_scene_service_groq_success(self):
        primary = DummyProvider(name="groq", response="Groq says hello")
        local = DummyProvider(name="local", response="Local says hello")

        service = SceneService(
            groq_provider=primary,
            local_provider=local,
            primary_name="groq",
            cooldown_seconds=0.0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "groq")
        self.assertEqual(res["text"], "Groq says hello")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(local.call_count, 0)

    def test_scene_service_fallback_to_local_when_groq_fails(self):
        primary = DummyProvider(name="groq", raise_error=True)
        local = DummyProvider(name="local", response="Local output")

        service = SceneService(
            groq_provider=primary,
            local_provider=local,
            primary_name="groq",
            cooldown_seconds=0.0,
            max_retries=0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "local")
        self.assertEqual(res["text"], "Local output")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(local.call_count, 1)

    def test_scene_service_fallback_to_local_when_cloud_unavailable(self):
        primary = DummyProvider(name="groq", available=False)
        local = DummyProvider(name="local", response="Local rule output")

        service = SceneService(
            groq_provider=primary,
            local_provider=local,
            cooldown_seconds=0.0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "local")
        self.assertEqual(res["text"], "Local rule output")
        self.assertEqual(primary.call_count, 0)
        self.assertEqual(local.call_count, 1)

    def test_scene_service_cooldown(self):
        primary = DummyProvider(name="groq", response="Hello")
        service = SceneService(
            groq_provider=primary,
            cooldown_seconds=5.0,
        )

        # First call succeeds
        res1 = service.describe_scene(self.test_jpeg)
        self.assertTrue(res1["success"])
        self.assertEqual(primary.call_count, 1)

        # Immediate second call is rate-limited
        res2 = service.describe_scene(self.test_jpeg)
        self.assertFalse(res2["success"])
        self.assertEqual(res2["provider"], "rate_limit")
        self.assertEqual(primary.call_count, 1)

        # Force bypasses cooldown
        res3 = service.describe_scene(self.test_jpeg, force=True)
        self.assertTrue(res3["success"])
        self.assertEqual(primary.call_count, 2)

    def test_scene_worker_async_execution(self):
        local = DummyProvider(name="local", response="Async description done")
        service = SceneService(primary_name="local", local_provider=local, cooldown_seconds=0.0)
        worker = SceneWorker(service=service)

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        accepted = worker.request(frame)
        self.assertTrue(accepted)

        # Wait for worker thread to process
        deadline = time.monotonic() + 3.0
        result = None
        while time.monotonic() < deadline:
            result = worker.poll_result()
            if result is not None:
                break
            time.sleep(0.05)

        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Async description done")
        worker.close(timeout=1.0)

    def test_world_model_preserves_distance_m(self):
        wm = WorldModel()
        det = Detection("chair", 0.95, (10, 20, 100, 200), distance_m=1.85, relative_depth=0.8)
        scene = wm.build([det], (480, 640))
        self.assertEqual(len(scene["objects"]), 1)
        obj = scene["objects"][0]
        self.assertEqual(obj.distance_m, 1.85)

    def test_command_parser_scene_phrases(self):
        parser = CommandParser()
        phrases = [
            "what is in front of me",
            "describe the scene",
            "what do you see",
            "tell me what is ahead",
            "what's in front of me",
        ]
        for phrase in phrases:
            parsed = parser.parse(phrase)
            self.assertEqual(parsed["intent"], "scene", f"Failed for phrase: '{phrase}'")

    def test_groq_decommissioned_model_classification(self):
        provider = GroqSceneProvider(api_key="gsk_fake_key_12345")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("The model is decommissioned.")
        provider._client = mock_client

        with self.assertRaises(RuntimeError) as ctx:
            provider.describe(self.test_jpeg)
        self.assertIn("decommissioned", str(ctx.exception))

    def test_worker_clean_shutdown(self):
        local = DummyProvider(name="local", response="Done")
        service = SceneService(primary_name="local", local_provider=local, cooldown_seconds=0.0)
        worker = SceneWorker(service=service)
        self.assertTrue(worker._thread.is_alive())
        worker.close(timeout=1.0)
        self.assertFalse(worker._thread.is_alive())

    def test_permanent_error_skips_retries_and_fast_failovers_to_local(self):
        """Authentication or invalid model errors must NOT retry and must failover immediately to local."""
        primary = DummyProvider(name="groq", raise_error=True)
        def raise_auth_error(*args, **kwargs):
            primary.call_count += 1
            raise RuntimeError("API_KEY_INVALID: 401 unauthorized")
        primary.describe = raise_auth_error

        local = DummyProvider(name="local", response="Local fallback")

        service = SceneService(
            groq_provider=primary,
            local_provider=local,
            primary_name="groq",
            cooldown_seconds=0.0,
            max_retries=2,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "local")
        self.assertEqual(res["text"], "Local fallback")
        # Critical verification: called exactly 1 time, did NOT waste 2 retries!
        self.assertEqual(primary.call_count, 1)

    def test_prompt_builder_includes_distances_and_corridor_context(self):
        """Verify prompt builder grounds LLM input in 3D distances and corridor hazard warnings."""
        det = Detection("chair", 0.95, (100, 100, 200, 200), direction="centre", distance_m=1.8)
        ctx = {"free_space": {"regions": {"centre": "blocked"}}}

        prompt = build_scene_prompt([det], ctx)
        self.assertIn("chair", prompt)
        self.assertIn("1.8m", prompt)
        self.assertIn("centre", prompt)
        self.assertIn("obstructed", prompt.lower())

    def test_scene_worker_duplicate_request_protection(self):
        """Submitting a second request while worker is busy must return False."""
        local = DummyProvider(name="local", response="Processed")
        service = SceneService(primary_name="local", local_provider=local, cooldown_seconds=0.0)
        worker = SceneWorker(service=service)

        try:
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            # Fill pending frame
            with worker._condition:
                worker._is_busy = True
            rejected = worker.request(frame)
            self.assertFalse(rejected, "Worker accepted request while busy.")
        finally:
            with worker._condition:
                worker._is_busy = False
            worker.close(timeout=1.0)

    def test_groq_timeout_configured_and_passed(self):
        """Verify timeout is accurately configured in GroqSceneProvider and SceneService."""
        cfg = {"scene_description": {"timeout_seconds": 12.0, "groq_model": "qwen/qwen3.8-27b"}}
        service = SceneService.from_config(cfg)
        self.assertEqual(service.groq.timeout_seconds, 12.0)
        self.assertEqual(service.groq.model_name, "qwen/qwen3.8-27b")

    def test_groq_503_transient_retry_and_local_fallback(self):
        """Verify transient 503/504 errors trigger fast failover to local synthesizer."""
        primary = DummyProvider(name="groq", raise_error=True)
        def raise_503_error(*args, **kwargs):
            primary.call_count += 1
            raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")
        primary.describe = raise_503_error

        local = DummyProvider(name="local", response="Local response after 503 failover")

        service = SceneService(
            groq_provider=primary,
            local_provider=local,
            primary_name="groq",
            cooldown_seconds=0.0,
            max_retries=1,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "local")
        self.assertEqual(res["text"], "Local response after 503 failover")
        # Fast failover skips redundant retry on server overload
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(local.call_count, 1)


if __name__ == "__main__":
    unittest.main()
