"""Comprehensive tests for AI Scene Description feature."""
import os
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from perception.detector import Detection
from scene_description.providers.base import BaseSceneProvider
from scene_description.providers.gemini_provider import GeminiSceneProvider
from scene_description.providers.groq_provider import GroqSceneProvider
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

    def test_gemini_provider_availability(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini = GeminiSceneProvider()
            self.assertFalse(gemini.is_available())

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}):
            gemini = GeminiSceneProvider()
            self.assertTrue(gemini.is_available())

    def test_groq_provider_availability(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            groq = GroqSceneProvider()
            self.assertFalse(groq.is_available())

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake_test_key"}):
            groq = GroqSceneProvider()
            self.assertTrue(groq.is_available())

    def test_gemini_provider_mock_describe(self):
        gemini = GeminiSceneProvider(api_key="fake_key")
        mock_response = MagicMock()
        mock_response.text = "A doorway is on your left and a table is ahead."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        gemini._client = mock_client

        text = gemini.describe(self.test_jpeg)
        self.assertEqual(text, "A doorway is on your left and a table is ahead.")
        self.assertTrue(mock_client.models.generate_content.called)

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

    def test_scene_service_primary_success(self):
        primary = DummyProvider(name="gemini", response="Gemini says hello")
        fallback = DummyProvider(name="groq", response="Groq says hello")
        local = DummyProvider(name="local", response="Local says hello")

        service = SceneService(
            gemini_provider=primary,
            groq_provider=fallback,
            local_provider=local,
            primary_name="gemini",
            cooldown_seconds=0.0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "gemini")
        self.assertEqual(res["text"], "Gemini says hello")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 0)

    def test_scene_service_fallback_to_groq_when_gemini_fails(self):
        primary = DummyProvider(name="gemini", raise_error=True)
        fallback = DummyProvider(name="groq", response="Groq fallback output")
        local = DummyProvider(name="local", response="Local output")

        service = SceneService(
            gemini_provider=primary,
            groq_provider=fallback,
            local_provider=local,
            primary_name="gemini",
            cooldown_seconds=0.0,
            max_retries=0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "groq")
        self.assertEqual(res["text"], "Groq fallback output")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 1)

    def test_scene_service_fallback_to_local_when_cloud_unavailable(self):
        primary = DummyProvider(name="gemini", available=False)
        fallback = DummyProvider(name="groq", available=False)
        local = DummyProvider(name="local", response="Local rule output")

        service = SceneService(
            gemini_provider=primary,
            groq_provider=fallback,
            local_provider=local,
            cooldown_seconds=0.0,
        )

        res = service.describe_scene(self.test_jpeg)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "local")
        self.assertEqual(res["text"], "Local rule output")
        self.assertEqual(primary.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(local.call_count, 1)

    def test_scene_service_cooldown(self):
        primary = DummyProvider(name="gemini", response="Hello")
        service = SceneService(
            gemini_provider=primary,
            cooldown_seconds=10.0,
        )

        first = service.describe_scene(self.test_jpeg)
        self.assertTrue(first["success"])

        second = service.describe_scene(self.test_jpeg)
        self.assertFalse(second["success"])
        self.assertEqual(second["provider"], "rate_limit")
        self.assertIn("wait a moment", second["text"].lower())

    def test_scene_worker_async_execution(self):
        local = DummyProvider(name="local", response="Async description done")
        service = SceneService(local_provider=local, primary_name="local", cooldown_seconds=0.0)

        results = []
        worker = SceneWorker(service=service, on_completed=lambda r: results.append(r))

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        accepted = worker.request(frame)
        self.assertTrue(accepted)

        # Wait up to 2 seconds for worker to process
        timeout = time.time() + 2.0
        while not results and time.time() < timeout:
            time.sleep(0.05)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Async description done")
        worker.close()

    def test_command_parser_scene_phrases(self):
        parser = CommandParser()
        phrases = [
            "describe the scene",
            "describe scene",
            "what is in front of me",
            "what is around me",
            "what is ahead",
            "tell me what you see",
            "what do you see",
            "describe surroundings",
            "describe the room",
        ]
        for p in phrases:
            res = parser.parse(p)
            self.assertEqual(res["intent"], "scene", f"Failed for phrase: {p}")

    def test_world_model_preserves_distance_m(self):
        world = WorldModel()
        det = Detection("chair", 0.9, (10, 10, 50, 50), distance_m=1.85)
        scene = world.build([det], (100, 100, 3))
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0].distance_m, 1.85)


if __name__ == "__main__":
    unittest.main()
