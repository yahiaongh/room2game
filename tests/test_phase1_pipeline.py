import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from openai import APITimeoutError
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import main
from services import metrics_service
from services.polygon_service import mask_png_to_polygon


def make_mask(size: tuple[int, int], bounds: tuple[int, int, int, int]) -> bytes:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(bounds, fill=255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def valid_rules(platforms: list[dict] | None = None) -> dict:
    return {
        "player_start": {"x": 20.0, "y": 20.0},
        "platforms": (
            [{"object_index": 0, "role": "solid"}]
            if platforms is None
            else platforms
        ),
        "hazards": [],
        "collectibles": [],
        "win_condition": "Reach the end",
        "theme": "pixel art",
    }


class PolygonServiceTests(unittest.TestCase):
    def test_extracts_bounded_polygons_and_rejects_small_or_invalid_masks(self):
        polygon = mask_png_to_polygon(
            make_mask((100, 100), (10, 10, 90, 90)), 100, 100
        )
        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertGreaterEqual(polygon["area_ratio"], 0.15)
        for key in ("physics", "display"):
            self.assertGreaterEqual(len(polygon[key]), 3)
            self.assertLessEqual(len(polygon[key]), 32)
            for vertex in polygon[key]:
                self.assertGreaterEqual(vertex["x"], 0)
                self.assertLessEqual(vertex["x"], 100)
                self.assertGreaterEqual(vertex["y"], 0)
                self.assertLessEqual(vertex["y"], 100)

        tiny_mask = make_mask((100, 100), (10, 10, 14, 14))
        self.assertIsNone(mask_png_to_polygon(tiny_mask, 100, 100))
        self.assertIsNone(mask_png_to_polygon(b"not a PNG", 100, 100))

    def test_nonconvex_display_polygon_uses_convex_physics_hull(self):
        image = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(image).polygon(
            [(10, 10), (80, 10), (80, 30), (30, 30), (30, 80), (10, 80)],
            fill=255,
        )
        output = io.BytesIO()
        image.save(output, format="PNG")

        polygon = mask_png_to_polygon(output.getvalue(), 100, 100)
        self.assertIsNotNone(polygon)
        assert polygon is not None
        display = np.asarray(
            [[round(point["x"]), round(point["y"])] for point in polygon["display"]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        physics = np.asarray(
            [[round(point["x"]), round(point["y"])] for point in polygon["physics"]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        self.assertFalse(cv2.isContourConvex(display))
        self.assertTrue(cv2.isContourConvex(physics))


class GenerateEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.metrics_directory = tempfile.TemporaryDirectory()
        self.metrics_path = Path(self.metrics_directory.name) / "demo_metrics.json"
        self.metrics_patch = patch.object(
            metrics_service, "METRICS_PATH", self.metrics_path
        )
        self.metrics_patch.start()
        self.addCleanup(self.metrics_patch.stop)
        self.addCleanup(self.metrics_directory.cleanup)

        self.local_mask = make_mask((100, 60), (0, 0, 99, 59))
        self.segment_patch = patch.object(
            main,
            "segment_objects",
            return_value=[
                {
                    "label": "mug",
                    "bbox": [20, 30, 120, 90],
                    "mask_png_base64": base64.b64encode(
                        self.local_mask
                    ).decode("ascii"),
                    "width": 100,
                    "height": 60,
                }
            ],
        )
        self.segment_patch.start()
        self.addCleanup(self.segment_patch.stop)

        image_buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), "white").save(image_buffer, format="JPEG")
        self.photo_bytes = image_buffer.getvalue()

    def post_generate(self):
        return self.client.post(
            "/generate",
            files={"image": ("room.jpg", self.photo_bytes, "image/jpeg")},
        )

    def test_returns_filtered_objects_rules_and_resized_canvas(self):
        with patch.object(main, "generate_game_rules", return_value=valid_rules()):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(set(payload), {"objects", "rules", "canvas"})
        self.assertEqual(payload["canvas"], {"width": 1280, "height": 720})
        self.assertEqual(
            set(payload["objects"][0]),
            {"object_index", "label", "bbox", "physics", "display", "area_ratio"},
        )
        self.assertEqual(payload["objects"][0]["label"], "mug")
        self.assertEqual(payload["rules"]["platforms"][0]["object_index"], 0)
        main.segment_objects.assert_called_once_with(
            ANY,
            concepts=[
                "chair",
                "desk",
                "mug",
                "laptop",
                "plant",
                "monitor",
                "keyboard",
            ],
        )

    def test_injects_ground_when_spark_returns_no_platforms(self):
        with patch.object(
            main, "generate_game_rules", return_value=valid_rules(platforms=[])
        ):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        ground_index = payload["rules"]["platforms"][0]["object_index"]
        self.assertEqual(payload["objects"][ground_index]["label"], "ground")
        self.assertEqual(payload["objects"][ground_index]["bbox"][1], 648.0)

    def test_invalid_spark_index_uses_fallback_and_logs_provider(self):
        invalid_rules = valid_rules(
            platforms=[{"object_index": 99, "role": "solid"}]
        )
        with patch.object(main, "generate_game_rules", return_value=invalid_rules):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            payload["rules"]["win_condition"],
            "Collect every object and reach the right edge of the scene.",
        )
        self.assertEqual(payload["rules"]["collectibles"][0]["object_index"], 0)
        self.assertEqual(payload["objects"][-1]["label"], "ground")
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["scene_fallbacks"][0]["provider"], "fallback")

    def test_spark_timeout_uses_fallback_and_records_provider(self):
        timeout = APITimeoutError(request=None)
        with patch.object(main, "generate_game_rules", side_effect=timeout):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["scene_fallbacks"][0]["provider"], "fallback")

    def test_out_of_bounds_player_start_uses_fallback(self):
        invalid_rules = valid_rules()
        invalid_rules["player_start"] = {"x": 2000.0, "y": 20.0}
        with patch.object(main, "generate_game_rules", return_value=invalid_rules):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["rules"]["player_start"],
            {"x": 128.0, "y": 144.0},
        )

    def test_upstream_segmentation_failure_is_reported_as_502(self):
        with patch.object(
            main, "segment_objects", side_effect=ValueError("test SAM outage")
        ):
            response = self.post_generate()

        self.assertEqual(response.status_code, 502)
        self.assertIn("test SAM outage", response.json()["detail"])

    def test_fallback_roles_follow_area_and_plant_rules(self):
        objects = [
            {"object_index": 0, "label": "desk", "area_ratio": 0.2},
            {"object_index": 1, "label": "plant", "area_ratio": 0.04},
            {"object_index": 2, "label": "mug", "area_ratio": 0.03},
        ]
        rules = main._fallback_rules(objects, 1000, 500, "retro")
        self.assertEqual(rules["platforms"], [{"object_index": 0, "role": "solid"}])
        self.assertEqual(
            rules["hazards"],
            [{"object_index": 1, "role": "hazard", "damage": 1}],
        )
        self.assertEqual(
            rules["collectibles"],
            [{"object_index": 2, "role": "collectible", "points": 10}],
        )


if __name__ == "__main__":
    unittest.main()
