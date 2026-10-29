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
from services.image_service import _remove_corner_background
from services.polygon_service import mask_png_to_polygon
from services.sam_service import _decoded_mask_png_base64


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
        self.assertGreater(len(display), len(physics))

    def test_curved_display_outline_keeps_higher_fidelity_vertices(self):
        image = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(image).ellipse((20, 20, 180, 180), fill=255)
        output = io.BytesIO()
        image.save(output, format="PNG")

        polygon = mask_png_to_polygon(output.getvalue(), 200, 200)

        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertGreaterEqual(len(polygon["display"]), 8)
        self.assertLessEqual(len(polygon["display"]), 32)


class SAMMaskTests(unittest.TestCase):
    def test_scales_parser_binary_mask_values_for_polygon_threshold(self):
        raw_mask = bytearray(100 * 100)
        for y in range(10, 90):
            for x in range(10, 90):
                raw_mask[y * 100 + x] = 1

        encoded_mask = _decoded_mask_png_base64(bytes(raw_mask), 100, 100)
        decoded = base64.b64decode(encoded_mask)
        polygon = mask_png_to_polygon(decoded, 100, 100)

        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertGreaterEqual(polygon["area_ratio"], 0.15)


class MuseImageTransparencyTests(unittest.TestCase):
    def test_removes_edge_connected_black_and_white_backgrounds(self):
        for background in ("black", "white"):
            image = Image.new("RGB", (20, 20), background)
            ImageDraw.Draw(image).rectangle((5, 5, 14, 14), fill="red")

            result = _remove_corner_background(image)

            self.assertEqual(result.getpixel((0, 0))[3], 0)
            self.assertEqual(result.getpixel((10, 10))[3], 255)


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
        self.restyle_directory = tempfile.TemporaryDirectory()
        self.restyle_path = Path(self.restyle_directory.name) / "restyle_cache"
        self.restyle_patch = patch.object(main, "RESTYLE_CACHE_DIR", self.restyle_path)
        self.restyle_patch.start()
        self.addCleanup(self.restyle_patch.stop)
        self.addCleanup(self.restyle_directory.cleanup)

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
        self.assertEqual(
            set(payload),
            {"objects", "rules", "canvas", "canvas_base64", "image_id", "meta"},
        )
        self.assertRegex(payload["image_id"], r"^[a-f0-9]{64}$")
        background_prefix = "data:image/jpeg;base64,"
        self.assertTrue(payload["canvas_base64"].startswith(background_prefix))
        background_bytes = base64.b64decode(
            payload["canvas_base64"][len(background_prefix):]
        )
        with Image.open(io.BytesIO(background_bytes)) as background:
            self.assertEqual(background.format, "JPEG")
            self.assertEqual(background.size, (1280, 720))
        self.assertEqual(
            payload["meta"]["debug_sam_status"],
            "SAM returned 1 masks; 1 objects retained",
        )
        self.assertEqual(payload["canvas"], {"width": 1280, "height": 720})
        self.assertEqual(
            set(payload["objects"][0]),
            {"object_index", "label", "bbox", "physics", "display", "area_ratio"},
        )
        self.assertEqual(payload["objects"][0]["label"], "mug")
        self.assertEqual(payload["rules"]["platforms"][0]["object_index"], 0)
        first_object = payload["objects"][0]
        self.assertTrue(all(type(value) is float for value in first_object["bbox"]))
        for polygon_name in ("physics", "display"):
            for point in first_object[polygon_name]:
                self.assertIs(type(point["x"]), float)
                self.assertIs(type(point["y"]), float)
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

    def test_restyle_crops_source_caches_png_and_returns_cache_hit(self):
        with patch.object(main, "generate_game_rules", return_value=valid_rules()):
            generated = self.post_generate()
        image_id = generated.json()["image_id"]
        png_bytes = make_mask((8, 8), (1, 1, 6, 6))
        with patch.object(main, "restyle_object", return_value=png_bytes) as restyle_mock:
            response = self.client.post(
                "/restyle",
                json={
                    "image_id": image_id,
                    "object_index": 0,
                    "theme": "pixel art",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(response.json()["cached"])
            self.assertEqual(
                base64.b64decode(response.json()["png_base64"]),
                png_bytes,
            )
            crop_bytes = restyle_mock.call_args.args[0]
            with Image.open(io.BytesIO(crop_bytes)) as crop:
                self.assertEqual(crop.size, (100, 60))

            cached = self.client.post(
                "/restyle",
                json={
                    "image_id": image_id,
                    "object_index": 0,
                    "theme": "pixel art",
                },
            )
            self.assertEqual(cached.status_code, 200, cached.text)
            self.assertTrue(cached.json()["cached"])
            self.assertEqual(
                base64.b64decode(cached.json()["png_base64"]),
                png_bytes,
            )
            restyle_mock.assert_called_once()

    def test_restyle_rejects_unknown_image_id(self):
        response = self.client.post(
            "/restyle",
            json={
                "image_id": "0" * 64,
                "object_index": 0,
                "theme": "pixel art",
            },
        )
        self.assertEqual(response.status_code, 404)

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
            {"x": 640.0, "y": 598.0},
        )

    def test_upstream_segmentation_failure_returns_visible_fallback_scene(self):
        with patch.object(
            main, "segment_objects", side_effect=ValueError("test SAM outage")
        ):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            payload["meta"]["debug_sam_status"],
            "ValueError: test SAM outage",
        )
        self.assertEqual(len(payload["objects"]), 1)
        self.assertEqual(payload["objects"][0]["label"], "ground")
        self.assertEqual(
            payload["rules"]["player_start"],
            {"x": 640.0, "y": 598.0},
        )

    def test_zero_sam_objects_are_reported_and_start_above_ground(self):
        with (
            patch.object(main, "segment_objects", return_value=[]),
            patch.object(main, "generate_game_rules") as spark_mock,
        ):
            response = self.post_generate()

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            payload["meta"]["debug_sam_status"],
            "Zero objects returned from API",
        )
        self.assertEqual([obj["label"] for obj in payload["objects"]], ["ground"])
        self.assertEqual(payload["rules"]["player_start"], {"x": 640.0, "y": 598.0})
        spark_mock.assert_not_called()

    def test_fallback_player_starts_above_widest_platform(self):
        objects = [
            {"bbox": [20.0, 200.0, 180.0, 250.0]},
            {"bbox": [100.0, 400.0, 700.0, 600.0]},
        ]
        rules = {
            "platforms": [
                {"object_index": 0, "role": "solid"},
                {"object_index": 1, "role": "solid"},
            ]
        }

        main._fallback_player_start(objects, rules, 800, 700)

        self.assertEqual(rules["player_start"], {"x": 400.0, "y": 350.0})

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
