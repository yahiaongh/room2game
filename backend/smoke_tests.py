import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PHOTO = PROJECT_ROOT / "assets" / "desk_smoke_test.jpeg"
METRICS_PATH = PROJECT_ROOT / "demo_metrics.json"
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
EXPECTED_RULE_KEYS = {
    "player_start",
    "platforms",
    "hazards",
    "collectibles",
    "win_condition",
    "theme",
}


def _print_result(name: str, result: str) -> None:
    print(f"{name}: PASS {result[:300]}")


def _validate_rules(rules: dict) -> None:
    if set(rules) != EXPECTED_RULE_KEYS:
        raise AssertionError("Spark rules do not contain exactly the six required keys")
    if not isinstance(rules["player_start"].get("x"), float):
        raise AssertionError("player_start.x is not a float")
    if not isinstance(rules["player_start"].get("y"), float):
        raise AssertionError("player_start.y is not a float")


def _run_baseten_spark(objects: list[dict]) -> dict:
    if METRICS_PATH.is_file():
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            previous_call_count = len(json.load(metrics_file).get("spark_calls", []))
    else:
        previous_call_count = 0

    child_env = os.environ.copy()
    child_env["FORCE_BASETEN"] = "1"
    code = (
        "import json; "
        "from services.spark_service import generate_game_rules; "
        f"print(json.dumps(generate_game_rules({objects!r}), separators=(',', ':')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT / "backend",
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    rules = json.loads(completed.stdout.strip().splitlines()[-1])
    _validate_rules(rules)
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    baseten_calls = [
        call
        for call in metrics.get("spark_calls", [])[previous_call_count:]
        if call.get("provider") == "baseten"
        and call.get("model_id") == "meta-models/muse-spark-open-1026"
    ]
    if not baseten_calls:
        raise AssertionError("Baseten Muse Spark call was not recorded in demo_metrics.json")
    return rules


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not SAMPLE_PHOTO.is_file():
        raise FileNotFoundError(f"Smoke-test photo not found: {SAMPLE_PHOTO}")
    photo_bytes = SAMPLE_PHOTO.read_bytes()

    os.environ["FORCE_BASETEN"] = "0"
    from services.sam_service import segment_objects
    from services.spark_service import generate_game_rules

    objects = segment_objects(photo_bytes, concepts=["mug"])
    if not objects:
        raise AssertionError("SAM did not return a mug object")
    for obj in objects:
        if not obj.get("bbox"):
            raise AssertionError("SAM object is missing its bounding box")
        mask_png = base64.b64decode(obj["mask_png_base64"], validate=True)
        with Image.open(io.BytesIO(mask_png)) as mask:
            if mask.format != "PNG":
                raise AssertionError("SAM mask did not decode as PNG")
    _print_result("SAM 3.1", json.dumps(objects, separators=(",", ":")))

    representative_objects = [
        {"label": "desk", "bbox": [0, 0, 1408, 768], "width": 1408, "height": 768},
        {"label": "mug", "bbox": [620, 250, 750, 370], "width": 1408, "height": 768},
    ]
    meta_rules = generate_game_rules(representative_objects)
    _validate_rules(meta_rules)
    _print_result("Muse Spark Meta", json.dumps(meta_rules, separators=(",", ":")))

    baseten_rules = _run_baseten_spark(representative_objects)
    _print_result(
        "Muse Spark Baseten",
        json.dumps(baseten_rules, separators=(",", ":")),
    )

    from services.image_service import restyle_object

    with Image.open(io.BytesIO(photo_bytes)) as photo:
        crop = io.BytesIO()
        photo.crop((0, 0, 320, 320)).save(crop, format="PNG")
    image_bytes = restyle_object(crop.getvalue(), "pixel art", label="mug")
    if image_bytes is None or not image_bytes.startswith(b"\x89PNG"):
        raise AssertionError("Muse Image did not return PNG bytes")
    _print_result(
        "Muse Image",
        f"{len(image_bytes)} PNG bytes; signature={image_bytes[:8]!r}",
    )

    from main import app

    health = TestClient(app).get("/health")
    if health.status_code != 200 or health.json() != {"status": "ok"}:
        raise AssertionError(f"Health check failed: {health.status_code} {health.text}")
    _print_result("Health", health.text)


if __name__ == "__main__":
    main()
