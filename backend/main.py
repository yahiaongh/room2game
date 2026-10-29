import base64
import io
import json
import logging
import math
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from models.schemas import GameRules, RestyleRequest
from services.image_service import restyle_object
from services.metrics_service import record_fallback
from services.polygon_service import mask_png_to_polygon
from services.sam_service import SAM_CONCEPTS, segment_objects
from services.spark_service import generate_game_rules

logger = logging.getLogger(__name__)
MAX_CANVAS_DIMENSION = 1280
HAZARD_LABELS = {"plant", "cactus"}
RESTYLE_CACHE_DIR = Path(__file__).resolve().parents[1] / "assets" / "restyle_cache"

app = FastAPI(title="Room2Game")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _canvas_image(image_bytes: bytes) -> tuple[bytes, int, int]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            source_image.thumbnail(
                (MAX_CANVAS_DIMENSION, MAX_CANVAS_DIMENSION),
                Image.Resampling.LANCZOS,
            )
            canvas = source_image.convert("RGB")
            canvas_width, canvas_height = canvas.size
            canvas_buffer = io.BytesIO()
            canvas.save(canvas_buffer, format="PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from error
    return canvas_buffer.getvalue(), canvas_width, canvas_height


def _persist_scene_source(
    image_id: str, image_bytes: bytes, objects: list[dict]
) -> None:
    RESTYLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = [
        {"label": obj["label"], "bbox": obj["bbox"]}
        for obj in objects
    ]
    for path, content in (
        (RESTYLE_CACHE_DIR / f"{image_id}.png", image_bytes),
        (
            RESTYLE_CACHE_DIR / f"{image_id}.json",
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        ),
    ):
        descriptor, temporary_path = tempfile.mkstemp(
            dir=RESTYLE_CACHE_DIR, prefix=".scene."
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
            Path(temporary_path).replace(path)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise


def _restyle_cache_path(image_id: str, object_index: int, theme: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", image_id):
        raise HTTPException(status_code=400, detail="image_id must be a SHA-256 hash")
    if object_index < 0:
        raise HTTPException(status_code=400, detail="object_index must not be negative")
    if (
        not theme.strip()
        or len(theme) > 60
        or not re.fullmatch(r"[a-zA-Z0-9 _-]+", theme)
    ):
        raise HTTPException(
            status_code=400,
            detail="theme may only contain letters, numbers, spaces, underscores, and hyphens",
        )
    theme_filename = re.sub(r"\s+", "-", theme.strip().lower())
    return RESTYLE_CACHE_DIR / f"{image_id}_{object_index}_{theme_filename}.png"


@app.post("/restyle")
def restyle(request: RestyleRequest) -> dict[str, str | bool]:
    cache_path = _restyle_cache_path(
        request.image_id,
        request.object_index,
        request.theme,
    )
    if cache_path.is_file():
        return {
            "png_base64": base64.b64encode(cache_path.read_bytes()).decode("ascii"),
            "cached": True,
        }

    image_path = RESTYLE_CACHE_DIR / f"{request.image_id}.png"
    metadata_path = RESTYLE_CACHE_DIR / f"{request.image_id}.json"
    if not image_path.is_file() or not metadata_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Generated image not found; call /generate with this photo first",
        )
    try:
        objects = json.loads(metadata_path.read_text(encoding="utf-8"))
        object_data = objects[request.object_index]
        left, top, right, bottom = (float(value) for value in object_data["bbox"])
        with Image.open(image_path) as source:
            crop = source.crop(
                (
                    max(0, round(left)),
                    max(0, round(top)),
                    min(source.width, round(right)),
                    min(source.height, round(bottom)),
                )
            )
            if crop.width <= 0 or crop.height <= 0:
                raise ValueError("Object bounding box does not overlap the source image")
            crop_buffer = io.BytesIO()
            crop.save(crop_buffer, format="PNG")
        png_bytes = restyle_object(
            crop_buffer.getvalue(),
            request.theme,
            label=object_data["label"],
            run_id=request.image_id,
        )
        if png_bytes is None:
            raise HTTPException(
                status_code=429,
                detail="Muse Image limit reached for this photo; using the polygon fallback",
            )
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Muse Image returned data that is not a PNG")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=cache_path.parent, prefix=".restyle."
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(png_bytes)
            Path(temporary_path).replace(cache_path)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise
    except HTTPException:
        raise
    except (IndexError, KeyError, TypeError, ValueError, OSError) as error:
        logger.exception("Could not restyle generated object")
        raise HTTPException(
            status_code=422,
            detail=f"Could not restyle object {request.object_index}: {error}",
        ) from error
    except Exception as error:
        logger.exception("Muse Image restyle failed")
        raise HTTPException(
            status_code=502,
            detail=f"Muse Image restyle failed: {type(error).__name__}: {error}",
        ) from error
    return {
        "png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "cached": False,
    }


def _full_canvas_mask(
    mask_png_base64: str,
    bbox: list[float],
    canvas_width: int,
    canvas_height: int,
) -> bytes:
    import base64

    try:
        mask_bytes = base64.b64decode(mask_png_base64, validate=True)
        with Image.open(io.BytesIO(mask_bytes)) as image:
            mask = image.convert("L")
            left, top, right, bottom = (float(value) for value in bbox)
            target_width = max(1, round(right - left))
            target_height = max(1, round(bottom - top))
            mask = mask.resize(
                (target_width, target_height), Image.Resampling.NEAREST
            )
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise ValueError("SAM returned a mask that could not be decoded") from error

    left = round(float(bbox[0]))
    top = round(float(bbox[1]))
    right = left + mask.width
    bottom = top + mask.height
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(canvas_width, right)
    clipped_bottom = min(canvas_height, bottom)
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return b""

    clipped_mask = mask.crop(
        (
            clipped_left - left,
            clipped_top - top,
            clipped_right - left,
            clipped_bottom - top,
        )
    )
    full_mask = Image.new("L", (canvas_width, canvas_height), 0)
    full_mask.paste(clipped_mask, (clipped_left, clipped_top))
    output = io.BytesIO()
    full_mask.save(output, format="PNG")
    return output.getvalue()


def _validated_rules(
    raw_rules: dict,
    object_count: int,
    width: int,
    height: int,
) -> dict:
    model_validate = getattr(GameRules, "model_validate", None)
    rules = (
        model_validate(raw_rules)
        if callable(model_validate)
        else GameRules.parse_obj(raw_rules)
    )
    rules_data = (
        rules.model_dump()
        if callable(getattr(rules, "model_dump", None))
        else rules.dict()
    )
    start = rules_data["player_start"]
    if not (
        math.isfinite(start["x"])
        and math.isfinite(start["y"])
        and 0 <= start["x"] <= width
        and 0 <= start["y"] <= height
    ):
        raise ValueError("Muse Spark player_start is outside the canvas")

    for role_name in ("platforms", "hazards", "collectibles"):
        for role in rules_data[role_name]:
            object_index = role["object_index"]
            if not 0 <= object_index < object_count:
                raise ValueError(
                    f"Muse Spark {role_name} object_index {object_index} "
                    "is outside the object list"
                )
    return rules_data


def _fallback_rules(objects: list[dict], width: int, height: int, theme: str) -> dict:
    platforms = []
    hazards = []
    collectibles = []
    for obj in objects:
        object_index = obj["object_index"]
        if obj["area_ratio"] >= 0.15:
            platforms.append({"object_index": object_index, "role": "solid"})
        elif obj["area_ratio"] < 0.05:
            if obj["label"].lower() in HAZARD_LABELS:
                hazards.append(
                    {"object_index": object_index, "role": "hazard", "damage": 1}
                )
            else:
                collectibles.append(
                    {
                        "object_index": object_index,
                        "role": "collectible",
                        "points": 10,
                    }
                )
    return {
        "player_start": {"x": float(width * 0.1), "y": float(height * 0.5)},
        "platforms": platforms,
        "hazards": hazards,
        "collectibles": collectibles,
        "win_condition": "Collect every object and reach the right edge of the scene.",
        "theme": theme,
    }


def _inject_ground(objects: list[dict], rules: dict, width: int, height: int) -> None:
    if rules["platforms"]:
        return

    ground_top = float(height * 0.9)
    object_index = len(objects)
    objects.append(
        {
            "object_index": object_index,
            "label": "ground",
            "bbox": [0.0, ground_top, float(width), float(height)],
            "area_ratio": 0.1,
            "physics": [
                {"x": 0.0, "y": ground_top},
                {"x": float(width), "y": ground_top},
                {"x": float(width), "y": float(height)},
                {"x": 0.0, "y": float(height)},
            ],
            "display": [
                {"x": 0.0, "y": ground_top},
                {"x": float(width), "y": ground_top},
                {"x": float(width), "y": float(height)},
                {"x": 0.0, "y": float(height)},
            ],
        }
    )
    rules["platforms"].append(
        {"object_index": object_index, "role": "solid"}
    )


@app.post("/generate")
def generate(
    image: UploadFile = File(...),
    theme: str = Query(default="pixel art"),
) -> dict:
    if image is None:
        raise HTTPException(status_code=400, detail="An image upload is required")
    image_bytes, canvas_width, canvas_height = _canvas_image(image.file.read())
    image_id = sha256(image_bytes).hexdigest()

    debug_sam_status = ""
    rules_fallback = False
    try:
        try:
            segmented_objects = segment_objects(
                image_bytes,
                concepts=SAM_CONCEPTS,
            )
        except Exception as error:
            response = getattr(error, "response", None)
            status_code = getattr(error, "status_code", None)
            debug_sam_status = (
                f"HTTP {status_code}"
                if status_code is not None
                else f"{type(error).__name__}: {error}"
            )
            logger.exception("SAM segmentation failed: %s", debug_sam_status)
            if response is not None:
                try:
                    raw_response = response.text
                except (AttributeError, OSError, ValueError):
                    raw_response = repr(response)
                print(f"SAM raw API response: {raw_response}")
            segmented_objects = []

        if not segmented_objects and not debug_sam_status:
            debug_sam_status = "Zero objects returned from API"
            logger.warning("SAM debug status: %s", debug_sam_status)

        objects = []
        filtered_by_polygon = 0
        for segmented in segmented_objects:
            bbox = segmented.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                logger.warning("Skipping SAM object with an invalid bounding box")
                continue
            full_mask = _full_canvas_mask(
                segmented["mask_png_base64"],
                bbox,
                canvas_width,
                canvas_height,
            )
            if not full_mask:
                continue
            polygon = mask_png_to_polygon(
                full_mask,
                canvas_width,
                canvas_height,
            )
            if polygon is None:
                filtered_by_polygon += 1
                continue
            objects.append(
                {
                    "object_index": len(objects),
                    "label": segmented["label"],
                    "bbox": [float(value) for value in bbox],
                    **polygon,
                }
            )

        if segmented_objects and not objects:
            if filtered_by_polygon == len(segmented_objects):
                debug_sam_status = (
                    f"SAM returned {len(segmented_objects)} masks, but all were "
                    "filtered out by area_ratio"
                )
                logger.warning("%s", debug_sam_status)
            elif not debug_sam_status:
                debug_sam_status = (
                    f"SAM returned {len(segmented_objects)} masks, but none "
                    "could be converted to usable objects"
                )
                logger.warning("SAM debug status: %s", debug_sam_status)
        elif segmented_objects and not debug_sam_status:
            debug_sam_status = (
                f"SAM returned {len(segmented_objects)} masks; "
                f"{len(objects)} objects retained"
            )
        logger.info("SAM debug status: %s", debug_sam_status)
        print(f"SAM debug status: {debug_sam_status}")

        spark_objects = [
            {
                "object_index": obj["object_index"],
                "label": obj["label"],
                "bbox": obj["bbox"],
                "area_ratio": obj["area_ratio"],
            }
            for obj in objects
        ]
        if objects:
            try:
                raw_rules = generate_game_rules(spark_objects)
                rules = _validated_rules(
                    raw_rules, len(objects), canvas_width, canvas_height
                )
            except (
                APITimeoutError,
                APIConnectionError,
                ValidationError,
                ValueError,
            ) as error:
                rules_fallback = True
                fallback_reason = f"{type(error).__name__}: {error}"
                logger.warning("Using fallback game rules: %s", fallback_reason)
                rules = _fallback_rules(
                    objects,
                    canvas_width,
                    canvas_height,
                    theme,
                )
                record_fallback(fallback_reason)
        else:
            rules_fallback = True
            fallback_reason = (
                f"SAM produced no usable objects: {debug_sam_status}"
            )
            logger.warning("Using fallback game rules: %s", fallback_reason)
            rules = _fallback_rules(
                objects,
                canvas_width,
                canvas_height,
                theme,
            )
            record_fallback(fallback_reason)
        _inject_ground(objects, rules, canvas_width, canvas_height)
        if not rules["platforms"]:
            raise RuntimeError("Scene generation failed to produce a ground platform")
        _persist_scene_source(image_id, image_bytes, objects)

        response_data = {
            "objects": objects,
            "rules": rules,
            "canvas": {"width": canvas_width, "height": canvas_height},
            "image_id": image_id,
            "meta": {
                "debug_sam_status": debug_sam_status,
                "rules_fallback": rules_fallback,
            },
        }
        if objects:
            print(
                "First generated object JSON:",
                json.dumps(objects[0], allow_nan=False, separators=(",", ":")),
            )
        return response_data
    except HTTPException:
        raise
    except (APIConnectionError, APIStatusError, ValueError) as error:
        logger.exception("Photo-to-scene model pipeline failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Photo-to-scene generation failed in an upstream model or "
                f"geometry step: {type(error).__name__}: {error}"
            ),
        ) from error
    except (KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Scene generation returned invalid data: {error}",
        ) from error
