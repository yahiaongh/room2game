import io
import json
import logging
import math

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from models.schemas import GameRules
from services.metrics_service import record_fallback
from services.polygon_service import mask_png_to_polygon
from services.sam_service import SAM_CONCEPTS, segment_objects
from services.spark_service import generate_game_rules

logger = logging.getLogger(__name__)
MAX_CANVAS_DIMENSION = 1280
HAZARD_LABELS = {"plant", "cactus"}

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

    debug_sam_status = ""
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

        response_data = {
            "objects": objects,
            "rules": rules,
            "canvas": {"width": canvas_width, "height": canvas_height},
            "meta": {"debug_sam_status": debug_sam_status},
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
