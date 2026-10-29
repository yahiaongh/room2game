import base64
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from meta_sam_parser import (
    CompletedOutcome,
    image_segmentation_format,
    decode_mask_to_raster,
)
from openai import APITimeoutError, OpenAI
from PIL import Image

from services.metrics_service import record_sam_call_count

load_dotenv()

SAM_CONCEPTS = ["chair", "desk", "mug", "laptop", "plant", "monitor", "keyboard"]
MAX_CONCEPTS_PER_IMAGE = 5
MAX_CALLS_PER_REQUEST = 5
_ASSETS_PATH = Path(__file__).resolve().parents[2] / "assets"


def _parse_objects(output_text: str, label: str) -> list[dict]:
    parser = image_segmentation_format().create_parser()
    parser.push(output_text)
    result = parser.finish(CompletedOutcome()).result

    boxes = {
        record.object_id: record
        for record in result.records
        if record.kind == "box"
    }
    objects = []
    for record in result.records:
        if record.kind != "mask":
            continue
        box = boxes.get(record.object_id)
        if box is None:
            continue

        mask_png_base64 = _decoded_mask_png_base64(
            decode_mask_to_raster(record.mask),
            record.mask.width,
            record.mask.height,
        )
        objects.append(
            {
                "label": label,
                "bbox": [
                    float(box.left),
                    float(box.top),
                    float(box.right),
                    float(box.bottom),
                ],
                "mask_png_base64": mask_png_base64,
                "width": record.mask.width,
                "height": record.mask.height,
            }
        )
    return objects


def _decoded_mask_png_base64(
    mask_bytes: bytes, width: int, height: int
) -> str:
    if mask_bytes and max(mask_bytes) <= 1:
        mask_bytes = bytes(value * 255 for value in mask_bytes)
    mask_image = Image.frombytes("L", (width, height), mask_bytes).convert("RGBA")
    png_buffer = io.BytesIO()
    mask_image.save(png_buffer, format="PNG")
    return base64.b64encode(png_buffer.getvalue()).decode("ascii")


def segment_objects(
    image_bytes: bytes, concepts: list[str] | None = None
) -> list[dict]:
    """Segment at most five concepts and return decoded PNG masks and boxes."""
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is required to call the Meta Model API")

    source_image = Image.open(io.BytesIO(image_bytes))
    mime_type = Image.MIME.get(source_image.format or "")
    if mime_type is None:
        source_image.close()
        raise ValueError(f"Unsupported image format: {source_image.format}")
    source_image.close()
    image_data_url = (
        f"data:{mime_type};base64,"
        + base64.b64encode(image_bytes).decode("ascii")
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.meta.ai/v1",
        timeout=20.0,
        max_retries=0,
    )
    selected_concepts = SAM_CONCEPTS if concepts is None else concepts
    selected_concepts = selected_concepts[:MAX_CONCEPTS_PER_IMAGE]
    call_count = 0
    timeout_count = 0
    objects = []
    empty_responses = []
    wrote_sample = False

    def request_concept(concept: str):
        nonlocal call_count
        if call_count >= MAX_CALLS_PER_REQUEST:
            raise ValueError(
                f"SAM call limit exceeded: maximum {MAX_CALLS_PER_REQUEST} "
                "calls per image"
            )
        call_count += 1
        response = client.responses.create(
            model="sam-3.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            stream=False,
        )
        return response

    try:
        for concept in selected_concepts:
            try:
                response = request_concept(concept)
            except APITimeoutError:
                timeout_count += 1
                if timeout_count >= 3:
                    break
                continue

            output_text = response.output_text or ""
            if not output_text:
                empty_responses.append(response)
                continue

            if not wrote_sample:
                _ASSETS_PATH.mkdir(parents=True, exist_ok=True)
                (_ASSETS_PATH / "sam_raw_sample.txt").write_text(
                    output_text, encoding="utf-8"
                )
                wrote_sample = True
            objects.extend(_parse_objects(output_text, concept))

        if timeout_count >= 3:
            try:
                response = request_concept("object")
            except APITimeoutError:
                pass
            else:
                output_text = response.output_text or ""
                if not output_text:
                    empty_responses.append(response)
                else:
                    if not wrote_sample:
                        _ASSETS_PATH.mkdir(parents=True, exist_ok=True)
                        (_ASSETS_PATH / "sam_raw_sample.txt").write_text(
                            output_text, encoding="utf-8"
                        )
                        wrote_sample = True
                    objects.extend(_parse_objects(output_text, "object"))

        if not objects and empty_responses and not wrote_sample:
            response_data = empty_responses[-1].model_dump()
            raw_output_text = empty_responses[-1].output_text or ""
            print("SAM returned empty output_text; full response.model_dump():")
            print(json.dumps(response_data, indent=2, default=str))
            print("SAM raw output_text:")
            print(raw_output_text)
            raise ValueError(
                "SAM returned empty output_text for every attempted concept; "
                "see full response and raw output above"
            )
        return objects
    finally:
        record_sam_call_count(call_count)
