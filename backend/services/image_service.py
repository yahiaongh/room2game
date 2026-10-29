import base64
import io
import os
import threading
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from services.metrics_service import record_muse_image_count

load_dotenv()

MAX_IMAGES_PER_RUN = 3
_image_counts: dict[str, int] = {}
_IMAGE_COUNT_LOCK = threading.Lock()


def reset_image_count(run_id: str | None = None) -> None:
    with _IMAGE_COUNT_LOCK:
        if run_id is None:
            _image_counts.clear()
        else:
            _image_counts.pop(run_id, None)
        record_muse_image_count(sum(_image_counts.values()))


def _png_bytes(response) -> bytes:
    image_data = response.data[0]
    if image_data.b64_json:
        image_bytes = base64.b64decode(image_data.b64_json)
    elif image_data.url:
        with urlopen(image_data.url, timeout=30) as result:
            image_bytes = result.read()
    else:
        raise ValueError("Muse Image returned no PNG data or downloadable URL")

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                png_buffer = io.BytesIO()
                image.save(png_buffer, format="PNG")
                image_bytes = png_buffer.getvalue()
        except UnidentifiedImageError as error:
            raise ValueError(
                "Muse Image returned data that is not a decodable image"
            ) from error
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Muse Image data could not be converted to PNG")
    return image_bytes


def restyle_object(
    object_image_bytes: bytes,
    theme: str,
    label: str = "object",
    run_id: str | None = None,
) -> bytes | None:
    if not object_image_bytes:
        raise ValueError("An object crop is required to restyle an asset")

    count_key = run_id or "default"
    with _IMAGE_COUNT_LOCK:
        image_count = _image_counts.get(count_key, 0)
        if image_count >= MAX_IMAGES_PER_RUN:
            record_muse_image_count(sum(_image_counts.values()))
            return None

        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise RuntimeError("MODEL_API_KEY is required to call the Meta Model API")

        _image_counts[count_key] = image_count + 1
        record_muse_image_count(sum(_image_counts.values()))

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.meta.ai/v1",
        timeout=30.0,
        max_retries=1,
    )
    prompt = (
        f"A pixel art {label}, transparent background, clean silhouette, "
        f"128x128, in {theme} style."
    )
    response = client.images.generate(
        model="muse-image-1.0",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    return _png_bytes(response)
