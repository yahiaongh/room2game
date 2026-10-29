import base64
import io
import os
import threading
from collections import deque
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from services.metrics_service import record_muse_image_count

load_dotenv()

MAX_IMAGES_PER_RUN = 3
_image_counts: dict[str, int] = {}
_IMAGE_COUNT_LOCK = threading.Lock()


def _remove_corner_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    corners = (
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    )
    for background in ((0, 0, 0), (255, 255, 255)):
        visited = bytearray(width * height)
        pending = deque()
        for x, y in corners:
            red, green, blue, alpha = pixels[x, y]
            close_to_background = max(
                abs(red - background[0]),
                abs(green - background[1]),
                abs(blue - background[2]),
            ) <= 48
            index = y * width + x
            if alpha and close_to_background and not visited[index]:
                visited[index] = 1
                pending.append((x, y))

        while pending:
            x, y = pending.popleft()
            red, green, blue, alpha = pixels[x, y]
            if alpha == 0:
                continue
            if max(
                abs(red - background[0]),
                abs(green - background[1]),
                abs(blue - background[2]),
            ) > 48:
                continue
            pixels[x, y] = (red, green, blue, 0)
            for neighbor_x, neighbor_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                    continue
                index = neighbor_y * width + neighbor_x
                if not visited[index]:
                    visited[index] = 1
                    pending.append((neighbor_x, neighbor_y))
    return rgba


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

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            png_buffer = io.BytesIO()
            _remove_corner_background(image).save(png_buffer, format="PNG")
            return png_buffer.getvalue()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "Muse Image returned data that is not a decodable image"
        ) from error


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
        f"Redraw this {label} as a glowing neon holographic game asset in "
        f"{theme} style, transparent background, 128x128."
    )
    response = client.images.generate(
        model="muse-image-1.0",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    return _png_bytes(response)
