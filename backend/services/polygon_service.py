import io
import logging

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MIN_AREA_RATIO = 0.005
MAX_POLYGON_VERTICES = 32
MAX_APPROXIMATION_RETRIES = 3


def _vertices(contour: np.ndarray) -> list[dict[str, float]]:
    return [
        {"x": float(point[0][0]), "y": float(point[0][1])}
        for point in contour
    ]


def mask_png_to_polygon(
    mask_png_bytes: bytes, canvas_width: int, canvas_height: int
) -> dict | None:
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive")

    try:
        with Image.open(io.BytesIO(mask_png_bytes)) as mask_image:
            grayscale = mask_image.convert("L")
            mask = np.asarray(grayscale)
    except (UnidentifiedImageError, OSError):
        return None

    if mask.size == 0:
        return None
    resized_mask = cv2.resize(
        mask,
        (canvas_width, canvas_height),
        interpolation=cv2.INTER_NEAREST,
    )
    _, threshold = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(largest))
    area_ratio = contour_area / (canvas_width * canvas_height)
    if area_ratio < MIN_AREA_RATIO:
        return None

    epsilon = 0.02 * cv2.arcLength(largest, True)
    display = cv2.approxPolyDP(largest, epsilon, True)
    retries = 0
    while (
        len(display) > MAX_POLYGON_VERTICES
        and retries < MAX_APPROXIMATION_RETRIES
    ):
        epsilon *= 1.5
        display = cv2.approxPolyDP(largest, epsilon, True)
        retries += 1

    if len(display) < 3 or len(display) > MAX_POLYGON_VERTICES:
        return None

    physics = display if cv2.isContourConvex(display) else cv2.convexHull(display)
    if len(physics) < 3 or len(physics) > MAX_POLYGON_VERTICES:
        return None

    def clamp_vertices(contour: np.ndarray) -> list[dict[str, float]]:
        vertices = _vertices(contour)
        for vertex in vertices:
            rounded_x = round(vertex["x"])
            rounded_y = round(vertex["y"])
            if not (0 <= rounded_x <= canvas_width and 0 <= rounded_y <= canvas_height):
                logger.warning(
                    "Clamping polygon vertex outside canvas: (%s, %s)",
                    rounded_x,
                    rounded_y,
                )
                vertex["x"] = float(min(max(rounded_x, 0), canvas_width))
                vertex["y"] = float(min(max(rounded_y, 0), canvas_height))
        assert all(
            0 <= round(vertex["x"]) <= canvas_width
            and 0 <= round(vertex["y"]) <= canvas_height
            for vertex in vertices
        )
        return vertices

    return {
        "physics": clamp_vertices(physics),
        "display": clamp_vertices(display),
        "area_ratio": float(area_ratio),
    }
