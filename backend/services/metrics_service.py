import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

METRICS_PATH = Path(__file__).resolve().parents[2] / "demo_metrics.json"
_METRICS_LOCK = threading.Lock()


def _update_metrics(update) -> None:
    with _METRICS_LOCK:
        if METRICS_PATH.exists():
            with METRICS_PATH.open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
        else:
            metrics = {}

        update(metrics)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=METRICS_PATH.parent, prefix=".demo_metrics."
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
                metrics_file.write("\n")
            os.replace(temporary_path, METRICS_PATH)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise


def record_sam_call_count(count: int) -> None:
    _update_metrics(lambda metrics: metrics.update(sam_call_count=count))


def record_spark_call(call: dict[str, Any]) -> None:
    def update(metrics):
        metrics.setdefault("spark_calls", []).append(call)

    _update_metrics(update)


def record_muse_image_count(count: int) -> None:
    _update_metrics(lambda metrics: metrics.update(muse_image_count=count))
