import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "demo_metrics.json"
README_PATH = PROJECT_ROOT / "README.md"
START_MARKER = "<!-- DEMO_METRICS_START -->"
END_MARKER = "<!-- DEMO_METRICS_END -->"


def _render_metrics() -> str:
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    calls = metrics.get("spark_calls", [])
    if not calls:
        return "No successful Muse Spark calls have been recorded yet."

    lines = [
        "| Provider | Model | Prompt tokens | Completion tokens | Latency (ms) | Estimated cost (USD) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for call in calls:
        estimated_cost = call.get("estimated_cost_usd")
        cost_text = (
            "unknown"
            if estimated_cost is None
            else f"{estimated_cost:.6f}"
        )
        lines.append(
            "| {provider} | `{model}` | {prompt} | {completion} | {latency} | {cost} |".format(
                provider=call["provider"],
                model=call["model_id"],
                prompt=call["prompt_tokens"],
                completion=call["completion_tokens"],
                latency=call["latency_ms"],
                cost=cost_text,
            )
        )
    return "\n".join(lines)


def main() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    start = readme.index(START_MARKER) + len(START_MARKER)
    end = readme.index(END_MARKER, start)
    rendered = _render_metrics()
    updated = readme[:start] + "\n\n" + rendered + "\n\n" + readme[end:]
    README_PATH.write_text(updated, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
