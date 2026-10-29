# Room2Game

## One-line pitch

Turn any real-world photo into a playable simulation for embodied-agent
training, with the game as the demo surface.

## How it maps to Push the Frontier

Room2Game explores real-to-simulation by extracting objects and their geometry
from an ordinary desk or room photo, then mapping them into a small interactive
world. `POST /generate` currently returns filtered object polygons, validated
game rules, and the resized canvas dimensions. This can become a lightweight
source of varied scenes for robotics
training and evaluation. The longer-term direction is interactive world models:
agents should be able to inspect a real environment, reason about its objects,
and act on a simulated version. The playable platformer is the demo surface for
that pipeline, not the end goal.

## Which Meta models are used and how

**SAM 3.1** segments one requested room object concept per API call, with at
most five concepts requested per image. The `meta-sam-parser` package decodes
SAM's special-token output into masks and bounding boxes; OpenCV extracts
simplified display and convex physics polygons. This project does not run local
ML models.

**Muse Spark 1.3 / Muse Spark Open** assigns game roles to detected objects and
returns schema-validated JSON rules. The Meta Model API is tried first;
timeouts and connection failures can fall back to Baseten's exact open model
identifier, `meta-models/muse-spark-open-1026`. `FORCE_BASETEN=1` exercises the
partner endpoint directly. If generated rules are invalid or the request times
out, `/generate` builds safe deterministic fallback rules.

**Muse Image** generates a pixel-art asset for an object label through the Meta
Images API. A per-run cap of three images limits generation cost.

## Which bonus prizes this targets

- **Best Build on Meta Open Weight Models:** Muse Spark Open is integrated
  through the Baseten partner endpoint using `meta-models/muse-spark-open-1026`;
  the forced-provider smoke test verifies that route independently.
- **Best Agentic Use-Case:** The planned Phase 3 `/validate` critic-and-patch
  retry loop targets this prize. The validation loop is not implemented in the
  current Phase 0 scaffold.

## What was built during the hackathon

The Phase 0 implementation and file-by-file provenance are listed in
[HACKATHON_SCOPE.md](./HACKATHON_SCOPE.md). Entries marked pre-existing were
provided before this implementation and are not claimed as original work.

## Demo mode fallback

Phase 4 is intended to add a pre-generated demo scene so judges can try the
experience when APIs or Wi-Fi are unavailable. That frontend fallback is not
implemented in Phase 1. The `/generate` endpoint can return fallback rules if
Spark rules are invalid or time out, but image generation still requires SAM.
The backend health endpoint can be checked without calling a model API.

## Setup

Copy `.env.example` to `.env`, set `MODEL_API_KEY` and `BASETEN_API_KEY`
locally, and leave the Baseten base URL and model identifier at their documented
values. Never commit `.env`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload
```

`GET http://localhost:8000/health` is the health check. Upload a photo as the
`image` multipart field to `POST http://localhost:8000/generate`; optionally set
the `theme` query parameter. Response coordinates use the resized canvas space.
The Phase 0 SAM, Spark, and Muse Image smoke tests are run with
`python backend/smoke_tests.py` from the project root after API access is
configured. Phase 1 geometry and endpoint tests run without model API calls:

```bash
python -m unittest discover -s tests -v
```

## Demo metrics

Spark token usage and latency are recorded in the ignored, regenerated
`demo_metrics.json`. The Phase 5 README update step is
`python scripts/update_readme_metrics.py`; it prints and inserts the recorded
provider, model, token, latency, and cost fields instead of making unsupported
performance claims. Estimated cost is left unknown when the provider does not
return pricing data.

<!-- DEMO_METRICS_START -->

| Provider | Model | Prompt tokens | Completion tokens | Latency (ms) | Estimated cost (USD) |
|---|---|---:|---:|---:|---:|
| meta | `muse-spark-1.3` | 369 | 1175 | 6383 | unknown |
| baseten | `meta-models/muse-spark-open-1026` | 502 | 1015 | 4881 | unknown |

<!-- DEMO_METRICS_END -->

## Submission status

The repository must be public on GitHub before submission. No GitHub remote is
configured yet; create the public repository and push the dated commits before
the submission deadline. Keep the demo video to three minutes maximum and
reserve Phase 6 for a single-take recording with demo mode enabled as the
safety net.
