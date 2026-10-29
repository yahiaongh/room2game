# Room2Game

## One-line pitch

Turn any real-world photo into a playable simulation for embodied-agent
training, with the game as the demo surface.

## How it maps to Push the Frontier

Room2Game explores real-to-simulation by extracting objects and their geometry
from an ordinary desk or room photo, then mapping them into a small interactive
world. `POST /generate` returns filtered object polygons, validated game rules,
and resized canvas dimensions; the Phaser 3 frontend turns that response into a
Matter.js platformer with keyboard and touch controls. The resized source photo
is the game background, with translucent cyan, red, and gold holographic
outlines tracing platforms, hazards, and collectibles; physics polygons remain
invisible. Use A/D or the arrow keys to move; Space, W, Up, or touch jumps.
Muse Image restyles optional object sprites asynchronously. This can become a
lightweight source of varied scenes for robotics training and evaluation. The
longer-term direction is interactive world models: agents should be able to
inspect a real environment, reason about its objects, and act on a simulated
version. The playable platformer is the demo surface for that pipeline, not the
end goal.

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
Images API. `POST /restyle` crops objects from the source photo and caches
generated PNGs by image, object, and theme. The browser requests sprites without
blocking gameplay; if generation is unavailable, the polygon remains visible.
A per-photo cap of three images limits generation cost.

## Which bonus prizes this targets

- **Best Build on Meta Open Weight Models:** Muse Spark Open is integrated
  through the Baseten partner endpoint using `meta-models/muse-spark-open-1026`;
  the forced-provider smoke test verifies that route independently.
- **Best Agentic Use-Case:** The planned Phase 3 `/validate` critic-and-patch
  retry loop targets this prize. The validation loop is not implemented yet.

## What was built during the hackathon

The implementation and file-by-file provenance are listed in
[HACKATHON_SCOPE.md](./HACKATHON_SCOPE.md). Entries marked pre-existing were
provided before this implementation and are not claimed as original work.

## Demo mode fallback

The frontend provides the live photo upload and game runtime. A pre-generated
scene fallback is planned for Phase 4 and is not implemented yet. The
`/generate` endpoint can return fallback rules if Spark rules are invalid or
time out, but image generation still requires SAM. The backend health endpoint
can be checked without calling a model API.

## Setup

Copy `.env.example` to `.env`, set `MODEL_API_KEY` and `BASETEN_API_KEY`
locally, and leave the Baseten base URL and model identifier at their documented
values. Never commit `.env`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

Run the API from the `backend` directory:

```bash
cd backend
uvicorn main:app --reload
```

In another terminal, serve the static frontend from the project root (no build
step or npm install):

```bash
python3 -m http.server 8001 --directory frontend
```

Open `http://localhost:8001`. The frontend sends photo uploads to
`http://localhost:8000/generate`; optionally set the `theme` query parameter.
`GET http://localhost:8000/health` is the API health check. Response
coordinates use the resized canvas space. `/generate` also returns a `meta`
object with `debug_sam_status` and `rules_fallback`, making SAM API failures,
empty results, filtered masks, and rules fallback visible in the response and
server log. The response `image_id` is used by `POST /restyle`; send
`{"image_id":"<id>","object_index":0,"theme":"pixel art"}` as JSON. Generated
assets and source crops are cached locally under `assets/restyle_cache/` and
ignored by git.
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

The public repository is [github.com/yahiaongh/room2game](https://github.com/yahiaongh/room2game).
Keep the demo video to three minutes maximum and reserve Phase 6 for a
single-take recording with demo mode enabled as the safety net.
