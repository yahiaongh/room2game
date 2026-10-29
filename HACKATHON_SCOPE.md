# Hackathon scope and file provenance

Work in this implementation was created or updated on October 29, 2026, during
the October 27-November 4 building window. Phase 2 now includes the static
Phaser 3/Matter.js game runtime and browser controls. Phase 3 validation,
Phase 4 demo-mode fallback, and Phase 5 submission materials remain planned and
are not claimed as completed. The project directory had no pre-existing
application scaffold when implementation began; the rules reference and demo
photos were supplied by the user and are not claimed as original implementation
work.

| File | What it contains / provenance |
|---|---|
| `.env.example` | Hackathon work: local API configuration template with the approved Baseten URL and exact Muse Spark Open model ID. |
| `.gitignore` | Hackathon work: excludes local secrets, generated files, and non-demo photos while retaining the designated demo image. |
| `README.md` | Hackathon work: track alignment, model usage, current status, setup, metrics, and submission reminders. |
| `HACKATHON_SCOPE.md` | Hackathon work: this file-by-file provenance and scope declaration. |
| `HACKATHON_RULES_AND_INFO.md` | User-provided track and submission reference; not original application work. |
| `assets/sam_raw_sample.txt` | Hackathon-generated: raw first successful SAM 3.1 output for grammar inspection; created when segmentation succeeds. |
| `backend/main.py` | Hackathon work: FastAPI app, demo CORS, health check, and Phase 1 multipart photo-to-scene pipeline. |
| `backend/models/__init__.py` | Hackathon work: Python package marker. |
| `backend/models/schemas.py` | Hackathon work: strict structured game-rules schema. |
| `backend/requirements.txt` | Hackathon work: backend, official SAM parser, and headless OpenCV dependencies. |
| `backend/services/__init__.py` | Hackathon work: Python package marker. |
| `backend/services/image_service.py` | Hackathon work: Muse Image generation, PNG decoding, and per-run image cap. |
| `backend/services/metrics_service.py` | Hackathon work: atomic writes for generated demo metrics. |
| `backend/services/polygon_service.py` | Hackathon work: mask thresholding, contour simplification, and bounded physics polygons. |
| `backend/services/sam_service.py` | Hackathon work: concept-limited SAM 3.1 requests and official grammar/mask parsing. |
| `backend/services/spark_service.py` | Hackathon work: Meta-first Muse Spark rules, Baseten fallback, strict schema validation, and provider metrics. |
| `backend/smoke_tests.py` | Hackathon work: ordered live smoke-test runner for SAM, Meta Spark, Baseten Spark, Muse Image, and health. |
| `frontend/assets/.gitkeep` | Hackathon work: preserves the future static frontend asset directory. |
| `frontend/game.js` | Hackathon work: Phaser 3/Matter.js runtime that renders generated polygons, creates physics and sensor bodies, and supports keyboard, touch, scoring, hazard resets, and replay. |
| `frontend/index.html` | Hackathon work: responsive static photo-upload interface, theme selector, game canvas, status, score, and Phaser CDN loading. |
| `scripts/update_readme_metrics.py` | Hackathon work: Phase 5 helper that prints real demo metrics into the README. |
| `tests/test_phase1_pipeline.py` | Hackathon work: standard-library tests for polygon filtering and the mocked `/generate` endpoint. |
| `demo_metrics.json` | Generated and ignored: per-run API counts and Spark usage; not committed. |
| `.env` | Local-only and ignored: contains credentials; not committed or described here. |

No local ML model, PyTorch dependency, GPU requirement, or non-available
Fireworks/Modal credit is used or claimed. Phase 1 adds backend segmentation,
polygon extraction, role validation, deterministic rules fallback, and an
injected ground platform. Phase 2 adds a static Phaser 3 frontend with Matter.js
physics; this project still has no database or authentication.
