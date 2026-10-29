# Hackathon scope and file provenance

Work in this implementation was created or updated on October 29, 2026, during
the October 27-November 4 building window. The application remains at Phase 0;
later-phase functionality is identified as planned in the README and is not
claimed as completed. The project directory had no pre-existing application
scaffold when implementation began; the rules reference and demo photo below
were supplied by the user and are not claimed as original implementation work.

| File | What it contains / provenance |
|---|---|
| `.env.example` | Hackathon work: local API configuration template with the approved Baseten URL and exact Muse Spark Open model ID. |
| `.gitignore` | Hackathon work: excludes local secrets, generated files, and non-demo photos while retaining the designated demo image. |
| `README.md` | Hackathon work: track alignment, model usage, current status, setup, metrics, and submission reminders. |
| `HACKATHON_SCOPE.md` | Hackathon work: this file-by-file provenance and scope declaration. |
| `HACKATHON_RULES_AND_INFO.md` | User-provided track and submission reference; not original application work. |
| `assets/desk_smoke_test.jpeg` | User-provided photo for API smoke tests; not original application work. |
| `assets/sam_raw_sample.txt` | Hackathon-generated: raw first successful SAM 3.1 output for grammar inspection; created when segmentation succeeds. |
| `backend/main.py` | Hackathon work: FastAPI app, demo CORS, and health check. |
| `backend/models/__init__.py` | Hackathon work: Python package marker. |
| `backend/models/schemas.py` | Hackathon work: strict structured game-rules schema. |
| `backend/requirements.txt` | Hackathon work: backend and official SAM parser dependencies. |
| `backend/services/__init__.py` | Hackathon work: Python package marker. |
| `backend/services/image_service.py` | Hackathon work: Muse Image generation, PNG decoding, and per-run image cap. |
| `backend/services/metrics_service.py` | Hackathon work: atomic writes for generated demo metrics. |
| `backend/services/sam_service.py` | Hackathon work: concept-limited SAM 3.1 requests and official grammar/mask parsing. |
| `backend/services/spark_service.py` | Hackathon work: Meta-first Muse Spark rules, Baseten fallback, strict schema validation, and provider metrics. |
| `backend/smoke_tests.py` | Hackathon work: ordered live smoke-test runner for SAM, Meta Spark, Baseten Spark, Muse Image, and health. |
| `frontend/assets/.gitkeep` | Hackathon work: preserves the future static frontend asset directory. |
| `scripts/update_readme_metrics.py` | Hackathon work: Phase 5 helper that prints real demo metrics into the README. |
| `demo_metrics.json` | Generated and ignored: per-run API counts and Spark usage; not committed. |
| `.env` | Local-only and ignored: contains credentials; not committed or described here. |

No local ML model, PyTorch dependency, GPU requirement, or non-available
Fireworks/Modal credit is used or claimed.
