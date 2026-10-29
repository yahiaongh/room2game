import json
import os
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import ValidationError

from models.schemas import GameRules
from services.metrics_service import record_spark_call

load_dotenv()

FORCE_BASETEN = os.getenv("FORCE_BASETEN") == "1"
BASETEN_SPARK_MODEL_ID = "meta-models/muse-spark-open-1026"
META_SPARK_MODEL_ID = "muse-spark-1.3"
SYSTEM_PROMPT = (
    "You are a game designer. Given a list of real-world objects with labels, "
    "assign each to a platform, hazard, or collectible. Be creative but safe: "
    "never make a laptop a collectible; make it a wall or a hazard if it has a "
    "screen. Output valid JSON only."
)
STRICT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " Return exactly one JSON object with all six required top-level keys, "
    "correctly typed values, no extra keys, and no surrounding prose."
)

RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "player_start": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "platforms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_index": {"type": "integer"},
                    "role": {"type": "string", "enum": ["solid"]},
                },
                "required": ["object_index", "role"],
                "additionalProperties": False,
            },
        },
        "hazards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_index": {"type": "integer"},
                    "role": {"type": "string", "enum": ["hazard"]},
                    "damage": {"type": "integer"},
                },
                "required": ["object_index", "role", "damage"],
                "additionalProperties": False,
            },
        },
        "collectibles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_index": {"type": "integer"},
                    "role": {"type": "string", "enum": ["collectible"]},
                    "points": {"type": "integer"},
                },
                "required": ["object_index", "role", "points"],
                "additionalProperties": False,
            },
        },
        "win_condition": {"type": "string"},
        "theme": {"type": "string"},
    },
    "required": [
        "player_start",
        "platforms",
        "hazards",
        "collectibles",
        "win_condition",
        "theme",
    ],
    "additionalProperties": False,
}


def _client(provider: str) -> tuple[OpenAI, str]:
    if provider == "meta":
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise RuntimeError("MODEL_API_KEY is required for Muse Spark")
        return (
            OpenAI(
                api_key=api_key,
                base_url="https://api.meta.ai/v1",
                timeout=20.0,
                max_retries=0,
            ),
            META_SPARK_MODEL_ID,
        )

    model_id = os.environ["BASETEN_SPARK_MODEL"]
    if model_id != BASETEN_SPARK_MODEL_ID:
        raise ValueError(
            "BASETEN_SPARK_MODEL must be exactly "
            f"{BASETEN_SPARK_MODEL_ID!r}"
        )
    return (
        OpenAI(
            api_key=os.environ["BASETEN_API_KEY"],
            base_url=os.environ["BASETEN_BASE_URL"],
            timeout=20.0,
            max_retries=0,
        ),
        model_id,
    )


def _validated_rules(content: str) -> dict[str, Any]:
    rules_data = json.loads(content)
    model_validate = getattr(GameRules, "model_validate", None)
    if callable(model_validate):
        return model_validate(rules_data).model_dump()
    return GameRules.parse_obj(rules_data).dict()


def _request_rules(
    provider: str, model_id: str, objects: list[dict], system_prompt: str
):
    client, configured_model_id = _client(provider)
    if configured_model_id != model_id:
        raise ValueError(f"Unexpected {provider} Muse Spark model id")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Return game rules matching this exact JSON schema:\n"
                    + json.dumps(RULES_SCHEMA, separators=(",", ":"))
                    + "\nSegmented objects:\n"
                    + json.dumps(objects, separators=(",", ":"))
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Muse Spark returned an empty rules response")
    return response, _validated_rules(content)


def generate_game_rules(objects: list[dict]) -> dict:
    providers = (
        [("baseten", BASETEN_SPARK_MODEL_ID)]
        if FORCE_BASETEN
        else [
            ("meta", META_SPARK_MODEL_ID),
            ("baseten", BASETEN_SPARK_MODEL_ID),
        ]
    )
    last_validation_error = None

    for provider, model_id in providers:
        for attempt in range(2):
            started_at = perf_counter()
            try:
                response, rules = _request_rules(
                    provider,
                    model_id,
                    objects,
                    SYSTEM_PROMPT if attempt == 0 else STRICT_SYSTEM_PROMPT,
                )
            except (APITimeoutError, APIConnectionError):
                if provider == "meta" and not FORCE_BASETEN:
                    break
                raise
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_validation_error = error
                if attempt == 1:
                    raise ValueError(
                        "Muse Spark returned invalid game rules after one retry"
                    ) from error
                continue

            elapsed_ms = round((perf_counter() - started_at) * 1000)
            usage = response.usage
            record_spark_call(
                {
                    "provider": provider,
                    "model_id": model_id,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "latency_ms": elapsed_ms,
                    "estimated_cost_usd": None,
                }
            )
            return rules

    if last_validation_error is not None:
        raise ValueError(
            "Muse Spark did not return valid game rules"
        ) from last_validation_error
    raise RuntimeError("No Muse Spark provider returned valid game rules")
