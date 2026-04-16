import json
import logging
from typing import Any

import requests

from config import get_ollama_config
from schemas import StructuredReport


logger = logging.getLogger(__name__)


def _compact_prediction_payload(prediction_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": prediction_result.get("prediction"),
        "confidence": prediction_result.get("confidence"),
        "class_probabilities": prediction_result.get("class_probabilities", {}),
        "active_classes": prediction_result.get("active_classes", []),
        "detections": prediction_result.get("detections", []),
    }


def _build_prompt(prediction_result: dict[str, Any]) -> str:
    payload = _compact_prediction_payload(prediction_result)
    payload_json = json.dumps(payload, ensure_ascii=True)

    return (
        "You are a medical imaging assistant. "
        "Generate a minimal structured report from model output only. "
        "Do not invent unsupported findings. Keep language concise.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "diagnosis_summary": "string",\n'
        '  "risk_level": "low|moderate|high",\n'
        '  "key_findings": ["string", "..."],\n'
        '  "recommendations": ["string", "..."],\n'
        '  "confidence_note": "string",\n'
        '  "disclaimer": "AI-generated summary for informational use only. Not a clinical diagnosis."\n'
        "}\n\n"
        "Rules:\n"
        "- Keep key_findings max 4 bullets, recommendations max 4 bullets.\n"
        "- If detections are empty and prediction is notumor, clearly state no tumor box detected.\n"
        "- Confidence note should mention confidence percentage quality (high/moderate/low confidence).\n"
        "- Do not include markdown or code fences.\n\n"
        f"Model output JSON:\n{payload_json}"
    )


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
    return stripped


def _call_ollama(prompt: str) -> tuple[str | None, str | None]:
    cfg = get_ollama_config()

    endpoint = f"{cfg.base_url.rstrip('/')}/api/generate"
    payload = {
        "model": cfg.model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.temperature,
        },
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=cfg.timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.info("Ollama connection failed: %s", exc)
        return None, "unavailable_ollama_connection_failed"

    if response.status_code == 404:
        logger.info("Ollama model not found: %s", cfg.model_name)
        return None, "unavailable_ollama_model_not_found"

    if response.status_code >= 400:
        logger.warning("Ollama generation failed: %s %s", response.status_code, response.text)
        return None, "unavailable_ollama_generation_failed"

    try:
        body = response.json()
    except ValueError:
        logger.warning("Ollama returned non-JSON response body.")
        return None, "unavailable_ollama_generation_failed"

    raw_text = str(body.get("response", "") or "").strip()
    if not raw_text:
        return None, "unavailable_ollama_generation_failed"

    return raw_text, None


def generate_structured_report(
    prediction_result: dict[str, Any],
) -> tuple[StructuredReport | None, str]:
    cfg = get_ollama_config()
    if not cfg.enabled:
        return None, "unavailable_ollama_not_configured"

    prompt = _build_prompt(prediction_result)
    raw_text, error_status = _call_ollama(prompt)
    if raw_text is None:
        return None, error_status or "unavailable_ollama_generation_failed"

    try:
        json_text = _extract_json(raw_text)
        parsed = json.loads(json_text)
        report = StructuredReport.model_validate(parsed)
        return report, "ok"
    except Exception as exc:
        logger.warning("Structured report parse/validation failed: %s", exc)
        return None, "unavailable_ollama_invalid_output"
