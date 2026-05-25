"""
Lightweight Lambda entrypoint for Bedrock moisture classification.

This handler avoids heavier dependencies (pandas, numpy, sklearn, etc.) so
the deployment package remains below the AWS Lambda size limits. It exposes a
minimal REST-like interface compatible with the original FastAPI routes that
only rely on the Bedrock text classifier.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

from aws_bedrock_agent.bedrock_client import BedrockMoistureClassifier

_classifier: BedrockMoistureClassifier | None = None


def _get_classifier() -> BedrockMoistureClassifier:
    """Reuse the classifier instance across invocations to preserve warm state."""
    global _classifier
    if _classifier is None:
        _classifier = BedrockMoistureClassifier()
    return _classifier


def _load_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Decode JSON body from API Gateway/Lambda proxy event."""
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Format a Lambda proxy integration response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


def _classify_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("input", "") or "")
    if not text:
        raise ValueError("'input' is required")
    label, confidence = _get_classifier().classify(text)
    return {"label": label, "confidence": confidence}


def _classify_many(payload: Dict[str, Any]) -> Dict[str, Any]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("'inputs' must be a non-empty list of strings")
    labels: List[str] = []
    confidences: List[float] = []
    clf = _get_classifier()
    for item in inputs:
        label, confidence = clf.classify(str(item or ""))
        labels.append(label)
        confidences.append(confidence)
    return {"labels": labels, "confidences": confidences}


def _unsupported(feature: str) -> Dict[str, Any]:
    return {
        "error": f"{feature} is not available in the lightweight Lambda bundle.",
        "hint": "Deploy the full application (see README 'Full pipeline' section) "
                "or back the GPR stage with a SageMaker endpoint.",
    }


def handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler compatible with API Gateway's proxy payload.

    Supported endpoints:
    - POST /classify        -> {"input": "..."}
    - POST /classify_many   -> {"inputs": ["...", "..."]}

    All other routes respond with 501 to signal that the heavy GPR pipeline is
    not bundled in the minimal deployment artifact.
    """
    method = event.get("httpMethod", "POST").upper()
    path = (event.get("rawPath") or event.get("path") or "").rstrip("/") or "/"

    if method != "POST":
        return _response(405, {"error": f"{method} not allowed"})

    try:
        body = _load_body(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    if path.endswith("/classify"):
        try:
            return _response(200, _classify_single(body))
        except ValueError as exc:
            return _response(400, {"error": str(exc)})

    if path.endswith("/classify_many"):
        try:
            return _response(200, _classify_many(body))
        except ValueError as exc:
            return _response(400, {"error": str(exc)})

    if path.endswith("/process_dataset") or path.endswith("/process_csv_text"):
        return _response(501, _unsupported("Dataset processing"))

    return _response(404, {"error": f"No route for {path}"})

