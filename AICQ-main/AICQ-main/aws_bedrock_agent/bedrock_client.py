import os
import json
import re
from typing import Tuple, Literal, Optional

import boto3

Label = Literal["dry", "medium", "wet"]


class BedrockMoistureClassifier:
    """Wraps Amazon Bedrock text classification for moisture level.

    Falls back to deterministic heuristic if Bedrock is unavailable unless BEDROCK_STRICT is set.
    """

    def __init__(self, region: Optional[str] = None, model_id: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 64):
        cfg_path = os.path.join(os.path.dirname(__file__), "config_aws.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or cfg.get("aws_region", "us-east-1")
        )
        model_cfg = cfg.get("bedrock", {})
        self.model_id = model_id or model_cfg.get("model_id")
        self.temperature = temperature if temperature is not None else float(model_cfg.get("temperature", 0.0))
        self.max_tokens = int(max_tokens or model_cfg.get("max_tokens", 64))
        self.strict = os.getenv("BEDROCK_STRICT", "0") not in ("0", "false", "False", None)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)

    @staticmethod
    def _prompt(user_text: str) -> str:
        return (
            "Classify the following human moisture description into one of: dry, medium, wet.\n"
            "Respond with only the single label.\n\n"
            f"Description: {user_text.strip()}\nLabel:"
        )

    @staticmethod
    def _postprocess_label(text: str) -> Label:
        text_norm = text.strip().lower()
        if "dry" in text_norm:
            return "dry"
        if "wet" in text_norm:
            return "wet"
        return "medium"

    def classify(self, user_text: str) -> Tuple[Label, float]:
        try:
            self._ensure_client()
            prompt = self._prompt(user_text)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
            })
            resp = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(resp["body"].read())
            text_out = ""
            for item in payload.get("content", []):
                if item.get("type") == "text":
                    text_out += item.get("text", "")
            label = self._postprocess_label(text_out)
            conf = 0.9 if label in text_out.lower() else 0.7
            return label, conf
        except Exception as e:
            if self.strict:
                raise
            return self._fallback(user_text)

    @staticmethod
    def _fallback(user_text: str) -> Tuple[Label, float]:
        text = user_text.lower()
        if re.search(r"\b(dry|arid|low|dusty)\b", text):
            return "dry", 0.7
        if re.search(r"\b(wet|soggy|high|muddy|waterlogged)\b", text):
            return "wet", 0.7
        return "medium", 0.6
