"""Ollama inference client implementation."""
from __future__ import annotations

from typing import Any

import httpx

from src.inference.base import BaseLLMClient, BaseVisionClient, ChatMessage, LLMResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _format_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage objects to Ollama API format."""
    formatted = []
    for msg in messages:
        if isinstance(msg.content, str):
            formatted.append({"role": msg.role, "content": msg.content})
        else:
            # Multimodal: extract text and images
            text_parts = []
            images = []
            for part in msg.content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        # Extract base64 data after the comma
                        b64 = url.split(",", 1)[1] if "," in url else url
                        images.append(b64)
            entry: dict[str, Any] = {"role": msg.role, "content": "\n".join(text_parts)}
            if images:
                entry["images"] = images
            formatted.append(entry)
    return formatted


def _build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an Ollama chat payload with pass-through support for advanced options."""
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    extra_options = kwargs.pop("options", None)
    if isinstance(extra_options, dict):
        options.update(extra_options)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    for key in ("format", "keep_alive", "raw", "template", "tools", "think"):
        if key in kwargs:
            payload[key] = kwargs.pop(key)

    payload.update(kwargs)
    return payload


class OllamaLLMClient(BaseLLMClient):
    """Ollama-based text LLM client."""

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        formatted = _format_messages(messages)
        payload = _build_payload(
            model=model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        logger.info("ollama_chat_request", model=model, num_messages=len(messages))
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.error("ollama_chat_error", error=str(e))
            raise

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            finish_reason=data.get("done_reason", ""),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


class OllamaVisionClient(BaseVisionClient):
    """Ollama-based vision client."""

    def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        model: str,
        mime_type: str = "image/png",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = _build_payload(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        logger.info("ollama_vision_request", model=model)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.error("ollama_vision_error", error=str(e))
            raise

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            finish_reason=data.get("done_reason", ""),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
