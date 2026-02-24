"""vLLM inference client implementation (OpenAI-compatible API)."""
from __future__ import annotations

from typing import Any

import httpx

from src.inference.base import BaseLLMClient, BaseVisionClient, ChatMessage, LLMResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _format_openai_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage objects to OpenAI API format."""
    formatted = []
    for msg in messages:
        if isinstance(msg.content, str):
            formatted.append({"role": msg.role, "content": msg.content})
        else:
            formatted.append({"role": msg.role, "content": msg.content})
    return formatted


class VLLMLLMClient(BaseLLMClient):
    """vLLM-based text LLM client using OpenAI-compatible API."""

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        formatted = _format_openai_messages(messages)
        payload = {
            "model": model,
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info("vllm_chat_request", model=model, num_messages=len(messages))
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.error("vllm_chat_error", error=str(e))
            raise

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason", ""),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


class VLLMVisionClient(BaseVisionClient):
    """vLLM-based vision client using OpenAI-compatible API."""

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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                        },
                    },
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info("vllm_vision_request", model=model)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.error("vllm_vision_error", error=str(e))
            raise

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason", ""),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
