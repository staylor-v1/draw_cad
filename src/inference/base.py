"""Abstract base classes for LLM inference clients."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "system", "user", "assistant"
    content: str | list[dict[str, Any]]  # text or multimodal content


@dataclass
class LLMResponse:
    """Response from an LLM inference call."""
    content: str
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(abc.ABC):
    """Abstract base class for text-based LLM clients."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is reachable."""
        ...


class BaseVisionClient(abc.ABC):
    """Abstract base class for vision-capable LLM clients."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abc.abstractmethod
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
        """Analyze an image with a text prompt."""
        ...

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is reachable."""
        ...
