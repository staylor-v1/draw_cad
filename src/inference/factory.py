"""Factory for creating inference clients based on configuration."""
from __future__ import annotations

from src.inference.base import BaseLLMClient, BaseVisionClient
from src.inference.llamacpp_client import LlamaCppLLMClient, LlamaCppVisionClient
from src.inference.ollama_client import OllamaLLMClient, OllamaVisionClient
from src.inference.vllm_client import VLLMLLMClient, VLLMVisionClient
from src.schemas.pipeline_config import PipelineConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_LLM_BACKENDS: dict[str, type[BaseLLMClient]] = {
    "ollama": OllamaLLMClient,
    "vllm": VLLMLLMClient,
    "llamacpp": LlamaCppLLMClient,
}

_VISION_BACKENDS: dict[str, type[BaseVisionClient]] = {
    "ollama": OllamaVisionClient,
    "vllm": VLLMVisionClient,
    "llamacpp": LlamaCppVisionClient,
}


def create_llm_client(config: PipelineConfig) -> BaseLLMClient:
    """Create a text LLM client from pipeline configuration."""
    backend_name = config.models.reasoning.backend
    cls = _LLM_BACKENDS.get(backend_name)
    if cls is None:
        raise ValueError(f"Unknown LLM backend: {backend_name}. Available: {list(_LLM_BACKENDS)}")
    backend_config = getattr(config.inference, backend_name)

    logger.info("creating_llm_client", backend=backend_name, base_url=backend_config.base_url)
    return cls(base_url=backend_config.base_url, timeout=backend_config.timeout)


def create_vision_client(config: PipelineConfig) -> BaseVisionClient:
    """Create a vision LLM client from pipeline configuration."""
    backend_name = config.models.vision.backend
    cls = _VISION_BACKENDS.get(backend_name)
    if cls is None:
        raise ValueError(f"Unknown vision backend: {backend_name}. Available: {list(_VISION_BACKENDS)}")
    backend_config = getattr(config.inference, backend_name)

    logger.info("creating_vision_client", backend=backend_name, base_url=backend_config.base_url)
    return cls(base_url=backend_config.base_url, timeout=backend_config.timeout)
