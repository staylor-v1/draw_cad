"""Tests for inference abstraction layer."""
import pytest
from unittest.mock import MagicMock, patch

from src.inference.base import BaseLLMClient, BaseVisionClient, ChatMessage, LLMResponse
from src.inference.factory import create_llm_client, create_vision_client
from src.inference.ollama_client import OllamaLLMClient, OllamaVisionClient
from src.inference.vllm_client import VLLMLLMClient, VLLMVisionClient
from src.inference.llamacpp_client import LlamaCppLLMClient, LlamaCppVisionClient
from src.schemas.pipeline_config import PipelineConfig


class TestChatMessage:
    def test_text_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_multimodal_message(self):
        content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        msg = ChatMessage(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestLLMResponse:
    def test_basic_response(self):
        resp = LLMResponse(content="Hello world", model="test-model")
        assert resp.content == "Hello world"
        assert resp.model == "test-model"
        assert resp.finish_reason == ""


class TestFactory:
    def test_create_llm_client_ollama(self):
        config = PipelineConfig()
        config.models.reasoning.backend = "ollama"
        client = create_llm_client(config)
        assert isinstance(client, OllamaLLMClient)

    def test_create_llm_client_vllm(self):
        config = PipelineConfig()
        config.models.reasoning.backend = "vllm"
        client = create_llm_client(config)
        assert isinstance(client, VLLMLLMClient)

    def test_create_llm_client_llamacpp(self):
        config = PipelineConfig()
        config.models.reasoning.backend = "llamacpp"
        client = create_llm_client(config)
        assert isinstance(client, LlamaCppLLMClient)

    def test_create_vision_client_ollama(self):
        config = PipelineConfig()
        config.models.vision.backend = "ollama"
        client = create_vision_client(config)
        assert isinstance(client, OllamaVisionClient)

    def test_create_llm_client_invalid_backend(self):
        config = PipelineConfig()
        config.models.reasoning.backend = "invalid"
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_llm_client(config)

    def test_create_vision_client_invalid_backend(self):
        config = PipelineConfig()
        config.models.vision.backend = "invalid"
        with pytest.raises(ValueError, match="Unknown vision backend"):
            create_vision_client(config)


class TestOllamaClient:
    @patch("httpx.Client")
    def test_chat_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Generated code here"},
            "model": "gpt-oss-120b",
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = OllamaLLMClient(base_url="http://localhost:11434")
        messages = [ChatMessage(role="user", content="Generate code")]
        response = client.chat(messages, model="gpt-oss-120b")

        assert response.content == "Generated code here"
        assert response.model == "gpt-oss-120b"

    @patch("httpx.Client")
    def test_health_check(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = OllamaLLMClient(base_url="http://localhost:11434")
        assert client.health_check() is True
