"""Tests for the OpenAI and Bedrock embedders."""

import json
from unittest.mock import Mock, patch

import pytest

from obsidian_rag_mcp.rag.embedder import (
    BedrockEmbedder,
    BedrockEmbedderConfig,
    EmbedderConfig,
    OpenAIEmbedder,
)


class TestEmbedderConfig:
    """Test EmbedderConfig defaults and behavior."""

    def test_default_values(self):
        """Test default configuration."""
        config = EmbedderConfig()

        assert config.model == "text-embedding-3-small"
        assert config.batch_size == 100
        assert config.dimensions is None

    def test_custom_values(self):
        """Test custom configuration."""
        config = EmbedderConfig(
            model="text-embedding-3-large",
            batch_size=50,
            dimensions=512,
        )

        assert config.model == "text-embedding-3-large"
        assert config.batch_size == 50
        assert config.dimensions == 512


class TestOpenAIEmbedder:
    """Test OpenAIEmbedder with mocked OpenAI client."""

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_embed_single_text(self, mock_openai_class):
        """Test embedding a single text."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        mock_response = Mock()
        mock_response.data = [Mock(index=0, embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        # Test
        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed_text("Hello world")

        assert len(result) == 1536
        assert result == [0.1] * 1536
        mock_client.embeddings.create.assert_called_once()

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_embed_multiple_texts(self, mock_openai_class):
        """Test embedding multiple texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, embedding=[0.1] * 1536),
            Mock(index=1, embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")
        results = embedder.embed_texts(["Hello", "World"])

        assert len(results) == 2
        assert results[0] == [0.1] * 1536
        assert results[1] == [0.2] * 1536

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_empty_input(self, mock_openai_class):
        """Test handling of empty input."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        embedder = OpenAIEmbedder(api_key="test-key")
        results = embedder.embed_texts([])

        assert results == []
        mock_client.embeddings.create.assert_not_called()

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_batching(self, mock_openai_class):
        """Test that large inputs are batched correctly."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        # Return different embeddings for each batch
        call_count = [0]

        def create_side_effect(**kwargs):
            batch = kwargs["input"]
            response = Mock()
            response.data = [
                Mock(index=i, embedding=[call_count[0] + i / 100] * 1536)
                for i in range(len(batch))
            ]
            call_count[0] += 1
            return response

        mock_client.embeddings.create.side_effect = create_side_effect

        config = EmbedderConfig(batch_size=5)
        embedder = OpenAIEmbedder(api_key="test-key", config=config)

        # 12 texts should result in 3 batches (5, 5, 2)
        texts = [f"Text {i}" for i in range(12)]
        results = embedder.embed_texts(texts)

        assert len(results) == 12
        assert mock_client.embeddings.create.call_count == 3

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_text_cleaning(self, mock_openai_class):
        """Test that text is cleaned before embedding."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        mock_response = Mock()
        mock_response.data = [Mock(index=0, embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")

        # Text with null bytes and trailing whitespace
        dirty_text = "Hello\x00 World  \nLine2  "
        embedder.embed_text(dirty_text)

        # Check the cleaned text was sent
        call_args = mock_client.embeddings.create.call_args
        sent_text = call_args.kwargs["input"][0]
        assert "\x00" not in sent_text  # Null bytes removed
        # Note: we preserve internal whitespace now to maintain code structure
        # but trailing whitespace on lines is stripped
        assert not sent_text.endswith(" ")  # Trailing whitespace stripped

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_embedding_dimension_property(self, mock_openai_class):
        """Test embedding dimension property."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        # Default model
        embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder.embedding_dimension == 1536

        # Large model
        config = EmbedderConfig(model="text-embedding-3-large")
        embedder = OpenAIEmbedder(api_key="test-key", config=config)
        assert embedder.embedding_dimension == 3072

        # Custom dimensions
        config = EmbedderConfig(dimensions=512)
        embedder = OpenAIEmbedder(api_key="test-key", config=config)
        assert embedder.embedding_dimension == 512

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_missing_api_key(self, mock_openai_class):
        """Test error when no API key provided."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = None  # No key

        with pytest.raises(ValueError, match="API key required"):
            OpenAIEmbedder()

    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_batch_mismatch_raises_error(self, mock_openai_class):
        """Test that mismatched embedding count raises RuntimeError.

        This catches the case where OpenAI filters/rejects some content,
        preventing silent data corruption.
        """
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        # Return fewer embeddings than inputs (simulating filtered content)
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, embedding=[0.1] * 1536),
            # Missing index 1 - simulates filtered content
        ]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")

        with pytest.raises(RuntimeError, match="returned 1 embeddings for 2 inputs"):
            embedder.embed_texts(["Hello", "World"])


def _mock_titan_response(dimensions: int = 1024):
    """Build a mock boto3 invoke_model response shaped like Titan's output."""
    response = Mock()
    response.__getitem__ = lambda self, key: self._body if key == "body" else None
    body_mock = Mock()
    body_mock.read.return_value = json.dumps(
        {"embedding": [0.1] * dimensions, "inputTextTokenCount": 4}
    ).encode()
    response._body = body_mock
    return response


class TestBedrockEmbedder:
    """Test BedrockEmbedder with a mocked boto3 client."""

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_embed_single_text(self, mock_session_class):
        mock_client = Mock()
        mock_client.invoke_model.return_value = _mock_titan_response()
        mock_session_class.return_value.client.return_value = mock_client

        embedder = BedrockEmbedder()
        result = embedder.embed_text("Hello world")

        assert len(result) == 1024
        mock_client.invoke_model.assert_called_once()

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_embed_multiple_texts_preserves_order(self, mock_session_class):
        mock_client = Mock()
        mock_client.invoke_model.return_value = _mock_titan_response()
        mock_session_class.return_value.client.return_value = mock_client

        embedder = BedrockEmbedder()
        results = embedder.embed_texts(["a", "b", "c"])

        assert len(results) == 3
        assert mock_client.invoke_model.call_count == 3

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_oversized_text_is_truncated_below_token_ceiling(self, mock_session_class):
        """
        Regression test: text exceeding Titan's 8192-token input limit must
        be truncated by actual token count before being sent, not passed
        through untouched (which previously caused a 400 ValidationException
        from Bedrock -- "Too many input tokens").
        """
        from obsidian_rag_mcp.utils.tokens import count_tokens

        mock_client = Mock()
        mock_client.invoke_model.return_value = _mock_titan_response()
        mock_session_class.return_value.client.return_value = mock_client

        # Build text well over the 8000-token config ceiling.
        oversized_text = "word " * 20000
        assert count_tokens(oversized_text) > 8000

        embedder = BedrockEmbedder(config=BedrockEmbedderConfig(max_input_tokens=8000))
        embedder.embed_text(oversized_text, is_query=False)

        sent_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        sent_text = sent_body["inputText"]
        assert count_tokens(sent_text) <= 8000

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_short_text_is_not_truncated(self, mock_session_class):
        mock_client = Mock()
        mock_client.invoke_model.return_value = _mock_titan_response()
        mock_session_class.return_value.client.return_value = mock_client

        embedder = BedrockEmbedder()
        embedder.embed_text("short text", is_query=False)

        sent_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert sent_body["inputText"] == "short text"

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_empty_input(self, mock_session_class):
        mock_client = Mock()
        mock_session_class.return_value.client.return_value = mock_client

        embedder = BedrockEmbedder()
        results = embedder.embed_texts([])

        assert results == []
        mock_client.invoke_model.assert_not_called()

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_no_embedding_in_response_raises(self, mock_session_class):
        mock_client = Mock()
        response = Mock()
        body_mock = Mock()
        body_mock.read.return_value = json.dumps({}).encode()
        response.__getitem__ = lambda self, key: body_mock if key == "body" else None
        mock_client.invoke_model.return_value = response
        mock_session_class.return_value.client.return_value = mock_client

        embedder = BedrockEmbedder()
        with pytest.raises(RuntimeError, match="no embedding"):
            embedder.embed_text("hello")

    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_embedding_dimension_property(self, mock_session_class):
        mock_session_class.return_value.client.return_value = Mock()

        embedder = BedrockEmbedder(config=BedrockEmbedderConfig(dimensions=512))
        assert embedder.embedding_dimension == 512


class TestCreateEmbedder:
    """Test the create_embedder provider-selection factory."""

    @patch.dict("os.environ", {"EMBEDDING_PROVIDER": "bedrock"}, clear=False)
    @patch("obsidian_rag_mcp.rag.embedder.boto3.Session")
    def test_bedrock_provider_selected_via_env(self, mock_session_class):
        from obsidian_rag_mcp.rag.embedder import create_embedder

        mock_session_class.return_value.client.return_value = Mock()

        embedder = create_embedder()
        assert isinstance(embedder, BedrockEmbedder)

    @patch.dict("os.environ", {"EMBEDDING_PROVIDER": "openai"}, clear=False)
    @patch("obsidian_rag_mcp.rag.embedder.OpenAI")
    def test_openai_provider_is_default(self, mock_openai_class):
        from obsidian_rag_mcp.rag.embedder import create_embedder

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.api_key = "test-key"

        embedder = create_embedder(api_key="test-key")
        assert isinstance(embedder, OpenAIEmbedder)
