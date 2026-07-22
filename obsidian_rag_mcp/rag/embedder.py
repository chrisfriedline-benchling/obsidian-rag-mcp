"""
Embedding wrappers with batching, retries, and logging.

Supports OpenAI, Azure OpenAI, and Amazon Bedrock (Titan Embeddings V2) as
embedding backends. Azure OpenAI is auto-detected when AZURE_OPENAI_ENDPOINT
and AZURE_API_KEY environment variables are set. Bedrock is selected by
setting EMBEDDING_PROVIDER=bedrock -- useful when running Claude via Bedrock
and no OpenAI API key is available; embeddings never touch OpenAI at all in
that mode.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """Configuration for the embedder."""

    model: str = "text-embedding-3-small"
    batch_size: int = 100  # OpenAI allows up to 2048
    dimensions: int | None = None  # Use model default
    query_max_chars: int = 8000  # Stricter limit for queries


def _create_openai_client(api_key: str | None = None) -> OpenAI:
    """
    Create an OpenAI client, auto-detecting Azure OpenAI when configured.

    Precedence:
    1. Explicit api_key parameter → standard OpenAI
    2. AZURE_OPENAI_ENDPOINT + AZURE_API_KEY → Azure OpenAI
    3. OPENAI_API_KEY env var → standard OpenAI
    """
    # Explicit api_key takes precedence over Azure env vars
    if api_key:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if azure_endpoint:
            logger.warning(
                "Explicit api_key provided; ignoring AZURE_OPENAI_ENDPOINT. "
                "Remove api_key to use Azure OpenAI."
            )
        return OpenAI(api_key=api_key)

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_api_key = os.getenv("AZURE_API_KEY", "")
    azure_api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-10-21")
    azure_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    if azure_endpoint and azure_api_key:
        base_url = f"{azure_endpoint}/openai/deployments/{azure_deployment}"
        logger.info(f"Using Azure OpenAI endpoint: {azure_endpoint}")
        return OpenAI(
            api_key=azure_api_key,
            base_url=base_url,
            default_query={"api-version": azure_api_version},
            http_client=httpx.Client(
                headers={"api-key": azure_api_key},
            ),
        )

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIEmbedder:
    """
    Wrapper around OpenAI's embedding API.

    Features:
    - Batched embedding for efficiency
    - Automatic retries with exponential backoff
    - Configurable model and dimensions
    - Simple interface
    - Auto-detects Azure OpenAI via environment variables
    """

    def __init__(
        self, api_key: str | None = None, config: EmbedderConfig | None = None
    ):
        self.config = config or EmbedderConfig()
        self.client = _create_openai_client(api_key)

        # Validate API key
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or AZURE_OPENAI_ENDPOINT + AZURE_API_KEY for Azure OpenAI."
            )

        logger.debug(f"Initialized embedder with model={self.config.model}")

    def embed_text(self, text: str, is_query: bool = True) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed
            is_query: If True, apply stricter length limit for queries

        Returns:
            Embedding vector as list of floats
        """
        result = self.embed_texts([text], is_query=is_query)
        return result[0]

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """
        Embed multiple texts efficiently with batching.

        Args:
            texts: List of texts to embed
            is_query: If True, apply stricter length limits

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (
            len(texts) + self.config.batch_size - 1
        ) // self.config.batch_size

        logger.debug(f"Embedding {len(texts)} texts in {total_batches} batches")

        # Process in batches
        for batch_num, i in enumerate(range(0, len(texts), self.config.batch_size)):
            batch = texts[i : i + self.config.batch_size]

            # Clean texts (remove null bytes, excessive whitespace)
            max_chars = self.config.query_max_chars if is_query else 30000
            batch = [self._clean_text(t, max_chars) for t in batch]

            # Call OpenAI API with retries
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Completed batch {batch_num + 1}/{total_batches}")

        return all_embeddings

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """
        Embed a single batch with retry logic.

        Args:
            batch: List of cleaned texts to embed

        Returns:
            List of embedding vectors
        """
        kwargs = {
            "model": self.config.model,
            "input": batch,
        }
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Extract embeddings in order
        batch_embeddings: list[list[float] | None] = [None] * len(batch)
        for item in response.data:
            batch_embeddings[item.index] = item.embedding

        # Verify we got all embeddings back - fail explicitly if count doesn't match
        result = [e for e in batch_embeddings if e is not None]
        if len(result) != len(batch):
            raise RuntimeError(
                f"OpenAI returned {len(result)} embeddings for {len(batch)} inputs. "
                "This indicates content was filtered or an API error occurred."
            )
        return result

    def _clean_text(self, text: str, max_chars: int = 30000) -> str:
        """Clean text for embedding while preserving code structure."""
        import re

        # Remove null bytes
        text = text.replace("\x00", "")
        # Dedupe excessive blank lines (3+ -> 2) but preserve structure
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace from lines
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text to {max_chars} characters")
        return text

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        if self.config.dimensions:
            return self.config.dimensions

        # Default dimensions by model
        defaults = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return defaults.get(self.config.model, 1536)


@dataclass
class BedrockEmbedderConfig:
    """Configuration for the Bedrock embedder."""

    model: str = "amazon.titan-embed-text-v2:0"
    dimensions: int = 1024  # Titan V2 supports 256, 512, or 1024
    region: str | None = None  # Falls back to AWS_REGION / boto3 default
    profile: str | None = None  # Falls back to AWS_PROFILE / boto3 default
    max_workers: int = 8  # Titan has no batch API; parallelize individual calls
    query_max_chars: int = 8000
    max_chars: int = 30000


class BedrockEmbedder:
    """
    Wrapper around Amazon Bedrock's Titan Embeddings V2 model.

    Titan's invoke_model API embeds one text per call (no batch endpoint), so
    embed_texts fans out individual calls across a thread pool while
    preserving input order. Uses the same AWS credential chain as the rest of
    this Bedrock account (SSO profile, env vars, instance role, etc.) via
    boto3 -- no separate API key is required.
    """

    def __init__(
        self,
        api_key: str | None = None,  # unused; accepted for interface parity
        config: BedrockEmbedderConfig | None = None,
    ):
        self.config = config or BedrockEmbedderConfig()

        session_kwargs = {}
        if self.config.profile:
            session_kwargs["profile_name"] = self.config.profile
        session = boto3.Session(**session_kwargs)

        client_kwargs = {}
        if self.config.region:
            client_kwargs["region_name"] = self.config.region
        self.client = session.client("bedrock-runtime", **client_kwargs)

        logger.debug(
            f"Initialized Bedrock embedder with model={self.config.model}, "
            f"dimensions={self.config.dimensions}"
        )

    def embed_text(self, text: str, is_query: bool = True) -> list[float]:
        """Embed a single text string."""
        result = self.embed_texts([text], is_query=is_query)
        return result[0]

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """
        Embed multiple texts by fanning out individual Titan calls.

        Args:
            texts: List of texts to embed
            is_query: If True, apply stricter length limits

        Returns:
            List of embedding vectors, in the same order as `texts`
        """
        if not texts:
            return []

        max_chars = self.config.query_max_chars if is_query else self.config.max_chars
        cleaned = [self._clean_text(t, max_chars) for t in texts]

        logger.debug(
            f"Embedding {len(cleaned)} texts via Bedrock "
            f"({self.config.max_workers} concurrent workers)"
        )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            embeddings = list(pool.map(self._embed_one, cleaned))

        return embeddings

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _embed_one(self, text: str) -> list[float]:
        """Embed a single (already-cleaned) text via Titan Embeddings V2."""
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self.config.dimensions,
                "normalize": True,
            }
        )

        response = self.client.invoke_model(
            modelId=self.config.model,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Bedrock returned no embedding for input (model={self.config.model})"
            )
        return [float(v) for v in embedding]

    def _clean_text(self, text: str, max_chars: int = 30000) -> str:
        """Clean text for embedding (same rules as OpenAIEmbedder)."""
        import re

        text = text.replace("\x00", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text to {max_chars} characters")
        return text

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.config.dimensions


def create_embedder(
    api_key: str | None = None,
    config: EmbedderConfig | BedrockEmbedderConfig | None = None,
) -> OpenAIEmbedder | BedrockEmbedder:
    """
    Create an embedder based on EMBEDDING_PROVIDER.

    - EMBEDDING_PROVIDER=bedrock -> BedrockEmbedder (Titan Embeddings V2, AWS creds)
    - EMBEDDING_PROVIDER unset/openai -> OpenAIEmbedder (default; original behavior)

    This is the single switch point -- callers should use this factory instead
    of instantiating OpenAIEmbedder/BedrockEmbedder directly, so provider
    selection stays in one place.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()

    if provider == "bedrock":
        bedrock_config = config if isinstance(config, BedrockEmbedderConfig) else None
        return BedrockEmbedder(api_key=api_key, config=bedrock_config)

    openai_config = config if isinstance(config, EmbedderConfig) else None
    return OpenAIEmbedder(api_key=api_key, config=openai_config)
