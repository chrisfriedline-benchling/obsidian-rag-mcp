"""RAG components for Obsidian vault indexing and search."""

from .chunker import Chunk, ChunkerConfig, MarkdownChunker
from .embedder import (
    BedrockEmbedder,
    BedrockEmbedderConfig,
    EmbedderConfig,
    OpenAIEmbedder,
    create_embedder,
)
from .engine import RAGEngine, SearchResponse, SearchResult
from .indexer import IndexerConfig, IndexStats, VaultIndexer

__all__ = [
    "Chunk",
    "ChunkerConfig",
    "MarkdownChunker",
    "EmbedderConfig",
    "OpenAIEmbedder",
    "BedrockEmbedder",
    "BedrockEmbedderConfig",
    "create_embedder",
    "RAGEngine",
    "SearchResponse",
    "SearchResult",
    "IndexerConfig",
    "IndexStats",
    "VaultIndexer",
]
