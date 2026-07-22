# CLAUDE.md — Project Context for AI Assistants

## Project Overview

**obsidian-rag-mcp** is an MCP server that provides semantic search over Obsidian vaults. It indexes markdown files with vector embeddings (OpenAI, Azure OpenAI, or Amazon Bedrock) stored in ChromaDB, then exposes search via the Model Context Protocol.

This is a fork of [danielscholl/obsidian-rag-mcp](https://github.com/danielscholl/obsidian-rag-mcp) (`upstream` remote) with Bedrock support added on top, for use with Claude on Bedrock accounts that have no OpenAI API key. `origin` is [chrisfriedline-benchling/obsidian-rag-mcp](https://github.com/chrisfriedline-benchling/obsidian-rag-mcp).

## Quick Commands

```bash
# Install
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_engine.py -v

# Format
uv run black obsidian_rag_mcp/ tests/

# Lint
uv run ruff check obsidian_rag_mcp/ tests/

# Type check
uv run mypy obsidian_rag_mcp/

# Security scan
uv run bandit -r obsidian_rag_mcp/ -ll -x tests/

# Run all quality checks (CI simulation)
uv run black --check obsidian_rag_mcp/ tests/ && \
uv run ruff check obsidian_rag_mcp/ tests/ && \
uv run mypy obsidian_rag_mcp/ && \
uv run pytest --cov=obsidian_rag_mcp --cov-fail-under=65

# Index sample vault
uv run obsidian-rag index --vault ./vault

# Search
uv run obsidian-rag search "query" --vault ./vault

# Start MCP server
uv run obsidian-rag serve --vault ./vault
```

## Architecture

```
obsidian_rag_mcp/
├── rag/                 # Core RAG pipeline
│   ├── indexer.py       # Vault scanning, chunking, embedding
│   ├── chunker.py       # Markdown-aware chunking (headers, code blocks, frontmatter)
│   ├── embedder.py      # OpenAI / Azure OpenAI / Bedrock embeddings + create_embedder() factory
│   └── engine.py        # Semantic search engine (query interface)
├── reasoning/           # Conclusion extraction layer
│   ├── extractor.py     # LLM-based conclusion extraction
│   ├── conclusion_store.py  # ChromaDB storage for conclusions
│   └── models.py        # Conclusion, ConclusionType dataclasses
├── mcp/
│   ├── server.py        # MCP server (9 tools)
│   └── __main__.py      # Entry point
├── cli/
│   └── main.py          # Click CLI (index, search, serve, stats)
└── utils/
    └── tokens.py        # Token counting utilities
```

### Key Patterns

- **Async everywhere**: MCP server is async; engine queries wrapped in `run_in_executor`
- **Global engine instance**: `server.py` initializes a single `RAGEngine` on startup
- **Click CLI**: Commands in `cli/main.py`
- **ChromaDB local**: Vectors stored locally in `.chroma/` directory
- **Incremental indexing**: Only re-indexes changed files (content hash)
- **Azure support**: `embedder.py` has `_create_openai_client()` factory for OpenAI/Azure
- **Embedding provider selection**: `embedder.py` has `create_embedder()` — the single switch point between `OpenAIEmbedder` (default, covers OpenAI + Azure OpenAI) and `BedrockEmbedder` (Amazon Titan Embeddings V2), selected via `EMBEDDING_PROVIDER`. `indexer.py` calls this factory rather than instantiating an embedder class directly — do not bypass it.
- **Chunk size is a hard ceiling, not a soft target**: `MarkdownChunker` hard-splits any single paragraph larger than `max_chunk_tokens` (e.g. one unbroken block of transcript/meeting dialogue with no blank lines) on line boundaries. This exists because embedding APIs reject oversized input outright — Bedrock Titan's 8192-token ceiling surfaced the bug first, but the same failure mode applies to OpenAI's 8191-token cap on a large enough input. Don't remove this without re-verifying against a large transcript-shaped fixture.

## Code Style

- **Formatter**: `black` (default settings)
- **Linter**: `ruff` (rules: E, F, I, N, W, UP; line-length 100)
- **Type checker**: `mypy --strict` (configured in pyproject.toml)
- **Type hints**: Required for all public functions
- **Imports**: Sorted by ruff (isort-compatible)

## Testing

- Framework: `pytest` with `pytest-asyncio`
- Coverage minimum: 65% (enforced in CI)
- Tests in `tests/` mirror the package structure
- Use existing fixtures in `tests/conftest.py`
- Mock external services (OpenAI, ChromaDB) in unit tests

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new search filter
fix: handle empty vault during indexing
docs: update architecture diagram
refactor(rag): simplify chunker logic
test: add embedder unit tests
chore(ci): add CodeQL scanning
```

Breaking changes: `feat!: require Python 3.12+`

## Dependencies

- **Runtime**: chromadb, openai, boto3, mcp, click, pydantic, python-frontmatter, tenacity, tiktoken, python-dotenv
- **Dev**: pytest, pytest-asyncio, pytest-cov, mypy, ruff, black, pre-commit
- **Build**: hatchling
- **Python**: >=3.11, <3.14

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EMBEDDING_PROVIDER` | No | `openai` (default) or `bedrock` — selects the embedding backend via `create_embedder()` |
| `OPENAI_API_KEY` | Yes* | OpenAI API key for embeddings (unused when `EMBEDDING_PROVIDER=bedrock`) |
| `AZURE_OPENAI_ENDPOINT` | No* | Azure OpenAI endpoint URL |
| `AZURE_API_KEY` | No* | Azure OpenAI API key |
| `AZURE_OPENAI_VERSION` | No | Azure API version (default: `2024-10-21`) |
| `AZURE_EMBEDDING_DEPLOYMENT` | No | Azure deployment name (default: `text-embedding-3-small`) |
| `AWS_PROFILE` | No† | AWS SSO/named profile for Bedrock (falls back to boto3 default credential chain) |
| `AWS_REGION` | No† | AWS region for the Bedrock runtime client |
| `OBSIDIAN_VAULT_PATH` | No | Default vault path |
| `REASONING_ENABLED` | No | Enable conclusion extraction (default: false) |

\* Either OpenAI or Azure OpenAI credentials required, unless `EMBEDDING_PROVIDER=bedrock`.
† Only relevant when `EMBEDDING_PROVIDER=bedrock`. No API key needed — uses the standard AWS credential chain (SSO profile, env vars, instance role, etc.) via boto3.

### Amazon Bedrock embeddings

Set `EMBEDDING_PROVIDER=bedrock` to embed via Amazon Titan Embeddings V2 (`amazon.titan-embed-text-v2:0`) instead of OpenAI/Azure — useful for accounts running Claude on Bedrock with no OpenAI API key. No separate credential is needed; it uses whatever AWS credentials are already active (`AWS_PROFILE` + `AWS_REGION`, or the default chain).

Notable differences from the OpenAI path:
- Titan's `invoke_model` API embeds one text per call (no batch endpoint) — `BedrockEmbedder.embed_texts()` fans calls out across a thread pool (`BedrockEmbedderConfig.max_workers`, default 8) while preserving input order.
- Titan hard-rejects any request over 8192 input tokens. `BedrockEmbedder` truncates by actual token count (via `utils/tokens.py`'s tiktoken-based counter, binary-searched to the largest fitting prefix) rather than a character-count estimate — Titan's ceiling is tight enough that a char proxy isn't reliable. Configurable via `BedrockEmbedderConfig.max_input_tokens` (default 8000, leaving headroom below the hard 8192 limit).
- `BedrockEmbedderConfig.dimensions` (default 1024) — Titan V2 supports 256, 512, or 1024.

## Key Documentation

- [Architecture](docs/ARCHITECTURE.md) -- System design and data flow
- [Development](docs/DEVELOPMENT.md) -- Local setup (includes Windows)
- [Getting Started](docs/GETTING_STARTED.md) -- 5-minute tutorial + Claude Desktop setup
- [ADRs](docs/decisions/) -- Architectural decision records
