"""Tests for the vault indexer."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_rag_mcp.rag.indexer import IndexerConfig, VaultIndexer


class TestVaultIndexer:
    """Test suite for VaultIndexer."""

    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_symlinks_are_skipped(self, mock_chroma, mock_embedder):
        """Test that symlinks are not indexed (security feature)."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_emb = Mock()
        mock_embedder.return_value = mock_emb

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)

            # Create a real markdown file
            real_file = vault / "real.md"
            real_file.write_text("# Real file\n\nContent here.")

            # Create a symlink to a file outside the vault
            outside_file = Path(tmpdir).parent / "outside_secret.md"
            try:
                outside_file.write_text("# SECRET DATA")
                symlink = vault / "sneaky_link.md"
                symlink.symlink_to(outside_file)

                # Create indexer
                config = IndexerConfig(
                    vault_path=str(vault),
                    persist_dir=str(vault / ".chroma"),
                )
                indexer = VaultIndexer(config, api_key="test-key")

                # Scan vault
                files = indexer.scan_vault()

                # Should only find the real file, not the symlink
                file_names = [f.name for f in files]
                assert "real.md" in file_names
                assert "sneaky_link.md" not in file_names
            finally:
                # Cleanup
                if outside_file.exists():
                    outside_file.unlink()

    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_vault_path_validation(self, mock_chroma, mock_embedder):
        """Test that invalid vault paths raise errors."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_client.get_or_create_collection.return_value = Mock()
        mock_embedder.return_value = Mock()

        # Non-existent path should raise
        with pytest.raises(ValueError, match="does not exist"):
            config = IndexerConfig(
                vault_path="/nonexistent/path/to/vault",
                persist_dir="/tmp/chroma",
            )
            VaultIndexer(config, api_key="test-key")

    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_ignore_patterns(self, mock_chroma, mock_embedder):
        """Test that ignore patterns work correctly."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embedder.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)

            # Create files
            (vault / "note.md").write_text("# Note")
            (vault / ".obsidian").mkdir()
            (vault / ".obsidian" / "config.md").write_text("# Config")
            (vault / "drawing.excalidraw.md").write_text("# Drawing")

            config = IndexerConfig(
                vault_path=str(vault),
                persist_dir=str(vault / ".chroma"),
            )
            indexer = VaultIndexer(config, api_key="test-key")

            files = indexer.scan_vault()
            file_names = [f.name for f in files]

            # Regular note should be found
            assert "note.md" in file_names
            # .obsidian contents should be ignored
            assert "config.md" not in file_names
            # Excalidraw files should be ignored
            assert "drawing.excalidraw.md" not in file_names


class TestIndexerConfig:
    """Test IndexerConfig defaults and behavior."""

    def test_default_ignore_patterns(self):
        """Test default ignore patterns are set."""
        config = IndexerConfig(vault_path="/tmp/vault")

        assert ".obsidian/*" in config.ignore_patterns
        assert ".trash/*" in config.ignore_patterns
        assert "*.excalidraw.md" in config.ignore_patterns

    def test_reasoning_disabled_by_default(self):
        """Test reasoning is disabled by default."""
        config = IndexerConfig(vault_path="/tmp/vault")
        assert config.reasoning_enabled is False
        assert config.extractor_config is None

    def test_reasoning_can_be_enabled(self):
        """Test reasoning can be enabled via config."""
        config = IndexerConfig(vault_path="/tmp/vault", reasoning_enabled=True)
        assert config.reasoning_enabled is True


class TestReasoningIndexer:
    """Test indexer with reasoning layer enabled."""

    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_reasoning_not_initialized_when_disabled(self, mock_chroma, mock_embedder):
        """Test reasoning components not initialized when disabled."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_client.get_or_create_collection.return_value = Mock()
        mock_embedder.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            (vault / "note.md").write_text("# Note")

            config = IndexerConfig(
                vault_path=str(vault),
                persist_dir=str(vault / ".chroma"),
                reasoning_enabled=False,
            )
            indexer = VaultIndexer(config, api_key="test-key")

            assert indexer.conclusion_extractor is None
            assert indexer.conclusion_store is None

    @patch("obsidian_rag_mcp.reasoning.extractor._create_openai_client")
    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_reasoning_initialized_when_enabled(
        self, mock_chroma, mock_embedder, mock_openai
    ):
        """Test reasoning components are initialized when enabled."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embedder.return_value = Mock()

        mock_openai_instance = Mock()
        mock_openai_instance.api_key = "test-key"
        mock_openai.return_value = mock_openai_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            (vault / "note.md").write_text("# Note")

            config = IndexerConfig(
                vault_path=str(vault),
                persist_dir=str(vault / ".chroma"),
                reasoning_enabled=True,
            )
            indexer = VaultIndexer(config, api_key="test-key")

            assert indexer.conclusion_extractor is not None
            assert indexer.conclusion_store is not None

    @patch("obsidian_rag_mcp.reasoning.extractor._create_openai_client")
    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_index_stats_include_reasoning_info(
        self, mock_chroma, mock_embedder, mock_openai
    ):
        """Test index stats include conclusion count when reasoning enabled."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embedder.return_value = Mock()

        mock_openai_instance = Mock()
        mock_openai_instance.api_key = "test-key"
        mock_openai.return_value = mock_openai_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            (vault / "note.md").write_text("# Note")

            config = IndexerConfig(
                vault_path=str(vault),
                persist_dir=str(vault / ".chroma"),
                reasoning_enabled=True,
            )
            indexer = VaultIndexer(config, api_key="test-key")

            stats = indexer.get_stats()
            stats_dict = stats.to_dict()

            assert stats.reasoning_enabled is True
            assert "total_conclusions" in stats_dict
            assert "reasoning_enabled" in stats_dict

    @patch("obsidian_rag_mcp.rag.indexer.create_embedder")
    @patch("obsidian_rag_mcp.rag.indexer.chromadb.PersistentClient")
    def test_index_stats_exclude_reasoning_when_disabled(
        self, mock_chroma, mock_embedder
    ):
        """Test index stats don't include reasoning info when disabled."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embedder.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            (vault / "note.md").write_text("# Note")

            config = IndexerConfig(
                vault_path=str(vault),
                persist_dir=str(vault / ".chroma"),
                reasoning_enabled=False,
            )
            indexer = VaultIndexer(config, api_key="test-key")

            stats = indexer.get_stats()
            stats_dict = stats.to_dict()

            assert stats.reasoning_enabled is False
            assert "total_conclusions" not in stats_dict
            assert "reasoning_enabled" not in stats_dict
