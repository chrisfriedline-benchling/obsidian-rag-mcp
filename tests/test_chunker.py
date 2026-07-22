"""Tests for the markdown chunker."""

from obsidian_rag_mcp.rag.chunker import Chunk, ChunkerConfig, MarkdownChunker


class TestMarkdownChunker:
    """Test suite for MarkdownChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = MarkdownChunker()

    def test_simple_document(self):
        """Test chunking a simple document."""
        content = """---
title: Test Document
tags: [test, example]
---

# Test Document

This is a simple test document.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        assert len(chunks) >= 1
        assert chunks[0].title == "Test Document"
        assert "test" in chunks[0].tags
        assert "example" in chunks[0].tags

    def test_h2_splitting(self):
        """Test that documents split on H2 headers."""
        content = """# Main Title

Introduction paragraph.

## Section One

Content for section one.

## Section Two

Content for section two.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        # Should have 3 chunks: intro + 2 sections
        assert len(chunks) == 3
        assert chunks[1].heading == "Section One"
        assert chunks[2].heading == "Section Two"

    def test_code_block_preservation(self):
        """Test that code blocks are preserved as atomic units."""
        content = """# Code Example

```python
def hello():
    print("Hello, World!")
```

Some text after code.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        # Code block should be in the content
        assert "```python" in chunks[0].content
        assert 'print("Hello, World!")' in chunks[0].content

    def test_inline_tag_extraction(self):
        """Test extraction of inline tags."""
        content = """# Document

This has #inline and #tags in the content.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        assert "inline" in chunks[0].tags
        assert "tags" in chunks[0].tags

    def test_frontmatter_tags(self):
        """Test extraction of frontmatter tags."""
        content = """---
tags: [frontend, react, typescript]
---

# Document

Content here.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        assert "frontend" in chunks[0].tags
        assert "react" in chunks[0].tags
        assert "typescript" in chunks[0].tags

    def test_title_from_h1(self):
        """Test title extraction from H1 when no frontmatter title."""
        content = """# My Document Title

Content here.
"""
        chunks = self.chunker.chunk_document(content, "test.md")

        assert chunks[0].title == "My Document Title"

    def test_title_from_filename(self):
        """Test title fallback to filename."""
        content = """Some content without title.
"""
        chunks = self.chunker.chunk_document(content, "my-document.md")

        assert chunks[0].title == "my-document"

    def test_large_section_splitting(self):
        """Test that large sections get split by paragraphs."""
        # Create content with many paragraphs
        paragraphs = [f"This is paragraph number {i}." * 50 for i in range(20)]
        content = "# Large Document\n\n" + "\n\n".join(paragraphs)

        config = ChunkerConfig(max_chunk_tokens=500)
        chunker = MarkdownChunker(config)
        chunks = chunker.chunk_document(content, "large.md")

        # Should have multiple chunks
        assert len(chunks) > 1

    def test_token_estimate(self):
        """Test token estimation property."""
        chunk = Chunk(
            content="Hello world this is a test",
            source_path="test.md",
            chunk_index=0,
        )

        # ~26 chars / 4 = ~6 tokens
        assert chunk.token_estimate >= 5
        assert chunk.token_estimate <= 10

    def test_empty_document(self):
        """Test handling of empty document."""
        content = ""
        chunks = self.chunker.chunk_document(content, "empty.md")

        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_source_path_preserved(self):
        """Test that source path is preserved in chunks."""
        content = "# Test\n\nContent"
        path = "folder/subfolder/document.md"

        chunks = self.chunker.chunk_document(content, path)

        assert chunks[0].source_path == path

    def test_oversized_unbroken_paragraph_is_split(self):
        """
        A single "paragraph" (no blank line inside it) larger than
        max_chunk_tokens must still be split into bounded pieces.

        Regression test: transcript-style content (e.g. a block of speaker
        dialogue with one line per turn but no blank lines) used to pass
        through _chunk_section's paragraph loop untouched, because the
        overflow check only fires once current_chunk already holds
        content. That produced a single oversized chunk which downstream
        embedding APIs reject outright (e.g. Bedrock Titan's 8192-token
        input ceiling).
        """
        max_chars = ChunkerConfig().max_chunk_tokens * 4  # ~4000 chars

        lines = [f"Speaker {i % 3 + 1}: dialogue line {i} " * 3 for i in range(400)]
        big_paragraph = "\n".join(lines)
        assert len(big_paragraph) > max_chars * 3  # sanity check the fixture

        content = f"# Meeting\n\n## Speaker Attribution\n\n{big_paragraph}\n"
        chunks = self.chunker.chunk_document(content, "transcript.md")

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= max_chars

        # No dialogue lines should be dropped or duplicated across pieces
        reassembled = "\n".join(
            c.content for c in chunks if c.heading == "Speaker Attribution"
        )
        for i in (0, 199, 399):
            assert f"dialogue line {i} " in reassembled


class TestChunkerConfig:
    """Test ChunkerConfig defaults and behavior."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkerConfig()

        assert config.max_chunk_tokens == 1000
        assert config.min_chunk_tokens == 100
        assert config.overlap_tokens == 50
        assert config.split_on_h2 is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = ChunkerConfig(
            max_chunk_tokens=500,
            split_on_h2=False,
        )

        assert config.max_chunk_tokens == 500
        assert config.split_on_h2 is False
