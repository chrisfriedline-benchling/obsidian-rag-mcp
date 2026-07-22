"""
Markdown-aware document chunking for Obsidian notes.

Splits documents into semantically meaningful chunks while preserving:
- Header hierarchy
- Code blocks (as atomic units)
- Frontmatter metadata
- Obsidian-specific syntax (tags, links)
"""

import logging
import re
from dataclasses import dataclass, field

import frontmatter
import yaml

from obsidian_rag_mcp.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    source_path: str
    chunk_index: int

    # Metadata
    title: str | None = None
    heading: str | None = None  # The heading this chunk falls under
    tags: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)

    # Position info for debugging
    start_line: int = 0
    end_line: int = 0

    @property
    def token_estimate(self) -> int:
        """Accurate token count using tiktoken."""
        return count_tokens(self.content)


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""

    max_chunk_tokens: int = 1000  # ~4000 chars
    min_chunk_tokens: int = 100  # Don't create tiny chunks
    overlap_tokens: int = 50  # Overlap between chunks

    # Splitting behavior
    split_on_h2: bool = True  # Primary split on ## headers
    split_on_h3: bool = False  # Also split on ### headers
    preserve_code_blocks: bool = True


class MarkdownChunker:
    """
    Chunks Obsidian markdown documents into semantic units.

    Strategy:
    1. Extract frontmatter and tags
    2. Split on H2 headers (configurable)
    3. Further split large sections by paragraphs
    4. Keep code blocks atomic
    5. Add overlap between chunks for context
    """

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

        # Regex patterns
        self.code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        self.h2_pattern = re.compile(r"^## .+$", re.MULTILINE)
        self.h3_pattern = re.compile(r"^### .+$", re.MULTILINE)
        self.tag_pattern = re.compile(r"#([a-zA-Z][a-zA-Z0-9_/-]*)")
        self.link_pattern = re.compile(r"\[\[([^\]]+)\]\]")

    def chunk_document(self, content: str, source_path: str) -> list[Chunk]:
        """
        Chunk a markdown document into semantic units.

        Args:
            content: Raw markdown content
            source_path: Path to the source file (for metadata)

        Returns:
            List of Chunk objects
        """
        # Parse frontmatter (with fallback for invalid YAML)
        try:
            post = frontmatter.loads(content)
            fm = dict(post.metadata)
            body = post.content
        except yaml.YAMLError as e:
            # Fallback: treat as plain content with no frontmatter
            logger.warning(f"Failed to parse frontmatter in {source_path}: {e}")
            fm = {}
            body = content

        # Extract title from frontmatter or first H1
        title = fm.get("title")
        if not title:
            h1_match = re.search(r"^# (.+)$", body, re.MULTILINE)
            title = (
                h1_match.group(1)
                if h1_match
                else source_path.split("/")[-1].replace(".md", "")
            )

        # Extract tags from frontmatter and body
        tags = self._extract_tags(fm, body)

        # Split into sections
        sections = self._split_into_sections(body)

        # Create chunks from sections
        chunks = []
        for i, (heading, section_content) in enumerate(sections):
            section_chunks = self._chunk_section(
                section_content,
                source_path=source_path,
                title=title,
                heading=heading,
                tags=tags,
                frontmatter=fm,
                base_index=len(chunks),
            )
            chunks.extend(section_chunks)

        return chunks

    def _extract_tags(self, fm: dict, body: str) -> list[str]:
        """Extract tags from frontmatter and inline tags."""
        tags = set()

        # Frontmatter tags
        fm_tags = fm.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        tags.update(fm_tags)

        # Inline tags (but not in code blocks)
        # First, remove code blocks
        body_no_code = self.code_block_pattern.sub("", body)
        inline_tags = self.tag_pattern.findall(body_no_code)
        tags.update(inline_tags)

        return sorted(tags)

    def _split_into_sections(self, body: str) -> list[tuple[str | None, str]]:
        """
        Split body into sections by headers.

        Returns list of (heading, content) tuples.
        """
        if not self.config.split_on_h2:
            return [(None, body)]

        # Find all H2 headers and their positions
        pattern = self.h2_pattern
        matches = list(pattern.finditer(body))

        if not matches:
            return [(None, body)]

        sections = []

        # Content before first header
        if matches[0].start() > 0:
            pre_content = body[: matches[0].start()].strip()
            if pre_content:
                sections.append((None, pre_content))

        # Each header section
        for i, match in enumerate(matches):
            heading = match.group(0).lstrip("#").strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            content = body[start:end].strip()
            if content:
                sections.append((heading, content))

        return sections

    def _chunk_section(
        self,
        content: str,
        source_path: str,
        title: str,
        heading: str | None,
        tags: list[str],
        frontmatter: dict,
        base_index: int,
    ) -> list[Chunk]:
        """
        Chunk a section, respecting max token limits.
        """
        max_chars = self.config.max_chunk_tokens * 4
        min_chars = self.config.min_chunk_tokens * 4

        # If section is small enough, return as single chunk
        if len(content) <= max_chars:
            return [
                Chunk(
                    content=content,
                    source_path=source_path,
                    chunk_index=base_index,
                    title=title,
                    heading=heading,
                    tags=tags,
                    frontmatter=frontmatter,
                )
            ]

        # Need to split further - by paragraphs
        chunks = []
        paragraphs = self._split_paragraphs(content)

        current_chunk = ""
        current_index = base_index

        for para in paragraphs:
            # A single paragraph larger than max_chars (e.g. one unbroken
            # block of transcript dialogue with no blank lines) can't be
            # bounded by the "does it fit in current_chunk" check below --
            # split it on its own before the normal accumulation logic runs.
            if len(para) > max_chars:
                if current_chunk.strip():
                    chunks.append(
                        Chunk(
                            content=current_chunk.strip(),
                            source_path=source_path,
                            chunk_index=current_index,
                            title=title,
                            heading=heading,
                            tags=tags,
                            frontmatter=frontmatter,
                        )
                    )
                    current_index += 1
                    current_chunk = ""

                for piece in self._hard_split(para, max_chars):
                    chunks.append(
                        Chunk(
                            content=piece.strip(),
                            source_path=source_path,
                            chunk_index=current_index,
                            title=title,
                            heading=heading,
                            tags=tags,
                            frontmatter=frontmatter,
                        )
                    )
                    current_index += 1
                continue

            # Check if adding this paragraph exceeds limit
            if (
                len(current_chunk) + len(para) > max_chars
                and len(current_chunk) >= min_chars
            ):
                # Save current chunk
                chunks.append(
                    Chunk(
                        content=current_chunk.strip(),
                        source_path=source_path,
                        chunk_index=current_index,
                        title=title,
                        heading=heading,
                        tags=tags,
                        frontmatter=frontmatter,
                    )
                )
                current_index += 1

                # Start new chunk with overlap
                overlap_chars = self.config.overlap_tokens * 4
                if len(current_chunk) > overlap_chars:
                    current_chunk = current_chunk[-overlap_chars:]
                else:
                    current_chunk = ""

            current_chunk += para + "\n\n"

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    source_path=source_path,
                    chunk_index=current_index,
                    title=title,
                    heading=heading,
                    tags=tags,
                    frontmatter=frontmatter,
                )
            )

        return chunks

    def _hard_split(self, text: str, max_chars: int) -> list[str]:
        """
        Split an oversized, unbreakable paragraph into pieces no larger than
        max_chars. Prefers splitting on line boundaries (e.g. one line per
        speaker turn in a transcript); falls back to a raw character slice
        for a single line that is itself larger than max_chars.
        """
        if len(text) <= max_chars:
            return [text]

        pieces = []
        current = ""

        for line in text.split("\n"):
            if len(line) > max_chars:
                if current:
                    pieces.append(current)
                    current = ""
                # Line itself exceeds the limit -- slice it directly.
                for i in range(0, len(line), max_chars):
                    pieces.append(line[i : i + max_chars])
                continue

            candidate = f"{current}\n{line}" if current else line
            if len(candidate) > max_chars:
                pieces.append(current)
                current = line
            else:
                current = candidate

        if current:
            pieces.append(current)

        return pieces

    def _split_paragraphs(self, content: str) -> list[str]:
        """
        Split content into paragraphs, keeping code blocks intact.
        """
        # Temporarily replace code blocks with placeholders
        code_blocks = []

        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        content = self.code_block_pattern.sub(save_code_block, content)

        # Split on double newlines
        paragraphs = re.split(r"\n\s*\n", content)

        # Restore code blocks
        result = []
        for para in paragraphs:
            for i, block in enumerate(code_blocks):
                para = para.replace(f"__CODE_BLOCK_{i}__", block)
            if para.strip():
                result.append(para.strip())

        return result
