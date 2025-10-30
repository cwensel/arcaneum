"""
Markdown file discovery with frontmatter extraction and content hashing (RDR-014).

This module handles:
- Recursive file discovery with glob patterns
- YAML frontmatter extraction using python-frontmatter
- Content hashing (SHA256) for change detection
- File metadata extraction (size, mtime)
- Graceful handling of files with/without frontmatter
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import frontmatter
    FRONTMATTER_AVAILABLE = True
except ImportError:
    FRONTMATTER_AVAILABLE = False
    logging.warning("python-frontmatter not available, frontmatter extraction will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class MarkdownFileMetadata:
    """Metadata for a discovered markdown file."""
    file_path: str  # Absolute path
    file_name: str  # Basename
    file_size: int  # Size in bytes
    content_hash: str  # SHA256 hash of content
    modified_time: float  # Unix timestamp

    # Frontmatter fields (optional)
    has_frontmatter: bool = False
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    project: Optional[str] = None
    created_at: Optional[str] = None

    # Custom frontmatter fields
    custom_metadata: Dict = field(default_factory=dict)


class MarkdownDiscovery:
    """Discover and analyze markdown files with frontmatter support."""

    # Default file extensions to search for
    MARKDOWN_EXTENSIONS = ['.md', '.markdown', '.mdown', '.mkd', '.mkdn']

    def __init__(
        self,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ):
        """Initialize markdown discovery.

        Args:
            extensions: File extensions to search for (default: .md, .markdown, etc.)
            exclude_patterns: Glob patterns to exclude (e.g., ['**/node_modules/**'])
        """
        self.extensions = extensions or self.MARKDOWN_EXTENSIONS
        self.exclude_patterns = exclude_patterns or []

        if not FRONTMATTER_AVAILABLE:
            logger.warning(
                "python-frontmatter not available. "
                "Frontmatter extraction will be disabled."
            )

    def discover_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """Discover markdown files in a directory.

        Args:
            directory: Directory to search
            recursive: If True, search recursively

        Returns:
            List of Path objects for discovered markdown files
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        markdown_files = []

        for ext in self.extensions:
            if recursive:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"

            files = list(directory.glob(pattern))
            markdown_files.extend(files)

        # Filter out excluded patterns
        if self.exclude_patterns:
            filtered_files = []
            for file_path in markdown_files:
                exclude = False
                for pattern in self.exclude_patterns:
                    if file_path.match(pattern):
                        exclude = True
                        break
                if not exclude:
                    filtered_files.append(file_path)
            markdown_files = filtered_files

        # Sort for deterministic ordering
        markdown_files.sort()

        logger.info(f"Discovered {len(markdown_files)} markdown files in {directory}")
        return markdown_files

    def extract_metadata(self, file_path: Path) -> MarkdownFileMetadata:
        """Extract metadata from a markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            MarkdownFileMetadata with file and frontmatter metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
            content = file_path.read_text(encoding='latin-1')

        # Compute content hash
        content_hash = self._compute_hash(content)

        # Get file stats
        stat = file_path.stat()
        file_size = stat.st_size
        modified_time = stat.st_mtime

        # Extract frontmatter if available
        has_frontmatter = False
        title = None
        author = None
        tags = []
        category = None
        project = None
        created_at = None
        custom_metadata = {}

        if FRONTMATTER_AVAILABLE:
            try:
                post = frontmatter.loads(content)

                if post.metadata:
                    has_frontmatter = True

                    # Extract standard fields
                    title = post.metadata.get('title')
                    author = post.metadata.get('author')

                    # Handle tags (can be string, list, or comma-separated)
                    tags_raw = post.metadata.get('tags', [])
                    if isinstance(tags_raw, str):
                        # Split comma-separated tags
                        tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
                    elif isinstance(tags_raw, list):
                        tags = [str(t) for t in tags_raw]

                    category = post.metadata.get('category')
                    project = post.metadata.get('project')
                    created_at = post.metadata.get('created_at') or post.metadata.get('date')

                    # Store all other fields as custom metadata
                    standard_fields = {'title', 'author', 'tags', 'category', 'project', 'created_at', 'date'}
                    for key, value in post.metadata.items():
                        if key not in standard_fields:
                            custom_metadata[key] = value

            except Exception as e:
                logger.debug(f"Frontmatter extraction failed for {file_path}: {e}")
                # Continue without frontmatter

        return MarkdownFileMetadata(
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_size=file_size,
            content_hash=content_hash,
            modified_time=modified_time,
            has_frontmatter=has_frontmatter,
            title=title,
            author=author,
            tags=tags,
            category=category,
            project=project,
            created_at=created_at,
            custom_metadata=custom_metadata
        )

    def discover_and_extract(
        self, directory: Path, recursive: bool = True
    ) -> List[MarkdownFileMetadata]:
        """Discover markdown files and extract metadata for each.

        Args:
            directory: Directory to search
            recursive: If True, search recursively

        Returns:
            List of MarkdownFileMetadata for all discovered files
        """
        files = self.discover_files(directory, recursive=recursive)

        metadata_list = []
        for file_path in files:
            try:
                metadata = self.extract_metadata(file_path)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Failed to extract metadata from {file_path}: {e}")
                # Continue with other files

        logger.info(f"Extracted metadata from {len(metadata_list)}/{len(files)} files")
        return metadata_list

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Text content to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @staticmethod
    def read_file_content(file_path: Path) -> str:
        """Read markdown file content (without frontmatter if present).

        Args:
            file_path: Path to markdown file

        Returns:
            File content (body only if frontmatter present)
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='latin-1')

        # Strip frontmatter if using frontmatter library
        if FRONTMATTER_AVAILABLE:
            try:
                post = frontmatter.loads(content)
                return post.content  # Return only body, not frontmatter
            except Exception:
                # Fallback to full content
                pass

        return content

    @staticmethod
    def read_file_with_frontmatter(file_path: Path) -> tuple[str, Dict]:
        """Read markdown file with frontmatter extraction.

        Args:
            file_path: Path to markdown file

        Returns:
            Tuple of (content, frontmatter_dict)
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='latin-1')

        if FRONTMATTER_AVAILABLE:
            try:
                post = frontmatter.loads(content)
                return post.content, dict(post.metadata)
            except Exception as e:
                logger.debug(f"Frontmatter parsing failed: {e}")

        # No frontmatter or parsing failed
        return content, {}


def discover_markdown_files(
    directory: Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[MarkdownFileMetadata]:
    """Convenience function to discover and extract markdown file metadata.

    Args:
        directory: Directory to search
        recursive: If True, search recursively
        extensions: File extensions to search for
        exclude_patterns: Glob patterns to exclude

    Returns:
        List of MarkdownFileMetadata
    """
    discovery = MarkdownDiscovery(
        extensions=extensions,
        exclude_patterns=exclude_patterns
    )
    return discovery.discover_and_extract(directory, recursive=recursive)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's content.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='latin-1')

    return hashlib.sha256(content.encode('utf-8')).hexdigest()
