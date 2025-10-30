"""
Markdown injection persistence module (RDR-014 / arcaneum-204).

This module handles persisting directly injected markdown content to disk
for agent memory and reference purposes. Content is stored in:
  ~/.arcaneum/agent-memory/{collection}/{date}_{agent}_{slug}.md

Features:
- Safe filename generation (slugify, collision handling)
- Metadata preservation (injection_id, injected_by, injected_at, collection)
- Directory permissions validation
- Concurrent injection handling
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to safe filename slug.

    Args:
        text: Text to slugify
        max_length: Maximum slug length

    Returns:
        Safe filename slug (lowercase, alphanumeric + hyphens)
    """
    # Convert to lowercase
    slug = text.lower()

    # Replace spaces and underscores with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)

    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r'[^a-z0-9\-]', '', slug)

    # Remove leading/trailing hyphens
    slug = slug.strip('-')

    # Collapse multiple hyphens
    slug = re.sub(r'-+', '-', slug)

    # Truncate to max length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')

    # Fallback if slug is empty
    if not slug:
        slug = 'untitled'

    return slug


def get_agent_memory_dir(collection: str) -> Path:
    """Get agent memory directory for collection.

    Args:
        collection: Collection name

    Returns:
        Path to agent memory directory
    """
    from arcaneum.paths import get_arcaneum_dir

    memory_dir = get_arcaneum_dir() / 'agent-memory' / collection
    memory_dir.mkdir(parents=True, exist_ok=True)

    return memory_dir


def generate_filename(
    title: Optional[str],
    metadata: Dict,
    agent: Optional[str] = None
) -> str:
    """Generate unique filename for injection.

    Format: {date}_{agent}_{slug}.md

    Args:
        title: Document title (from metadata or content)
        metadata: Full metadata dict
        agent: Agent identifier (defaults to 'claude')

    Returns:
        Filename (not full path)
    """
    # Extract date
    timestamp = datetime.utcnow()
    date_str = timestamp.strftime('%Y%m%d')

    # Extract or default agent
    if agent is None:
        agent = metadata.get('injected_by', 'claude')

    # Extract title for slug
    if title:
        slug = slugify(title)
    elif metadata.get('title'):
        slug = slugify(metadata['title'])
    else:
        # Fallback to timestamp
        slug = timestamp.strftime('%H%M%S')

    # Format: YYYYMMDD_agent_slug.md
    filename = f"{date_str}_{agent}_{slug}.md"

    return filename


def handle_collision(base_path: Path) -> Path:
    """Handle filename collision by adding counter.

    Args:
        base_path: Desired file path

    Returns:
        Available file path (may have -N suffix)
    """
    if not base_path.exists():
        return base_path

    # Extract components
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    # Try numbered suffixes
    counter = 1
    while True:
        new_path = parent / f"{stem}-{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

        # Safety limit
        if counter > 1000:
            raise RuntimeError(f"Too many collisions for {base_path}")


def build_frontmatter(metadata: Dict, injection_id: str) -> str:
    """Build YAML frontmatter for persisted markdown.

    Args:
        metadata: Metadata dict
        injection_id: Unique injection identifier

    Returns:
        YAML frontmatter block
    """
    from datetime import datetime

    # Required fields
    lines = [
        '---',
        f'injection_id: {injection_id}',
        f'injected_at: {datetime.utcnow().isoformat()}Z',
        f'injected_by: {metadata.get("injected_by", "claude")}',
    ]

    # Optional fields
    if 'title' in metadata:
        lines.append(f'title: {metadata["title"]}')

    if 'category' in metadata:
        lines.append(f'category: {metadata["category"]}')

    if 'tags' in metadata and metadata['tags']:
        tags_str = ', '.join(metadata['tags'])
        lines.append(f'tags: [{tags_str}]')

    if 'collection' in metadata:
        lines.append(f'collection: {metadata["collection"]}')

    # Add any custom metadata fields (exclude system fields)
    exclude_fields = {
        'injection_id', 'injected_at', 'injected_by', 'title',
        'category', 'tags', 'collection', 'store_type', 'injection_mode'
    }
    for key, value in metadata.items():
        if key not in exclude_fields:
            lines.append(f'{key}: {value}')

    lines.append('---')
    lines.append('')  # Blank line after frontmatter

    return '\n'.join(lines)


def persist_injection(
    content: str,
    collection: str,
    metadata: Dict,
    agent: Optional[str] = None
) -> Dict:
    """Persist injected content to disk.

    Args:
        content: Markdown content to persist
        collection: Collection name
        metadata: Metadata dict
        agent: Agent identifier (defaults to 'claude')

    Returns:
        Dict with persist info: {
            'persisted': bool,
            'path': str,
            'injection_id': str,
            'error': str (if failed)
        }
    """
    try:
        # Generate unique injection ID
        injection_id = str(uuid4())

        # Add injection metadata
        full_metadata = {
            **metadata,
            'injection_id': injection_id,
            'injected_by': agent or metadata.get('injected_by', 'claude'),
            'collection': collection,
        }

        # Get memory directory
        memory_dir = get_agent_memory_dir(collection)

        # Generate filename
        filename = generate_filename(
            title=metadata.get('title'),
            metadata=full_metadata,
            agent=agent
        )

        # Handle collisions
        file_path = handle_collision(memory_dir / filename)

        # Build frontmatter
        frontmatter = build_frontmatter(full_metadata, injection_id)

        # Write file
        full_content = frontmatter + content
        file_path.write_text(full_content, encoding='utf-8')

        logger.info(f"Persisted injection to {file_path}")

        return {
            'persisted': True,
            'path': str(file_path),
            'injection_id': injection_id,
        }

    except Exception as e:
        logger.error(f"Failed to persist injection: {e}")
        return {
            'persisted': False,
            'error': str(e),
        }
