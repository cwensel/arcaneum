"""
Markdown indexing pipeline orchestrator (RDR-014).

This module orchestrates the complete markdown indexing workflow:
- Discovery → Chunking → Embedding → Indexing
- Supports both directory sync and direct injection modes
- Incremental sync using metadata-based change detection
- Progress reporting and error handling
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

from ...embeddings.client import EmbeddingClient
from ..common.sync import MetadataBasedSync, compute_file_hash
from .discovery import MarkdownDiscovery
from .chunker import SemanticMarkdownChunker

logger = logging.getLogger(__name__)


class MarkdownIndexingPipeline:
    """Orchestrate markdown file indexing with incremental sync."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: EmbeddingClient,
        batch_size: int = 100,
        exclude_patterns: List[str] = None,
    ):
        """Initialize markdown indexing pipeline.

        Args:
            qdrant_client: Qdrant client instance
            embedding_client: Embedding client instance
            batch_size: Number of points per upload batch
            exclude_patterns: Patterns to exclude from discovery (default: node_modules, .git, venv)
        """
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.batch_size = batch_size

        # Initialize components with custom or default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = ['**/node_modules/**', '**/.git/**', '**/venv/**']
        self.discovery = MarkdownDiscovery(exclude_patterns=exclude_patterns)
        self.sync = MetadataBasedSync(qdrant_client)

    def index_directory(
        self,
        markdown_dir: Path,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        force_reindex: bool = False,
        verbose: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        recursive: bool = True
    ) -> Dict:
        """Index markdown files in directory with incremental sync.

        Args:
            markdown_dir: Directory containing markdown files
            collection_name: Qdrant collection name
            model_name: Embedding model to use
            model_config: Model configuration
            force_reindex: Bypass sync and reindex all files
            verbose: Show detailed progress
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens

        Returns:
            Statistics dict with files, chunks, errors counts
        """
        # Initialize chunker
        chunker = SemanticMarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_code_blocks=True
        )

        # Discover all markdown files
        all_markdown_files = self.discovery.discover_files(markdown_dir, recursive=recursive)
        logger.info(f"Found {len(all_markdown_files)} total markdown files")

        # Filter to unindexed files (unless force_reindex)
        if verbose:
            print(f"🔍 Scanning collection for existing files...")

        if force_reindex:
            markdown_files = all_markdown_files
            logger.info(f"Force reindex: processing all {len(markdown_files)} files")
            if verbose:
                print(f"🔄 Force reindex: {len(markdown_files)} files to process")
        else:
            markdown_files = self.sync.get_unindexed_files(collection_name, all_markdown_files)
            skipped = len(all_markdown_files) - len(markdown_files)
            logger.info(f"Incremental sync: {len(markdown_files)} new/modified, {skipped} already indexed")

            if verbose:
                print(f"📊 Found {len(all_markdown_files)} files: {len(markdown_files)} new/modified, {skipped} already indexed")

        if not markdown_files:
            logger.info("No markdown files to index")
            if not verbose:
                print("All markdown files up to date")
            else:
                print("✅ All markdown files are up to date")
            return {"files": 0, "chunks": 0, "errors": 0}

        # Show count
        if not verbose:
            print(f"Found {len(markdown_files)} markdown file(s)")
            print(f"Processing {len(markdown_files)} file(s)...")
        else:
            print()

        # Pre-load model to avoid "hang" during first file processing
        is_cached = self.embeddings.is_model_cached(model_name)
        if not is_cached and not verbose:
            print(f"⬇️  Downloading {model_name} model (first time only)...", flush=True)
        elif not is_cached and verbose:
            print(f"⬇️  Model not cached, downloading {model_name}...", flush=True)
        elif verbose:
            print(f"📦 Loading {model_name} model from cache...", flush=True)
        else:
            print(f"📦 Loading model...", flush=True)

        # Load model now (separate phase from file processing)
        self.embeddings.get_model(model_name)

        if not verbose:
            print(" ✓")
        else:
            print(f"✓ Model ready", flush=True)
            print()

        # Process files with progress tracking
        batch = []
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            total_files = len(markdown_files)

            # Use tqdm for verbose mode
            if verbose:
                pbar = tqdm(total=total_files, desc="Indexing markdown", unit="file",
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                pbar = None

            for file_idx, file_path in enumerate(markdown_files, 1):
                try:
                    # Show progress
                    if not verbose:
                        print(f"[{file_idx}/{total_files}] {file_path.name}", end="", flush=True)
                    else:
                        print(f"\n[{file_idx}/{total_files}] Processing {file_path.name}...", flush=True)

                    # Extract metadata and content
                    if verbose:
                        print(f"  → Extracting metadata", flush=True)

                    file_metadata = self.discovery.extract_metadata(file_path)
                    content, frontmatter = MarkdownDiscovery.read_file_with_frontmatter(file_path)

                    if verbose:
                        print(f"  → Read {len(content)} chars", flush=True)

                    # Build base metadata
                    base_metadata = {
                        'filename': file_metadata.file_name,
                        'file_path': file_metadata.file_path,
                        'file_hash': file_metadata.content_hash,
                        'file_size': file_metadata.file_size,
                        'store_type': 'markdown',
                        'has_frontmatter': file_metadata.has_frontmatter,
                    }

                    # Add frontmatter fields if present
                    if file_metadata.has_frontmatter:
                        if file_metadata.title:
                            base_metadata['title'] = file_metadata.title
                        if file_metadata.author:
                            base_metadata['author'] = file_metadata.author
                        if file_metadata.tags:
                            base_metadata['tags'] = file_metadata.tags
                        if file_metadata.category:
                            base_metadata['category'] = file_metadata.category
                        if file_metadata.project:
                            base_metadata['project'] = file_metadata.project

                    # Chunk the content
                    if verbose:
                        print(f"  → Chunking text ({len(content)} chars)", flush=True)

                    chunks = chunker.chunk(content, base_metadata)
                    file_chunk_count = len(chunks)

                    if verbose:
                        print(f"  → Created {file_chunk_count} chunks", flush=True)

                    # Generate embeddings
                    texts = [chunk.text for chunk in chunks]

                    if verbose:
                        print(f"  → Embedding {file_chunk_count} chunks...", flush=True)

                    # Batch embedding with progress updates (matching PDF indexing style)
                    EMBEDDING_BATCH_SIZE = 100
                    embeddings = []
                    for batch_start in range(0, file_chunk_count, EMBEDDING_BATCH_SIZE):
                        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, file_chunk_count)
                        batch_texts = texts[batch_start:batch_end]
                        batch_embeddings = self.embeddings.embed(batch_texts, model_name)
                        embeddings.extend(batch_embeddings)

                        # Show progress (PDF-style: continuous updates on same line)
                        if not verbose:
                            # Non-verbose: show embedding progress with carriage return
                            print(f"\r[{file_idx}/{total_files}] {file_path.name} → embedding {batch_end}/{file_chunk_count}",
                                  end="", flush=True)
                        elif file_chunk_count > EMBEDDING_BATCH_SIZE:
                            # Verbose: show as separate lines for large files
                            print(f"    Embedded {batch_end}/{file_chunk_count} chunks", flush=True)

                    # Create points
                    for chunk, embedding in zip(chunks, embeddings):
                        payload = {
                            **chunk.metadata,
                            'text': chunk.text,
                        }

                        # Handle named vectors if needed
                        vector_name = model_config.get('vector_name')
                        if vector_name:
                            vector = {vector_name: embedding}
                        else:
                            vector = embedding

                        point = PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload
                        )

                        batch.append(point)
                        point_id += 1

                        # Upload batch if full
                        if len(batch) >= self.batch_size:
                            self.qdrant.upsert(
                                collection_name=collection_name,
                                points=batch
                            )
                            stats["chunks"] += len(batch)
                            batch = []

                    stats["files"] += 1

                    # Show final status (PDF-style: uploading → done)
                    if not verbose:
                        print(f"\r[{file_idx}/{total_files}] {file_path.name} → uploading" + " " * 30, end="", flush=True)
                        print(f"\r[{file_idx}/{total_files}] {file_path.name}" + " " * 50, end="", flush=True)
                        print(f"\r[{file_idx}/{total_files}] {file_path.name} ✓ ({file_chunk_count} chunks)")
                    else:
                        print(f"  ✓ Uploaded {file_chunk_count} chunks")

                    if pbar:
                        pbar.update(1)

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Error processing {file_path}: {e}")
                    if not verbose:
                        print(f" ✗ Error")
                    else:
                        print(f"  ✗ Error: {e}")
                    continue

            # Upload final batch
            if batch:
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                stats["chunks"] += len(batch)

            if pbar:
                pbar.close()

            # Summary
            if verbose:
                print(f"\n✅ Indexed {stats['files']} files ({stats['chunks']} chunks)")
                if stats['errors'] > 0:
                    print(f"⚠️  {stats['errors']} files had errors")
            else:
                print(f"Indexed {stats['files']} file(s), {stats['chunks']} chunk(s)")
                if stats['errors'] > 0:
                    print(f"{stats['errors']} error(s)")

            return stats

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if verbose:
                print(f"\n❌ Pipeline error: {e}")
            raise

    def inject_content(
        self,
        content: str,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        metadata: Optional[Dict] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        persist: bool = True,
        persist_dir: Optional[Path] = None
    ) -> Dict:
        """Inject markdown content directly (for agent memory).

        Args:
            content: Markdown content to inject
            collection_name: Qdrant collection name
            model_name: Embedding model to use
            model_config: Model configuration
            metadata: Optional metadata to attach
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            persist: Whether to persist to disk
            persist_dir: Directory to persist to (default: ~/.arcaneum/agent-memory)

        Returns:
            Statistics dict
        """
        # Initialize chunker
        chunker = SemanticMarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Build base metadata
        base_metadata = metadata or {}
        base_metadata.update({
            'store_type': 'markdown',
            'injection_mode': True,
        })

        # Chunk content
        chunks = chunker.chunk(content, base_metadata)

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.embed(texts, model_name)

        # Create points
        point_id = self._get_next_point_id(collection_name)
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                **chunk.metadata,
                'text': chunk.text,
            }

            vector_name = model_config.get('vector_name')
            if vector_name:
                vector = {vector_name: embedding}
            else:
                vector = embedding

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )

            points.append(point)
            point_id += 1

        # Upload to Qdrant
        self.qdrant.upsert(
            collection_name=collection_name,
            points=points
        )

        # Persist to disk if requested
        persist_result = {}
        if persist:
            from .injection import persist_injection
            persist_result = persist_injection(
                content=content,
                collection=collection_name,
                metadata=metadata or {},
                agent=None  # Use default 'claude'
            )

            if persist_result.get('persisted'):
                logger.info(f"Persisted to {persist_result['path']}")
            else:
                logger.warning(f"Persistence failed: {persist_result.get('error')}")

        return {
            "chunks": len(chunks),
            "errors": 0,
            **persist_result
        }

    def _get_next_point_id(self, collection_name: str) -> int:
        """Get next available point ID for collection."""
        try:
            collection_info = self.qdrant.get_collection(collection_name)
            return collection_info.points_count
        except Exception:
            return 0
