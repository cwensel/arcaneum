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
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ...embeddings.client import EmbeddingClient
from ..common.sync import MetadataBasedSync, compute_text_file_hash
from .discovery import MarkdownDiscovery
from .chunker import SemanticMarkdownChunker

logger = logging.getLogger(__name__)


class MarkdownIndexingPipeline:
    """Orchestrate markdown file indexing with incremental sync."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: EmbeddingClient,
        batch_size: int = 300,
        exclude_patterns: List[str] = None,
        file_workers: int = 1,
        embedding_workers: int = 4,
        embedding_batch_size: int = 256,
    ):
        """Initialize markdown indexing pipeline.

        Args:
            qdrant_client: Qdrant client instance
            embedding_client: Embedding client instance
            batch_size: Number of points per upload batch (default: 300, optimized from 100)
            exclude_patterns: Patterns to exclude from discovery (default: node_modules, .git, venv)
            file_workers: Number of markdown files to process in parallel (default: 1)
            embedding_workers: Number of parallel workers for embedding generation (default: 4)
            embedding_batch_size: Batch size for embedding generation (default: 256, optimized from 200 per arcaneum-9kgg)
        """
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.batch_size = batch_size
        self.file_workers = file_workers
        self.embedding_workers = embedding_workers
        self.embedding_batch_size = embedding_batch_size

        # Initialize components with custom or default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = ['**/node_modules/**', '**/.git/**', '**/venv/**']
        self.discovery = MarkdownDiscovery(exclude_patterns=exclude_patterns)
        self.sync = MetadataBasedSync(qdrant_client)

    def _process_single_markdown(
        self,
        file_path: Path,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        chunker: SemanticMarkdownChunker,
        point_id_start: int,
        verbose: bool,
        file_idx: int,
        total_files: int
    ) -> Tuple[List[PointStruct], int, Optional[str]]:
        """Process a single markdown file: read, chunk, embed, create points.

        Args:
            file_path: Path to markdown file
            collection_name: Collection name (for metadata)
            model_name: Embedding model name
            model_config: Model configuration
            chunker: SemanticMarkdownChunker instance
            point_id_start: Starting point ID for this file
            verbose: Verbose output flag
            file_idx: Current file index (for progress)
            total_files: Total number of files (for progress)

        Returns:
            Tuple of (points list, chunk count, error message or None)
        """
        try:
            # Stage 1: Read and extract metadata
            if not verbose:
                print(f"\r[{file_idx}/{total_files}] {file_path.name} → reading{' '*20}", end="", flush=True)
            else:
                print(f"\n[{file_idx}/{total_files}] {file_path.name}", flush=True)
                print(f"  → reading file", flush=True)

            file_metadata = self.discovery.extract_metadata(file_path)
            content, frontmatter = MarkdownDiscovery.read_file_with_frontmatter(file_path)

            if verbose:
                print(f"     read {len(content)} chars", flush=True)

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

            # Stage 2: Chunk the content
            if not verbose:
                print(f"\r[{file_idx}/{total_files}] {file_path.name} → chunking ({len(content)} chars){' '*15}", end="", flush=True)
            else:
                print(f"  → chunking ({len(content)} chars)", flush=True)

            chunks = chunker.chunk(content, base_metadata)
            file_chunk_count = len(chunks)

            if verbose:
                print(f"     created {file_chunk_count} chunks", flush=True)

            # Stage 3: Generate embeddings (parallel)
            texts = [chunk.text for chunk in chunks]

            if not verbose:
                print(f"\r[{file_idx}/{total_files}] {file_path.name} → embedding ({file_chunk_count} chunks){' '*15}", end="", flush=True)
            else:
                print(f"  → embedding ({file_chunk_count} chunks)", flush=True)

            # Generate embeddings in parallel using ThreadPoolExecutor
            embeddings = self.embeddings.embed_parallel(
                texts,
                model_name,
                max_workers=self.embedding_workers,
                batch_size=self.embedding_batch_size
            )

            if verbose:
                print(f"     embedded {file_chunk_count} chunks", flush=True)

            # Stage 4: Create points
            points = []
            point_id = point_id_start
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

                points.append(point)
                point_id += 1

            return (points, file_chunk_count, None)

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            return ([], 0, f"{error_type}: {error_msg}")

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
            markdown_files, renames, already_indexed = self.sync.get_unindexed_files(
                collection_name, all_markdown_files, hash_fn=compute_text_file_hash
            )

            # Handle renames first (metadata update only, no reindexing)
            if renames:
                self.sync.handle_renames(collection_name, renames)

            logger.info(f"Incremental sync: {len(markdown_files)} new/modified, "
                       f"{len(renames)} renamed, {len(already_indexed)} already indexed")

            if verbose:
                if renames:
                    print(f"📝 Renamed {len(renames)} files (metadata updated)")
                print(f"📊 Found {len(all_markdown_files)} files: {len(markdown_files)} new/modified, "
                      f"{len(renames)} renamed, {len(already_indexed)} already indexed")

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
            print(" ✓\n")  # Add newline to prevent overwriting
        else:
            print(f"✓ Model ready", flush=True)
            print()

        # Process files with optional parallel processing (arcaneum-ce28)
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            total_files = len(markdown_files)

            # Use parallel processing if file_workers > 1
            if self.file_workers > 1:
                # Parallel mode: Use ThreadPoolExecutor to process multiple files concurrently
                # Pre-allocate point ID ranges (generous allocation: 500 chunks per file)
                point_id_step = 500

                with ThreadPoolExecutor(max_workers=self.file_workers) as executor:
                    # Submit all file processing jobs
                    future_to_file = {}
                    for file_idx, file_path in enumerate(markdown_files, 1):
                        future = executor.submit(
                            self._process_single_markdown,
                            file_path,
                            collection_name,
                            model_name,
                            model_config,
                            chunker,
                            point_id + (file_idx - 1) * point_id_step,
                            verbose,
                            file_idx,
                            total_files
                        )
                        future_to_file[future] = (file_idx, file_path)

                    # Collect results as they complete and upload
                    for future in as_completed(future_to_file):
                        file_idx, file_path = future_to_file[future]
                        points, file_chunk_count, error = future.result()

                        if error:
                            # Show error
                            if not verbose:
                                status_line = f"[{file_idx}/{total_files}] {file_path.name} ✗ ({error.split(':')[0]})"
                                print(f"\r{status_line:<80}")
                            else:
                                print(f"  ✗ ERROR: {error}", flush=True)
                            stats["errors"] += 1
                        else:
                            # Upload this file's chunks
                            if points:
                                if not verbose:
                                    print(f"\r[{file_idx}/{total_files}] {file_path.name} → uploading ({len(points)} chunks){' '*15}", end="", flush=True)
                                else:
                                    print(f"  → uploading ({len(points)} chunks)", flush=True)

                                self.qdrant.upsert(
                                    collection_name=collection_name,
                                    points=points
                                )
                                stats["chunks"] += len(points)
                                stats["files"] += 1

                                # Complete
                                if not verbose:
                                    status_line = f"[{file_idx}/{total_files}] {file_path.name} ✓ ({file_chunk_count} chunks)"
                                    print(f"\r{status_line:<80}")
                                else:
                                    print(f"  ✓ complete ({file_chunk_count} chunks)", flush=True)

            else:
                # Sequential mode: Process files one at a time
                for file_idx, file_path in enumerate(markdown_files, 1):
                    points, file_chunk_count, error = self._process_single_markdown(
                        file_path,
                        collection_name,
                        model_name,
                        model_config,
                        chunker,
                        point_id,
                        verbose,
                        file_idx,
                        total_files
                    )

                    if error:
                        # Show error
                        if not verbose:
                            status_line = f"[{file_idx}/{total_files}] {file_path.name} ✗ ({error.split(':')[0]})"
                            print(f"\r{status_line:<80}")
                        else:
                            print(f"  ✗ ERROR: {error}", flush=True)
                        stats["errors"] += 1
                    else:
                        # Upload this file's chunks
                        if points:
                            if not verbose:
                                print(f"\r[{file_idx}/{total_files}] {file_path.name} → uploading ({len(points)} chunks){' '*15}", end="", flush=True)
                            else:
                                print(f"  → uploading ({len(points)} chunks)", flush=True)

                            self.qdrant.upsert(
                                collection_name=collection_name,
                                points=points
                            )
                            stats["chunks"] += len(points)
                            stats["files"] += 1
                            point_id += len(points)

                            # Complete
                            if not verbose:
                                status_line = f"[{file_idx}/{total_files}] {file_path.name} ✓ ({file_chunk_count} chunks)"
                                print(f"\r{status_line:<80}")
                            else:
                                print(f"  ✓ complete ({file_chunk_count} chunks)", flush=True)

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

        # Generate embeddings (parallel)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.embed_parallel(
            texts,
            model_name,
            max_workers=self.embedding_workers,
            batch_size=self.embedding_batch_size
        )

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
