"""Directory sync command for dual indexing (RDR-009).

This module implements the 'corpus sync' command that indexes documents
to both Qdrant and MeiliSearch in a single operation.
"""

import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..cli.output import print_json, print_error, print_info
from ..cli.utils import create_qdrant_client
from ..cli.interaction_logger import interaction_logger
from ..cli.errors import InvalidArgumentError, ResourceNotFoundError
from ..embeddings.client import EmbeddingClient, EMBEDDING_MODELS
from ..fulltext.client import FullTextClient
from ..schema.document import DualIndexDocument
from ..indexing.dual_indexer import DualIndexer
from ..indexing.collection_metadata import get_collection_type, get_collection_metadata
from ..indexing.common.sync import MetadataBasedSync, compute_quick_hash
from ..config import DEFAULT_MODELS

console = Console()
logger = logging.getLogger(__name__)


def get_meili_client() -> FullTextClient:
    """Get MeiliSearch client from environment or auto-generated key."""
    from ..paths import get_meilisearch_api_key

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def discover_files(
    directory: Path,
    file_types: Optional[str],
    corpus_type: str
) -> List[Path]:
    """Discover files to index based on corpus type and file filters.

    Args:
        directory: Directory to scan
        file_types: Comma-separated file extensions (e.g., ".py,.js")
        corpus_type: Type of corpus (pdf, code, markdown)

    Returns:
        List of file paths to index
    """
    # Determine extensions to look for
    if file_types:
        extensions = set(ext.strip().lower() for ext in file_types.split(','))
        # Ensure extensions start with '.'
        extensions = set(e if e.startswith('.') else f'.{e}' for e in extensions)
    else:
        # Default extensions based on corpus type
        type_extensions = {
            "pdf": {".pdf"},
            "code": {".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".cpp", ".c", ".h", ".hpp"},
            "markdown": {".md", ".markdown"},
        }
        extensions = type_extensions.get(corpus_type, set())

    if not extensions:
        logger.warning(f"No file extensions defined for corpus type: {corpus_type}")
        return []

    # Discover files
    files = []
    for ext in extensions:
        pattern = f"*{ext}"  # rglob already handles recursive search
        found = list(directory.rglob(pattern))
        files.extend(found)

    # Sort for consistent ordering
    files.sort()
    return files


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file for change detection."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # First 16 chars is enough


def chunk_pdf_file(file_path: Path, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunk a PDF file using existing PDF chunking logic.

    Args:
        file_path: Path to PDF file
        model_config: Model configuration with chunk_size, etc.

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.pdf.chunker import PDFChunker
    from ..indexing.pdf.extractor import PDFExtractor

    # Extract text from PDF using PDFExtractor class
    extractor = PDFExtractor()
    text, metadata = extractor.extract(file_path)

    if not text or not text.strip():
        logger.warning(f"No text extracted from {file_path}")
        return []

    # Create chunker and chunk the text
    chunker = PDFChunker(model_config)
    base_metadata = {
        'file_path': str(file_path),
        'filename': file_path.name,
        'page_boundaries': metadata.get('page_boundaries', []),
    }

    chunks = chunker.chunk(text, base_metadata)

    return [{'text': c.text, 'metadata': c.metadata} for c in chunks]


def chunk_markdown_file(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk a markdown file using semantic markdown chunking.

    Args:
        file_path: Path to markdown file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.markdown.chunker import SemanticMarkdownChunker

    text = file_path.read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    chunker = SemanticMarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    base_metadata = {
        'file_path': str(file_path),
        'filename': file_path.name,
    }

    chunks = chunker.chunk(text, base_metadata)
    return [{'text': c.text, 'metadata': c.metadata} for c in chunks]


def chunk_code_file(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk a source code file using AST-aware chunking.

    Args:
        file_path: Path to source file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.ast_chunker import ASTCodeChunker

    text = file_path.read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    # Determine language from extension
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
    }
    language = ext_to_lang.get(file_path.suffix.lower(), 'unknown')

    chunker = ASTCodeChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = chunker.chunk(text, language)

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            'text': chunk.content,
            'metadata': {
                'file_path': str(file_path),
                'filename': file_path.name,
                'language': language,
                'chunk_index': i,
                'method': chunk.method,
            }
        })

    return result


def sync_directory_command(
    directory: str,
    corpus: str,
    models: str,
    file_types: Optional[str],
    force: bool,
    verify: bool,
    verbose: bool,
    output_json: bool
):
    """Sync a directory to both Qdrant and MeiliSearch.

    This implements the second command of the 2-command workflow:
    1. corpus create - creates both systems
    2. corpus sync (this command) - indexes documents to both systems

    Args:
        directory: Directory to sync
        corpus: Corpus name (must exist)
        models: Comma-separated list of embedding models
        file_types: File extensions to index (e.g., ".py,.js")
        force: If True, reindex all files (bypass change detection)
        verify: If True, verify collection integrity after indexing
        verbose: If True, show detailed progress
        output_json: If True, output JSON format
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "corpus", "sync",
        corpus=corpus,
        directory=directory,
        models=models,
        file_types=file_types,
    )

    try:
        dir_path = Path(directory).resolve()

        if not dir_path.exists():
            raise InvalidArgumentError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise InvalidArgumentError(f"Not a directory: {directory}")

        if not output_json:
            print_info(f"Syncing '{directory}' to corpus '{corpus}'")

        # Initialize clients
        qdrant = create_qdrant_client()
        meili = get_meili_client()

        # Verify corpus exists in both systems
        try:
            qdrant.get_collection(corpus)
        except Exception:
            raise ResourceNotFoundError(
                f"Qdrant collection '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        if not meili.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        if not meili.index_exists(corpus):
            raise ResourceNotFoundError(
                f"MeiliSearch index '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        # Get corpus type and configured models from collection metadata
        corpus_type = get_collection_type(qdrant, corpus)
        metadata = get_collection_metadata(qdrant, corpus)
        configured_models = metadata.get('model', models)

        if not corpus_type:
            corpus_type = 'pdf'  # Default
            logger.warning(f"Collection type not set, defaulting to {corpus_type}")

        if not output_json:
            print_info(f"Corpus type: {corpus_type}")
            print_info(f"Models: {configured_models}")
            print_info(f"Dual indexing to:")
            print_info(f"  - Qdrant collection: {corpus} (semantic search)")
            print_info(f"  - MeiliSearch index: {corpus} (full-text search)")

        # Parse models
        model_list = [m.strip() for m in configured_models.split(',')]

        # Discover files
        files = discover_files(dir_path, file_types, corpus_type)

        if not files:
            if output_json:
                print_json("success", "No files to index", data={"indexed": 0})
            else:
                print_info("No files found to index")
            interaction_logger.finish(result_count=0)
            return

        if not output_json:
            print_info(f"Found {len(files)} files in directory")

        # Apply change detection (skip already indexed files) unless --force
        already_indexed_count = 0
        meili_backfill_paths = []  # Files in Qdrant but missing from MeiliSearch
        qdrant_backfill_paths = []  # Files in MeiliSearch but missing from Qdrant

        if force:
            if not output_json:
                print_info("Force mode: reindexing all files")
        else:
            if not output_json:
                print_info("Checking index parity between Qdrant and MeiliSearch...")

            # Get file paths from both systems
            sync_manager = MetadataBasedSync(qdrant)
            meili_file_paths = meili.get_all_file_paths(corpus)
            qdrant_file_paths = sync_manager._get_indexed_file_paths_set(corpus)

            # Calculate set operations
            in_both_systems = qdrant_file_paths & meili_file_paths
            missing_from_meili = qdrant_file_paths - meili_file_paths
            missing_from_qdrant = meili_file_paths - qdrant_file_paths

            # Files in Qdrant but not in MeiliSearch need backfill
            if missing_from_meili:
                meili_backfill_paths = [p for p in missing_from_meili if Path(p).exists()]
                if not output_json and meili_backfill_paths:
                    print_info(f"Found {len(meili_backfill_paths)} files in Qdrant missing from MeiliSearch")

            # Files in MeiliSearch but not in Qdrant need backfill
            if missing_from_qdrant:
                qdrant_backfill_paths = [p for p in missing_from_qdrant if Path(p).exists()]
                if not output_json and qdrant_backfill_paths:
                    print_info(f"Found {len(qdrant_backfill_paths)} files in MeiliSearch missing from Qdrant")

            # Check for new/modified files not in either system
            # Convert discovered files to absolute path strings for comparison
            discovered_file_paths = {str(f.absolute()) for f in files}
            all_indexed_paths = qdrant_file_paths | meili_file_paths
            new_file_paths = discovered_file_paths - all_indexed_paths

            # Filter to only process new files (not in either system)
            files_to_process = [f for f in files if str(f.absolute()) in new_file_paths]

            # Count files in both systems (truly skipped)
            already_indexed_count = len(in_both_systems & discovered_file_paths)

            if not output_json:
                if already_indexed_count > 0:
                    print_info(f"Skipping {already_indexed_count} files already in both systems")
                if len(files_to_process) > 0:
                    print_info(f"Processing {len(files_to_process)} new files")

            files = files_to_process

        # If no new files but there are files to backfill, continue
        if not files and not meili_backfill_paths and not qdrant_backfill_paths:
            if output_json:
                print_json("success", "All files already indexed", data={
                    "indexed": 0,
                    "skipped": already_indexed_count
                })
            else:
                print_info("All files are already indexed in both systems (use --force to reindex)")
            interaction_logger.finish(result_count=0)
            return

        # Process files
        total_indexed = 0
        total_chunks = 0
        total_qdrant = 0
        total_meili = 0

        # Only initialize embedding infrastructure if there are new files to process
        if files:
            # Initialize embedding client
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Create dual indexer
            dual_indexer = DualIndexer(
                qdrant_client=qdrant,
                meili_client=meili,
                collection_name=corpus,
                index_name=corpus
            )

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                # Fallback config
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                task = progress.add_task("Indexing...", total=len(files))

                for file_path in files:
                    progress.update(task, description=f"Processing {file_path.name}...")

                    try:
                        # Chunk file based on corpus type
                        if verbose and not output_json:
                            progress.console.print(f"[dim]Extracting text from {file_path.name}...[/dim]")

                        if corpus_type == 'pdf':
                            chunks = chunk_pdf_file(file_path, model_config)
                        elif corpus_type == 'markdown':
                            chunks = chunk_markdown_file(
                                file_path,
                                model_config.get('chunk_size', 512),
                                model_config.get('chunk_overlap', 50)
                            )
                        elif corpus_type == 'code':
                            chunks = chunk_code_file(
                                file_path,
                                model_config.get('chunk_size', 400),
                                model_config.get('chunk_overlap', 20)
                            )
                        else:
                            logger.warning(f"Unknown corpus type: {corpus_type}, skipping {file_path}")
                            continue

                        if not chunks:
                            if verbose and not output_json:
                                progress.console.print(f"[yellow]  No text extracted from {file_path.name}[/yellow]")
                            progress.advance(task)
                            continue

                        if verbose and not output_json:
                            progress.console.print(f"[dim]  Created {len(chunks)} chunks, generating embeddings...[/dim]")

                        # Build dual index documents
                        documents = []
                        file_hash = compute_file_hash(file_path)
                        quick_hash = compute_quick_hash(file_path)

                        for i, chunk in enumerate(chunks):
                            # Generate embeddings for all models
                            vectors = {}
                            for model in model_list:
                                embeddings = embedding_client.embed([chunk['text']], model)
                                # Handle both list and numpy array returns
                                if hasattr(embeddings, 'tolist'):
                                    vectors[model] = embeddings[0].tolist()
                                else:
                                    vectors[model] = list(embeddings[0])

                            # Create document with shared metadata
                            doc = DualIndexDocument(
                                id=str(uuid4()),
                                content=chunk['text'],
                                file_path=str(file_path.absolute()),  # Use absolute path for change detection
                                filename=file_path.name,
                                file_extension=file_path.suffix,
                                chunk_index=i,
                                chunk_count=len(chunks),
                                file_hash=file_hash,
                                file_size=file_path.stat().st_size,
                                quick_hash=quick_hash,
                                vectors=vectors,
                            )

                            # Add type-specific metadata
                            chunk_meta = chunk.get('metadata', {})

                            if corpus_type == 'pdf':
                                doc.page_number = chunk_meta.get('page_number')
                                doc.document_type = 'pdf'

                            elif corpus_type == 'markdown':
                                doc.language = 'markdown'
                                doc.section = chunk_meta.get('header_path')
                                if chunk_meta.get('has_code_blocks'):
                                    doc.tags = ['has-code']

                            elif corpus_type == 'code':
                                doc.language = chunk_meta.get('language', 'unknown')
                                doc.line_number = chunk_meta.get('line_number')

                            documents.append(doc)

                        # Index to both systems
                        if documents:
                            if verbose and not output_json:
                                progress.console.print(f"[dim]  Indexing {len(documents)} chunks to Qdrant + MeiliSearch...[/dim]")

                            qdrant_count, meili_count = dual_indexer.index_batch(documents)
                            total_chunks += len(documents)
                            total_qdrant += qdrant_count
                            total_meili += meili_count

                            if verbose and not output_json:
                                progress.console.print(f"[green]  ✓ {file_path.name}: {len(chunks)} chunks → Qdrant({qdrant_count}) + MeiliSearch({meili_count})[/green]")

                        total_indexed += 1

                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        if not output_json:
                            progress.console.print(f"[yellow]Warning: Failed to process {file_path.name}: {e}[/yellow]")

                    progress.advance(task)

        # Backfill MeiliSearch for files already in Qdrant but missing from MeiliSearch
        meili_backfilled = 0
        meili_backfill_chunks = 0
        meili_backfill_failed = 0

        if meili_backfill_paths and not force:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(meili_backfill_paths)} files to MeiliSearch...[/blue]")

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling...", total=len(meili_backfill_paths))

                for file_path_str in meili_backfill_paths:
                    progress.update(backfill_task, description=f"Backfilling {Path(file_path_str).name}...")

                    try:
                        # Fetch all chunks for this file from Qdrant
                        points, _ = qdrant.scroll(
                            collection_name=corpus,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="file_path",
                                        match=MatchValue(value=file_path_str)
                                    )
                                ]
                            ),
                            limit=1000,
                            with_payload=True,
                            with_vectors=False
                        )

                        if not points:
                            progress.advance(backfill_task)
                            continue

                        # Convert Qdrant points to MeiliSearch documents
                        meili_docs = []
                        for point in points:
                            payload = point.payload
                            # Create a minimal doc for MeiliSearch
                            meili_doc = {
                                "id": str(point.id),
                                "content": payload.get("text", ""),
                                "file_path": payload.get("file_path", ""),
                                "filename": payload.get("filename", ""),
                                "file_extension": payload.get("file_extension", ""),
                                "chunk_index": payload.get("chunk_index", 0),
                            }

                            # Add optional fields
                            if payload.get("page_number"):
                                meili_doc["page_number"] = payload["page_number"]
                            if payload.get("programming_language"):
                                meili_doc["language"] = payload["programming_language"]
                            if payload.get("document_type"):
                                meili_doc["document_type"] = payload["document_type"]

                            meili_docs.append(meili_doc)

                        # Index to MeiliSearch with longer timeout for backfill (2 minutes)
                        if meili_docs:
                            meili.add_documents_sync(corpus, meili_docs, timeout_ms=120000)
                            meili_backfill_chunks += len(meili_docs)

                            if verbose and not output_json:
                                progress.console.print(f"[green]  ✓ {Path(file_path_str).name}: {len(meili_docs)} chunks → MeiliSearch[/green]")

                        meili_backfilled += 1

                    except Exception as e:
                        meili_backfill_failed += 1
                        logger.error(f"Failed to backfill {file_path_str}: {e}")
                        if not output_json:
                            progress.console.print(f"[yellow]Warning: Failed to backfill {Path(file_path_str).name}: {e}[/yellow]")

                    progress.advance(backfill_task)

            total_meili += meili_backfill_chunks

        # Backfill Qdrant for files in MeiliSearch but missing from Qdrant
        # This requires re-processing the files since we need embeddings
        qdrant_backfilled = 0
        qdrant_backfill_chunks = 0
        qdrant_backfill_failed = 0

        if qdrant_backfill_paths and not force:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(qdrant_backfill_paths)} files to Qdrant (requires embedding)...[/blue]")

            # Need embedding client for Qdrant backfill
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling Qdrant...", total=len(qdrant_backfill_paths))

                for file_path_str in qdrant_backfill_paths:
                    file_path = Path(file_path_str)
                    progress.update(backfill_task, description=f"Backfilling {file_path.name}...")

                    try:
                        # Chunk file based on corpus type
                        if corpus_type == 'pdf':
                            chunks = chunk_pdf_file(file_path, model_config)
                        elif corpus_type == 'markdown':
                            chunks = chunk_markdown_file(
                                file_path,
                                model_config.get('chunk_size', 512),
                                model_config.get('chunk_overlap', 50)
                            )
                        elif corpus_type == 'code':
                            chunks = chunk_code_file(
                                file_path,
                                model_config.get('chunk_size', 400),
                                model_config.get('chunk_overlap', 20)
                            )
                        else:
                            progress.advance(backfill_task)
                            continue

                        if not chunks:
                            progress.advance(backfill_task)
                            continue

                        # Build Qdrant documents with embeddings
                        from qdrant_client.models import PointStruct
                        points = []
                        file_hash = compute_file_hash(file_path)
                        quick_hash = compute_quick_hash(file_path)

                        for i, chunk in enumerate(chunks):
                            # Generate embeddings for all models
                            vectors = {}
                            for model in model_list:
                                embeddings = embedding_client.embed([chunk['text']], model)
                                if hasattr(embeddings, 'tolist'):
                                    vectors[model] = embeddings[0].tolist()
                                else:
                                    vectors[model] = list(embeddings[0])

                            # Build payload
                            payload = {
                                "text": chunk['text'],
                                "file_path": str(file_path.absolute()),
                                "filename": file_path.name,
                                "file_extension": file_path.suffix,
                                "chunk_index": i,
                                "chunk_count": len(chunks),
                                "file_hash": file_hash,
                                "file_size": file_path.stat().st_size,
                                "quick_hash": quick_hash,
                            }

                            # Add type-specific metadata
                            chunk_meta = chunk.get('metadata', {})
                            if corpus_type == 'pdf':
                                if chunk_meta.get('page_number'):
                                    payload["page_number"] = chunk_meta["page_number"]
                                payload["document_type"] = "pdf"

                            points.append(PointStruct(
                                id=str(uuid4()),
                                vector=vectors,
                                payload=payload
                            ))

                        # Upload to Qdrant
                        if points:
                            qdrant.upsert(collection_name=corpus, points=points, wait=True)
                            qdrant_backfill_chunks += len(points)

                            if verbose and not output_json:
                                progress.console.print(f"[green]  ✓ {file_path.name}: {len(points)} chunks → Qdrant[/green]")

                        qdrant_backfilled += 1

                    except Exception as e:
                        qdrant_backfill_failed += 1
                        logger.error(f"Failed to backfill {file_path_str} to Qdrant: {e}")
                        if not output_json:
                            progress.console.print(f"[yellow]Warning: Failed to backfill {file_path.name}: {e}[/yellow]")

                    progress.advance(backfill_task)

            total_qdrant += qdrant_backfill_chunks

        # Output results
        data = {
            "corpus": corpus,
            "directory": str(dir_path),
            "files_indexed": total_indexed,
            "files_skipped": already_indexed_count,
            "meili_backfilled": meili_backfilled,
            "meili_backfill_failed": meili_backfill_failed,
            "qdrant_backfilled": qdrant_backfilled,
            "qdrant_backfill_failed": qdrant_backfill_failed,
            "total_chunks": total_chunks,
            "qdrant_indexed": total_qdrant,
            "meili_indexed": total_meili,
            "meili_backfilled_chunks": meili_backfill_chunks,
            "qdrant_backfilled_chunks": qdrant_backfill_chunks,
            "models": model_list,
        }

        if output_json:
            print_json("success", f"Indexed {total_indexed} files ({total_chunks} chunks)", data=data)
        else:
            console.print(f"\n[green]✅ Sync complete for corpus '{corpus}'[/green]")

            # New files (indexed to both systems)
            if total_indexed > 0:
                console.print(f"   New files (both systems): {total_indexed} files ({total_chunks} chunks)")

            # Already in both systems
            if already_indexed_count > 0:
                console.print(f"   Already synced:           {already_indexed_count} files")

            # MeiliSearch backfill (from Qdrant)
            if meili_backfilled > 0 or meili_backfill_failed > 0:
                status = f"{meili_backfilled} files ({meili_backfill_chunks} chunks)"
                if meili_backfill_failed > 0:
                    status += f" [yellow]({meili_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to MeiliSearch: {status}")

            # Qdrant backfill (requires re-embedding)
            if qdrant_backfilled > 0 or qdrant_backfill_failed > 0:
                status = f"{qdrant_backfilled} files ({qdrant_backfill_chunks} chunks)"
                if qdrant_backfill_failed > 0:
                    status += f" [yellow]({qdrant_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to Qdrant:      {status}")

            # Totals
            console.print(f"\n   Total in Qdrant:      {total_qdrant} vectors")
            console.print(f"   Total in MeiliSearch: {total_meili} documents")

            console.print(f"\n[dim]Search with:[/dim]")
            console.print(f"  arc search semantic \"your query\" --collection {corpus}")
            console.print(f"  arc search text \"your query\" --index {corpus}")

        # Post-verify if requested (uses same verifier as index commands)
        verification_result = None
        if verify:
            from ..indexing.verify import CollectionVerifier

            if not output_json:
                console.print("\n[dim]Verifying collection integrity...[/dim]")

            verifier = CollectionVerifier(qdrant)
            verification_result = verifier.verify_collection(corpus, verbose=verbose)

            if verification_result.is_healthy:
                if not output_json:
                    console.print(f"[green]✓ Collection verified - all {verification_result.complete_items} files complete[/green]")
            else:
                incomplete = verification_result.get_items_needing_repair()
                if not output_json:
                    console.print(f"[yellow]⚠ Found {len(incomplete)} incomplete files[/yellow]")
                    for item in incomplete[:5]:
                        console.print(f"  [yellow]{item}[/yellow]")
                    if len(incomplete) > 5:
                        console.print(f"  [dim]... and {len(incomplete) - 5} more[/dim]")
                    console.print("[dim]Re-run with --force to repair incomplete files[/dim]")

            data["verification"] = {
                "is_healthy": verification_result.is_healthy,
                "total_files": verification_result.total_items,
                "complete_files": verification_result.complete_items,
                "incomplete_files": verification_result.incomplete_items,
            }

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=total_indexed,
            total_chunks=total_chunks,
        )

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to sync directory: {e}", output_json)
        sys.exit(1)


def _backfill_qdrant_to_meili(
    qdrant,
    meili: FullTextClient,
    corpus: str,
    file_paths: List[str],
    verbose: bool,
    output_json: bool,
    progress,
    backfill_task
) -> tuple:
    """Backfill files from Qdrant to MeiliSearch.

    Copies chunk data from Qdrant to MeiliSearch without needing file access.

    Args:
        qdrant: Qdrant client
        meili: MeiliSearch client
        corpus: Corpus name
        file_paths: List of file paths to backfill
        verbose: Show verbose output
        output_json: JSON output mode
        progress: Rich progress instance
        backfill_task: Progress task ID

    Returns:
        Tuple of (files_success, chunks_success, files_failed)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    files_success = 0
    chunks_success = 0
    files_failed = 0

    for file_path_str in file_paths:
        progress.update(backfill_task, description=f"Backfilling {Path(file_path_str).name}...")

        try:
            # Fetch all chunks for this file from Qdrant
            points, _ = qdrant.scroll(
                collection_name=corpus,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path_str)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            if not points:
                progress.advance(backfill_task)
                continue

            # Convert Qdrant points to MeiliSearch documents
            meili_docs = []
            for point in points:
                payload = point.payload
                meili_doc = {
                    "id": str(point.id),
                    "content": payload.get("text", ""),
                    "file_path": payload.get("file_path", ""),
                    "filename": payload.get("filename", ""),
                    "file_extension": payload.get("file_extension", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                }

                # Add optional fields
                if payload.get("page_number"):
                    meili_doc["page_number"] = payload["page_number"]
                if payload.get("programming_language"):
                    meili_doc["language"] = payload["programming_language"]
                if payload.get("document_type"):
                    meili_doc["document_type"] = payload["document_type"]

                meili_docs.append(meili_doc)

            # Index to MeiliSearch with longer timeout (2 minutes)
            if meili_docs:
                meili.add_documents_sync(corpus, meili_docs, timeout_ms=120000)
                chunks_success += len(meili_docs)

                if verbose and not output_json:
                    progress.console.print(f"[green]  ✓ {Path(file_path_str).name}: {len(meili_docs)} chunks → MeiliSearch[/green]")

            files_success += 1

        except Exception as e:
            files_failed += 1
            logger.error(f"Failed to backfill {file_path_str}: {e}")
            if not output_json:
                progress.console.print(f"[yellow]Warning: Failed to backfill {Path(file_path_str).name}: {e}[/yellow]")

        progress.advance(backfill_task)

    return files_success, chunks_success, files_failed


def _backfill_meili_to_qdrant(
    qdrant,
    embedding_client: EmbeddingClient,
    corpus: str,
    corpus_type: str,
    model_list: List[str],
    model_config: Dict[str, Any],
    file_paths: List[str],
    verbose: bool,
    output_json: bool,
    progress,
    backfill_task
) -> tuple:
    """Backfill files from MeiliSearch to Qdrant.

    Re-chunks files and generates embeddings. Requires file access.
    Files that don't exist on disk are skipped with a warning.

    Args:
        qdrant: Qdrant client
        embedding_client: Embedding client
        corpus: Corpus name
        corpus_type: Type of corpus (pdf, code, markdown)
        model_list: List of embedding model names
        model_config: Chunking configuration
        file_paths: List of file paths to backfill
        verbose: Show verbose output
        output_json: JSON output mode
        progress: Rich progress instance
        backfill_task: Progress task ID

    Returns:
        Tuple of (files_success, chunks_success, files_failed, skipped_paths)
    """
    from qdrant_client.models import PointStruct

    files_success = 0
    chunks_success = 0
    files_failed = 0
    skipped_paths = []

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        progress.update(backfill_task, description=f"Backfilling {file_path.name}...")

        # Check if file exists on disk
        if not file_path.exists():
            skipped_paths.append(file_path_str)
            if not output_json:
                progress.console.print(f"[yellow]  ⚠ {file_path.name}: File not found, skipping[/yellow]")
            progress.advance(backfill_task)
            continue

        try:
            # Chunk file based on corpus type
            if corpus_type == 'pdf':
                chunks = chunk_pdf_file(file_path, model_config)
            elif corpus_type == 'markdown':
                chunks = chunk_markdown_file(
                    file_path,
                    model_config.get('chunk_size', 512),
                    model_config.get('chunk_overlap', 50)
                )
            elif corpus_type == 'code':
                chunks = chunk_code_file(
                    file_path,
                    model_config.get('chunk_size', 400),
                    model_config.get('chunk_overlap', 20)
                )
            else:
                progress.advance(backfill_task)
                continue

            if not chunks:
                progress.advance(backfill_task)
                continue

            # Build Qdrant documents with embeddings
            points = []
            file_hash = compute_file_hash(file_path)
            quick_hash = compute_quick_hash(file_path)

            for i, chunk in enumerate(chunks):
                # Generate embeddings for all models
                vectors = {}
                for model in model_list:
                    embeddings = embedding_client.embed([chunk['text']], model)
                    if hasattr(embeddings, 'tolist'):
                        vectors[model] = embeddings[0].tolist()
                    else:
                        vectors[model] = list(embeddings[0])

                # Build payload
                payload = {
                    "text": chunk['text'],
                    "file_path": str(file_path.absolute()),
                    "filename": file_path.name,
                    "file_extension": file_path.suffix,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "file_hash": file_hash,
                    "file_size": file_path.stat().st_size,
                    "quick_hash": quick_hash,
                }

                # Add type-specific metadata
                chunk_meta = chunk.get('metadata', {})
                if corpus_type == 'pdf':
                    if chunk_meta.get('page_number'):
                        payload["page_number"] = chunk_meta["page_number"]
                    payload["document_type"] = "pdf"

                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=vectors,
                    payload=payload
                ))

            # Upload to Qdrant
            if points:
                qdrant.upsert(collection_name=corpus, points=points, wait=True)
                chunks_success += len(points)

                if verbose and not output_json:
                    progress.console.print(f"[green]  ✓ {file_path.name}: {len(points)} chunks → Qdrant[/green]")

            files_success += 1

        except Exception as e:
            files_failed += 1
            logger.error(f"Failed to backfill {file_path_str} to Qdrant: {e}")
            if not output_json:
                progress.console.print(f"[yellow]Warning: Failed to backfill {file_path.name}: {e}[/yellow]")

        progress.advance(backfill_task)

    return files_success, chunks_success, files_failed, skipped_paths


def parity_command(
    corpus: str,
    dry_run: bool,
    verbose: bool,
    output_json: bool
):
    """Check and restore parity between Qdrant and MeiliSearch.

    Compares indexed files in both systems and backfills missing entries:
    - Qdrant -> MeiliSearch: Copies metadata (no file access needed)
    - MeiliSearch -> Qdrant: Re-chunks and embeds files (requires file access)

    Files that don't exist on disk are skipped with a warning.

    Args:
        corpus: Corpus name
        dry_run: If True, show what would be backfilled without making changes
        verbose: If True, show detailed progress
        output_json: If True, output JSON format
    """
    interaction_logger.start(
        "corpus", "parity",
        corpus=corpus,
        dry_run=dry_run,
    )

    try:
        if not output_json:
            print_info(f"Checking parity for corpus '{corpus}'...")

        # Initialize clients
        qdrant = create_qdrant_client()
        meili = get_meili_client()

        # Verify corpus exists in both systems
        try:
            qdrant.get_collection(corpus)
        except Exception:
            raise ResourceNotFoundError(
                f"Qdrant collection '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        if not meili.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        if not meili.index_exists(corpus):
            raise ResourceNotFoundError(
                f"MeiliSearch index '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        # Get corpus metadata
        corpus_type = get_collection_type(qdrant, corpus)
        metadata = get_collection_metadata(qdrant, corpus)
        configured_models = metadata.get('model', 'stella')
        model_list = [m.strip() for m in configured_models.split(',')]

        if not corpus_type:
            corpus_type = 'pdf'
            logger.warning(f"Collection type not set, defaulting to {corpus_type}")

        if not output_json:
            print_info(f"Corpus type: {corpus_type}, Models: {configured_models}")

        # Get file paths from both systems
        sync_manager = MetadataBasedSync(qdrant)
        qdrant_file_paths = sync_manager._get_indexed_file_paths_set(corpus)
        meili_file_paths = meili.get_all_file_paths(corpus)

        # Calculate set operations
        in_both = qdrant_file_paths & meili_file_paths
        missing_from_meili = qdrant_file_paths - meili_file_paths
        missing_from_qdrant = meili_file_paths - qdrant_file_paths

        # Report status
        if not output_json:
            console.print(f"\n[bold]Index Status:[/bold]")
            console.print(f"  Files in both systems:     {len(in_both)}")
            console.print(f"  Files in Qdrant only:      {len(missing_from_meili)}")
            console.print(f"  Files in MeiliSearch only: {len(missing_from_qdrant)}")

        # Check which files missing from Qdrant exist on disk
        qdrant_backfill_paths = []
        qdrant_skip_paths = []
        for path in missing_from_qdrant:
            if Path(path).exists():
                qdrant_backfill_paths.append(path)
            else:
                qdrant_skip_paths.append(path)

        meili_backfill_paths = list(missing_from_meili)

        # Dry-run mode: report and exit
        if dry_run:
            if not output_json:
                console.print(f"\n[bold yellow]DRY RUN - No changes will be made[/bold yellow]")

                if meili_backfill_paths:
                    console.print(f"\nWould backfill to MeiliSearch: {len(meili_backfill_paths)} files")
                    if verbose:
                        for p in meili_backfill_paths[:10]:
                            console.print(f"  {Path(p).name}")
                        if len(meili_backfill_paths) > 10:
                            console.print(f"  ... and {len(meili_backfill_paths) - 10} more")

                if qdrant_backfill_paths:
                    console.print(f"\nWould backfill to Qdrant: {len(qdrant_backfill_paths)} files")
                    if verbose:
                        for p in qdrant_backfill_paths[:10]:
                            console.print(f"  {Path(p).name}")
                        if len(qdrant_backfill_paths) > 10:
                            console.print(f"  ... and {len(qdrant_backfill_paths) - 10} more")

                if qdrant_skip_paths:
                    console.print(f"\n[yellow]Would skip (file not found): {len(qdrant_skip_paths)} files[/yellow]")
                    if verbose:
                        for p in qdrant_skip_paths[:10]:
                            console.print(f"  [yellow]{Path(p).name}[/yellow]")
                        if len(qdrant_skip_paths) > 10:
                            console.print(f"  ... and {len(qdrant_skip_paths) - 10} more")

                if not meili_backfill_paths and not qdrant_backfill_paths:
                    console.print(f"\n[green]✓ Indexes are already in parity[/green]")
            else:
                data = {
                    "corpus": corpus,
                    "dry_run": True,
                    "files_in_both": len(in_both),
                    "would_backfill_to_meili": len(meili_backfill_paths),
                    "would_backfill_to_qdrant": len(qdrant_backfill_paths),
                    "would_skip_not_found": len(qdrant_skip_paths),
                    "skipped_files": qdrant_skip_paths,
                }
                print_json("success", "Dry run complete", data=data)

            interaction_logger.finish(result_count=0)
            return

        # Nothing to do
        if not meili_backfill_paths and not qdrant_backfill_paths:
            if output_json:
                data = {
                    "corpus": corpus,
                    "files_in_both": len(in_both),
                    "meili_backfilled": 0,
                    "qdrant_backfilled": 0,
                }
                print_json("success", "Indexes are already in parity", data=data)
            else:
                console.print(f"\n[green]✓ Indexes are already in parity[/green]")
            interaction_logger.finish(result_count=0)
            return

        # Backfill MeiliSearch from Qdrant
        meili_backfilled = 0
        meili_backfill_chunks = 0
        meili_backfill_failed = 0

        if meili_backfill_paths:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(meili_backfill_paths)} files to MeiliSearch...[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling...", total=len(meili_backfill_paths))
                meili_backfilled, meili_backfill_chunks, meili_backfill_failed = _backfill_qdrant_to_meili(
                    qdrant, meili, corpus, meili_backfill_paths,
                    verbose, output_json, progress, backfill_task
                )

        # Backfill Qdrant from MeiliSearch
        qdrant_backfilled = 0
        qdrant_backfill_chunks = 0
        qdrant_backfill_failed = 0

        if qdrant_backfill_paths:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(qdrant_backfill_paths)} files to Qdrant (requires embedding)...[/blue]")

            # Initialize embedding client
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling Qdrant...", total=len(qdrant_backfill_paths))
                qdrant_backfilled, qdrant_backfill_chunks, qdrant_backfill_failed, _ = _backfill_meili_to_qdrant(
                    qdrant, embedding_client, corpus, corpus_type, model_list, model_config,
                    qdrant_backfill_paths, verbose, output_json, progress, backfill_task
                )

        # Output results
        data = {
            "corpus": corpus,
            "files_in_both": len(in_both),
            "meili_backfilled": meili_backfilled,
            "meili_backfill_chunks": meili_backfill_chunks,
            "meili_backfill_failed": meili_backfill_failed,
            "qdrant_backfilled": qdrant_backfilled,
            "qdrant_backfill_chunks": qdrant_backfill_chunks,
            "qdrant_backfill_failed": qdrant_backfill_failed,
            "qdrant_skipped_not_found": len(qdrant_skip_paths),
            "skipped_files": qdrant_skip_paths,
        }

        if output_json:
            print_json("success", f"Parity restored for corpus '{corpus}'", data=data)
        else:
            console.print(f"\n[green]✅ Parity restored for corpus '{corpus}'[/green]")

            if meili_backfilled > 0 or meili_backfill_failed > 0:
                status = f"{meili_backfilled} files ({meili_backfill_chunks} chunks)"
                if meili_backfill_failed > 0:
                    status += f" [yellow]({meili_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to MeiliSearch: {status}")

            if qdrant_backfilled > 0 or qdrant_backfill_failed > 0:
                status = f"{qdrant_backfilled} files ({qdrant_backfill_chunks} chunks)"
                if qdrant_backfill_failed > 0:
                    status += f" [yellow]({qdrant_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to Qdrant:      {status}")

            if qdrant_skip_paths:
                console.print(f"   [yellow]Skipped (not found):      {len(qdrant_skip_paths)} files[/yellow]")

        interaction_logger.finish(
            result_count=meili_backfilled + qdrant_backfilled,
        )

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to check parity: {e}", output_json)
        sys.exit(1)
