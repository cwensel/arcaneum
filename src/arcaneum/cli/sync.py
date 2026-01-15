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
        pattern = f"**/*{ext}"
        found = list(directory.rglob(pattern.lstrip('*/')))
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
    from ..indexing.pdf.extractor import extract_pdf_text

    # Extract text from PDF
    text_result = extract_pdf_text(file_path)
    if not text_result or not text_result.get('text'):
        logger.warning(f"No text extracted from {file_path}")
        return []

    # Create chunker and chunk the text
    chunker = PDFChunker(model_config)
    base_metadata = {
        'file_path': str(file_path),
        'filename': file_path.name,
        'page_boundaries': text_result.get('page_boundaries', []),
    }

    chunks = chunker.chunk(text_result['text'], base_metadata)

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
            print_info(f"Found {len(files)} files to index")

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

        # Process files
        total_indexed = 0
        total_chunks = 0

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
                        logger.debug(f"No chunks from {file_path}")
                        progress.advance(task)
                        continue

                    # Build dual index documents
                    documents = []
                    file_hash = compute_file_hash(file_path)

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
                            file_path=str(file_path),
                            filename=file_path.name,
                            file_extension=file_path.suffix,
                            chunk_index=i,
                            chunk_count=len(chunks),
                            file_hash=file_hash,
                            file_size=file_path.stat().st_size,
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
                        qdrant_count, meili_count = dual_indexer.index_batch(documents)
                        total_chunks += len(documents)

                    total_indexed += 1

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    if not output_json:
                        console.print(f"[yellow]Warning: Failed to process {file_path.name}: {e}[/yellow]")

                progress.advance(task)

        # Output results
        data = {
            "corpus": corpus,
            "directory": str(dir_path),
            "files_indexed": total_indexed,
            "total_chunks": total_chunks,
            "models": model_list,
        }

        if output_json:
            print_json("success", f"Indexed {total_indexed} files ({total_chunks} chunks)", data=data)
        else:
            console.print(f"\n[green]✅ Indexed {total_indexed} files ({total_chunks} chunks) to corpus '{corpus}'[/green]")
            console.print(f"\n[dim]Search with:[/dim]")
            console.print(f"  arc search semantic \"your query\" --collection {corpus}")
            console.print(f"  arc search text \"your query\" --index {corpus}")

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
