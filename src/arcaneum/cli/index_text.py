"""CLI commands for full-text indexing to MeiliSearch (RDR-010).

Provides commands for indexing PDFs, code, and markdown to MeiliSearch
for exact phrase and keyword search.
"""

import hashlib
import json
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .interaction_logger import interaction_logger
from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import set_process_priority
from .errors import InvalidArgumentError, ResourceNotFoundError
from .output import print_error, print_success

console = Console()
logger = logging.getLogger(__name__)


def get_meili_client():
    """Get MeiliSearch client from environment or auto-generated key."""
    from ..paths import get_meilisearch_api_key
    from ..fulltext.client import FullTextClient

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def index_text_pdf_command(
    path: str,
    from_file: str,
    index_name: str,
    recursive: bool,
    ocr_enabled: bool,
    ocr_language: str,
    ocr_workers: int,
    normalize_only: bool,
    batch_size: int,
    force: bool,
    process_priority: str,
    verbose: bool,
    debug: bool,
    output_json: bool
):
    """Index PDF files to MeiliSearch for full-text search.

    Args:
        path: Directory containing PDF files (or None if using from_file)
        from_file: Path to file containing list of PDF paths, or "-" for stdin
        index_name: Target MeiliSearch index name
        recursive: Search subdirectories recursively
        ocr_enabled: Enable OCR for scanned PDFs
        ocr_language: OCR language code
        ocr_workers: Number of parallel OCR workers
        normalize_only: Skip markdown conversion
        batch_size: Documents per batch upload
        force: Force reindex all files
        process_priority: Process scheduling priority
        verbose: Verbose output
        debug: Debug mode
        output_json: Output JSON format
    """
    # Set process priority early
    set_process_priority(process_priority)

    # Setup logging
    if debug:
        setup_logging_debug()
    elif verbose:
        setup_logging_verbose()
    else:
        setup_logging_default()

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "index", "text-pdf",
        index=index_name,
        path=path,
        from_file=from_file,
        force=force,
    )

    try:
        # Handle file list if provided
        file_list = None
        pdf_dir = None

        if from_file:
            from .utils import read_file_list
            file_list = read_file_list(from_file, allowed_extensions={'.pdf'})
            if not file_list:
                raise ValueError("No valid PDF files found in the provided list")
            pdf_dir = file_list[0].parent
        else:
            pdf_dir = Path(path)
            if not pdf_dir.exists():
                raise ValueError(f"Path does not exist: {path}")

        # Initialize MeiliSearch client
        meili_client = get_meili_client()

        # Verify server is available
        if not meili_client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Import settings and indexer
        from ..fulltext.indexes import PDF_DOCS_SETTINGS
        from ..indexing.fulltext.pdf_indexer import PDFFullTextIndexer

        # Ensure index exists with correct settings
        if not meili_client.index_exists(index_name):
            if not output_json:
                console.print(f"Creating index '{index_name}'...")
            meili_client.create_index(
                name=index_name,
                primary_key='id',
                settings=PDF_DOCS_SETTINGS
            )
            if not output_json:
                console.print(f"[green]Created index '{index_name}' with PDF settings[/green]")
        else:
            # Verify index has correct filterable attributes for change detection
            current_settings = meili_client.get_index_settings(index_name)
            required_attrs = {'file_path', 'file_hash'}
            current_filterable = set(current_settings.get('filterableAttributes', []))

            if not required_attrs.issubset(current_filterable):
                # Update settings to add missing filterable attributes
                if not output_json:
                    console.print(
                        f"[yellow]Updating index '{index_name}' with required "
                        f"filterable attributes for change detection...[/yellow]"
                    )
                meili_client.update_index_settings(index_name, PDF_DOCS_SETTINGS)
                if not output_json:
                    console.print(f"[green]Updated index settings[/green]")
            else:
                if not output_json:
                    console.print(f"[blue]Using existing index '{index_name}'[/blue]")

        # Initialize indexer
        indexer = PDFFullTextIndexer(
            meili_client=meili_client,
            index_name=index_name,
            ocr_enabled=ocr_enabled,
            ocr_language=ocr_language,
            ocr_workers=ocr_workers,
            batch_size=batch_size,
            markdown_conversion=not normalize_only,
        )

        # Show configuration
        if not output_json:
            console.print(f"\n[bold blue]PDF Full-Text Indexing Configuration[/bold blue]")
            console.print(f"  Index: {index_name}")
            console.print(f"  Path: {pdf_dir}")
            console.print(f"  Recursive: {recursive}")

            if normalize_only:
                console.print(f"  Extraction: Normalization-only (max token savings)")
            else:
                console.print(f"  Extraction: Markdown conversion (semantic structure)")

            if ocr_enabled:
                from multiprocessing import cpu_count
                workers_display = ocr_workers if ocr_workers else cpu_count()
                console.print(f"  OCR: tesseract ({ocr_language}, {workers_display} workers)")
            else:
                console.print(f"  OCR: disabled")

            console.print(f"  Batch size: {batch_size}")
            if force:
                console.print(f"  [yellow]Force: Reindexing all files[/yellow]")
            console.print()

        # Index PDFs
        stats = indexer.index_directory(
            directory=pdf_dir,
            recursive=recursive,
            force_reindex=force,
            verbose=verbose,
            file_list=file_list
        )

        # Output results
        if output_json:
            result = {
                "success": True,
                "index": index_name,
                "stats": {
                    "total_pdfs": stats['total_pdfs'],
                    "indexed_pdfs": stats['indexed_pdfs'],
                    "skipped_pdfs": stats['skipped_pdfs'],
                    "failed_pdfs": stats['failed_pdfs'],
                    "total_pages": stats['total_pages'],
                }
            }
            if stats.get('errors'):
                result["errors"] = stats['errors']
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            if not verbose:
                # Minimal output
                console.print(
                    f"\n[green]Indexed {stats['indexed_pdfs']} PDF(s): "
                    f"{stats['total_pages']} pages[/green]"
                )
                if stats['skipped_pdfs'] > 0:
                    console.print(f"[dim]Skipped {stats['skipped_pdfs']} unchanged PDF(s)[/dim]")
                if stats['failed_pdfs'] > 0:
                    console.print(f"[yellow]Failed: {stats['failed_pdfs']} PDF(s)[/yellow]")
            else:
                # Verbose: detailed table
                console.print("\n[bold green]Indexing Complete[/bold green]")

                table = Table(title="Full-Text Indexing Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("Total PDFs", str(stats['total_pdfs']))
                table.add_row("Indexed", str(stats['indexed_pdfs']))
                table.add_row("Skipped (unchanged)", str(stats['skipped_pdfs']))
                table.add_row("Failed", str(stats['failed_pdfs']))
                table.add_row("Total Pages", str(stats['total_pages']))

                console.print(table)

                if stats.get('errors'):
                    console.print("\n[bold red]Errors:[/bold red]")
                    for error in stats['errors'][:5]:
                        console.print(f"  - {error['file']}: {error['error']}")
                    if len(stats['errors']) > 5:
                        console.print(f"  ... and {len(stats['errors']) - 5} more")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=stats.get('indexed_pdfs', 0),
            pages=stats.get('total_pages', 0),
            errors=stats.get('failed_pdfs', 0),
        )

    except KeyboardInterrupt:
        interaction_logger.finish(error="interrupted by user")
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    except (InvalidArgumentError, ResourceNotFoundError):
        raise

    except Exception as e:
        interaction_logger.finish(error=str(e))
        if output_json:
            result = {
                "success": False,
                "error": str(e)
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            if debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file for change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_already_indexed(meili_client, index_name: str, file_path: Path, file_hash: str) -> bool:
    """Check if file is already indexed with same hash."""
    try:
        filter_expr = f'file_path = "{file_path}" AND file_hash = "{file_hash}"'
        results = meili_client.search(index_name, '', filter=filter_expr, limit=1)
        return len(results.get('hits', [])) > 0
    except Exception:
        return False


def index_text_code_command(
    path: str,
    from_file: str,
    index_name: str,
    recursive: bool,
    batch_size: int,
    force: bool,
    verbose: bool,
    output_json: bool
):
    """Index source code files to MeiliSearch for full-text search.

    Args:
        path: Directory containing source files (or None if using from_file)
        from_file: Path to file containing list of paths, or "-" for stdin
        index_name: Target MeiliSearch index name
        recursive: Search subdirectories recursively
        batch_size: Documents per batch upload
        force: Force reindex all files
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Setup logging
    if verbose:
        setup_logging_verbose()
    else:
        setup_logging_default()

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "index", "text-code",
        index=index_name,
        path=path,
        from_file=from_file,
        force=force,
    )

    try:
        # Handle file list if provided
        file_list = None
        source_dir = None

        if from_file:
            from .utils import read_file_list
            file_list = read_file_list(from_file, allowed_extensions=None)
            if not file_list:
                raise ValueError("No valid files found in the provided list")
            source_dir = file_list[0].parent
        else:
            source_dir = Path(path)
            if not source_dir.exists():
                raise ValueError(f"Path does not exist: {path}")

        # Initialize MeiliSearch client
        meili_client = get_meili_client()

        # Verify server is available
        if not meili_client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Import settings
        from ..fulltext.indexes import SOURCE_CODE_SETTINGS

        # Ensure index exists with correct settings
        if not meili_client.index_exists(index_name):
            if not output_json:
                console.print(f"Creating index '{index_name}'...")
            meili_client.create_index(
                name=index_name,
                primary_key='id',
                settings=SOURCE_CODE_SETTINGS
            )
            if not output_json:
                console.print(f"[green]Created index '{index_name}' with source-code settings[/green]")
        else:
            if not output_json:
                console.print(f"[blue]Using existing index '{index_name}'[/blue]")

        # Discover source files
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.c', '.cpp',
                          '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash'}

        if file_list:
            source_files = [f for f in file_list if f.suffix.lower() in code_extensions]
        else:
            source_files = []
            pattern = '**/*' if recursive else '*'
            for ext in code_extensions:
                source_files.extend(source_dir.glob(f'{pattern}{ext}'))

        stats = {
            'total_files': len(source_files),
            'indexed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'errors': [],
        }

        if not source_files:
            if output_json:
                print(json.dumps({"success": True, "stats": stats}))
            else:
                console.print("[yellow]No source code files found to index[/yellow]")
            interaction_logger.finish(result_count=0)
            return

        # Show configuration
        if not output_json:
            console.print(f"\n[bold blue]Source Code Full-Text Indexing[/bold blue]")
            console.print(f"  Index: {index_name}")
            console.print(f"  Path: {source_dir}")
            console.print(f"  Files found: {len(source_files)}")
            console.print(f"  Batch size: {batch_size}")
            if force:
                console.print(f"  [yellow]Force: Reindexing all files[/yellow]")
            console.print()

        # Index files with progress tracking
        documents = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Indexing source code to MeiliSearch",
                total=len(source_files)
            )

            for file_path in source_files:
                try:
                    file_hash = _compute_file_hash(file_path)

                    # Check if already indexed
                    if not force and _is_already_indexed(meili_client, index_name, file_path, file_hash):
                        stats['skipped_files'] += 1
                        if verbose:
                            progress.console.print(f"  [dim]Skipped:[/dim] {file_path.name}")
                        progress.update(task, advance=1)
                        continue

                    # Read file content
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                    except Exception as e:
                        raise ValueError(f"Failed to read file: {e}")

                    # Detect language from extension
                    ext_to_lang = {
                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                        '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
                        '.go': 'go', '.rs': 'rust', '.c': 'c', '.cpp': 'cpp',
                        '.h': 'c', '.hpp': 'cpp', '.cs': 'csharp', '.rb': 'ruby',
                        '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
                        '.scala': 'scala', '.sh': 'shell', '.bash': 'shell',
                    }
                    language = ext_to_lang.get(file_path.suffix.lower(), 'unknown')

                    # Generate unique ID
                    path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)[:200]
                    doc_id = f"{sanitized_name}_{path_hash}"

                    # Build document
                    doc = {
                        'id': doc_id,
                        'content': content,
                        'filename': file_path.name,
                        'file_path': str(file_path.absolute()),
                        'file_hash': file_hash,
                        'file_extension': file_path.suffix,
                        'language': language,
                        'document_type': 'source_code',
                    }

                    documents.append(doc)
                    stats['indexed_files'] += 1

                    if verbose:
                        progress.console.print(f"  [green]Indexed:[/green] {file_path.name}")

                    # Upload in batches
                    if len(documents) >= batch_size:
                        meili_client.add_documents(index_name, documents)
                        stats['total_documents'] += len(documents)
                        documents = []

                except Exception as e:
                    stats['failed_files'] += 1
                    stats['errors'].append({'file': str(file_path), 'error': str(e)})
                    if verbose:
                        progress.console.print(f"  [red]Failed:[/red] {file_path.name}: {e}")

                progress.update(task, advance=1)

        # Upload remaining documents
        if documents:
            meili_client.add_documents(index_name, documents)
            stats['total_documents'] += len(documents)

        # Output results
        if output_json:
            result = {"success": True, "index": index_name, "stats": stats}
            if stats['errors']:
                result["errors"] = stats['errors']
            print(json.dumps(result, indent=2))
        else:
            console.print(f"\n[green]Indexed {stats['indexed_files']} files ({stats['total_documents']} documents)[/green]")
            if stats['skipped_files'] > 0:
                console.print(f"[dim]Skipped {stats['skipped_files']} unchanged files[/dim]")
            if stats['failed_files'] > 0:
                console.print(f"[yellow]Failed: {stats['failed_files']} files[/yellow]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=stats.get('indexed_files', 0),
            documents=stats.get('total_documents', 0),
            errors=stats.get('failed_files', 0),
        )

    except KeyboardInterrupt:
        interaction_logger.finish(error="interrupted by user")
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    except (InvalidArgumentError, ResourceNotFoundError):
        raise

    except Exception as e:
        interaction_logger.finish(error=str(e))
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
            sys.exit(1)
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)


def index_text_markdown_command(
    path: str,
    from_file: str,
    index_name: str,
    recursive: bool,
    batch_size: int,
    force: bool,
    verbose: bool,
    output_json: bool
):
    """Index markdown files to MeiliSearch for full-text search.

    Args:
        path: Directory containing markdown files (or None if using from_file)
        from_file: Path to file containing list of paths, or "-" for stdin
        index_name: Target MeiliSearch index name
        recursive: Search subdirectories recursively
        batch_size: Documents per batch upload
        force: Force reindex all files
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Setup logging
    if verbose:
        setup_logging_verbose()
    else:
        setup_logging_default()

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "index", "text-markdown",
        index=index_name,
        path=path,
        from_file=from_file,
        force=force,
    )

    try:
        # Handle file list if provided
        file_list = None
        markdown_dir = None

        if from_file:
            from .utils import read_file_list
            file_list = read_file_list(from_file, allowed_extensions={'.md', '.markdown', '.mdown', '.mkd'})
            if not file_list:
                raise ValueError("No valid markdown files found in the provided list")
            markdown_dir = file_list[0].parent
        else:
            markdown_dir = Path(path)
            if not markdown_dir.exists():
                raise ValueError(f"Path does not exist: {path}")

        # Initialize MeiliSearch client
        meili_client = get_meili_client()

        # Verify server is available
        if not meili_client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Import settings
        from ..fulltext.indexes import MARKDOWN_DOCS_SETTINGS

        # Ensure index exists with correct settings
        if not meili_client.index_exists(index_name):
            if not output_json:
                console.print(f"Creating index '{index_name}'...")
            meili_client.create_index(
                name=index_name,
                primary_key='id',
                settings=MARKDOWN_DOCS_SETTINGS
            )
            if not output_json:
                console.print(f"[green]Created index '{index_name}' with markdown-docs settings[/green]")
        else:
            if not output_json:
                console.print(f"[blue]Using existing index '{index_name}'[/blue]")

        # Discover markdown files
        md_extensions = {'.md', '.markdown', '.mdown', '.mkd', '.mkdn'}

        if file_list:
            md_files = [f for f in file_list if f.suffix.lower() in md_extensions]
        else:
            md_files = []
            pattern = '**/*' if recursive else '*'
            for ext in md_extensions:
                md_files.extend(markdown_dir.glob(f'{pattern}{ext}'))

        stats = {
            'total_files': len(md_files),
            'indexed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'errors': [],
        }

        if not md_files:
            if output_json:
                print(json.dumps({"success": True, "stats": stats}))
            else:
                console.print("[yellow]No markdown files found to index[/yellow]")
            interaction_logger.finish(result_count=0)
            return

        # Show configuration
        if not output_json:
            console.print(f"\n[bold blue]Markdown Full-Text Indexing[/bold blue]")
            console.print(f"  Index: {index_name}")
            console.print(f"  Path: {markdown_dir}")
            console.print(f"  Files found: {len(md_files)}")
            console.print(f"  Batch size: {batch_size}")
            if force:
                console.print(f"  [yellow]Force: Reindexing all files[/yellow]")
            console.print()

        # Index files with progress tracking
        documents = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Indexing markdown to MeiliSearch",
                total=len(md_files)
            )

            for file_path in md_files:
                try:
                    file_hash = _compute_file_hash(file_path)

                    # Check if already indexed
                    if not force and _is_already_indexed(meili_client, index_name, file_path, file_hash):
                        stats['skipped_files'] += 1
                        if verbose:
                            progress.console.print(f"  [dim]Skipped:[/dim] {file_path.name}")
                        progress.update(task, advance=1)
                        continue

                    # Read file content
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                    except Exception as e:
                        raise ValueError(f"Failed to read file: {e}")

                    # Extract title from first H1 or filename
                    title = file_path.stem
                    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()

                    # Extract headings for searchability
                    headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)

                    # Generate unique ID
                    path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)[:200]
                    doc_id = f"{sanitized_name}_{path_hash}"

                    # Build document
                    doc = {
                        'id': doc_id,
                        'content': content,
                        'title': title,
                        'filename': file_path.name,
                        'file_path': str(file_path.absolute()),
                        'file_hash': file_hash,
                        'headings': ' | '.join(headings) if headings else '',
                        'document_type': 'markdown',
                    }

                    documents.append(doc)
                    stats['indexed_files'] += 1

                    if verbose:
                        progress.console.print(f"  [green]Indexed:[/green] {file_path.name}")

                    # Upload in batches
                    if len(documents) >= batch_size:
                        meili_client.add_documents(index_name, documents)
                        stats['total_documents'] += len(documents)
                        documents = []

                except Exception as e:
                    stats['failed_files'] += 1
                    stats['errors'].append({'file': str(file_path), 'error': str(e)})
                    if verbose:
                        progress.console.print(f"  [red]Failed:[/red] {file_path.name}: {e}")

                progress.update(task, advance=1)

        # Upload remaining documents
        if documents:
            meili_client.add_documents(index_name, documents)
            stats['total_documents'] += len(documents)

        # Output results
        if output_json:
            result = {"success": True, "index": index_name, "stats": stats}
            if stats['errors']:
                result["errors"] = stats['errors']
            print(json.dumps(result, indent=2))
        else:
            console.print(f"\n[green]Indexed {stats['indexed_files']} files ({stats['total_documents']} documents)[/green]")
            if stats['skipped_files'] > 0:
                console.print(f"[dim]Skipped {stats['skipped_files']} unchanged files[/dim]")
            if stats['failed_files'] > 0:
                console.print(f"[yellow]Failed: {stats['failed_files']} files[/yellow]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=stats.get('indexed_files', 0),
            documents=stats.get('total_documents', 0),
            errors=stats.get('failed_files', 0),
        )

    except KeyboardInterrupt:
        interaction_logger.finish(error="interrupted by user")
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    except (InvalidArgumentError, ResourceNotFoundError):
        raise

    except Exception as e:
        interaction_logger.finish(error=str(e))
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
            sys.exit(1)
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
