"""Main CLI entry point for Arcaneum (RDR-001 with RDR-006 enhancements)."""

import sys
import click
import os
from arcaneum import __version__
from arcaneum.cli.errors import (
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_INVALID_ARGS,
    EXIT_NOT_FOUND,
    ArcaneumError,
    InvalidArgumentError,
    ResourceNotFoundError,
    HelpfulGroup,
)

# Version check (RDR-006: Best practice from Beads)
MIN_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON:
    print(f"[ERROR] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
    sys.exit(1)

# SSL configuration (must happen BEFORE any embedding library imports)
# Check for ARC_SSL_VERIFY=false to disable SSL certificate verification
# This is needed for corporate proxies with self-signed certificates
from arcaneum.ssl_config import configure_ssl_from_env
configure_ssl_from_env()


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """Arcaneum: Semantic and full-text search tools for Qdrant and MeiliSearch"""
    # Run migration from legacy ~/.arcaneum/ to XDG-compliant structure if needed
    # Only show verbose output if user passed --verbose or similar flags
    verbose = ctx.parent and ctx.parent.params.get('verbose', False) if ctx.parent else False

    # Auto-migrate silently on first access (only logs errors)
    from arcaneum.migrations import run_migration_if_needed
    run_migration_if_needed(verbose=verbose)


# Collection management commands (RDR-003)
@cli.group(cls=HelpfulGroup, usage_examples=[
    'arc collection list',
    'arc collection create MyCollection --type code',
    'arc collection info MyCollection',
    'arc collection verify MyCollection',
    'arc collection delete MyCollection --confirm',
])
def collection():
    """Manage Qdrant collections"""
    pass


@collection.command('create')
@click.argument('name')
@click.option('--model', default=None, help='Embedding model (stella, modernbert, bge, jina-code). If not specified, inferred from --type.')
@click.option('--type', 'collection_type', type=click.Choice(['pdf', 'code', 'markdown']), help='Collection type (pdf, code, or markdown). Model will be inferred from type if not specified.')
@click.option('--hnsw-m', type=int, default=16, help='HNSW index parameter m')
@click.option('--hnsw-ef', type=int, default=100, help='HNSW index parameter ef_construct')
@click.option('--on-disk', is_flag=True, help='Store vectors on disk')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_collection(name, model, collection_type, hnsw_m, hnsw_ef, on_disk, output_json):
    """Create a new collection"""
    from arcaneum.cli.collections import create_collection_command
    create_collection_command(name, model, hnsw_m, hnsw_ef, on_disk, output_json, collection_type)


@collection.command('list')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_collections(verbose, output_json):
    """List all collections"""
    from arcaneum.cli.collections import list_collections_command
    list_collections_command(verbose, output_json)


@collection.command('info')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def collection_info(name, output_json):
    """Show collection details"""
    from arcaneum.cli.collections import info_collection_command
    info_collection_command(name, output_json)


@collection.command('delete')
@click.argument('name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def delete_collection(name, confirm, output_json):
    """Delete a collection"""
    from arcaneum.cli.collections import delete_collection_command
    delete_collection_command(name, confirm, output_json)


@collection.command('items')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def collection_items(name, output_json):
    """List all indexed files/repos in collection"""
    from arcaneum.cli.collections import items_collection_command
    items_collection_command(name, output_json)


@collection.command('verify')
@click.argument('name')
@click.option('--project', help='Verify specific project identifier only (code collections)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed file-level results')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def collection_verify(name, project, verbose, output_json):
    """Verify collection integrity (fsck-like check).

    Scans the collection to detect items with incomplete chunk sets.
    For code collections, verifies all files in each repo have complete chunks.
    For PDF/markdown, verifies all file chunks are present.

    Examples:
      arc collection verify MyCode
      arc collection verify MyCode --project myrepo#main
      arc collection verify MyPDFs --verbose
    """
    from arcaneum.cli.collections import verify_collection_command
    verify_collection_command(name, project, verbose, output_json)


@collection.command('export')
@click.argument('name')
@click.option('-o', '--output', required=True, type=click.Path(),
              help='Output file path (.arcexp or .jsonl)')
@click.option('--format', 'fmt', type=click.Choice(['binary', 'jsonl']),
              default='binary', help='Export format (default: binary)')
@click.option('--include', 'includes', multiple=True,
              help='Include files matching glob pattern (file_path)')
@click.option('--exclude', 'excludes', multiple=True,
              help='Exclude files matching glob pattern (file_path)')
@click.option('--repo', 'repos', multiple=True,
              help='Filter by repo name or repo#branch (code collections)')
@click.option('--detach', is_flag=True,
              help='Strip root prefix, store relative paths (shareable)')
@click.option('--json', 'output_json', is_flag=True, help='Output stats as JSON')
def collection_export(name, output, fmt, includes, excludes, repos, detach, output_json):
    """Export collection to portable format.

    Default format is compressed binary (.arcexp) for efficiency.
    Use --format jsonl for human-readable debug output.

    Filter options (all filters combined with AND):

    \b
      --include   Include files matching glob pattern on file_path (multiple = OR)
      --exclude   Exclude files matching glob pattern on file_path (multiple = AND)
      --repo      Filter by repo name (all branches) or repo#branch (code only)

    Path options:

    \b
      --detach    Strip common root prefix from paths, storing relative paths.
                  Use --attach on import to prepend new root. Enables sharing
                  collections without exposing your directory structure.

    Examples:

    \b
      arc collection export MyPDFs -o backup.arcexp
      arc collection export MyPDFs -o reports.arcexp --include "*/reports/*.pdf"
      arc collection export Code -o arcaneum.arcexp --repo arcaneum#main
      arc collection export Code -o subset.arcexp --include "~/repos/*" --repo arcaneum#main
      arc collection export MyCode -o shareable.arcexp --detach
    """
    from arcaneum.cli.collections import export_collection_command
    export_collection_command(name, output, fmt, includes, excludes, repos, detach, output_json)


@collection.command('import')
@click.argument('file', type=click.Path(exists=True))
@click.option('--into', 'target_name', help='Target collection name')
@click.option('--attach', 'attach_root',
              help='Attach root path to relative paths (for detached exports)')
@click.option('--remap', 'remaps', multiple=True,
              help='Path substitution: old:new prefix mapping (for non-detached exports)')
@click.option('--json', 'output_json', is_flag=True, help='Output stats as JSON')
def collection_import(file, target_name, attach_root, remaps, output_json):
    """Import collection from export file.

    Automatically detects format from file content (binary .arcexp or JSONL).

    Path handling options:

    \b
      --attach     Prepend root path to relative paths. Use with detached exports.
                   Symmetric with --detach on export.
      --remap      Explicit path substitution (old:new format). Use with non-detached
                   exports to update absolute paths for new machine.

    Examples:

    \b
      arc collection import backup.arcexp
      arc collection import backup.arcexp --into MyPDFs-restored
      arc collection import shareable.arcexp --attach /home/bob/projects
      arc collection import backup.arcexp --remap /Users/alice/docs:/home/bob/docs
    """
    from arcaneum.cli.collections import import_collection_command
    import_collection_command(file, target_name, attach_root, remaps, output_json)


# Models commands
@cli.group(cls=HelpfulGroup, usage_examples=[
    'arc models list',
    'arc models list --json',
])
def models():
    """Manage embedding models"""
    pass


@models.command('list')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_models(output_json):
    """List available models"""
    from arcaneum.cli.models import list_models_command
    list_models_command(output_json)


# Indexing commands (RDR-004, RDR-005, RDR-010)
@cli.group(cls=HelpfulGroup, usage_examples=[
    'arc index pdf /path/to/pdfs --collection MyPDFs',
    'arc index code /path/to/repo --collection MyCode',
    'arc index markdown /path/to/docs --collection MyDocs',
    'arc index text pdf /path/to/pdfs --index MyIndex',
])
def index():
    """Index content into collections"""
    pass


@index.command('pdf')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default=None, help='(Deprecated: model is now set at collection creation time) Embedding model to use')
@click.option('--embedding-batch-size', type=int, default=None, help='Batch size for embedding generation. Auto-tuned for GPU memory if not specified. Larger batches (300-500) improve throughput 10-20%.')
@click.option('--no-ocr', is_flag=True, help='Disable OCR (enabled by default for scanned PDFs)')
@click.option('--ocr-language', default='eng', help='OCR language code')
@click.option('--ocr-workers', type=int, default=None, help='Parallel OCR workers for page processing (default: cpu_count, effective for scanned PDFs only)')
@click.option('--normalize-only', is_flag=True, help='Skip markdown conversion, only normalize whitespace (RDR-016: max 47%% token savings)')
@click.option('--preserve-images', is_flag=True, help='Extract images for future multimodal search (RDR-016: slower processing)')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal). Use low for background indexing.')
@click.option('--not-nice', is_flag=True, help='Disable process priority reduction for worker processes (use normal priority)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only, 2-3x slower)')
@click.option('--offline', is_flag=True, help='Offline mode (use cached models only, no network)')
@click.option('--randomize', is_flag=True, help='Randomize file processing order (useful for parallel indexing)')
@click.option('--verify', is_flag=True, help='Verify collection integrity after indexing (fsck-like check)')
@click.option('--no-streaming', is_flag=True, help='Disable streaming mode (accumulate all embeddings before upload, uses more memory)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_pdf(path, from_file, collection, model, embedding_batch_size, no_ocr, ocr_language, ocr_workers, normalize_only, preserve_images, process_priority, not_nice, force, no_gpu, offline, randomize, verify, no_streaming, verbose, debug, output_json):
    """Index PDF files"""
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    from arcaneum.cli.index_pdfs import index_pdfs_command
    streaming = not no_streaming  # Default is streaming=True (--no-streaming disables it)
    index_pdfs_command(path, from_file, collection, model, embedding_batch_size, no_ocr, ocr_language, ocr_workers, normalize_only, preserve_images, process_priority, not_nice, force, no_gpu, offline, randomize, verify, streaming, verbose, debug, output_json)


@index.command('code')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default=None, help='(Deprecated: model is now set at collection creation time) Embedding model to use')
@click.option('--embedding-batch-size', type=int, default=None, help='Batch size for embedding generation. Auto-tuned for GPU memory if not specified. Larger batches (300-500) improve throughput 10-20%.')
@click.option('--chunk-size', type=int, help='Target chunk size in tokens (default: 400)')
@click.option('--chunk-overlap', type=int, help='Overlap between chunks in tokens (default: 20)')
@click.option('--depth', type=int, help='Git discovery depth')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal). Use low for background indexing.')
@click.option('--not-nice', is_flag=True, help='Disable process priority reduction for worker processes (use normal priority)')
@click.option('--force', is_flag=True, help='Force reindex all projects')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only, 2-3x slower)')
@click.option('--verify', is_flag=True, help='Verify and repair incomplete items after indexing (fsck-like check)')
@click.option('--no-streaming', is_flag=True, help='Disable streaming mode (accumulate all embeddings before upload, uses more memory)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--profile', is_flag=True, help='Show pipeline performance profiling (stage breakdown, throughput)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_code(path, from_file, collection, model, embedding_batch_size, chunk_size, chunk_overlap, depth, process_priority, not_nice, force, no_gpu, verify, no_streaming, verbose, debug, profile, output_json):
    """Index source code"""
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    from arcaneum.cli.index_source import index_source_command
    streaming = not no_streaming  # Default is streaming=True (--no-streaming disables it)
    index_source_command(path, from_file, collection, model, embedding_batch_size, chunk_size, chunk_overlap, depth, process_priority, not_nice, force, no_gpu, verify, streaming, verbose, debug, profile, output_json)


@index.command('markdown')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default=None, help='(Deprecated: model is now set at collection creation time) Embedding model to use')
@click.option('--embedding-batch-size', type=int, default=None, help='Batch size for embedding generation. Auto-tuned for GPU memory if not specified. Larger batches (300-500) improve throughput 10-20%.')
@click.option('--chunk-size', type=int, help='Target chunk size in tokens')
@click.option('--chunk-overlap', type=int, help='Overlap between chunks in tokens')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories recursively')
@click.option('--exclude', multiple=True, help='Patterns to exclude (e.g., node_modules, .obsidian)')
@click.option('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal). Use low for background indexing.')
@click.option('--not-nice', is_flag=True, help='Disable process priority reduction for worker processes (use normal priority)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only, 2-3x slower)')
@click.option('--offline', is_flag=True, help='Offline mode (use cached models only, no network)')
@click.option('--randomize', is_flag=True, help='Randomize file processing order (useful for parallel indexing)')
@click.option('--verify', is_flag=True, help='Verify collection integrity after indexing (fsck-like check)')
@click.option('--no-streaming', is_flag=True, help='Disable streaming mode (accumulate all embeddings before upload, uses more memory)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_markdown(path, from_file, collection, model, embedding_batch_size, chunk_size, chunk_overlap, recursive, exclude, qdrant_url, process_priority, not_nice, force, no_gpu, offline, randomize, verify, no_streaming, verbose, debug, output_json):
    """Index markdown files"""
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    from arcaneum.cli.index_markdown import index_markdown_command
    streaming = not no_streaming  # Default is streaming=True (--no-streaming disables it)
    index_markdown_command(path, from_file, collection, model, embedding_batch_size, chunk_size, chunk_overlap, recursive, exclude, qdrant_url, process_priority, not_nice, force, no_gpu, offline, randomize, verify, streaming, verbose, debug, output_json)


# Full-text indexing subgroup (RDR-010: arc index text ...)
@index.group('text', cls=HelpfulGroup, usage_examples=[
    'arc index text pdf /path/to/pdfs --index MyPDFs',
])
def index_text():
    """Index content to MeiliSearch for full-text search (RDR-010)"""
    pass


@index_text.command('code')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index name')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories recursively (simple mode only)')
@click.option('--depth', type=int, help='Git discovery depth (git-aware mode only)')
@click.option('--batch-size', type=int, default=1000, help='Documents per batch upload (default: 1000)')
@click.option('--workers', type=int, default=None, help='Parallel workers for AST extraction (default: auto=cpu/2, 0=sequential)')
@click.option('--force', is_flag=True, help='Force reindex all files/projects')
@click.option('--no-git', 'no_git', is_flag=True, help='Disable git-aware mode (use simple file-based indexing)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_text_code(path, from_file, index_name, recursive, depth, batch_size, workers, force, no_git, verbose, output_json):
    """Index source code to MeiliSearch for full-text search (RDR-011).

    By default uses git-aware mode: discovers git repositories, extracts
    function/class definitions with line ranges, supports multi-branch.

    Use --no-git for simple file-based indexing without git awareness.
    Use --workers to control parallel processing (default: auto=cpu/2, 0=sequential).

    Examples:
        arc index text code ./repos --index code-index
        arc index text code ./src --index code-index --no-git
        arc index text code ./repos --index code-index --depth 2 --force
        arc index text code ./repos --index code-index --workers 8
    """
    from arcaneum.cli.index_text import index_text_code_command
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    git_aware = not no_git
    index_text_code_command(path, from_file, index_name, recursive, batch_size, workers, force, verbose, output_json, depth, git_aware)


@index_text.command('markdown')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index name')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories recursively')
@click.option('--batch-size', type=int, default=1000, help='Documents per batch upload (default: 1000)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_text_markdown(path, from_file, index_name, recursive, batch_size, force, verbose, output_json):
    """Index markdown files to MeiliSearch for full-text search.

    Indexes markdown files for exact keyword and phrase search.
    Complements semantic search via Qdrant (arc index markdown).

    Examples:
        arc index text markdown ./docs --index docs-index
        arc index text markdown ./wiki --index wiki-index --force
    """
    from arcaneum.cli.index_text import index_text_markdown_command
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    index_text_markdown_command(path, from_file, index_name, recursive, batch_size, force, verbose, output_json)


@index_text.command('pdf')
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read file paths from list (one per line, or "-" for stdin)')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index name')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories recursively')
@click.option('--no-ocr', is_flag=True, help='Disable OCR (enabled by default for scanned PDFs)')
@click.option('--ocr-language', default='eng', help='OCR language code')
@click.option('--ocr-workers', type=int, default=None, help='Parallel OCR workers (default: cpu_count)')
@click.option('--normalize-only', is_flag=True, help='Skip markdown conversion, only normalize whitespace')
@click.option('--batch-size', type=int, default=1000, help='Documents per batch upload (default: 1000)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_text_pdf(path, from_file, index_name, recursive, no_ocr, ocr_language, ocr_workers, normalize_only, batch_size, force, process_priority, verbose, debug, output_json):
    """Index PDFs to MeiliSearch for full-text search.

    Extracts text from PDFs and indexes to MeiliSearch for exact phrase
    and keyword search. Complements semantic search via Qdrant (arc index pdf).

    Examples:
        arc index text pdf ./research-papers --index research-pdfs
        arc index text pdf ./docs --index docs --no-ocr --force
    """
    # Validate that exactly one of path or from_file is provided
    if not path and not from_file:
        click.echo("Error: Either PATH or --from-file must be provided", err=True)
        raise click.Abort()
    if path and from_file:
        click.echo("Error: Cannot use both PATH and --from-file", err=True)
        raise click.Abort()

    from arcaneum.cli.index_text import index_text_pdf_command
    ocr_enabled = not no_ocr
    index_text_pdf_command(path, from_file, index_name, recursive, ocr_enabled, ocr_language, ocr_workers, normalize_only, batch_size, force, process_priority, verbose, debug, output_json)


@cli.command('store')
@click.argument('file', type=click.Path())
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='stella', help='Embedding model (default: stella for documents)')
@click.option('--title', help='Document title')
@click.option('--category', help='Document category')
@click.option('--tags', help='Comma-separated tags')
@click.option('--metadata', help='Additional metadata as JSON')
@click.option('--chunk-size', type=int, help='Target chunk size in tokens')
@click.option('--chunk-overlap', type=int, help='Overlap between chunks in tokens')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def store(file, collection, model, title, category, tags, metadata, chunk_size, chunk_overlap, verbose, output_json):
    """Store agent-generated content for long-term memory.

    Designed for AI agents (Claude skills) to store research, analysis, and
    synthesized information. Content is persisted to disk for re-indexing
    and full-text retrieval, then indexed to Qdrant for semantic search.

    Storage location: ~/.arcaneum/agent-memory/{collection}/

    Examples:
      # Store from stdin (agent workflow)
      echo "# Research\\n\\nFindings..." | arc store - --collection knowledge

      # Store from file with metadata
      arc store analysis.md --collection research \\
          --title "Security Findings" --category security --tags "audit,critical"

    For indexing existing markdown directories, use 'arc index markdown' instead.
    """
    from arcaneum.cli.index_markdown import store_command
    store_command(file, collection, model, title, category, tags, metadata, chunk_size, chunk_overlap, verbose, output_json)


# Search commands (RDR-007, RDR-012)
@cli.group(cls=HelpfulGroup, usage_examples=[
    'arc search semantic "your query" --corpus CorpusName',
    'arc search semantic "your query" --corpus Corp1 --corpus Corp2',
    'arc search text "your query" --corpus CorpusName',
])
def search():
    """Search collections"""
    pass


@search.command('semantic')
@click.argument('query')
@click.option('--corpus', 'corpora', multiple=True, help='Corpus to search (can specify multiple)')
@click.option('--collection', 'legacy_collection', default=None, hidden=True, help='(Deprecated) Collection to search')
@click.option('--vector-name', help='Vector name to use (auto-detects if not specified)')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--offset', type=int, default=0, help='Number of results to skip (for pagination)')
@click.option('--score-threshold', type=float, help='Minimum score threshold')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_semantic(query, corpora, legacy_collection, vector_name, filter_arg, limit, offset, score_threshold, output_json, verbose):
    """Vector-based semantic search"""
    from arcaneum.cli.search import search_command, resolve_corpora
    resolved_corpora = resolve_corpora(corpora, legacy_collection, 'collection')
    search_command(query, resolved_corpora, vector_name, filter_arg, limit, offset, score_threshold, output_json, verbose)


@search.command('text')
@click.argument('query')
@click.option('--corpus', 'corpora', multiple=True, help='Corpus to search (can specify multiple)')
@click.option('--index', 'legacy_index', default=None, hidden=True, help='(Deprecated) MeiliSearch index to search')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--offset', type=int, default=0, help='Number of results to skip (for pagination)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_text(query, corpora, legacy_index, filter_arg, limit, offset, output_json, verbose):
    """Keyword-based full-text search"""
    from arcaneum.cli.fulltext import search_text_command, resolve_corpora
    resolved_corpora = resolve_corpora(corpora, legacy_index, 'index')
    search_text_command(query, resolved_corpora, filter_arg, limit, offset, output_json, verbose)


# Dual indexing commands (RDR-009)
@cli.group(cls=HelpfulGroup, usage_examples=[
    'arc corpus create MyCorpus --type code',
    'arc corpus sync MyCorpus /path/to/files',
    'arc corpus sync MyCorpus /path/one /path/two',
    'arc corpus items MyCorpus',
])
def corpus():
    """Manage dual-index corpora (Qdrant + MeiliSearch)"""
    pass


@corpus.command('list')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_corpora(verbose, output_json):
    """List all corpora with parity status."""
    from arcaneum.cli.corpus import list_corpora_command
    list_corpora_command(verbose, output_json)


@corpus.command('create')
@click.argument('name')
@click.option('--type', 'corpus_type', type=click.Choice(['pdf', 'code', 'markdown']), required=True, help='Corpus type')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_corpus(name, corpus_type, models, output_json):
    """Create both collection and index"""
    from arcaneum.cli.corpus import create_corpus_command
    create_corpus_command(name, corpus_type, models, output_json)


@corpus.command('delete')
@click.argument('name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def delete_corpus(name, confirm, output_json):
    """Delete both collection and index for a corpus."""
    from arcaneum.cli.corpus import delete_corpus_command
    delete_corpus_command(name, confirm, output_json)


@corpus.command('sync')
@click.argument('corpus')
@click.argument('paths', nargs=-1, type=click.Path(exists=True), required=False)
@click.option('--from-file', help='Read paths from file (one per line, or "-" for stdin)')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--file-types', help='File extensions to index (e.g., .py,.md)')
@click.option('--force', is_flag=True, help='Force reindex all files (bypass change detection)')
@click.option('--dry-run', is_flag=True, help='Show what would be synced without making changes')
@click.option('--verify', is_flag=True, help='Verify collection integrity after indexing')
@click.option('--text-workers', type=int, default=None,
              help='Parallel workers for code AST chunking (default: auto=cpu/2, 0=sequential)')
@click.option('--max-embedding-batch', type=int, default=None,
              help='Cap embedding batch size (default: auto from GPU memory, use 8-16 for OOM)')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only, slower but stable)')
@click.option('--cpu-workers', type=int, default=None,
              help='Batch parallelization workers for --no-gpu mode (default: 1, conservative to prevent system crashes)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress (files, chunks, indexing)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--git-update', is_flag=True,
              help='Skip repos with unchanged commit hash (git-aware fast path)')
@click.option('--git-version', is_flag=True,
              help='Keep multiple versions indexed (different commits coexist)')
@click.option('--skip-dir-prefix', multiple=True, default=('_',),
              help='Skip directories starting with PREFIX (default: _). Repeatable.')
@click.option('--no-skip-dir-prefix', is_flag=True,
              help='Disable all directory prefix skipping')
def sync_directory(corpus, paths, from_file, models, file_types, force, dry_run, verify, text_workers, max_embedding_batch, no_gpu, cpu_workers, verbose, output_json, git_update, git_version, skip_dir_prefix, no_skip_dir_prefix):
    """Index to both vector and full-text.

    Examples:
        arc corpus sync MyCorpus /path/to/files
        arc corpus sync MyCorpus /path/one /path/two /path/three
        arc corpus sync MyCorpus document.pdf
        arc corpus sync MyCorpus notes.md /path/to/dir
        arc corpus sync MyCorpus --from-file paths.txt
        find . -name "*.pdf" | arc corpus sync MyCorpus --from-file -

    Use --text-workers to parallelize AST chunking for code corpora.
    Use --no-gpu for CPU-only mode (avoids MPS instability with large models).
    """
    # Validate that at least one of paths or from_file is provided
    if not paths and not from_file:
        click.echo("Error: Either PATH(s) or --from-file must be provided", err=True)
        raise SystemExit(1)

    # Validate mutually exclusive git flags
    if git_update and git_version:
        click.echo("Error: --git-update and --git-version are mutually exclusive", err=True)
        raise SystemExit(1)

    # Resolve skip prefixes: --no-skip-dir-prefix disables all, otherwise use provided prefixes
    effective_prefixes = () if no_skip_dir_prefix else skip_dir_prefix

    from arcaneum.cli.sync import sync_directory_command
    sync_directory_command(corpus, paths, from_file, models, file_types, force, verify, text_workers, max_embedding_batch, no_gpu, cpu_workers, verbose, output_json, git_update, git_version, effective_prefixes, dry_run=dry_run)


@corpus.command('parity')
@click.argument('name', required=False)
@click.option('--dry-run', is_flag=True, help='Show what would be backfilled without making changes')
@click.option('--verify', is_flag=True, help='Verify chunk counts match between systems (detects partial uploads)')
@click.option('--repair-metadata', is_flag=True, help='Update MeiliSearch docs with missing git metadata from Qdrant')
@click.option('--text-workers', type=int, default=None,
              help='Parallel workers for fetching/chunking (default: auto=cpu/2, 0=sequential)')
@click.option('--timeout', type=int, default=120,
              help='Qdrant timeout in seconds for fetch operations (default: 120)')
@click.option('--create-missing', is_flag=True, help='Create missing MeiliSearch indexes for qdrant_only corpora')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt when processing all corpora')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress for each file')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def corpus_parity(name, dry_run, verify, repair_metadata, text_workers, timeout, create_missing, confirm, verbose, output_json):
    """Check and restore parity between Qdrant and MeiliSearch.

    When NAME is provided, operates on a single corpus. When NAME is omitted,
    discovers all corpora and shows a summary before proceeding.

    Compares indexed files in both systems and backfills missing entries:
    - Qdrant -> MeiliSearch: Copies metadata (no file access needed)
    - MeiliSearch -> Qdrant: Re-chunks and embeds files (requires file access)

    Files that don't exist on disk are skipped with a warning.

    Use --verify to check that chunk counts match for files in both systems.
    This detects partial uploads from previous failed syncs.

    Use --repair-metadata to update existing MeiliSearch documents that are
    missing git metadata (git_project_identifier, etc.) by copying from Qdrant.

    Use --text-workers to control parallelism for fetching and chunking.

    Use --confirm to skip the confirmation prompt when processing all corpora.

    Use --create-missing to create MeiliSearch indexes for corpora that only
    exist in Qdrant. This promotes single-sided collections into full corpora.
    """
    from arcaneum.cli.sync import parity_command
    parity_command(name, dry_run, verify, repair_metadata, text_workers, timeout, create_missing, confirm, verbose, output_json)


@corpus.command('info')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def corpus_info(name, output_json):
    """Show combined corpus information (collection + index)."""
    from arcaneum.cli.corpus import corpus_info_command
    corpus_info_command(name, output_json)


@corpus.command('items')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def corpus_items(name, output_json):
    """List all indexed items with parity status.

    Shows items from both Qdrant collection and MeiliSearch index,
    with chunk counts from each system (Q and M columns).

    Examples:
        arc corpus items MyCorpus
        arc corpus items MyCorpus --json
    """
    from arcaneum.cli.corpus import corpus_items_command
    corpus_items_command(name, output_json)


@corpus.command('verify')
@click.argument('name')
@click.option('--project', help='Filter by project identifier (code corpora only)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed file-level results')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def corpus_verify(name, project, verbose, output_json):
    """Verify corpus health across both Qdrant and MeiliSearch.

    Performs fsck-like integrity checks on both systems and reports
    collection/index health and parity status.

    Examples:
        arc corpus verify MyCorpus
        arc corpus verify MyCorpus --verbose
        arc corpus verify MyCorpus --json
    """
    from arcaneum.cli.corpus import corpus_verify_command
    corpus_verify_command(name, project, verbose, output_json)


# Diagnostics command (RDR-006 enhancement)
@cli.command('doctor')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed diagnostic information')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def doctor(verbose, output_json):
    """Verify Arcaneum setup and prerequisites (from RDR-006 enhancement)"""
    from arcaneum.cli.doctor import doctor_command
    return doctor_command(verbose, output_json)


# Configuration and cache management commands
from arcaneum.cli.config import config_group
cli.add_command(config_group, name='config')

# Container management commands
from arcaneum.cli.docker import container_group
cli.add_command(container_group, name='container')

# MeiliSearch index management commands (RDR-008, RDR-010)
# Named 'indexes' to mirror 'collection' for Qdrant
from arcaneum.cli.fulltext import fulltext as indexes_group
cli.add_command(indexes_group, name='indexes')


def main():
    """Main CLI entry point with structured error handling (RDR-006)."""
    try:
        cli()
        return EXIT_SUCCESS
    except click.ClickException as e:
        # Click handles its own exceptions (usage errors, etc.)
        e.show()
        return EXIT_INVALID_ARGS
    except (InvalidArgumentError, click.BadParameter) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except ResourceNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user", file=sys.stderr)
        return EXIT_ERROR
    except ArcaneumError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return e.exit_code
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
