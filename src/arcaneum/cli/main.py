"""Main CLI entry point for Arcaneum (RDR-001 with RDR-006 enhancements)."""

import sys
import click
from arcaneum import __version__
from arcaneum.cli.errors import (
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_INVALID_ARGS,
    EXIT_NOT_FOUND,
    ArcaneumError,
    InvalidArgumentError,
    ResourceNotFoundError,
)

# Version check (RDR-006: Best practice from Beads)
MIN_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON:
    print(f"[ERROR] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
    sys.exit(1)


@click.group()
@click.version_option(version=__version__)
def cli():
    """Arcaneum: Semantic and full-text search tools for Qdrant and MeiliSearch"""
    pass


# Collection management commands (RDR-003)
@cli.group()
def collection():
    """Manage Qdrant collections"""
    pass


@collection.command('create')
@click.argument('name')
@click.option('--model', required=True, help='Embedding model (stella, modernbert, bge, jina-code)')
@click.option('--type', 'collection_type', type=click.Choice(['pdf', 'code', 'markdown']), help='Collection type (pdf, code, or markdown)')
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


# Models commands
@cli.group()
def models():
    """Manage embedding models"""
    pass


@models.command('list')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_models(output_json):
    """List available models"""
    from arcaneum.cli.models import list_models_command
    list_models_command(output_json)


# Indexing commands (RDR-004, RDR-005)
@cli.group()
def index():
    """Index content into collections"""
    pass


@index.command('pdf')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='stella', help='Embedding model (default: stella for documents)')
@click.option('--file-workers', type=int, default=None, help='Number of PDF files to process in parallel (absolute, overrides multiplier)')
@click.option('--file-worker-mult', type=float, default=None, help='File worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-workers', type=int, default=None, help='Parallel workers for embedding generation (absolute, overrides multiplier)')
@click.option('--embedding-worker-mult', type=float, default=None, help='Embedding worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-batch-size', type=int, default=200, help='Batch size for embedding generation (default: 200)')
@click.option('--no-ocr', is_flag=True, help='Disable OCR (enabled by default for scanned PDFs)')
@click.option('--ocr-language', default='eng', help='OCR language code')
@click.option('--ocr-workers', type=int, default=None, help='Parallel OCR workers for page processing (default: cpu_count)')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal)')
@click.option('--max-perf', is_flag=True, help='Maximum performance preset (1.0 worker mult, batch 500, low priority)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--batch-across-files', is_flag=True, help='Batch uploads across files (faster but less atomic)')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only)')
@click.option('--offline', is_flag=True, help='Offline mode (use cached models only, no network)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_pdf(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, no_ocr, ocr_language, ocr_workers, process_priority, max_perf, force, batch_across_files, no_gpu, offline, verbose, debug, output_json):
    """Index PDF files"""
    from arcaneum.cli.index_pdfs import index_pdfs_command
    index_pdfs_command(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, no_ocr, ocr_language, ocr_workers, process_priority, max_perf, force, batch_across_files, no_gpu, offline, verbose, debug, output_json)


@index.command('code')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='jina-code', help='Embedding model (default: jina-code for source code)')
@click.option('--file-workers', type=int, default=None, help='Number of source files to process in parallel within each repo (absolute, overrides multiplier)')
@click.option('--file-worker-mult', type=float, default=None, help='File worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-workers', type=int, default=None, help='Parallel workers for embedding generation (absolute, overrides multiplier)')
@click.option('--embedding-worker-mult', type=float, default=None, help='Embedding worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-batch-size', type=int, default=200, help='Batch size for embedding generation (default: 200)')
@click.option('--chunk-size', type=int, help='Target chunk size in tokens (default: 400)')
@click.option('--chunk-overlap', type=int, help='Overlap between chunks in tokens (default: 20)')
@click.option('--depth', type=int, help='Git discovery depth')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal)')
@click.option('--max-perf', is_flag=True, help='Maximum performance preset (1.0 worker mult, batch 500, low priority)')
@click.option('--force', is_flag=True, help='Force reindex all projects')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_code(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, chunk_size, chunk_overlap, depth, process_priority, max_perf, force, no_gpu, verbose, debug, output_json):
    """Index source code"""
    from arcaneum.cli.index_source import index_source_command
    index_source_command(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, chunk_size, chunk_overlap, depth, process_priority, max_perf, force, no_gpu, verbose, debug, output_json)


@index.command('markdown')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='stella', help='Embedding model (default: stella for documents)')
@click.option('--file-workers', type=int, default=None, help='Number of markdown files to process in parallel (absolute, overrides multiplier)')
@click.option('--file-worker-mult', type=float, default=None, help='File worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-workers', type=int, default=None, help='Parallel workers for embedding generation (absolute, overrides multiplier)')
@click.option('--embedding-worker-mult', type=float, default=None, help='Embedding worker multiplier of cpu_count (e.g., 1.0 = all cores, 0.5 = half cores)')
@click.option('--embedding-batch-size', type=int, default=200, help='Batch size for embedding generation (default: 200)')
@click.option('--chunk-size', type=int, help='Target chunk size in tokens')
@click.option('--chunk-overlap', type=int, help='Overlap between chunks in tokens')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories recursively')
@click.option('--exclude', multiple=True, help='Patterns to exclude (e.g., node_modules, .obsidian)')
@click.option('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
@click.option('--process-priority', type=click.Choice(['low', 'normal', 'high']), default='normal', help='Process scheduling priority (default: normal)')
@click.option('--max-perf', is_flag=True, help='Maximum performance preset (1.0 worker mult, batch 500, low priority)')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration (use CPU only)')
@click.option('--offline', is_flag=True, help='Offline mode (use cached models only, no network)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode (show all library warnings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_markdown(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, chunk_size, chunk_overlap, recursive, exclude, qdrant_url, process_priority, max_perf, force, no_gpu, offline, verbose, debug, output_json):
    """Index markdown files"""
    from arcaneum.cli.index_markdown import index_markdown_command
    index_markdown_command(path, collection, model, file_workers, file_worker_mult, embedding_workers, embedding_worker_mult, embedding_batch_size, chunk_size, chunk_overlap, recursive, exclude, qdrant_url, process_priority, max_perf, force, no_gpu, offline, verbose, debug, output_json)


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
@cli.group()
def search():
    """Search collections"""
    pass


@search.command('semantic')
@click.argument('query')
@click.option('--collection', required=True, help='Collection to search')
@click.option('--vector-name', help='Vector name to use (auto-detects if not specified)')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--score-threshold', type=float, help='Minimum score threshold')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_semantic(query, collection, vector_name, filter_arg, limit, score_threshold, output_json, verbose):
    """Vector-based semantic search"""
    from arcaneum.cli.search import search_command
    search_command(query, collection, vector_name, filter_arg, limit, score_threshold, output_json, verbose)


@search.command('text')
@click.argument('query')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index to search')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_text(query, index_name, filter_arg, limit, output_json, verbose):
    """Keyword-based full-text search"""
    from arcaneum.cli.fulltext import search_text_command
    search_text_command(query, index_name, filter_arg, limit, output_json, verbose)


# Dual indexing commands (RDR-009)
@cli.group()
def corpus():
    """Manage dual-index corpora (Qdrant + MeiliSearch)"""
    pass


@corpus.command('create')
@click.argument('name')
@click.option('--type', 'corpus_type', type=click.Choice(['pdf', 'code', 'markdown']), required=True, help='Corpus type')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_corpus(name, corpus_type, models, output_json):
    """Create both collection and index"""
    from arcaneum.cli.corpus import create_corpus_command
    create_corpus_command(name, corpus_type, models, output_json)


@corpus.command('sync')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--corpus', required=True, help='Corpus name')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--file-types', help='File extensions to index (e.g., .py,.md)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def sync_directory(directory, corpus, models, file_types, output_json):
    """Index to both vector and full-text"""
    from arcaneum.cli.sync import sync_directory_command
    sync_directory_command(directory, corpus, models, file_types, output_json)


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
