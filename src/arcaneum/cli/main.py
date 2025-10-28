"""Main CLI entry point for Arcaneum (RDR-001 with RDR-006 enhancements)."""

import sys
import click
from arcaneum import __version__

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
@cli.command('create-collection')
@click.argument('name')
@click.option('--model', required=True, help='Embedding model (stella, modernbert, bge, jina-code)')
@click.option('--type', 'collection_type', type=click.Choice(['pdf', 'code']), help='Collection type (pdf or code)')
@click.option('--hnsw-m', type=int, default=16, help='HNSW index parameter m')
@click.option('--hnsw-ef', type=int, default=100, help='HNSW index parameter ef_construct')
@click.option('--on-disk', is_flag=True, help='Store vectors on disk')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_collection(name, model, collection_type, hnsw_m, hnsw_ef, on_disk, output_json):
    """Create Qdrant collection (from RDR-003)"""
    from arcaneum.cli.collections import create_collection_command
    create_collection_command(name, model, hnsw_m, hnsw_ef, on_disk, output_json, collection_type)


@cli.command('list-collections')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_collections(verbose, output_json):
    """List all Qdrant collections (from RDR-003)"""
    from arcaneum.cli.collections import list_collections_command
    list_collections_command(verbose, output_json)


@cli.command('collection-info')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def collection_info(name, output_json):
    """Show detailed information about a collection (from RDR-003)"""
    from arcaneum.cli.collections import info_collection_command
    info_collection_command(name, output_json)


@cli.command('delete-collection')
@click.argument('name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def delete_collection(name, confirm, output_json):
    """Delete a Qdrant collection (from RDR-003)"""
    from arcaneum.cli.collections import delete_collection_command
    delete_collection_command(name, confirm, output_json)


# Indexing commands (RDR-004, RDR-005)
@cli.command('index-pdfs')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='stella', help='Embedding model')
@click.option('--workers', type=int, default=4, help='Parallel workers')
@click.option('--ocr-enabled', is_flag=True, help='Enable OCR for scanned PDFs')
@click.option('--ocr-language', default='eng', help='OCR language code')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--batch-across-files', is_flag=True, help='Batch uploads across files (faster but less atomic)')
@click.option('--offline', is_flag=True, help='Offline mode (use cached models only, no network)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_pdfs(path, collection, model, workers, ocr_enabled, ocr_language, force, batch_across_files, offline, verbose, output_json):
    """Index PDF files to Qdrant collection (from RDR-004)"""
    from arcaneum.cli.index_pdfs import index_pdfs_command
    index_pdfs_command(path, collection, model, workers, ocr_enabled, ocr_language, force, batch_across_files, offline, verbose, output_json)


@cli.command('index-source')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Target collection name')
@click.option('--model', default='bge', help='Embedding model (default: bge for 1024D)')
@click.option('--workers', type=int, default=4, help='Parallel workers')
@click.option('--depth', type=int, help='Git discovery depth')
@click.option('--force', is_flag=True, help='Force reindex all projects')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_source(path, collection, model, workers, depth, force, verbose, output_json):
    """Index source code to Qdrant collection (from RDR-005)"""
    from arcaneum.cli.index_source import index_source_command
    index_source_command(path, collection, model, workers, depth, force, verbose, output_json)


# Search commands (RDR-007, RDR-012)
@cli.command('search')
@click.argument('query')
@click.option('--collection', required=True, help='Collection to search')
@click.option('--vector-name', help='Vector name to use (auto-detects if not specified)')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--score-threshold', type=float, help='Minimum score threshold')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search(query, collection, vector_name, filter_arg, limit, score_threshold, output_json, verbose):
    """Search Qdrant collection semantically (from RDR-007)"""
    from arcaneum.cli.search import search_command
    search_command(query, collection, vector_name, filter_arg, limit, score_threshold, output_json, verbose)


@cli.command('search-text')
@click.argument('query')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index to search')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_text(query, index_name, filter_arg, limit, output_json, verbose):
    """Full-text search MeiliSearch index (from RDR-012)"""
    from arcaneum.cli.fulltext import search_text_command
    search_text_command(query, index_name, filter_arg, limit, output_json, verbose)


# Dual indexing commands (RDR-009)
@cli.command('create-corpus')
@click.argument('name')
@click.option('--type', 'corpus_type', type=click.Choice(['source-code', 'pdf']), required=True, help='Corpus type')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_corpus(name, corpus_type, models, output_json):
    """Create both Qdrant collection and MeiliSearch index (from RDR-009)"""
    from arcaneum.cli.corpus import create_corpus_command
    create_corpus_command(name, corpus_type, models, output_json)


@cli.command('sync-directory')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--corpus', required=True, help='Corpus name')
@click.option('--models', default='stella,jina', help='Embedding models (comma-separated)')
@click.option('--file-types', help='File extensions to index (e.g., .py,.md)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def sync_directory(directory, corpus, models, file_types, output_json):
    """Index directory to both Qdrant and MeiliSearch (from RDR-009)"""
    from arcaneum.cli.sync import sync_directory_command
    sync_directory_command(directory, corpus, models, file_types, output_json)


def main():
    """Main CLI entry point."""
    try:
        cli()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
