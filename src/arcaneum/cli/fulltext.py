"""CLI commands for full-text search operations (RDR-008)."""

import json
import logging
import os
import sys
import time
import click
from typing import Optional
from rich.console import Console
from rich.table import Table

from ..fulltext.client import FullTextClient
from ..fulltext.indexes import get_index_settings, get_available_index_types
from .output import print_json, print_error, print_info, print_success
from .interaction_logger import interaction_logger
from .errors import InvalidArgumentError, ResourceNotFoundError

console = Console()
logger = logging.getLogger(__name__)


def get_client() -> FullTextClient:
    """Get MeiliSearch client from environment or auto-generated key."""
    from ..paths import get_meilisearch_api_key

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    # Use auto-generated key if not set in environment
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def search_text_command(
    query: str,
    index_name: str,
    filter_arg: Optional[str],
    limit: int,
    offset: int,
    output_json: bool,
    verbose: bool
):
    """
    Implementation for 'arc search text' command.
    Called from main.py search group.

    Args:
        query: Search query (use quotes for exact phrases)
        index_name: MeiliSearch index to search
        filter_arg: Metadata filter expression (e.g., 'language = python')
        limit: Maximum number of results
        offset: Number of results to skip (for pagination)
        output_json: If True, output JSON format
        verbose: If True, show detailed output
    """
    # Setup logging based on verbose flag
    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(index_name):
            raise ResourceNotFoundError(f"Index '{index_name}' not found")

        if verbose:
            logger.info(f"Searching index '{index_name}' for: \"{query}\"")
            if filter_arg:
                logger.info(f"Filter: {filter_arg}")

        start_time = time.time()

        # Start interaction logging (RDR-018)
        interaction_logger.start(
            "search", "text",
            index=index_name,
            query=query,
            limit=limit,
            offset=offset,
            filters=filter_arg if filter_arg else None,
        )

        try:
            results = client.search(
                index_name,
                query,
                filter=filter_arg,
                limit=limit,
                offset=offset,
                attributes_to_highlight=['content']
            )

            execution_time_ms = (time.time() - start_time) * 1000

            if verbose:
                logger.info(f"Search completed in {execution_time_ms:.1f}ms")

            # Log successful search
            interaction_logger.finish(result_count=len(results.get('hits', [])))
        except Exception as e:
            # Log failed search
            interaction_logger.finish(error=str(e))
            raise

        # Format and output results
        if output_json:
            # JSON output mode
            output = {
                "status": "success",
                "message": f"Found {results.get('estimatedTotalHits', 0)} results",
                "data": {
                    "query": query,
                    "index": index_name,
                    "hits": results.get('hits', []),
                    "estimatedTotalHits": results.get('estimatedTotalHits', 0),
                    "processingTimeMs": results.get('processingTimeMs', 0),
                    "limit": limit,
                    "offset": offset,
                },
                "errors": []
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            processing_time = results.get('processingTimeMs', 0)
            estimated_total = results.get('estimatedTotalHits', 0)
            hits = results.get('hits', [])

            console.print(f"\n[bold]Search Results[/bold] ({processing_time}ms)")
            console.print(f"Found {estimated_total} matches in '{index_name}'\n")

            if not hits:
                console.print("[dim]No results found[/dim]")
            else:
                for i, hit in enumerate(hits, 1):
                    # Get file location
                    location = hit.get('filename', hit.get('file_path', 'Unknown'))
                    if 'line_number' in hit:
                        location += f":{hit['line_number']}"

                    console.print(f"[cyan]{i}. {location}[/cyan]")

                    # Show metadata if available
                    if verbose:
                        if 'language' in hit:
                            console.print(f"   Language: {hit['language']}")
                        if 'project' in hit:
                            console.print(f"   Project: {hit['project']}")
                        if 'page_number' in hit:
                            console.print(f"   Page: {hit['page_number']}")

                    # Show highlighted content
                    if '_formatted' in hit and 'content' in hit['_formatted']:
                        content = hit['_formatted']['content']
                        # Truncate long content
                        if len(content) > 200:
                            content = content[:200] + "..."
                        # Replace MeiliSearch highlight markers with rich formatting
                        content = content.replace('<em>', '[yellow]').replace('</em>', '[/yellow]')
                        console.print(f"   {content}")
                    elif 'content' in hit:
                        content = hit['content']
                        if len(content) > 200:
                            content = content[:200] + "..."
                        console.print(f"   {content}")

                    console.print()  # Blank line between results

        sys.exit(0)

    except (InvalidArgumentError, ResourceNotFoundError):
        raise  # Re-raise our custom exceptions for main() to handle
    except Exception as e:
        console.print(f"[ERROR] Search failed: {e}", style="red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Index management commands (arc indexes ...)
# Named 'indexes' to mirror 'arc collection' for Qdrant
@click.group()
def fulltext():
    """MeiliSearch index management commands (mirrors arc collection)."""
    pass


@fulltext.command('create')
@click.argument('name')
@click.option('--type', 'index_type',
              type=click.Choice(['source-code', 'pdf-docs', 'markdown-docs', 'code', 'pdf', 'markdown']),
              help='Index type (determines settings)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_index(name, index_type, output_json):
    """Create a new MeiliSearch index."""
    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index already exists
        if client.index_exists(name):
            raise InvalidArgumentError(f"Index '{name}' already exists")

        # Get settings if type specified
        settings = None
        if index_type:
            settings = get_index_settings(index_type)
            if not output_json:
                print_info(f"Using {index_type} settings")

        client.create_index(name, primary_key='id', settings=settings)

        data = {"name": name}
        if settings:
            data["searchableAttributes"] = len(settings.get('searchableAttributes', []))
            data["filterableAttributes"] = len(settings.get('filterableAttributes', []))

        print_success(f"Created index '{name}'", json_output=output_json, data=data)

        if not output_json and settings:
            print_info(f"  Searchable attributes: {data['searchableAttributes']}")
            print_info(f"  Filterable attributes: {data['filterableAttributes']}")

    except (InvalidArgumentError, ResourceNotFoundError):
        raise
    except Exception as e:
        print_error(f"Failed to create index: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('list')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_indexes(output_json):
    """List all MeiliSearch indexes."""
    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        indexes = client.list_indexes()

        if output_json:
            print_json("success", f"Found {len(indexes)} indexes", data={"indexes": indexes})
            return

        if not indexes:
            print_info("No indexes found")
            return

        table = Table(title="MeiliSearch Indexes")
        table.add_column("Name", style="cyan")
        table.add_column("Primary Key", style="green")
        table.add_column("Documents", style="yellow")
        table.add_column("Created", style="dim")

        for idx in indexes:
            # Get document count
            try:
                stats = client.get_index_stats(idx['uid'])
                doc_count = str(stats.get('numberOfDocuments', 0))
            except Exception:
                doc_count = "?"

            # Format created date - handle both string and datetime
            created_at = idx.get('createdAt')
            if created_at:
                if hasattr(created_at, 'strftime'):
                    created_str = created_at.strftime('%Y-%m-%d')
                elif isinstance(created_at, str):
                    created_str = created_at[:10]
                else:
                    created_str = str(created_at)[:10]
            else:
                created_str = 'N/A'

            table.add_row(
                idx['uid'],
                idx.get('primaryKey', 'N/A'),
                doc_count,
                created_str
            )

        console.print(table)

    except ResourceNotFoundError:
        raise
    except Exception as e:
        print_error(f"Failed to list indexes: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('info')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_info(name, output_json):
    """Show detailed information about an index."""
    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        # Get index details
        index = client.get_index(name)
        stats = client.get_index_stats(name)
        settings = client.get_index_settings(name)

        data = {
            "name": name,
            "primaryKey": index.primary_key,
            "stats": stats,
            "settings": settings,
        }

        if output_json:
            print_json("success", f"Index '{name}' details", data=data)
            return

        console.print(f"\n[bold]Index: {name}[/bold]\n")

        # Stats table
        stats_table = Table(title="Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Documents", str(stats.get('numberOfDocuments', 0)))
        stats_table.add_row("Is Indexing", str(stats.get('isIndexing', False)))
        console.print(stats_table)
        console.print()

        # Settings table
        settings_table = Table(title="Settings")
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="dim")

        searchable = settings.get('searchableAttributes', [])
        if searchable == ['*']:
            settings_table.add_row("Searchable Attributes", "All (*)")
        else:
            settings_table.add_row("Searchable Attributes", ", ".join(searchable[:5]) + ("..." if len(searchable) > 5 else ""))

        filterable = settings.get('filterableAttributes', [])
        settings_table.add_row("Filterable Attributes", ", ".join(filterable[:5]) + ("..." if len(filterable) > 5 else ""))

        sortable = settings.get('sortableAttributes', [])
        settings_table.add_row("Sortable Attributes", ", ".join(sortable) if sortable else "None")

        typo_config = settings.get('typoTolerance', {})
        settings_table.add_row("Typo Tolerance", "Enabled" if typo_config.get('enabled', True) else "Disabled")

        console.print(settings_table)

    except ResourceNotFoundError:
        raise
    except Exception as e:
        print_error(f"Failed to get index info: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('delete')
@click.argument('name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def delete_index(name, confirm, output_json):
    """Delete a MeiliSearch index."""
    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        if not confirm and not output_json:
            if not click.confirm(f"Delete index '{name}'? This cannot be undone."):
                print_info("Cancelled.")
                return

        client.delete_index(name)
        print_success(f"Deleted index '{name}'", json_output=output_json)

    except ResourceNotFoundError:
        raise
    except Exception as e:
        print_error(f"Failed to delete index: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('update-settings')
@click.argument('name')
@click.option('--type', 'index_type',
              type=click.Choice(['source-code', 'pdf-docs', 'markdown-docs', 'code', 'pdf', 'markdown']),
              required=True,
              help='Index type to apply settings from')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def update_settings(name, index_type, output_json):
    """Update index settings from a preset type."""
    try:
        client = get_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        settings = get_index_settings(index_type)
        client.update_index_settings(name, settings)

        data = {
            "name": name,
            "type": index_type,
            "searchableAttributes": len(settings.get('searchableAttributes', [])),
            "filterableAttributes": len(settings.get('filterableAttributes', [])),
        }

        print_success(f"Updated settings for index '{name}' using {index_type} template", json_output=output_json, data=data)

    except ResourceNotFoundError:
        raise
    except Exception as e:
        print_error(f"Failed to update settings: {e}", json_output=output_json)
        sys.exit(1)
