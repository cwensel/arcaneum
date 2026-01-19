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
from .utils import create_meili_client

console = Console()
logger = logging.getLogger(__name__)


def format_location(hit: dict) -> str:
    """Format location with full context from RDR-011 metadata.

    Handles three cases:
    1. Source code with line ranges and function/class context
    2. PDF documents with page numbers
    3. Simple file paths (fallback)

    Args:
        hit: Search result hit containing metadata

    Returns:
        Formatted location string (e.g., "file.py:42-67 (my_func function)")
    """
    file_path = hit.get('file_path') or hit.get('filename', 'Unknown')

    # Source code: Include line range and function/class name
    if 'start_line' in hit:
        start = hit['start_line']
        end = hit.get('end_line', start)
        if start == end:
            location = f"{file_path}:{start}"
        else:
            location = f"{file_path}:{start}-{end}"

        if hit.get('function_name'):
            location += f" ({hit['function_name']} function)"
        elif hit.get('class_name'):
            location += f" ({hit['class_name']} class)"
        return location

    # PDF: Include page number
    if 'page_number' in hit:
        return f"{file_path}:page{hit['page_number']}"

    # Legacy support: single line_number field
    if 'line_number' in hit:
        return f"{file_path}:{hit['line_number']}"

    return file_path


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
        client = create_meili_client()

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
                    # Get file location with full context (RDR-012 enhancement)
                    location = format_location(hit)

                    console.print(f"[cyan]{i}. {location}[/cyan]")

                    # Show metadata if available
                    if verbose:
                        if 'language' in hit:
                            console.print(f"   Language: {hit['language']}")
                        if 'project' in hit:
                            console.print(f"   Project: {hit['project']}")

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
        error_str = str(e)
        # Check for MeiliSearch filter attribute error
        if "is not filterable" in error_str or "invalid_search_filter" in error_str:
            import re
            attr_match = re.search(r"Attribute `(\w+)`", error_str)
            bad_attr = attr_match.group(1) if attr_match else None

            # Fetch filterable attributes for this index
            try:
                settings = client.get_index_settings(index_name)
                filterable = settings.get('filterableAttributes', [])

                if bad_attr:
                    console.print(f"[red][ERROR] Filter attribute '{bad_attr}' is not filterable.[/red]")
                else:
                    console.print(f"[red][ERROR] Invalid filter attribute.[/red]")

                if filterable:
                    console.print(f"\nAvailable filterable attributes for index '{index_name}':")
                    for attr in filterable:
                        console.print(f"  - {attr}")
                    if bad_attr:
                        # Suggest similar attribute if there's a close match
                        for attr in filterable:
                            if bad_attr.lower() in attr.lower() or attr.lower() in bad_attr.lower():
                                console.print(f"\n[yellow]Hint: Try --filter \"{attr}=<value>\"[/yellow]")
                                break
                else:
                    console.print(f"\n[yellow]Index '{index_name}' has no filterable attributes configured.[/yellow]")
                    console.print("[dim]Filters cannot be used with this index.[/dim]")
            except Exception:
                console.print(f"[red][ERROR] Search failed: {e}[/red]")
        else:
            console.print(f"[red][ERROR] Search failed: {e}[/red]")

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
              type=click.Choice(['source-code', 'source-code-fulltext', 'pdf-docs', 'markdown-docs',
                                 'code', 'code-fulltext', 'pdf', 'markdown']),
              required=True,
              help='Index type (determines searchable/filterable attributes)')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def create_index(name, index_type, output_json):
    """Create a new MeiliSearch index."""
    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index already exists
        if client.index_exists(name):
            raise InvalidArgumentError(f"Index '{name}' already exists")

        # Get settings based on type
        settings = get_index_settings(index_type)
        if not output_json:
            print_info(f"Using {index_type} settings")

        client.create_index(name, primary_key='id', settings=settings)

        data = {
            "name": name,
            "type": index_type,
            "searchableAttributes": len(settings.get('searchableAttributes', [])),
            "filterableAttributes": len(settings.get('filterableAttributes', [])),
        }

        print_success(f"Created index '{name}'", json_output=output_json, data=data)

        if not output_json:
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
        client = create_meili_client()

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
        client = create_meili_client()

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
        client = create_meili_client()

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
              type=click.Choice(['source-code', 'source-code-fulltext', 'pdf-docs', 'markdown-docs',
                                 'code', 'code-fulltext', 'pdf', 'markdown']),
              required=True,
              help='Index type to apply settings from')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def update_settings(name, index_type, output_json):
    """Update index settings from a preset type."""
    try:
        client = create_meili_client()

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


@fulltext.command('verify')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def verify_index(name, output_json):
    """Verify index health and integrity.

    Checks that the index is accessible, documents are retrievable,
    and settings are properly configured.

    Examples:
        arc indexes verify MyIndex
        arc indexes verify MyIndex --json
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "verify", index=name)

    try:
        client = create_meili_client()

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
        stats = client.get_index_stats(name)
        settings = client.get_index_settings(name)

        # Perform health checks
        issues = []
        warnings = []

        doc_count = stats.get('numberOfDocuments', 0)
        is_indexing = stats.get('isIndexing', False)

        # Check if index is currently processing
        if is_indexing:
            warnings.append("Index is currently processing documents")

        # Check searchable attributes
        searchable = settings.get('searchableAttributes', [])
        if not searchable or searchable == ['*']:
            warnings.append("Searchable attributes not explicitly defined (using wildcard)")

        # Check filterable attributes
        filterable = settings.get('filterableAttributes', [])
        if not filterable:
            warnings.append("No filterable attributes defined")

        # Try to retrieve a sample document to verify accessibility
        sample_accessible = False
        try:
            sample_results = client.search(name, "", limit=1)
            if sample_results.get('hits'):
                sample_accessible = True
        except Exception as e:
            issues.append(f"Failed to retrieve sample document: {e}")

        is_healthy = len(issues) == 0

        data = {
            "name": name,
            "is_healthy": is_healthy,
            "document_count": doc_count,
            "is_indexing": is_indexing,
            "searchable_attributes": len(searchable) if searchable != ['*'] else "all",
            "filterable_attributes": len(filterable),
            "sample_accessible": sample_accessible,
            "issues": issues,
            "warnings": warnings,
        }

        if output_json:
            status = "success" if is_healthy else "warning"
            msg = f"Index '{name}' is healthy" if is_healthy else f"Index '{name}' has issues"
            print_json(status, msg, data)
        else:
            console.print(f"\n[bold cyan]Index: {name}[/bold cyan]\n")

            # Status
            if is_healthy:
                console.print(f"[green]Status: Healthy[/green]")
            else:
                console.print(f"[red]Status: Issues detected[/red]")

            # Stats
            console.print(f"Documents: {doc_count:,}")
            if is_indexing:
                console.print(f"[yellow]Currently indexing...[/yellow]")

            # Settings summary
            if searchable == ['*']:
                console.print(f"Searchable: All attributes")
            else:
                console.print(f"Searchable: {len(searchable)} attributes")
            console.print(f"Filterable: {len(filterable)} attributes")

            # Sample accessibility
            if sample_accessible:
                console.print(f"[green]Sample retrieval: OK[/green]")
            elif doc_count > 0:
                console.print(f"[red]Sample retrieval: Failed[/red]")

            # Issues
            if issues:
                console.print(f"\n[red]Issues:[/red]")
                for issue in issues:
                    console.print(f"  [red]• {issue}[/red]")

            # Warnings
            if warnings:
                console.print(f"\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  [yellow]• {warning}[/yellow]")

            if is_healthy and not warnings:
                console.print(f"\n[green]All checks passed[/green]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            is_healthy=is_healthy,
            document_count=doc_count,
            issues=len(issues),
            warnings=len(warnings),
        )

    except ResourceNotFoundError:
        interaction_logger.finish(error="resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to verify index: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('items')
@click.argument('name')
@click.option('--limit', type=int, default=100, help='Maximum number of items to show')
@click.option('--offset', type=int, default=0, help='Number of items to skip')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_items(name, limit, offset, output_json):
    """List indexed documents in an index.

    Shows unique source files/paths that have been indexed,
    similar to 'arc collection items' for Qdrant.

    Examples:
        arc indexes items MyIndex
        arc indexes items MyIndex --limit 50
        arc indexes items MyIndex --json
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "items", index=name, limit=limit, offset=offset)

    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        # Get all documents to find unique files
        # MeiliSearch doesn't have direct aggregation, so we fetch and dedupe
        items_by_path = {}
        batch_offset = offset
        batch_size = 1000  # Internal batch size for fetching

        while True:
            # Use search with empty query to get all documents
            results = client.search(
                name,
                "",
                limit=batch_size,
                offset=batch_offset,
                attributes_to_retrieve=['file_path', 'filename', 'file_hash', 'page_number', 'language', 'project']
            )

            hits = results.get('hits', [])
            if not hits:
                break

            for hit in hits:
                file_path = hit.get('file_path') or hit.get('filename')
                if file_path and file_path not in items_by_path:
                    items_by_path[file_path] = {
                        'file_path': file_path,
                        'filename': hit.get('filename'),
                        'file_hash': hit.get('file_hash'),
                        'language': hit.get('language'),
                        'project': hit.get('project'),
                        'chunk_count': 1,
                    }
                elif file_path:
                    items_by_path[file_path]['chunk_count'] += 1

            batch_offset += len(hits)

            # Stop if we've collected enough unique items
            if len(items_by_path) >= limit + offset:
                break

            # Also stop if we got fewer results than requested (no more data)
            if len(hits) < batch_size:
                break

        # Convert to list and apply limit
        items_list = list(items_by_path.values())
        total_items = len(items_list)
        items_list = items_list[:limit]

        if output_json:
            data = {
                "index": name,
                "total_items": total_items,
                "showing": len(items_list),
                "offset": offset,
                "items": items_list,
            }
            print_json("success", f"Found {total_items} unique items in index '{name}'", data)
        else:
            console.print(f"\n[bold cyan]Index: {name}[/bold cyan]")
            console.print(f"Unique items: {total_items}\n")

            if not items_list:
                print_info("No items found")
            else:
                table = Table(title="Indexed Files")
                table.add_column("File", style="cyan", no_wrap=False)
                table.add_column("Language", style="green")
                table.add_column("Project", style="yellow")
                table.add_column("Chunks", style="magenta")

                for item in sorted(items_list, key=lambda x: x.get('filename') or x['file_path']):
                    display_name = item.get('filename') or item['file_path']
                    table.add_row(
                        display_name,
                        item.get('language') or '-',
                        item.get('project') or '-',
                        str(item['chunk_count']),
                    )

                console.print(table)

                if total_items > limit:
                    console.print(f"\n[dim]Showing {len(items_list)} of {total_items} items. Use --limit to see more.[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(result_count=len(items_list), total_items=total_items)

    except ResourceNotFoundError:
        interaction_logger.finish(error="resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list index items: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('export')
@click.argument('name')
@click.option('-o', '--output', required=True, type=click.Path(), help='Output file path (.jsonl)')
@click.option('--json', 'output_json', is_flag=True, help='Output stats as JSON')
def export_index(name, output, output_json):
    """Export index documents to JSONL file.

    Exports all documents from a MeiliSearch index for backup or migration.

    Examples:
        arc indexes export MyIndex -o backup.jsonl
        arc indexes export MyIndex -o backup.jsonl --json
    """
    from pathlib import Path

    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "export", index=name, output=output)

    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        output_path = Path(output)
        exported_count = 0
        batch_size = 1000
        batch_offset = 0

        if not output_json:
            console.print(f"Exporting index '{name}'...")

        with open(output_path, 'w') as f:
            # Write header with index metadata
            stats = client.get_index_stats(name)
            settings = client.get_index_settings(name)
            index_obj = client.get_index(name)

            header = {
                '_type': 'index_metadata',
                'name': name,
                'primary_key': index_obj.primary_key,
                'settings': settings,
                'exported_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
            f.write(json.dumps(header) + '\n')

            # Export documents in batches
            while True:
                results = client.search(name, "", limit=batch_size, offset=batch_offset)
                hits = results.get('hits', [])

                if not hits:
                    break

                for hit in hits:
                    doc = {'_type': 'document', **hit}
                    f.write(json.dumps(doc) + '\n')
                    exported_count += 1

                batch_offset += len(hits)

                if not output_json:
                    console.print(f"  Exported {exported_count} documents...", end='\r')

                if len(hits) < batch_size:
                    break

        file_size = output_path.stat().st_size

        if output_json:
            data = {
                "index": name,
                "output_path": str(output_path),
                "exported_count": exported_count,
                "file_size_bytes": file_size,
            }
            print_json("success", f"Exported {exported_count} documents", data)
        else:
            size_mb = file_size / (1024 * 1024)
            console.print(f"\n[green]Exported {exported_count} documents to {output_path}[/green]")
            console.print(f"File size: {size_mb:.2f} MB")

        # Log successful operation (RDR-018)
        interaction_logger.finish(exported_count=exported_count, file_size_bytes=file_size)

    except ResourceNotFoundError:
        interaction_logger.finish(error="resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to export index: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('list-projects')
@click.argument('name')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_projects(name, output_json):
    """List indexed git projects in an index (RDR-011).

    Shows all unique git_project_identifier values with their commit hashes.
    Only applicable for indexes with git-aware source code.

    Examples:
        arc indexes list-projects MyCode
        arc indexes list-projects MyCode --json
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "list-projects", index=name)

    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(name):
            raise ResourceNotFoundError(f"Index '{name}' not found")

        # Query for unique projects
        from ..indexing.fulltext.sync import GitCodeMetadataSync
        sync = GitCodeMetadataSync(client)
        indexed_projects = sync.get_indexed_projects(name)

        if output_json:
            data = {
                "index": name,
                "total_projects": len(indexed_projects),
                "projects": [
                    {"identifier": identifier, "commit_hash": commit}
                    for identifier, commit in sorted(indexed_projects.items())
                ]
            }
            print_json("success", f"Found {len(indexed_projects)} indexed projects", data)
        else:
            console.print(f"\n[bold cyan]Index: {name}[/bold cyan]")
            console.print(f"Indexed Projects: {len(indexed_projects)}\n")

            if not indexed_projects:
                print_info("No git projects found in this index")
                print_info("This index may use simple file-based indexing (not git-aware)")
            else:
                table = Table(title="Git Projects")
                table.add_column("Project Identifier", style="cyan")
                table.add_column("Commit Hash", style="dim")

                for identifier, commit in sorted(indexed_projects.items()):
                    table.add_row(identifier, commit[:12] + "...")

                console.print(table)

        # Log successful operation (RDR-018)
        interaction_logger.finish(result_count=len(indexed_projects))

    except ResourceNotFoundError:
        interaction_logger.finish(error="resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list projects: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('delete-project')
@click.argument('identifier')
@click.option('--index', 'index_name', required=True, help='MeiliSearch index name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def delete_project(identifier, index_name, confirm, output_json):
    """Delete all documents for a specific git project/branch (RDR-011).

    Removes all documents matching the git_project_identifier.
    Other projects/branches in the same index are unaffected.

    Examples:
        arc indexes delete-project arcaneum#main --index MyCode
        arc indexes delete-project myrepo#feature-x --index MyCode --confirm
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "delete-project", index=index_name, identifier=identifier)

    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        # Check if index exists
        if not client.index_exists(index_name):
            raise ResourceNotFoundError(f"Index '{index_name}' not found")

        # Check how many documents will be deleted
        from ..indexing.fulltext.sync import GitCodeMetadataSync
        sync = GitCodeMetadataSync(client)
        doc_count = sync.get_project_document_count(index_name, identifier)

        if doc_count == 0:
            if output_json:
                print_json("warning", f"No documents found for project '{identifier}'", {
                    "index": index_name,
                    "identifier": identifier,
                    "deleted_count": 0
                })
            else:
                print_info(f"No documents found for project '{identifier}' in index '{index_name}'")
            interaction_logger.finish(deleted_count=0)
            return

        # Confirm deletion
        if not confirm and not output_json:
            if not click.confirm(
                f"Delete {doc_count} documents for project '{identifier}' from index '{index_name}'? "
                f"This cannot be undone."
            ):
                print_info("Cancelled.")
                interaction_logger.finish(error="cancelled by user")
                return

        # Delete documents
        deleted_count = sync.delete_project_documents(index_name, identifier)

        if output_json:
            print_json("success", f"Deleted {deleted_count} documents", {
                "index": index_name,
                "identifier": identifier,
                "deleted_count": deleted_count
            })
        else:
            print_success(f"Deleted {deleted_count} documents for project '{identifier}'")

        # Log successful operation (RDR-018)
        interaction_logger.finish(deleted_count=deleted_count)

    except ResourceNotFoundError:
        interaction_logger.finish(error="resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to delete project: {e}", json_output=output_json)
        sys.exit(1)


@fulltext.command('import')
@click.argument('file', type=click.Path(exists=True))
@click.option('--into', 'target_name', help='Target index name (defaults to original name)')
@click.option('--json', 'output_json', is_flag=True, help='Output stats as JSON')
def import_index(file, target_name, output_json):
    """Import index documents from JSONL file.

    Imports documents from a previously exported JSONL file.
    Creates the index if it doesn't exist, using the exported settings.

    Examples:
        arc indexes import backup.jsonl
        arc indexes import backup.jsonl --into NewIndex
        arc indexes import backup.jsonl --json
    """
    from pathlib import Path

    # Start interaction logging (RDR-018)
    interaction_logger.start("indexes", "import", source_file=file, target_name=target_name)

    try:
        client = create_meili_client()

        # Verify server is available
        if not client.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        input_path = Path(file)
        metadata = None
        documents = []
        imported_count = 0

        if not output_json:
            console.print(f"Reading export file...")

        # Read the file
        with open(input_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                record_type = record.pop('_type', 'document')

                if record_type == 'index_metadata':
                    metadata = record
                elif record_type == 'document':
                    documents.append(record)

        if not metadata:
            raise InvalidArgumentError("Export file missing index metadata header")

        # Determine target index name
        index_name = target_name or metadata['name']

        # Create index if it doesn't exist
        if not client.index_exists(index_name):
            if not output_json:
                console.print(f"Creating index '{index_name}'...")

            client.create_index(
                name=index_name,
                primary_key=metadata.get('primary_key', 'id'),
                settings=metadata.get('settings'),
            )

        # Import documents in batches
        batch_size = 1000
        if not output_json:
            console.print(f"Importing {len(documents)} documents...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            client.add_documents(index_name, batch)
            imported_count += len(batch)

            if not output_json:
                console.print(f"  Imported {imported_count} documents...", end='\r')

        if output_json:
            data = {
                "index": index_name,
                "imported_count": imported_count,
                "source_file": str(input_path),
            }
            print_json("success", f"Imported {imported_count} documents into '{index_name}'", data)
        else:
            console.print(f"\n[green]Imported {imported_count} documents into '{index_name}'[/green]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(imported_count=imported_count, index=index_name)

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to import index: {e}", json_output=output_json)
        sys.exit(1)
