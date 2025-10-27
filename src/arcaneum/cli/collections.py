"""Collection management CLI commands (RDR-003)."""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

from arcaneum.config import load_config, ArcaneumConfig, DEFAULT_MODELS
from arcaneum.embeddings.client import EMBEDDING_MODELS

console = Console()


def get_distance(distance_str: str) -> Distance:
    """Convert distance string to Qdrant Distance enum.

    Args:
        distance_str: Distance metric name (cosine, euclid, dot)

    Returns:
        Distance enum value
    """
    distance_map = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    return distance_map.get(distance_str.lower(), Distance.COSINE)


def create_collection_command(
    name: str,
    model: str,
    hnsw_m: int,
    hnsw_ef: int,
    on_disk: bool,
    output_json: bool,
):
    """Create a new Qdrant collection with named vectors.

    Args:
        name: Collection name
        model: Embedding model to use (or comma-separated list)
        hnsw_m: HNSW m parameter
        hnsw_ef: HNSW ef_construct parameter
        on_disk: Store vectors on disk
        output_json: Output as JSON
    """
    try:
        # Parse models (support comma-separated list)
        model_list = [m.strip() for m in model.split(',')]

        # Validate models
        for m in model_list:
            if m not in EMBEDDING_MODELS:
                error_msg = f"Unknown model: {m}. Available: {list(EMBEDDING_MODELS.keys())}"
                if output_json:
                    print(json.dumps({"error": error_msg}))
                else:
                    console.print(f"[red]❌ {error_msg}[/red]")
                sys.exit(1)

        # Build vectors config
        vectors_config = {}
        for m in model_list:
            model_info = EMBEDDING_MODELS[m]
            vectors_config[m] = VectorParams(
                size=model_info["dimensions"],
                distance=Distance.COSINE,
            )

        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:6333")

        # Create collection
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef),
            on_disk_payload=on_disk,
        )

        # Output success
        if output_json:
            result = {
                "collection": name,
                "models": model_list,
                "vectors": {m: EMBEDDING_MODELS[m]["dimensions"] for m in model_list},
                "hnsw": {"m": hnsw_m, "ef_construct": hnsw_ef},
                "on_disk_payload": on_disk,
            }
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✅ Created collection '{name}' with {len(model_list)} models[/green]")
            for m in model_list:
                dims = EMBEDDING_MODELS[m]["dimensions"]
                console.print(f"  • {m}: {dims}D")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Failed to create collection: {e}[/red]")
        sys.exit(1)


def list_collections_command(verbose: bool, output_json: bool):
    """List all Qdrant collections.

    Args:
        verbose: Show detailed information
        output_json: Output as JSON
    """
    try:
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()

        if output_json:
            result = []
            for col in collections.collections:
                col_info = client.get_collection(col.name)
                vectors = {}
                if hasattr(col_info.config.params, 'vectors') and isinstance(col_info.config.params.vectors, dict):
                    for vector_name, vector_params in col_info.config.params.vectors.items():
                        vectors[vector_name] = {
                            "size": vector_params.size,
                            "distance": str(vector_params.distance),
                        }
                result.append({
                    "name": col.name,
                    "points_count": col_info.points_count,
                    "vectors": vectors,
                })
            print(json.dumps(result, indent=2))
        else:
            table = Table(title="Qdrant Collections")
            table.add_column("Name", style="cyan")
            table.add_column("Points", style="yellow")
            if verbose:
                table.add_column("Vectors", style="green")

            for col in collections.collections:
                col_info = client.get_collection(col.name)
                row = [col.name, str(col_info.points_count)]

                if verbose:
                    vectors_str = ""
                    if hasattr(col_info.config.params, 'vectors') and isinstance(col_info.config.params.vectors, dict):
                        vectors_list = [f"{name}({params.size}D)" for name, params in col_info.config.params.vectors.items()]
                        vectors_str = ", ".join(vectors_list)
                    row.append(vectors_str)

                table.add_row(*row)

            console.print(table)

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Failed to list collections: {e}[/red]")
        sys.exit(1)


def delete_collection_command(name: str, confirm: bool, output_json: bool):
    """Delete a Qdrant collection.

    Args:
        name: Collection name to delete
        confirm: Skip confirmation prompt
        output_json: Output as JSON
    """
    try:
        if not confirm:
            if output_json:
                console.print("[red]Error: --confirm flag required for non-interactive deletion[/red]")
                sys.exit(1)

            response = console.input(f"[yellow]Delete collection '{name}'? This cannot be undone. (yes/no): [/yellow]")
            if response.lower() != 'yes':
                console.print("Cancelled.")
                return

        client = QdrantClient(url="http://localhost:6333")
        client.delete_collection(name)

        if output_json:
            print(json.dumps({"deleted": name}))
        else:
            console.print(f"[green]✅ Deleted collection '{name}'[/green]")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Failed to delete collection: {e}[/red]")
        sys.exit(1)


def info_collection_command(name: str, output_json: bool):
    """Show detailed information about a collection.

    Args:
        name: Collection name
        output_json: Output as JSON
    """
    try:
        client = QdrantClient(url="http://localhost:6333")
        info = client.get_collection(name)

        if output_json:
            vectors = {}
            if hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
                for vector_name, vector_params in info.config.params.vectors.items():
                    vectors[vector_name] = {
                        "size": vector_params.size,
                        "distance": str(vector_params.distance),
                    }
            result = {
                "name": name,
                "points_count": info.points_count,
                "status": str(info.status),
                "vectors": vectors,
                "hnsw_config": {
                    "m": info.config.hnsw_config.m,
                    "ef_construct": info.config.hnsw_config.ef_construct,
                },
            }
            print(json.dumps(result, indent=2))
        else:
            console.print(f"\n[bold cyan]Collection: {name}[/bold cyan]")
            console.print(f"Points: {info.points_count}")
            console.print(f"Status: {info.status}")
            console.print(f"\n[bold]Vectors:[/bold]")

            if hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
                for vector_name, vector_params in info.config.params.vectors.items():
                    console.print(f"  • {vector_name}: {vector_params.size}D ({vector_params.distance})")

            console.print(f"\n[bold]HNSW Config:[/bold]")
            hnsw = info.config.hnsw_config
            console.print(f"  m: {hnsw.m}")
            console.print(f"  ef_construct: {hnsw.ef_construct}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Failed to get collection info: {e}[/red]")
        sys.exit(1)
