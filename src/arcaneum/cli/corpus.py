"""Corpus management commands for dual indexing (RDR-009).

This module implements the 'corpus create' command that creates both a Qdrant
collection and a MeiliSearch index in a single operation.
"""

import sys
from typing import Dict, Any

import click
from rich.console import Console
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

from ..cli.output import print_json, print_error, print_info, print_success
from ..cli.utils import create_qdrant_client
from ..cli.interaction_logger import interaction_logger
from ..cli.errors import InvalidArgumentError, ResourceNotFoundError
from ..embeddings.client import EMBEDDING_MODELS
from ..fulltext.indexes import get_index_settings
from ..indexing.collection_metadata import set_collection_metadata, CollectionType

console = Console()


def get_meili_client():
    """Get MeiliSearch client from environment or auto-generated key."""
    import os
    from ..paths import get_meilisearch_api_key
    from ..fulltext.client import FullTextClient

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def get_model_dimensions(model_name: str) -> int:
    """Get vector dimensions for an embedding model.

    Args:
        model_name: Model identifier

    Returns:
        Number of dimensions

    Raises:
        ValueError: If model is unknown
    """
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(EMBEDDING_MODELS.keys())}"
        )
    return EMBEDDING_MODELS[model_name]["dimensions"]


def map_corpus_type_to_canonical(corpus_type: str) -> str:
    """Map corpus type aliases to canonical names.

    Args:
        corpus_type: Type from CLI (pdf, code, markdown)

    Returns:
        Canonical type name for MeiliSearch (pdf-docs, source-code, markdown-docs)
    """
    type_map = {
        "pdf": "pdf-docs",
        "code": "source-code",
        "markdown": "markdown-docs",
    }
    return type_map.get(corpus_type, corpus_type)


def create_corpus_command(
    name: str,
    corpus_type: str,
    models: str,
    output_json: bool
):
    """Create both Qdrant collection and MeiliSearch index.

    This implements the first command of the 2-command workflow:
    1. corpus create (this command) - creates both systems
    2. corpus sync - indexes documents to both systems

    Args:
        name: Corpus name (used for both collection and index)
        corpus_type: Type of corpus (pdf, code, markdown)
        models: Comma-separated list of embedding models
        output_json: If True, output JSON format
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "corpus", "create",
        corpus=name,
        corpus_type=corpus_type,
        models=models,
    )

    try:
        if not output_json:
            print_info(f"Creating corpus '{name}'")
            print_info(f"Type: {corpus_type}, Models: {models}")

        # Parse and validate models
        model_list = [m.strip() for m in models.split(',')]
        for model in model_list:
            if model not in EMBEDDING_MODELS:
                error_msg = f"Unknown model: {model}. Available: {list(EMBEDDING_MODELS.keys())}"
                raise InvalidArgumentError(error_msg)

        # Build vectors config for Qdrant
        vectors_config = {}
        for model in model_list:
            vectors_config[model] = VectorParams(
                size=get_model_dimensions(model),
                distance=Distance.COSINE,
            )

        # Step 1: Create Qdrant collection
        if not output_json:
            print_info("Step 1/2: Creating Qdrant collection...")

        qdrant = create_qdrant_client()

        try:
            qdrant.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )

            # Set collection metadata (type and model)
            set_collection_metadata(
                client=qdrant,
                collection_name=name,
                collection_type=corpus_type,
                model=models
            )

            if not output_json:
                console.print(f"[green]✅ Qdrant collection '{name}' created[/green]")

        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str:
                raise InvalidArgumentError(f"Qdrant collection '{name}' already exists")
            raise

        # Step 2: Create MeiliSearch index
        if not output_json:
            print_info("Step 2/2: Creating MeiliSearch index...")

        meili = get_meili_client()

        try:
            # Verify MeiliSearch is available
            if not meili.health_check():
                raise ResourceNotFoundError(
                    "MeiliSearch server not available. "
                    "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
                )

            # Check if index already exists
            if meili.index_exists(name):
                raise InvalidArgumentError(f"MeiliSearch index '{name}' already exists")

            # Get settings based on corpus type
            canonical_type = map_corpus_type_to_canonical(corpus_type)
            settings = get_index_settings(canonical_type)

            # Create index with settings
            meili.create_index(name, primary_key='id', settings=settings)

            if not output_json:
                console.print(f"[green]✅ MeiliSearch index '{name}' created[/green]")

        except InvalidArgumentError:
            raise
        except ResourceNotFoundError:
            raise
        except Exception as e:
            # MeiliSearch failed but Qdrant succeeded
            if not output_json:
                print_error(f"MeiliSearch index creation failed: {e}")
                print_info(f"Note: Qdrant collection '{name}' was created successfully")
                print_info("You may need to create the MeiliSearch index manually or delete the Qdrant collection")
            raise

        # Success output
        data = {
            "corpus": name,
            "type": corpus_type,
            "models": model_list,
            "vectors": {m: get_model_dimensions(m) for m in model_list},
            "qdrant_collection": name,
            "meilisearch_index": name,
        }

        if output_json:
            print_json(
                "success",
                f"Corpus '{name}' created with {len(model_list)} models",
                data=data
            )
        else:
            console.print(f"\n[green]✅ Corpus '{name}' ready for indexing![/green]")
            for model in model_list:
                dims = get_model_dimensions(model)
                console.print(f"  • {model}: {dims}D")
            console.print(f"\n[dim]Next: arc corpus sync /path/to/files --corpus {name}[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to create corpus: {e}", output_json)
        sys.exit(1)
