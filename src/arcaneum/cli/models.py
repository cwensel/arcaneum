"""Model listing CLI command."""

import json
from rich.console import Console
from rich.table import Table

from arcaneum.embeddings.client import EMBEDDING_MODELS

console = Console()


def list_models_command(output_json: bool):
    """List available embedding models.

    Args:
        output_json: Output as JSON
    """
    if output_json:
        # JSON output
        models_data = []
        for alias, config in EMBEDDING_MODELS.items():
            models_data.append({
                "alias": alias,
                "model": config["name"],
                "dimensions": config["dimensions"],
                "description": config.get("description", "")
            })
        print(json.dumps(models_data, indent=2))
    else:
        # Table output
        table = Table(title="Available Embedding Models")
        table.add_column("Alias", style="cyan")
        table.add_column("Actual Model", style="yellow")
        table.add_column("Dims", style="magenta", justify="right")
        table.add_column("Description", style="green")

        for alias, config in EMBEDDING_MODELS.items():
            # Only show available models
            if config.get("available", True):
                table.add_row(
                    alias,
                    config["name"],
                    str(config["dimensions"]),
                    config.get("description", "")
                )

        console.print(table)
        console.print("\n[cyan]Usage:[/cyan]")
        console.print("  arc index-pdfs ~/docs --collection docs --model bge-large")
        console.print("  arc index-source ~/code --collection code --model bge")
