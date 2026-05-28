"""Model listing CLI command (RDR-003 with RDR-006 enhancements)."""

from rich.console import Console
from rich.table import Table

from arcaneum.cli.corpus_defaults import DEFAULT_MODELS_BY_CORPUS_TYPE
from arcaneum.cli.output import print_json
from arcaneum.embeddings.client import EMBEDDING_MODELS, get_embedding_prompt_policy
from arcaneum.utils.memory import get_batch_size_for_model_params

console = Console()


def _default_for(alias: str) -> list[str]:
    """Return corpus types that use alias in their default model list."""
    defaults = []
    for corpus_type, model_list in DEFAULT_MODELS_BY_CORPUS_TYPE.items():
        aliases = [part.strip() for part in model_list.split(",") if part.strip()]
        if alias in aliases:
            defaults.append(corpus_type)
    return defaults


def _risk_tier(config: dict) -> str:
    """Classify model size/cost risk for LLM-driven selection."""
    params = config.get("params_billions")
    if params is None:
        return "low"
    if params >= 5.0:
        return "very-high"
    if params >= 1.0:
        return "high"
    if params >= 0.3:
        return "medium"
    return "low"


def _support_tier(config: dict, default_for: list[str]) -> str:
    """Return the user-facing operational support tier."""
    if default_for:
        return "stable-default"
    if config.get("backend", "fastembed") == "fastembed":
        return "stable"
    if _risk_tier(config) in {"high", "very-high"}:
        return "gpu-opt-in"
    return "opt-in"


def _prompt_policy_summary(config: dict) -> str:
    """Condense prompt/task policy into a compact stable string."""
    parts = []
    for role in ("query", "document"):
        for field in ("prompt_name", "task", "prompt"):
            value = config.get(f"{role}_{field}")
            if value:
                parts.append(f"{role}_{field}={value}")
    if not parts:
        backend = config.get("backend", "fastembed")
        return "fastembed query_embed/embed" if backend == "fastembed" else "encode"
    return "; ".join(parts)


def _suggested_batches(config: dict) -> dict:
    """Return conservative batch hints from the same registry fields runtime uses."""
    params = config.get("params_billions")
    if params is None:
        outer = 128 if config.get("backend", "fastembed") == "fastembed" else 32
    else:
        outer = get_batch_size_for_model_params(params)
        if config.get("backend", "fastembed") == "fastembed":
            outer = max(128, outer)
    mps_inner = config.get("mps_max_batch")
    return {
        "cpu_outer": f"1-{outer}",
        "gpu_outer": f"8-{outer}",
        "mps_inner_max": mps_inner,
    }


def _hardware_support(config: dict) -> dict:
    """Describe runtime backend support without probing the local machine."""
    backend = config.get("backend", "fastembed")
    if backend == "sentence-transformers":
        return {
            "cpu": True,
            "cuda": True,
            "mps": True,
            "mps_note": "supported; use registry mps_max_batch",
        }
    return {
        "cpu": True,
        "cuda": False,
        "mps": "experimental-coreml",
        "mps_note": "requires ARC_EXPERIMENTAL_COREML=1",
    }


def _reindex_warning(config: dict) -> str:
    """Explain when model metadata changes require corpus reindexing."""
    policy = get_embedding_prompt_policy(config["alias"])
    has_prompt_policy = bool(policy.get("query") or policy.get("document"))
    if has_prompt_policy:
        return "reindex after prompt/task/backend/default changes"
    return "reindex after dimension/backend/default changes"


def _model_catalog_row(alias: str, config: dict) -> dict:
    """Build the stable LLM-readable model catalog row."""
    config = {"alias": alias, **config}
    default_for = _default_for(alias)
    params = config.get("params_billions")
    return {
        "alias": alias,
        "model": config["name"],
        "backend": config.get("backend", "fastembed"),
        "recommended_for": config.get("recommended_for"),
        "default_for": default_for,
        "support_tier": _support_tier(config, default_for),
        "install_extra": "core",
        "prompt_policy": _prompt_policy_summary(config),
        "context_limit": config.get("max_seq_length"),
        "dimensions": config["dimensions"],
        "params_billions": params,
        "risk_tier": _risk_tier(config),
        "hardware": _hardware_support(config),
        "suggested_batches": _suggested_batches(config),
        "reindex_warning": _reindex_warning(config),
        "description": config.get("description", ""),
    }


def list_models_command(output_json: bool):
    """List available embedding models.

    Args:
        output_json: Output as JSON
    """
    if output_json:
        # JSON output
        models_data = [
            _model_catalog_row(alias, config)
            for alias, config in EMBEDDING_MODELS.items()
        ]
        print_json("success", f"Found {len(models_data)} embedding models", {"models": models_data})
    else:
        # Table output
        table = Table(title="Available Embedding Models")
        table.add_column("Alias", style="cyan")
        table.add_column("Backend", style="blue")
        table.add_column("Use", style="green")
        table.add_column("Default", style="cyan")
        table.add_column("Tier", style="yellow")
        table.add_column("Ctx", style="magenta", justify="right")
        table.add_column("Dims", style="magenta", justify="right")
        table.add_column("Risk", style="red")
        table.add_column("Batch", style="white")
        table.add_column("Description", style="green")

        for alias, config in EMBEDDING_MODELS.items():
            # Only show available models
            if config.get("available", True):
                row = _model_catalog_row(alias, config)
                table.add_row(
                    alias,
                    row["backend"],
                    row["recommended_for"] or "-",
                    ",".join(row["default_for"]) or "-",
                    row["support_tier"],
                    str(row["context_limit"] or "default"),
                    str(row["dimensions"]),
                    row["risk_tier"],
                    row["suggested_batches"]["cpu_outer"],
                    row["description"],
                )

        console.print(table)
        console.print("\n[cyan]Usage:[/cyan]")
        console.print("  arc corpus create docs --type pdf        # defaults to arctic-m")
        console.print("  arc corpus create code --type code       # defaults to jina-code")
        console.print("  arc sync docs ~/docs --no-gpu            # CPU-first stable path")
        console.print("  arc models list --json                   # full LLM-readable policy")
        console.print("\n[yellow]GPU is opt-in.[/yellow] Prefer stable defaults unless you need a larger model.")
        console.print("Reindex a corpus after changing model dimensions, backend, prompts, tasks, or defaults.")
