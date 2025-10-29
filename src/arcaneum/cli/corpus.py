"""Corpus management commands - stub for RDR-009 (not yet implemented)."""

import click
import sys


def create_corpus_command(name: str, corpus_type: str, models: str, output_json: bool):
    """Create corpus - stub implementation."""
    error_msg = (
        "The 'create-corpus' command is not yet implemented (RDR-009).\n\n"
        "For now, use 'create-collection' instead:\n"
        f"  bin/arc collection create {name} --model stella\n\n"
        "Then index with:\n"
        f"  bin/arc index pdfs <path> --collection {name} --model stella"
    )

    if output_json:
        import json
        print(json.dumps({"success": False, "error": error_msg}))
    else:
        click.echo(click.style(error_msg, fg='yellow'))

    sys.exit(1)
