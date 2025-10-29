"""Directory sync commands - stub for RDR-009 (not yet implemented)."""

import click
import sys


def sync_directory_command(
    directory: str,
    corpus: str,
    models: str,
    file_types: str,
    output_json: bool
):
    """Sync directory - stub implementation."""
    error_msg = (
        "The 'sync-directory' command is not yet implemented (RDR-009).\n\n"
        "For now, use 'index-pdfs' instead:\n"
        f"  bin/arc index pdfs {directory} --collection {corpus} --model stella"
    )

    if output_json:
        import json
        print(json.dumps({"success": False, "error": error_msg}))
    else:
        click.echo(click.style(error_msg, fg='yellow'))

    sys.exit(1)
