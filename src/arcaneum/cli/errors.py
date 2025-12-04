"""Custom exception classes for CLI error handling (RDR-006).

This module defines exception classes that map to specific exit codes following
Beads best practices for structured error handling.

Exit Codes:
- 0: Success
- 1: General error
- 2: Invalid arguments
- 3: Resource not found
"""


# Exit codes (same as main.py)
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARGS = 2
EXIT_NOT_FOUND = 3


class ArcaneumError(Exception):
    """Base exception for Arcaneum CLI errors.

    All custom Arcaneum exceptions should inherit from this class.
    Default exit code is EXIT_ERROR (1).
    """
    exit_code = EXIT_ERROR


class InvalidArgumentError(ArcaneumError):
    """Invalid command line arguments or configuration.

    Examples:
    - Invalid model name
    - Invalid collection type
    - Mutually exclusive options used together
    - Required file/directory doesn't exist

    Exit code: 2
    """
    exit_code = EXIT_INVALID_ARGS


class ResourceNotFoundError(ArcaneumError):
    """Resource not found (collection, file, directory, etc.).

    Examples:
    - Collection doesn't exist in Qdrant
    - File path not found
    - Git repository not found
    - Model not downloaded

    Exit code: 3
    """
    exit_code = EXIT_NOT_FOUND


class CollectionNotFoundError(ResourceNotFoundError):
    """Specific case: Qdrant collection not found."""
    pass


class ModelNotFoundError(ResourceNotFoundError):
    """Specific case: Embedding model not found or not downloaded."""
    pass


class GitRepositoryNotFoundError(ResourceNotFoundError):
    """Specific case: Git repository not found at specified path."""
    pass


# Custom Click classes for better error messages
import click


class HelpfulGroup(click.Group):
    """Custom Click group that provides helpful error messages for missing subcommands.

    When a user runs a command without a required subcommand (e.g., `arc search "query"`
    instead of `arc search semantic "query"`), this provides clear guidance on correct usage.
    """

    def __init__(self, *args, usage_examples: list[str] | None = None, **kwargs):
        """Initialize with optional usage examples.

        Args:
            usage_examples: List of example commands to show in error messages
        """
        super().__init__(*args, **kwargs)
        self.usage_examples = usage_examples or []

    def resolve_command(self, ctx, args):
        """Override to provide helpful error when subcommand is missing or invalid."""
        # If no args, show help with examples
        if not args:
            return super().resolve_command(ctx, args)

        # Get the first arg to check if it's a valid subcommand
        cmd_name = args[0]

        # Try to resolve normally first
        cmd = self.get_command(ctx, cmd_name)
        if cmd is not None:
            return super().resolve_command(ctx, args)

        # Not a valid subcommand - provide helpful error
        valid_commands = list(self.commands.keys())
        group_name = ctx.info_name or self.name

        # Build helpful error message
        error_lines = [
            f"[ERROR] '{cmd_name}' is not a valid subcommand for '{group_name}'.",
            "",
            f"Available subcommands: {', '.join(valid_commands)}",
            "",
            "Correct syntax:",
        ]

        # Add usage examples
        if self.usage_examples:
            for example in self.usage_examples:
                error_lines.append(f"  {example}")
        else:
            # Generate generic examples from available commands
            for cmd in valid_commands:
                error_lines.append(f"  arc {group_name} {cmd} --help")

        error_lines.append("")
        error_lines.append(f"Run 'arc {group_name} --help' for more information.")

        raise click.UsageError("\n".join(error_lines))
