"""Migration utilities for arcaneum directory structure.

Handles migration from legacy ~/.arcaneum/ to XDG-compliant structure:
- ~/.arcaneum/models → ~/.cache/arcaneum/models
- ~/.arcaneum/data → ~/.local/share/arcaneum
"""

import shutil
import logging
from pathlib import Path
from typing import Tuple

from .paths import get_legacy_arcaneum_dir, get_models_dir, get_data_dir

logger = logging.getLogger(__name__)


def check_needs_migration() -> bool:
    """Check if legacy directory exists and needs migration.

    Returns:
        True if ~/.arcaneum/ exists with data that needs migration
    """
    legacy_dir = get_legacy_arcaneum_dir()
    if not legacy_dir.exists():
        return False

    # Check if it has any content that needs migration
    legacy_models = legacy_dir / "models"
    legacy_data = legacy_dir / "data"

    has_models = legacy_models.exists() and any(legacy_models.iterdir())
    has_data = legacy_data.exists() and any(legacy_data.iterdir())

    return has_models or has_data


def migrate_legacy_directory(verbose: bool = False) -> Tuple[bool, str]:
    """Migrate from legacy ~/.arcaneum/ to XDG-compliant structure.

    Copies models cache and moves user data to new locations.

    Args:
        verbose: If True, print detailed progress messages

    Returns:
        Tuple of (success, message)
    """
    legacy_dir = get_legacy_arcaneum_dir()
    if not legacy_dir.exists():
        return True, "No legacy directory found"

    legacy_models = legacy_dir / "models"
    legacy_data = legacy_dir / "data"

    try:
        # Migrate models (copy, not move - cache is re-downloadable)
        if legacy_models.exists() and any(legacy_models.iterdir()):
            new_models = get_models_dir()
            if verbose:
                logger.info(f"Migrating models: {legacy_models} → {new_models}")

            # Copy models directory contents
            for item in legacy_models.iterdir():
                dest = new_models / item.name
                if item.is_dir():
                    if not dest.exists():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                        if verbose:
                            logger.info(f"  Copied {item.name}")
                else:
                    if not dest.exists():
                        shutil.copy2(item, dest)
                        if verbose:
                            logger.info(f"  Copied {item.name}")

        # Migrate data (move, not copy - user data is essential)
        if legacy_data.exists() and any(legacy_data.iterdir()):
            new_data = get_data_dir()
            if verbose:
                logger.info(f"Migrating data: {legacy_data} → {new_data}")

            # Move data directory contents
            for item in legacy_data.iterdir():
                dest = new_data / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                    if verbose:
                        logger.info(f"  Moved {item.name}")

        # Create migration marker with explanation
        readme_path = legacy_dir / "MIGRATED.txt"
        readme_content = f"""Arcaneum Directory Migration
============================

This directory has been migrated to XDG-compliant locations:

Models (cache):
  OLD: {legacy_models}
  NEW: {get_models_dir()}

Data (databases):
  OLD: {legacy_data}
  NEW: {get_data_dir()}

Why the change?
- Follows XDG Base Directory Specification
- Separates cache (re-downloadable) from data (essential)
- Improves compatibility with system tools

You can safely delete this directory (~/.arcaneum/) if it's empty or only contains this file.

For more information, see: https://specifications.freedesktop.org/basedir/latest/
"""
        readme_path.write_text(readme_content)

        return True, f"Migration completed successfully"

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False, f"Migration failed: {e}"


def run_migration_if_needed(verbose: bool = False) -> None:
    """Auto-run migration if legacy directory detected.

    Called on startup to transparently migrate existing installations.

    Args:
        verbose: If True, print detailed progress messages
    """
    if check_needs_migration():
        if verbose:
            logger.info("Legacy directory detected, migrating to XDG-compliant structure...")

        success, message = migrate_legacy_directory(verbose=verbose)

        if success:
            if verbose:
                logger.info(message)
                logger.info("Migration complete! Using new directory structure.")
        else:
            logger.error(f"Migration failed: {message}")
            logger.error("Continuing with new directory structure, but some data may not be available.")
