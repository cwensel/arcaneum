"""Shared multi-corpus search merge helpers (S1 dedup).

Semantic search (Qdrant) and text search (MeiliSearch) both need to:
  - Over-fetch from each corpus so merged top-N is stable when score
    distributions differ across corpora
  - Tolerate a missing corpus (warn + skip) when more than one is requested,
    but error when a single requested corpus is missing
  - Error when every requested corpus is missing

Merge strategies differ (score-sort vs. positional round-robin), so the
strategy-specific code stays in each caller. This module captures the
common skeleton.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple, TypeVar

from .errors import ResourceNotFoundError

T = TypeVar("T")


def per_corpus_limit(corpora: List[str], limit: int, offset: int) -> int:
    """Return per-corpus fetch size.

    Single-corpus search needs no headroom; multi-corpus search over-fetches
    by 2x so the post-merge top-N is stable even when score distributions
    differ across corpora.
    """
    if len(corpora) > 1:
        return (limit + offset) * 2
    return limit + offset


def fetch_from_corpora(
    corpora: List[str],
    fetch_fn: Callable[[str], T],
    is_missing: Callable[[Exception], bool],
    logger: logging.Logger,
    verbose: bool,
) -> Tuple[List[T], List[str]]:
    """Call ``fetch_fn`` for each corpus, handling missing-corpus cases.

    Args:
        corpora: Corpus names to query.
        fetch_fn: Callable(corpus_name) -> corpus-specific result object.
            May raise to signal a missing corpus; ``is_missing`` decides.
        is_missing: Predicate on an exception — True if it means "corpus does
            not exist". Raised exceptions that aren't "missing" propagate.
        logger: Logger for verbose skip warnings.
        verbose: Log skip warnings when True.

    Returns:
        (per_corpus_results, missing_corpora).

        Raises ``ResourceNotFoundError`` if a single requested corpus is
        missing, or if every requested corpus is missing.
    """
    results: List[T] = []
    missing: List[str] = []

    for corpus_name in corpora:
        try:
            results.append(fetch_fn(corpus_name))
        except Exception as exc:
            if not is_missing(exc):
                raise
            missing.append(corpus_name)
            if len(corpora) > 1:
                if verbose:
                    logger.warning(f"Corpus '{corpus_name}' not found, skipping")
            else:
                raise ResourceNotFoundError(f"Corpus '{corpus_name}' not found")

    if missing and len(missing) == len(corpora):
        raise ResourceNotFoundError(
            f"No matching corpora found: {', '.join(corpora)}"
        )

    return results, missing
