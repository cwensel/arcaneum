"""Centralized logging configuration for CLI commands.

Provides three logging levels:
- Default: Clean output, suppress library warnings
- Verbose: Show user progress/stats, suppress library warnings
- Debug: Show everything including library internals
"""

import logging
import os
import warnings


def setup_logging_default():
    """Default logging: Clean output, only critical errors.

    Suppresses:
    - HuggingFace transformers model initialization warnings
    - HuggingFace hub connection messages
    - FastEmbed download progress (keeps progress bars)
    - HTTP client verbose output
    - Qdrant client verbose output

    Shows:
    - User-facing error messages
    - Critical failures
    """
    # Set tokenizers parallelism warning (must be set before import)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # Suppress transformers warnings via environment (before import)
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    # Root logger: Only warnings and errors
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    # Suppress library INFO logs
    logging.getLogger('arcaneum').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    logging.getLogger('qdrant_client').setLevel(logging.ERROR)
    logging.getLogger('fastembed').setLevel(logging.WARNING)  # Keep progress bars

    # Suppress HuggingFace transformers warnings (Phase 2 RDR-013)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.utils.generic').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

    # Use transformers' own logging API (most reliable method)
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass

    # Suppress Python warnings from transformers (not using logging module)
    warnings.filterwarnings('ignore', message='Some weights of.*were not initialized.*')
    warnings.filterwarnings('ignore', message='.*BertSdpaSelfAttention.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
    # Suppress optimum library warnings from custom model code (jina models)
    warnings.filterwarnings('ignore', message='.*optimum is not installed.*')


def setup_logging_verbose():
    """Verbose logging: Show user-relevant progress and stats.

    Shows:
    - User progress information (file counts, chunk counts)
    - Performance statistics
    - Configuration details

    Suppresses:
    - Library DEBUG logs
    - HuggingFace transformers warnings
    - HTTP client verbose output
    """
    # Set tokenizers parallelism warning
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # Suppress transformers warnings via environment (before import)
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    # Root logger: INFO level for user messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Show arcaneum INFO logs (progress, stats)
    logging.getLogger('arcaneum').setLevel(logging.INFO)

    # Suppress DEBUG from libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.INFO)
    logging.getLogger('fastembed').setLevel(logging.WARNING)

    # Suppress HuggingFace transformers warnings even in verbose mode
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.utils.generic').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

    # Use transformers' own logging API (most reliable method)
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass

    # Suppress Python warnings from transformers (not using logging module)
    warnings.filterwarnings('ignore', message='Some weights of.*were not initialized.*')
    warnings.filterwarnings('ignore', message='.*BertSdpaSelfAttention.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
    # Suppress optimum library warnings from custom model code (jina models)
    warnings.filterwarnings('ignore', message='.*optimum is not installed.*')


def setup_logging_debug():
    """Debug logging: Show everything including library internals.

    Shows:
    - All user messages
    - Library DEBUG logs
    - HuggingFace transformers warnings
    - HTTP client details
    - Full stack traces

    Use for:
    - Debugging model loading issues
    - Troubleshooting GPU acceleration
    - Investigating performance problems
    """
    # Set tokenizers parallelism warning
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # Root logger: DEBUG level
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(name)s - %(levelname)s: %(message)s'
    )

    # Allow all loggers to show their natural levels
    # (No suppression in debug mode)
