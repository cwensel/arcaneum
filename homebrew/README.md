# homebrew-arcaneum

Homebrew tap for [Arcaneum](https://github.com/cwensel/arcaneum) — CLI tools for
semantic and full-text search.

## Install

```bash
brew tap cwensel/arcaneum https://github.com/cwensel/homebrew-arcaneum
brew install arcaneum
```

## Update

```bash
brew update
brew upgrade arcaneum
```

## Requirements

- macOS (Apple Silicon or Intel) or Linux
- Docker (for Qdrant and MeiliSearch services)

## Notes

This formula installs Arcaneum into a Python virtualenv managed by Homebrew.
Embedding models (~1-2GB) are downloaded on first use and cached in
`~/.cache/arcaneum/models/`.

The formula is auto-updated on each
[Arcaneum release](https://github.com/cwensel/arcaneum/releases).
