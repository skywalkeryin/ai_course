# AI Course

This project now uses [uv](https://github.com/astral-sh/uv) to manage the virtual environment and dependencies instead of Poetry.

## Getting Started

1. Install uv if you have not already:
   ```powershell
   pip install uv
   ```
2. Create or update the local environment:
   ```powershell
   uv sync
   ```
   This will create `.venv/` (if missing) and install the dependencies listed in `pyproject.toml`.
3. Run commands inside the environment with `uv run`, for example:
   ```powershell
   uv run python -m ipykernel install --user --name ai-course
   ```

## Useful Commands

- `uv venv` – create the virtual environment explicitly.
- `uv pip list` – inspect installed packages.
- `uv add PACKAGE` / `uv remove PACKAGE` – modify dependencies and update the lock file.

The Poetry files (`poetry.lock`, `[tool.poetry]` sections) have been removed. Use uv commands for all future dependency and environment management tasks.

