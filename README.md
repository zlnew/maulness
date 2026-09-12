# Maulness

> **Maul + Harness:** Personal AI Agent Harness (Antigravity ACP + Gemini API + Discord + systemd)

`maulness` is a CLI‑driven, systemd‑managed agent harness that lets you run AI‑augmented coding tasks from the terminal or Discord. It isolates work on Git worktrees, supports multiple execution profiles, and persists state in a local SQLite DB.

## Features

- **Antigravity ACP** – Executes safe, human‑in‑the‑loop workspace changes via the Google Antigravity binary.
- **Gemini (Flash/Pro)** – Direct LLM calls for fast conversational answers.
- **Discord gateway** – Optional bot that mirrors the CLI surface (`run`, `task`, `status`, …) in a Discord forum channel.
- **Profiles** – Pre‑defined execution contexts (`builder`, `planner`, `reviewer`, …) selectable at runtime.
- **Isolated worktrees** – `--worktree` runs tasks in a temporary Git worktree, protecting the main repo.
- **YOLO mode** – `--yolo` bypasses human confirmation for mutating tools.
- **Systemd daemon** – Background service (`maulness.service`) keeps the agent alive and watches for queued jobs.
- **SQLite persistence** – `maulness.db` stores tasks, sessions, and execution history.

## Installation

```bash
# Create an isolated virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode with dev extras
uv pip install -e "[dev]"
```

The package expects a config directory at `~/.config/maulness`:
- `config.yaml` – optional overrides for workspace root, DB path, execution limits, sandbox mode, etc.
- `.env` or `env` – environment variables (Discord token, Gemini API key, `AGY_CMD`, …).

If no config is present the defaults are:
- Workspace root: `/home/zlnew/www/personal`
- Repo directory: `<workspace_root>/repo`
- DB path: `~/.config/maulness/maulness.db`

## Quick Start

```bash
# Verify installation
maulness --help

# Show system status (daemon, Discord, DB, profiles)
maulness status

# Run a one‑off task against a repository
maulness run expense-tracker "Fix CSV export null date" \
    --profile builder --worktree
```

Use `--yolo` to skip confirmation when you are sure the command is safe.

## CLI Overview

| Command | Description |
|---------|-------------|
| `maulness status` | Summarise daemon, Discord, DB and loaded profiles |
| `maulness run <repo> <prompt>` | Execute a single task with the chosen profile. Options: `--profile`, `--worktree`, `--yolo` |
| `maulness daemon start|stop|restart|log` | Manage the background systemd user service |
| `maulness task list|show <id>` | Inspect queued or completed tasks |
| `maulness session list|show <id>` | View active or historic agent sessions |
| `maulness profile list|activate <name>` | Enumerate and switch execution profiles |
| `maulness discord start|stop|status` | Control the Discord bot gateway |
| `maulness db migrate|reset|vacuum` | Database maintenance commands |
| `maulness soul|memory|user edit` | Open and edit the SOUL/ MEMORY/ USER doctrine files |

All sub‑commands are Typer‑powered and provide `--help` for detailed flags.

## Development & Docker Override

A local Docker compose override lives at `docker/maulness/docker-compose.override.yml`. It runs the harness in the `personal-network` without exposing host ports, using the same configuration files as the host environment.

```yaml
services:
  maulness:
    container_name: maulness-dev
    volumes:
      - ~/.config/maulness:/root/.config/maulness:rw
    # No host ports – interact via the CLI inside the container
```

## Documentation & Specs

The full product requirement document, technical design, and wire contracts are stored in the workspace knowledge base:
- Decision Record: `_memory/decisions/2026-09-09-maulness-agent-harness-acp-discord.md`
- Master Spec: `_memory/plans/2026-09-09-maulness-agent-harness-spec.md`

Keep those files up‑to‑date as the code evolves.

## Contributing

1. Fork the repo and create a feature branch.
2. Run the test suite: `pytest -q` (or `uv run pytest`).
3. Lint with `ruff`/`black`.
4. Submit a PR; Maul will give the final sign‑off before merging.

---

*Maulness is a personal tool – treat the config and DB as private data.*
