# Maulness

> **Maul + Harness:** Personal AI Agent Harness (Antigravity ACP + Gemini API + Discord + systemd)

`maulness` is an executive remote control for workstation coding tasks, bridging mobile Discord interactions with local Google Antigravity execution via the Agent Client Protocol (ACP).

## Architecture

- **Execution Engines:**
  - **Antigravity via ACP:** Local workspace modifications, AST tools, shell terminal execution with Human-in-the-Loop approvals.
  - **Gemini Flash / Pro (Direct API):** Instant sub-second conversational answers and prompt triage.
- **Interfaces:**
  - **Discord Forum (`#workbench`):** Visual task cards using native Discord tags (`Direct`, `Pipeline`, `Building`, `Done`).
  - **CLI (`maulness`):** Hermes-style operational CLI (`run`, `task`, `status`, `daemon`).
- **Daemon:** Managed 24/7 via `systemd` user service with auto-restart on crash.
- **Persistence:** Local SQLite database (`maulness.db`).

## Quick Start

```bash
# Setup virtual environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Check status
maulness status

# Direct run against a local repository
maulness run expense-tracker "Fix CSV export null date"
```

## Specification

Full PRD, Tech Spec, and Protocol Wire Contracts are documented in the workspace knowledge base:
- Decision Record: `_memory/decisions/2026-09-09-maulness-agent-harness-acp-discord.md`
- Master Spec: `_memory/plans/2026-09-09-maulness-agent-harness-spec.md`
