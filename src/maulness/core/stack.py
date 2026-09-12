"""Workspace stack auto-detection engine.

Inspects workspace root markers (pyproject.toml with uv, go.mod, package.json,
docker-compose, etc.) to determine the active stack and provide strictly capped
stack-specific rules (max 3-5 bullet points) and deterministic tool commands.
"""

import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class WorkspaceStack(BaseModel):
    name: str = "generic"
    markers: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    test_command: Optional[str] = None
    format_command: Optional[str] = None
    lint_command: Optional[str] = None


def detect_stack(workspace_path: Path) -> WorkspaceStack:
    """Inspect workspace root markers and return detected stack with rules capped at 5."""
    ws = Path(workspace_path).resolve()
    markers: list[str] = []

    # 1. Python Check
    pyproject = ws / "pyproject.toml"
    requirements = ws / "requirements.txt"
    setup_py = ws / "setup.py"

    if pyproject.exists():
        markers.append("pyproject.toml")
        try:
            content = pyproject.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""

        if "[tool.uv]" in content or "uv." in content:
            rules = [
                "Always run Python tools via `uv run <command>` (e.g. `uv run pytest`, `uv run ruff`).",
                "Manage dependencies via `uv add` and `uv remove` rather than plain pip.",
                "Format code with `uv run ruff format` and check lints with `uv run ruff check`.",
                "Keep imports sorted and type annotations consistent with PEP 484 / PEP 526.",
            ]
            return _finalize_stack(
                name="python-uv",
                markers=markers,
                rules=rules,
                test_cmd="uv run pytest",
                format_cmd="uv run ruff format",
                lint_cmd="uv run ruff check",
                ws=ws,
            )
        else:
            rules = [
                "Run test suites via `pytest` or `python -m unittest`.",
                "Follow PEP 8 formatting conventions across all Python source files.",
                "Keep dependencies pinned in pyproject.toml or requirements.txt.",
                "Type-check modifications using mypy where annotations exist.",
            ]
            return _finalize_stack(
                name="python",
                markers=markers,
                rules=rules,
                test_cmd="pytest",
                format_cmd="ruff format",
                lint_cmd="ruff check",
                ws=ws,
            )

    if requirements.exists() or setup_py.exists():
        if requirements.exists():
            markers.append("requirements.txt")
        if setup_py.exists():
            markers.append("setup.py")
        rules = [
            "Run test suites via `pytest` or `python -m unittest`.",
            "Follow PEP 8 formatting conventions across all Python source files.",
            "Keep dependencies recorded in requirements.txt.",
        ]
        return _finalize_stack(
            name="python",
            markers=markers,
            rules=rules,
            test_cmd="pytest",
            format_cmd="ruff format",
            lint_cmd="ruff check",
            ws=ws,
        )

    # 2. Golang Check
    go_mod = ws / "go.mod"
    if go_mod.exists():
        markers.append("go.mod")
        rules = [
            "Run package test suites with `go test ./...`.",
            "Format Go source files deterministically with `gofmt -w .`.",
            "Run `go mod tidy` after modifying imported packages.",
            "Verify correctness and static analysis via `go vet ./...`.",
        ]
        return _finalize_stack(
            name="golang",
            markers=markers,
            rules=rules,
            test_cmd="go test ./...",
            format_cmd="gofmt -w .",
            lint_cmd="go vet ./...",
            ws=ws,
        )

    # 3. Node / Web Check
    pkg_json = ws / "package.json"
    if pkg_json.exists():
        markers.append("package.json")
        pm = "npm"
        if (ws / "pnpm-lock.yaml").exists():
            pm = "pnpm"
            markers.append("pnpm-lock.yaml")
        elif (ws / "bun.lockb").exists() or (ws / "bun.lock").exists():
            pm = "bun"
            markers.append("bun.lock")
        elif (ws / "yarn.lock").exists():
            pm = "yarn"
            markers.append("yarn.lock")
        elif (ws / "package-lock.json").exists():
            markers.append("package-lock.json")

        sub_name = "node"
        try:
            raw_pkg = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
            deps = {
                **raw_pkg.get("dependencies", {}),
                **raw_pkg.get("devDependencies", {}),
            }
            if "vue" in deps:
                sub_name = "vue"
            elif "react" in deps or "next" in deps:
                sub_name = "react"
        except Exception:
            pass

        rules = [
            f"Execute project scripts and dependencies using `{pm} run <script>`.",
            f"Run automated tests using `{pm} test`.",
            "Maintain consistent styling using prettier/eslint before finishing tasks.",
            "Avoid introducing unnecessary heavy npm packages when standard web APIs suffice.",
        ]
        return _finalize_stack(
            name=sub_name,
            markers=markers,
            rules=rules,
            test_cmd=f"{pm} test",
            format_cmd=f"{pm} run format",
            lint_cmd=f"{pm} run lint",
            ws=ws,
        )

    # 4. Rust Check
    cargo_toml = ws / "Cargo.toml"
    if cargo_toml.exists():
        markers.append("Cargo.toml")
        rules = [
            "Run automated tests via `cargo test`.",
            "Format Rust code with `cargo fmt`.",
            "Run compiler lint checks with `cargo clippy`.",
        ]
        return _finalize_stack(
            name="rust",
            markers=markers,
            rules=rules,
            test_cmd="cargo test",
            format_cmd="cargo fmt",
            lint_cmd="cargo clippy",
            ws=ws,
        )

    # 5. Docker Check
    docker_markers = (
        [f.name for f in ws.glob("docker-compose*.yml")]
        + [f.name for f in ws.glob("docker-compose*.yaml")]
        + [
            f.name
            for f in [ws / "compose.yaml", ws / "compose.yml", ws / "Dockerfile"]
            if f.exists()
        ]
    )
    if docker_markers:
        markers.extend(docker_markers[:2])
        rules = [
            "Manage container lifecycles via `docker compose`.",
            "Use internal service-name addressing on isolated networks.",
            "Never bind ports to host unless explicitly needed for external access.",
        ]
        return _finalize_stack(
            name="docker",
            markers=markers,
            rules=rules,
            test_cmd="docker compose ps",
            format_cmd=None,
            lint_cmd=None,
            ws=ws,
        )

    # 6. Generic Fallback
    return WorkspaceStack(
        name="generic",
        markers=[],
        rules=[
            "Inspect workspace files and structure before proposing changes.",
            "Run workspace test suites to verify modifications before completing tasks.",
            "Keep changes minimal and self-contained.",
        ],
    )


def _finalize_stack(
    name: str,
    markers: list[str],
    rules: list[str],
    test_cmd: Optional[str],
    format_cmd: Optional[str],
    lint_cmd: Optional[str],
    ws: Path,
) -> WorkspaceStack:
    """Helper to check for auxiliary docker files and cap rules at 5."""
    has_docker = (
        (ws / "docker-compose.yml").exists()
        or (ws / "docker-compose.yaml").exists()
        or (ws / "compose.yaml").exists()
        or (ws / "Dockerfile").exists()
    )
    if has_docker and name != "docker" and len(rules) < 5:
        markers.append("docker")
        rules.append(
            "Shared infra services run in Docker; communicate via container service names."
        )

    return WorkspaceStack(
        name=name,
        markers=markers,
        rules=rules[:5],  # Strict cap of 5 rules
        test_command=test_cmd,
        format_command=format_cmd,
        lint_command=lint_cmd,
    )


def get_stack_doctrine(workspace_path: Path) -> str:
    """Return formatted markdown doctrine block for prompt injection."""
    stack = detect_stack(workspace_path)
    if not stack.rules:
        return ""
    bullet_points = "\n".join(f"- {r}" for r in stack.rules)
    return f"## Workspace Stack Rules ({stack.name})\n{bullet_points}"
