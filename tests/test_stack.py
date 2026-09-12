from pathlib import Path
from maulness.core.stack import detect_stack, get_stack_doctrine


def test_detect_stack_python_uv(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'demo'\n\n[tool.uv]\ndev-dependencies = []\n",
        encoding="utf-8",
    )

    stack = detect_stack(tmp_path)
    assert stack.name == "python-uv"
    assert "pyproject.toml" in stack.markers
    assert stack.test_command == "uv run pytest"
    assert stack.format_command == "uv run ruff format"
    assert len(stack.rules) <= 5
    assert any("uv run" in r for r in stack.rules)

    doc = get_stack_doctrine(tmp_path)
    assert "## Workspace Stack Rules (python-uv)" in doc


def test_detect_stack_standard_python(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'plain-python'\n", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert stack.name == "python"
    assert stack.test_command == "pytest"
    assert len(stack.rules) <= 5


def test_detect_stack_requirements_txt(tmp_path: Path):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("flask>=2.0\nrequests\n", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert stack.name == "python"
    assert "requirements.txt" in stack.markers


def test_detect_stack_golang(tmp_path: Path):
    go_mod = tmp_path / "go.mod"
    go_mod.write_text("module github.com/user/demo\n\ngo 1.22\n", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert stack.name == "golang"
    assert "go.mod" in stack.markers
    assert stack.test_command == "go test ./..."
    assert stack.format_command == "gofmt -w ."
    assert len(stack.rules) <= 5


def test_detect_stack_node_vue(tmp_path: Path):
    pkg = tmp_path / "package.json"
    pkg.write_text(
        '{"name": "web-app", "dependencies": {"vue": "^3.4.0"}}', encoding="utf-8"
    )
    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert stack.name == "vue"
    assert "package.json" in stack.markers
    assert "pnpm-lock.yaml" in stack.markers
    assert "pnpm" in stack.test_command
    assert len(stack.rules) <= 5


def test_detect_stack_rust(tmp_path: Path):
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        '[package]\nname = "myrust"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    stack = detect_stack(tmp_path)
    assert stack.name == "rust"
    assert "Cargo.toml" in stack.markers
    assert stack.test_command == "cargo test"
    assert len(stack.rules) <= 5


def test_detect_stack_docker(tmp_path: Path):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "version: '3.8'\nservices:\n  web:\n    image: nginx\n", encoding="utf-8"
    )

    stack = detect_stack(tmp_path)
    assert stack.name == "docker"
    assert "docker-compose.yml" in stack.markers
    assert len(stack.rules) <= 5


def test_detect_stack_docker_combined(tmp_path: Path):
    # Coexisting python and docker
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'app'\n", encoding="utf-8"
    )
    (tmp_path / "docker-compose.yml").write_text("services: {}", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert stack.name == "python"
    assert "docker" in stack.markers
    assert len(stack.rules) <= 5  # Strict cap
