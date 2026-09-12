import json
import time
from pathlib import Path
from maulness.core.indexing.repo_map import (
    FileIndex,
    extract_python_symbols,
    extract_golang_symbols,
    extract_js_ts_symbols,
    compute_pagerank,
    RepoMapManager,
)


def test_extract_python_symbols():
    code = """
import os
from pathlib import Path
from typing import Optional

class Calculator:
    \"\"\"A simple calculator utility.\"\"\"
    def __init__(self, base: int = 0):
        self.base = base

    async def add(self, value: int) -> int:
        return self.base + value

def standalone_helper(x: int) -> bool:
    \"\"\"Checks if positive.\"\"\"
    return x > 0
"""
    symbols, imports = extract_python_symbols(code, "calc.py")
    assert "os" in imports
    assert "pathlib" in imports

    sym_names = {s.name: s for s in symbols}
    assert "Calculator" in sym_names
    assert sym_names["Calculator"].kind == "class"
    assert "calculator utility" in sym_names["Calculator"].docstring

    assert "add" in sym_names
    assert sym_names["add"].kind == "method"
    assert sym_names["add"].parent == "Calculator"

    assert "standalone_helper" in sym_names
    assert sym_names["standalone_helper"].kind == "function"
    assert "Checks if positive" in sym_names["standalone_helper"].docstring


def test_extract_golang_symbols():
    code = """package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
}

func (s *Server) Start() error {
    return nil
}

func NewServer(port int) *Server {
    return &Server{port: port}
}
"""
    symbols, imports = extract_golang_symbols(code, "server.go")
    assert "fmt" in imports
    assert "net/http" in imports

    sym_names = {s.name: s for s in symbols}
    assert "Server" in sym_names
    assert sym_names["Server"].kind == "type"

    assert "Start" in sym_names
    assert sym_names["Start"].kind == "method"

    assert "NewServer" in sym_names
    assert sym_names["NewServer"].kind == "function"


def test_extract_js_ts_symbols():
    code = """import { useState } from 'react';
import axios from 'axios';

export interface UserState {
    id: string;
}

export class AuthService {
    login() {}
}

export async function fetchUser(id: string) {
    return null;
}

export const logOut = () => {};
"""
    symbols, imports = extract_js_ts_symbols(code, "auth.ts")
    assert "react" in imports
    assert "axios" in imports

    sym_names = {s.name: s for s in symbols}
    assert "UserState" in sym_names
    assert sym_names["UserState"].kind == "type"

    assert "AuthService" in sym_names
    assert sym_names["AuthService"].kind == "class"

    assert "fetchUser" in sym_names
    assert sym_names["fetchUser"].kind == "function"

    assert "logOut" in sym_names
    assert sym_names["logOut"].kind == "function"


def test_compute_pagerank():
    # File A (models.py) is imported by B (service.py) and C (main.py)
    # File B (service.py) is imported by C (main.py)
    # File C is not imported by anything
    file_indices = {
        "models.py": FileIndex("models.py", "python", [], [], 1.0),
        "service.py": FileIndex("service.py", "python", [], ["models"], 1.0),
        "main.py": FileIndex("main.py", "python", [], ["models", "service"], 1.0),
    }

    ranks = compute_pagerank(file_indices)
    assert len(ranks) == 3
    # models.py should have highest rank because it has the most incoming references
    assert ranks["models.py"] > ranks["service.py"]
    assert ranks["service.py"] > ranks["main.py"]


def test_repo_map_manager_cache_and_outline(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    models_py = src_dir / "models.py"
    models_py.write_text(
        """class User:
    \"\"\"User entity.\"\"\"
    def __init__(self, name: str):
        self.name = name
""",
        encoding="utf-8",
    )

    app_py = src_dir / "app.py"
    app_py.write_text(
        """from src.models import User

def run_app():
    return User("alice")
""",
        encoding="utf-8",
    )

    mgr = RepoMapManager(tmp_path)
    repo_map = mgr.build_repo_map(max_tokens=500)
    assert "src/models.py" in repo_map
    assert "class User" in repo_map

    # Verify cache file exists
    cache_file = tmp_path / ".maulness" / "cache" / "repo_map.json"
    assert cache_file.exists()
    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "symbols" in cached_data
    assert len(cached_data["symbols"]) > 0

    # Test outline
    outline = mgr.get_outline("src/models.py")
    assert "src/models.py Outline" in outline
    assert "class User" in outline

    # Test find_symbol
    results = mgr.find_symbol("User", kind="class")
    assert len(results) >= 1
    assert results[0].name == "User"
    assert results[0].kind == "class"

    # Invalidate cache by modifying a file
    time.sleep(0.01)
    models_py.write_text(
        """class SuperUser:
    def is_admin(self):
        return True
""",
        encoding="utf-8",
    )

    # Rebuild should detect mtime change and update
    new_map = mgr.build_repo_map(max_tokens=500)
    assert "SuperUser" in new_map
