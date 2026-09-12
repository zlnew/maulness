"""Tree-sitter and AST-based Codebase Indexer and PageRank Repo Map Generator.

Parses workspace source files (Python, Go, JS/TS) into AST symbols (classes,
methods, exported functions, docstrings, and imports). Computes PageRank over
import dependencies to prioritize architecturally critical files and generates
a dense ~1,000 token Repo Map cached in .maulness/cache/repo_map.json.
"""

import ast
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("maulness.indexing")

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".maulness",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".tox",
    "target",
}

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".go": "golang",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


@dataclass
class SymbolInfo:
    name: str
    kind: str  # "class", "function", "method", "type", "variable"
    file_path: str  # relative path
    line_number: int
    signature: str = ""
    docstring: str = ""
    parent: Optional[str] = None


@dataclass
class FileIndex:
    file_path: str
    language: str
    symbols: list[SymbolInfo]
    imports: list[str]  # imported modules or file paths
    mtime: float


def extract_python_symbols(
    content: str, rel_path: str
) -> tuple[list[SymbolInfo], list[str]]:
    """Extract classes, methods, functions, docstrings, and imports using Python AST."""
    symbols: list[SymbolInfo] = []
    imports: list[str] = []

    try:
        tree = ast.parse(content)
    except Exception:
        return _extract_python_regex(content, rel_path)

    for node in tree.body:
        # 1. Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append(mod)
            for alias in node.names:
                if mod:
                    imports.append(f"{mod}.{alias.name}")

        # 2. Classes
        elif isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(f"{getattr(b.value, 'id', '')}.{b.attr}")
            base_str = f"({', '.join(bases)})" if bases else ""
            doc = ast.get_docstring(node) or ""
            first_doc = doc.strip().splitlines()[0] if doc.strip() else ""

            symbols.append(
                SymbolInfo(
                    name=node.name,
                    kind="class",
                    file_path=rel_path,
                    line_number=node.lineno,
                    signature=f"class {node.name}{base_str}:",
                    docstring=first_doc,
                )
            )

            # Class methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = (
                        "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    )
                    m_doc = ast.get_docstring(item) or ""
                    m_first_doc = m_doc.strip().splitlines()[0] if m_doc.strip() else ""
                    args = [a.arg for a in item.args.args]
                    arg_str = ", ".join(args)
                    symbols.append(
                        SymbolInfo(
                            name=item.name,
                            kind="method",
                            file_path=rel_path,
                            line_number=item.lineno,
                            signature=f"{prefix} {item.name}({arg_str})",
                            docstring=m_first_doc,
                            parent=node.name,
                        )
                    )

        # 3. Top-level functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            doc = ast.get_docstring(node) or ""
            first_doc = doc.strip().splitlines()[0] if doc.strip() else ""
            args = [a.arg for a in node.args.args]
            arg_str = ", ".join(args)
            symbols.append(
                SymbolInfo(
                    name=node.name,
                    kind="function",
                    file_path=rel_path,
                    line_number=node.lineno,
                    signature=f"{prefix} {node.name}({arg_str})",
                    docstring=first_doc,
                )
            )

    return symbols, imports


def _extract_python_regex(
    content: str, rel_path: str
) -> tuple[list[SymbolInfo], list[str]]:
    symbols: list[SymbolInfo] = []
    imports: list[str] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        m_cls = re.match(r"^class\s+([a-zA-Z0-9_]+)(?:\((.*?)\))?:", line)
        if m_cls:
            symbols.append(
                SymbolInfo(
                    name=m_cls.group(1),
                    kind="class",
                    file_path=rel_path,
                    line_number=idx,
                    signature=line.strip(),
                )
            )
            continue
        m_fn = re.match(r"^(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", line)
        if m_fn:
            symbols.append(
                SymbolInfo(
                    name=m_fn.group(1),
                    kind="function",
                    file_path=rel_path,
                    line_number=idx,
                    signature=line.strip(),
                )
            )
            continue
        m_imp = re.match(
            r"^(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))", line
        )
        if m_imp:
            imports.append(m_imp.group(1) or m_imp.group(2))
    return symbols, imports


def extract_golang_symbols(
    content: str, rel_path: str
) -> tuple[list[SymbolInfo], list[str]]:
    """Extract Go types, methods, functions, and imports."""
    symbols: list[SymbolInfo] = []
    imports: list[str] = []

    lines = content.splitlines()
    in_import_block = False

    for idx, line in enumerate(lines, start=1):
        clean = line.strip()
        if clean.startswith("import ("):
            in_import_block = True
            continue
        if in_import_block:
            if clean.startswith(")"):
                in_import_block = False
                continue
            m = re.search(r'"([^"]+)"', clean)
            if m:
                imports.append(m.group(1))
            continue
        if clean.startswith("import "):
            m = re.search(r'"([^"]+)"', clean)
            if m:
                imports.append(m.group(1))
            continue

        # Types / Structs / Interfaces
        m_type = re.match(r"^type\s+([a-zA-Z0-9_]+)\s+(struct|interface)", clean)
        if m_type:
            symbols.append(
                SymbolInfo(
                    name=m_type.group(1),
                    kind="type",
                    file_path=rel_path,
                    line_number=idx,
                    signature=f"type {m_type.group(1)} {m_type.group(2)}",
                )
            )
            continue

        # Methods: func (r *Recv) Name(...) ...
        m_method = re.match(
            r"^func\s+\((?:[^)]+)\)\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", clean
        )
        if m_method:
            symbols.append(
                SymbolInfo(
                    name=m_method.group(1),
                    kind="method",
                    file_path=rel_path,
                    line_number=idx,
                    signature=clean.split("{")[0].strip(),
                )
            )
            continue

        # Functions: func Name(...) ...
        m_func = re.match(r"^func\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", clean)
        if m_func:
            symbols.append(
                SymbolInfo(
                    name=m_func.group(1),
                    kind="function",
                    file_path=rel_path,
                    line_number=idx,
                    signature=clean.split("{")[0].strip(),
                )
            )

    return symbols, imports


def extract_js_ts_symbols(
    content: str, rel_path: str
) -> tuple[list[SymbolInfo], list[str]]:
    """Extract JavaScript/TypeScript classes, interfaces, exported functions, and imports."""
    symbols: list[SymbolInfo] = []
    imports: list[str] = []

    for idx, line in enumerate(content.splitlines(), start=1):
        clean = line.strip()
        m_imp = re.search(r"from\s+['\"]([^'\"]+)['\"]", clean)
        if m_imp:
            imports.append(m_imp.group(1))

        # Classes
        m_cls = re.match(
            r"^(?:export\s+)?(?:default\s+)?class\s+([a-zA-Z0-9_]+)", clean
        )
        if m_cls:
            symbols.append(
                SymbolInfo(
                    name=m_cls.group(1),
                    kind="class",
                    file_path=rel_path,
                    line_number=idx,
                    signature=clean.split("{")[0].strip(),
                )
            )
            continue

        # Interfaces / Types (TS)
        m_type = re.match(r"^(?:export\s+)?(?:interface|type)\s+([a-zA-Z0-9_]+)", clean)
        if m_type:
            symbols.append(
                SymbolInfo(
                    name=m_type.group(1),
                    kind="type",
                    file_path=rel_path,
                    line_number=idx,
                    signature=clean.split("{")[0].strip(),
                )
            )
            continue

        # Functions
        m_fn = re.match(
            r"^(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)\s*\(", clean
        )
        if m_fn:
            symbols.append(
                SymbolInfo(
                    name=m_fn.group(1),
                    kind="function",
                    file_path=rel_path,
                    line_number=idx,
                    signature=clean.split("{")[0].strip(),
                )
            )
            continue

        # Exported arrow functions: export const name = ...
        m_arrow = re.match(
            r"^(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
            clean,
        )
        if m_arrow:
            symbols.append(
                SymbolInfo(
                    name=m_arrow.group(1),
                    kind="function",
                    file_path=rel_path,
                    line_number=idx,
                    signature=f"const {m_arrow.group(1)} = (...) =>",
                )
            )

    return symbols, imports


def parse_source_file(file_path: Path, workspace_root: Path) -> Optional[FileIndex]:
    """Parse a single source file into a FileIndex structure."""
    ext = file_path.suffix.lower()
    lang = SUPPORTED_EXTENSIONS.get(ext)
    if not lang:
        return None

    try:
        rel_path = str(file_path.relative_to(workspace_root))
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        mtime = file_path.stat().st_mtime

        if lang == "python":
            symbols, imports = extract_python_symbols(content, rel_path)
        elif lang == "golang":
            symbols, imports = extract_golang_symbols(content, rel_path)
        elif lang in ("javascript", "typescript"):
            symbols, imports = extract_js_ts_symbols(content, rel_path)
        else:
            return None

        return FileIndex(
            file_path=rel_path,
            language=lang,
            symbols=symbols,
            imports=imports,
            mtime=mtime,
        )
    except Exception as e:
        logger.debug("Failed parsing file %s: %s", file_path, e)
        return None


def compute_pagerank(
    file_indices: dict[str, FileIndex],
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> dict[str, float]:
    """Compute PageRank over file dependencies.

    Files that are imported/depended on by many other files receive higher ranks.
    """
    nodes = list(file_indices.keys())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    in_links: dict[str, set[str]] = {node: set() for node in nodes}
    out_degree: dict[str, int] = {node: 0 for node in nodes}

    for src_path, findex in file_indices.items():
        for imp in findex.imports:
            imp_normalized = imp.replace(".", "/").replace("\\", "/")
            for target_path in nodes:
                if target_path == src_path:
                    continue
                target_stem = Path(target_path).stem
                if imp_normalized.endswith(target_stem) or target_path.startswith(
                    imp_normalized
                ):
                    in_links[target_path].add(src_path)
                    out_degree[src_path] += 1
                    break

    scores = {node: 1.0 / n for node in nodes}

    for _ in range(max_iter):
        next_scores: dict[str, float] = {}
        dangling_sum = sum(scores[node] for node in nodes if out_degree[node] == 0)

        for u in nodes:
            rank_from_in = sum(
                scores[v] / out_degree[v] for v in in_links[u] if out_degree[v] > 0
            )
            next_scores[u] = (1.0 - damping) / n + damping * (
                rank_from_in + dangling_sum / n
            )

        diff = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if diff < tol:
            break

    return scores


class RepoMapManager:
    """Generates, formats, and caches PageRank repo maps and AST symbol indexes."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path).resolve()
        self.cache_dir = self.workspace_path / ".maulness" / "cache"
        self.cache_file = self.cache_dir / "repo_map.json"

    def scan_files(self) -> dict[str, FileIndex]:
        """Scan all supported source files in the workspace (excluding vendor/caches)."""
        file_indices: dict[str, FileIndex] = {}

        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [
                d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")
            ]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    fp = Path(root) / file
                    findex = parse_source_file(fp, self.workspace_path)
                    if findex:
                        file_indices[findex.file_path] = findex

        return file_indices

    def build_repo_map(self, max_tokens: int = 1000, force: bool = False) -> str:
        """Generate dense ~1,000-token repo map, caching results in .maulness/cache/repo_map.json."""
        if not force and self.cache_file.exists():
            try:
                cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
                file_mtimes = cached.get("file_mtimes", {})
                is_valid = True
                for rel_path, cached_mtime in file_mtimes.items():
                    fp = self.workspace_path / rel_path
                    if not fp.exists() or fp.stat().st_mtime != cached_mtime:
                        is_valid = False
                        break
                if is_valid and "repo_map" in cached:
                    return cached["repo_map"]
            except Exception:
                pass

        file_indices = self.scan_files()
        if not file_indices:
            return "(No indexable source code files found in workspace)"

        ranks = compute_pagerank(file_indices)
        sorted_files = sorted(
            file_indices.keys(), key=lambda f: ranks.get(f, 0.0), reverse=True
        )

        char_budget = max_tokens * 4
        lines: list[str] = [
            "# Repository Structural Map (ranked by PageRank importance)",
            "",
        ]
        current_len = sum(len(line) + 1 for line in lines)
        all_symbols_dump: list[dict[str, Any]] = []

        for rel_path in sorted_files:
            findex = file_indices[rel_path]
            for sym in findex.symbols:
                all_symbols_dump.append(asdict(sym))

        for rel_path in sorted_files:
            findex = file_indices[rel_path]
            if not findex.symbols:
                continue

            file_block = [f"{rel_path}:"]
            for sym in findex.symbols:
                if sym.kind == "class":
                    doc_snippet = f"  # {sym.docstring}" if sym.docstring else ""
                    file_block.append(f"  {sym.signature}{doc_snippet}")
                elif sym.kind == "method":
                    file_block.append(f"    {sym.signature}")
                elif sym.kind in ("function", "type"):
                    doc_snippet = f"  # {sym.docstring}" if sym.docstring else ""
                    file_block.append(f"  {sym.signature}{doc_snippet}")

            block_text = "\n".join(file_block) + "\n\n"
            if current_len + len(block_text) > char_budget:
                remaining_files = len(sorted_files) - sorted_files.index(rel_path)
                lines.append(
                    f"[... {remaining_files} additional modules indexed in cache ...]"
                )
                break

            lines.append("\n".join(file_block))
            current_len += len(block_text)

        repo_map_str = "\n".join(lines).strip()

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "version": 1,
                "generated_at": time.time(),
                "file_mtimes": {
                    rel: findex.mtime for rel, findex in file_indices.items()
                },
                "repo_map": repo_map_str,
                "symbols": all_symbols_dump,
            }
            self.cache_file.write_text(
                json.dumps(cache_payload, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Failed writing repo_map cache: %s", e)

        return repo_map_str

    def get_outline(self, rel_path: str) -> str:
        """Return concise symbol outline of a specific file without implementation bodies."""
        target = Path(rel_path)
        if not target.is_absolute():
            target = (self.workspace_path / target).resolve()
        else:
            target = target.resolve()

        if not target.exists():
            return f"Error: File '{rel_path}' does not exist."
        if not target.is_file():
            return f"Error: '{rel_path}' is not a regular file."

        findex = parse_source_file(target, self.workspace_path)
        if not findex or not findex.symbols:
            return f"[{target.name} Outline]\n(No structural symbols detected or file format not parsed)"

        lines = [f"[{findex.file_path} Outline ({len(findex.symbols)} symbols)]"]
        for sym in findex.symbols:
            if sym.kind == "class":
                doc = f' -- "{sym.docstring}"' if sym.docstring else ""
                lines.append(f"L{sym.line_number}: {sym.signature}{doc}")
            elif sym.kind == "method":
                lines.append(f"  L{sym.line_number}: {sym.signature}")
            else:
                doc = f' -- "{sym.docstring}"' if sym.docstring else ""
                lines.append(f"L{sym.line_number}: {sym.signature}{doc}")

        return "\n".join(lines)

    def find_symbol(
        self,
        name: str,
        kind: Optional[str] = None,
        file_filter: Optional[str] = None,
    ) -> list[SymbolInfo]:
        """Find exact or prefix-matching symbols across indexed workspace files."""
        self.build_repo_map()

        symbols: list[SymbolInfo] = []
        if self.cache_file.exists():
            try:
                cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
                for s in cached.get("symbols", []):
                    s_name = s.get("name", "")
                    s_kind = s.get("kind", "")
                    s_file = s.get("file_path", "")

                    if name.lower() not in s_name.lower():
                        continue
                    if (
                        kind
                        and kind.lower() != "all"
                        and s_kind.lower() != kind.lower()
                    ):
                        continue
                    if file_filter and file_filter.lower() not in s_file.lower():
                        continue

                    symbols.append(SymbolInfo(**s))
            except Exception:
                pass

        return symbols


def get_repo_map(workspace_path: Path, max_tokens: int = 1000) -> str:
    mgr = RepoMapManager(workspace_path)
    return mgr.build_repo_map(max_tokens=max_tokens)


def get_outline(workspace_path: Path, rel_path: str) -> str:
    mgr = RepoMapManager(workspace_path)
    return mgr.get_outline(rel_path)


def find_symbol(
    workspace_path: Path,
    name: str,
    kind: Optional[str] = None,
    file_filter: Optional[str] = None,
) -> list[SymbolInfo]:
    mgr = RepoMapManager(workspace_path)
    return mgr.find_symbol(name, kind, file_filter)
