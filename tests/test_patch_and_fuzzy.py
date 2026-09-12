import pytest
from pathlib import Path
from maulness.core.tools import _execute_replace_file_content, _execute_patch_file


def test_replace_file_content_exact(tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    res = _execute_replace_file_content(f, "return 'world'", "return 'maulness'")
    assert "Successfully replaced 1 occurrence(s)" in res
    assert "return 'maulness'" in f.read_text(encoding="utf-8")


def test_replace_file_content_fuzzy_indentation(tmp_path: Path):
    f = tmp_path / "fuzzy.py"
    # Target file has 4-space indent
    f.write_text("class Agent:\n    def run(self):\n        start()\n        execute()\n", encoding="utf-8")

    # LLM provided slightly misindented block (e.g. 2 spaces)
    target_block = "  def run(self):\n    start()\n    execute()"
    repl_block = "    def run(self):\n        start()\n        execute_upgraded()\n"

    res = _execute_replace_file_content(f, target_block, repl_block)
    assert "via fuzzy matching" in res
    assert "execute_upgraded()" in f.read_text(encoding="utf-8")


def test_patch_file_unified_diff(tmp_path: Path):
    f = tmp_path / "patchable.txt"
    f.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    patch_diff = """--- a/patchable.txt
+++ b/patchable.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+line two (patched)
 line 3
"""
    res = _execute_patch_file(f, patch_diff)
    assert "Successfully applied" in res
    assert "line two (patched)" in f.read_text(encoding="utf-8")
