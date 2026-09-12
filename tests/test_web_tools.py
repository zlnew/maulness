import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from maulness.core.tools import (
    execute_tool_call,
    _execute_web_search,
    _execute_fetch_doc_markdown,
    format_lean_tool_breadcrumb,
)


@pytest.mark.asyncio
async def test_execute_fetch_doc_markdown_html():
    sample_html = """
    <html>
        <head>
            <title>Test Documentation Guide</title>
            <style>body { font-size: 16px; }</style>
            <script>console.log("ignore me");</script>
        </head>
        <body>
            <nav><a href="/home">Home</a></nav>
            <h1>Getting Started</h1>
            <p>Welcome to the <b>official</b> documentation. Check out <a href="https://example.com/api">the API</a>.</p>
            <pre><code>def hello():
    print("world")
</code></pre>
            <ul>
                <li>Requirement 1</li>
                <li>Requirement 2</li>
            </ul>
            <footer>Footer notes</footer>
        </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = sample_html

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        md = await _execute_fetch_doc_markdown(
            "https://example.com/docs", max_chars=1000
        )

        assert "# Test Documentation Guide" in md
        assert "# Getting Started" in md
        assert "Welcome to the official documentation" in md
        assert "[the API](https://example.com/api)" in md
        assert "```" in md
        assert 'print("world")' in md
        assert "- Requirement 1" in md
        assert "- Requirement 2" in md
        assert "console.log" not in md
        assert "Footer notes" not in md


@pytest.mark.asyncio
async def test_execute_fetch_doc_markdown_truncation():
    long_text = "A" * 500
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.text = long_text

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await _execute_fetch_doc_markdown(
            "https://example.com/large.txt", max_chars=100
        )
        assert len(res) > 100
        assert "[... content truncated to 100 chars ...]" in res


@pytest.mark.asyncio
async def test_execute_fetch_doc_invalid_url():
    res = await _execute_fetch_doc_markdown("ftp://invalid.com/file")
    assert "Invalid URL scheme" in res


@pytest.mark.asyncio
async def test_execute_web_search_duckduckgo_mock():
    ddg_html = """
    <div class="results">
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpython&rut=1">
            Python is an interpreted programming language.
        </a>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fast&rut=1">
            The AST module provides abstract syntax tree capabilities.
        </a>
    </div>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = ddg_html

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await _execute_web_search("python ast", max_results=2)
        assert 'Web Search Results for "python ast"' in res
        assert "https://example.com/python" in res
        assert "Python is an interpreted" in res


@pytest.mark.asyncio
async def test_execute_web_search_github_fallback():
    # Simulate DDG failure followed by GitHub fallback success
    ddg_fail = MagicMock()
    ddg_fail.status_code = 500
    ddg_fail.text = ""

    gh_success = MagicMock()
    gh_success.status_code = 200
    gh_success.json = lambda: {
        "items": [
            {
                "full_name": "ast-org/parser",
                "html_url": "https://github.com/ast-org/parser",
                "description": "A fast AST parser in Python",
            }
        ]
    }

    async def mock_get(url, *args, **kwargs):
        if "duckduckgo" in url:
            return ddg_fail
        return gh_success

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        res = await _execute_web_search("ast parser", max_results=1)
        assert "ast-org/parser" in res
        assert "https://github.com/ast-org/parser" in res


@pytest.mark.asyncio
async def test_execute_tool_call_dispatch(tmp_path: Path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("class Foo:\n    def bar(self): pass\n", encoding="utf-8")

    # get_outline
    res_outline = await execute_tool_call(
        "get_outline", {"path": "sample.py"}, workspace_path=tmp_path
    )
    assert "sample.py Outline" in res_outline
    assert "class Foo" in res_outline

    # find_symbol
    res_symbol = await execute_tool_call(
        "find_symbol", {"name": "Foo"}, workspace_path=tmp_path
    )
    assert "Foo" in res_symbol
    assert "[class]" in res_symbol


def test_format_lean_tool_breadcrumb():
    bc1 = format_lean_tool_breadcrumb(
        "get_outline", {"path": "src/models.py"}, "Outline output"
    )
    assert "get_outline: src/models.py" in bc1

    bc2 = format_lean_tool_breadcrumb("find_symbol", {"name": "User"}, "Found symbol")
    assert "find_symbol: User" in bc2

    bc3 = format_lean_tool_breadcrumb(
        "web_search", {"query": "pytest fixtures"}, "Results"
    )
    assert "web_search: 'pytest fixtures'" in bc3
