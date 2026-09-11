import pytest
from pathlib import Path
from maulness.core.models import ApprovalRequestEvent
from maulness.core.tools import execute_tool_call, resolve_path, TOOL_DEFINITIONS


def test_tool_definitions():
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "run_command" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "list_dir" in names
    assert "git_status" in names


@pytest.mark.asyncio
async def test_execute_read_and_write_file(tmp_path: Path):
    test_file = tmp_path / "hello.txt"

    # Write file with yolo=True
    res = await execute_tool_call(
        name="write_file",
        args={"path": str(test_file), "content": "Hello world from test"},
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Successfully wrote" in res
    assert test_file.read_text() == "Hello world from test"

    # Read file
    read_res = await execute_tool_call(
        name="read_file",
        args={"path": "hello.txt"},
        workspace_path=tmp_path,
    )
    assert read_res == "Hello world from test"

    # Read non-existent file
    err_res = await execute_tool_call(
        name="read_file",
        args={"path": "missing.txt"},
        workspace_path=tmp_path,
    )
    assert "Error: File" in err_res


@pytest.mark.asyncio
async def test_execute_list_dir(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "test.txt").write_text("content")

    res = await execute_tool_call(
        name="list_dir",
        args={"path": str(tmp_path)},
        workspace_path=tmp_path,
    )
    assert "[dir] subdir" in res
    assert "[file] test.txt" in res


@pytest.mark.asyncio
async def test_execute_run_command_approval_rejected(tmp_path: Path):
    async def reject_approval(req: ApprovalRequestEvent) -> bool:
        assert req.tool_name == "run_command"
        return False

    res = await execute_tool_call(
        name="run_command",
        args={"command": "echo 'dangerous'"},
        workspace_path=tmp_path,
        on_approval=reject_approval,
        yolo=False,
    )
    assert "Execution cancelled: User rejected command" in res


@pytest.mark.asyncio
async def test_execute_run_command_yolo(tmp_path: Path):
    res = await execute_tool_call(
        name="run_command",
        args={"command": "echo 'all good'"},
        workspace_path=tmp_path,
        yolo=True,
    )
    assert res == "all good"


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    res = await execute_tool_call(
        name="non_existent_tool",
        args={},
    )
    assert "Error: Unknown tool" in res


def test_format_lean_tool_breadcrumb():
    from maulness.core.tools import format_lean_tool_breadcrumb

    # Single-line output
    b1 = format_lean_tool_breadcrumb("run_command", {"command": "uname -srm"}, "Linux 6.18.48-1-cachyos-lts x86_64")
    assert "> **`run_command: uname -srm`** -> `Linux 6.18.48-1-cachyos-lts x86_64`" in b1

    # Multiline output
    b2 = format_lean_tool_breadcrumb("git_status", {"repo_path": "/tmp"}, "M file1.py\n?? file2.py")
    assert "> **`git_status: /tmp`**:" in b2
    assert "M file1.py" in b2


def test_detect_simulated_tool_call():
    from maulness.core.tools import detect_simulated_tool_call

    # Detected cases
    sim1 = detect_simulated_tool_call("> ⚡ **`list_dir: yodu`** ➔\n```\n[dir] app\n```")
    assert sim1 == ("list_dir", {"path": "yodu"})

    sim2 = detect_simulated_tool_call("> **`run_command: ls -la /home/zlnew/www/yodu`**:\n```\ntotal 44\n```")
    assert sim2 == ("run_command", {"command": "ls -la /home/zlnew/www/yodu"})

    sim3 = detect_simulated_tool_call("> **`git_status`** -> clean")
    assert sim3 == ("git_status", {"repo_path": ""})

    # Non-simulated regular text
    assert detect_simulated_tool_call("Here is what you need to know about git_status.") is None
    assert detect_simulated_tool_call("") is None


def test_clean_history_message():
    from maulness.core.tools import clean_history_message

    # Message with tool breadcrumb and final narrative
    msg_with_narrative = (
        "> **`run_command: date`** -> `Jum 11 Sep 2026 07:38:49 WIB`\n\n"
        "07:38:49 WIB (Jum, 11 Sep 2026)"
    )
    cleaned1 = clean_history_message(msg_with_narrative)
    assert cleaned1 == "07:38:49 WIB (Jum, 11 Sep 2026)"

    # Message with only tool breadcrumb
    msg_only_crumb = "> ⚡ **`list_dir: yodu`** ➔\n```\n[dir] .git\n[dir] app\n```"
    cleaned2 = clean_history_message(msg_only_crumb)
    assert "[list_dir yodu]" in cleaned2
    assert "app" not in cleaned2

