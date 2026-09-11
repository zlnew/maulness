import pytest
from pathlib import Path
from maulness.core.models import ApprovalRequestEvent
from maulness.core.tools import execute_tool_call, resolve_path, TOOL_DEFINITIONS


def test_tool_definitions():
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "run_command" in names
    assert "read_file" in names
    assert "replace_file_content" in names
    assert "write_file" in names
    assert "search_files" in names
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


def test_resolve_path_resilient(tmp_path: Path):
    www_dir = tmp_path / "www"
    personal_dir = www_dir / "personal"
    personal_dir.mkdir(parents=True)

    # 1. Normal relative path
    assert resolve_path("personal", workspace_path=www_dir) == personal_dir

    # 2. Overlapping root segment (e.g. target is www/personal when cwd is /.../www)
    assert resolve_path("www/personal", workspace_path=www_dir) == personal_dir

    # 3. Absolute path
    assert resolve_path(str(personal_dir), workspace_path=www_dir) == personal_dir

    # 4. Fallback for non-existent target
    missing = resolve_path("missing/file.txt", workspace_path=www_dir)
    assert missing == (www_dir / "missing/file.txt").resolve()


def test_extract_checkpoint_info():
    from maulness.core.tools import extract_checkpoint_info

    # 1. Standard format
    text1 = (
        "[STATUS: IN_PROGRESS]\n"
        "Accomplished: Inspected directory structure.\n"
        "Key Findings: Found config and profiles.\n"
        "Next Step: Edit api_provider.py to add relay loop."
    )
    is_prog1, body1, step1 = extract_checkpoint_info(text1)
    assert is_prog1 is True
    assert step1 == "Edit api_provider.py to add relay loop."
    assert "Inspected directory" in body1

    # 2. Markdown bold format
    text2 = (
        "Here is the status:\n"
        "[status: in_progress]\n"
        "**Accomplished:** Step 1\n"
        "**Next Step:** Run unit tests with pytest"
    )
    is_prog2, body2, step2 = extract_checkpoint_info(text2)
    assert is_prog2 is True
    assert step2 == "Run unit tests with pytest"

    # 3. List bullet format
    text3 = (
        "[STATUS: IN_PROGRESS]\n"
        "- Next Step: Deploy service and check logs"
    )
    is_prog3, _, step3 = extract_checkpoint_info(text3)
    assert is_prog3 is True
    assert step3 == "Deploy service and check logs"

    # 4. No next step specified -> fallback
    text4 = "[STATUS: IN_PROGRESS]\nWork in progress..."
    is_prog4, _, step4 = extract_checkpoint_info(text4)
    assert is_prog4 is True
    assert step4 == "Continuing task execution"

    # 5. Natural progress map with unchecked checkboxes and Next Step (no literal [STATUS: IN_PROGRESS] tag)
    text5 = (
        "I have read internal/domain.\n\n"
        "Current Status:\n"
        "- [x] internal/domain -> COMPLETE\n"
        "- [ ] internal/application -> NEXT\n"
        "- [ ] internal/adapters -> PENDING\n\n"
        "Next Step: I am diving into internal/application to inspect services."
    )
    is_prog5, _, step5 = extract_checkpoint_info(text5)
    assert is_prog5 is True
    assert "diving into internal/application" in step5

    # 6. Negative cases
    assert extract_checkpoint_info("")[0] is False
    assert extract_checkpoint_info("Everything is finished! [STATUS: COMPLETE]")[0] is False
    assert extract_checkpoint_info("Here is your answer: 42")[0] is False


def test_clean_relay_completion_tags():
    from maulness.core.tools import clean_relay_completion_tags

    text = "Here is the final report.\n\n[STATUS: COMPLETE]"
    cleaned = clean_relay_completion_tags(text)
    assert cleaned == "Here is the final report."
    assert "[STATUS: COMPLETE]" not in cleaned


def test_build_sandboxed_command(tmp_path: Path):
    import shutil
    from maulness.core.tools import build_sandboxed_command

    # 1. Disabled mode
    cmd_disabled, is_shell = build_sandboxed_command("echo hello", tmp_path, sandbox_mode="none")
    assert cmd_disabled == "echo hello"
    assert is_shell is True

    # 2. bwrap / auto mode
    if shutil.which("bwrap"):
        args, is_shell_bwrap = build_sandboxed_command("echo hello", tmp_path, sandbox_mode="bwrap")
        assert is_shell_bwrap is False
        assert isinstance(args, list)
        assert "--ro-bind" in args
        assert "/" in args
        assert "--bind" in args
        assert str(tmp_path.resolve()) in args
        assert "--unshare-all" in args
        assert "--share-net" in args


@pytest.mark.asyncio
async def test_execute_run_command_sandboxed_blocks_external_write(tmp_path: Path):
    import shutil
    from maulness.core.tools import execute_tool_call

    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed on test runner")

    # Attempt to write outside workspace to /home/zlnew/leak_test.txt (or /etc/leak_test)
    res = await execute_tool_call(
        name="run_command",
        args={"command": "touch /home/zlnew/leak_test_blocked.txt"},
        workspace_path=tmp_path,
        yolo=True,
        sandbox_mode="bwrap",
    )
    assert "Read-only file system" in res
    assert not Path("/home/zlnew/leak_test_blocked.txt").exists()


@pytest.mark.asyncio
async def test_execute_run_command_sandboxed_allows_workspace_write(tmp_path: Path):
    import shutil
    from maulness.core.tools import execute_tool_call

    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed on test runner")

    res = await execute_tool_call(
        name="run_command",
        args={"command": "touch inside_test.txt && ls"},
        workspace_path=tmp_path,
        yolo=True,
        sandbox_mode="bwrap",
    )
    assert "inside_test.txt" in res
    assert (tmp_path / "inside_test.txt").exists()


@pytest.mark.asyncio
async def test_execute_read_file_line_slicing(tmp_path: Path):
    target = tmp_path / "numbers.txt"
    target.write_text("\n".join(f"Line {i}" for i in range(1, 21)))

    # Read slice 5 to 8
    res = await execute_tool_call(
        name="read_file",
        args={"path": "numbers.txt", "start_line": 5, "end_line": 8},
        workspace_path=tmp_path,
    )
    assert "[numbers.txt lines 5-8 of 20]" in res
    assert "L5: Line 5" in res
    assert "L8: Line 8" in res
    assert "Line 4" not in res
    assert "Line 9" not in res

    # Out of bounds start_line
    err_res = await execute_tool_call(
        name="read_file",
        args={"path": "numbers.txt", "start_line": 50},
        workspace_path=tmp_path,
    )
    assert "Error: start_line 50 exceeds total lines" in err_res


@pytest.mark.asyncio
async def test_execute_replace_file_content(tmp_path: Path):
    target = tmp_path / "code.py"
    target.write_text("def hello():\n    return 'old'\n")

    # Successful replacement
    res = await execute_tool_call(
        name="replace_file_content",
        args={
            "path": "code.py",
            "target_content": "return 'old'",
            "replacement_content": "return 'new'",
        },
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Successfully replaced 1 occurrence(s)" in res
    assert target.read_text() == "def hello():\n    return 'new'\n"

    # Missing target content
    fail_res = await execute_tool_call(
        name="replace_file_content",
        args={
            "path": "code.py",
            "target_content": "non_existent_code()",
            "replacement_content": "new_code()",
        },
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Error: target_content not found" in fail_res


@pytest.mark.asyncio
async def test_execute_replace_file_content_multiple(tmp_path: Path):
    target = tmp_path / "multi.txt"
    target.write_text("val = 1\nval = 1\n")

    # Multiple without allow_multiple flag -> refused for safety
    refused = await execute_tool_call(
        name="replace_file_content",
        args={
            "path": "multi.txt",
            "target_content": "val = 1",
            "replacement_content": "val = 2",
        },
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "found 2 times" in refused

    # Multiple with allow_multiple=True -> replaced all
    allowed = await execute_tool_call(
        name="replace_file_content",
        args={
            "path": "multi.txt",
            "target_content": "val = 1",
            "replacement_content": "val = 2",
            "allow_multiple": True,
        },
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Successfully replaced 2 occurrence(s)" in allowed
    assert target.read_text() == "val = 2\nval = 2\n"


@pytest.mark.asyncio
async def test_execute_search_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("def find_me():\n    pass\n")
    (tmp_path / "b.txt").write_text("not here\n")

    res = await execute_tool_call(
        name="search_files",
        args={"pattern": "find_me"},
        workspace_path=tmp_path,
    )
    assert "find_me" in res
    assert "a.py" in res

    # Glob filtering
    filtered_res = await execute_tool_call(
        name="search_files",
        args={"pattern": "find_me", "glob_pattern": "*.txt"},
        workspace_path=tmp_path,
    )
    assert "No matches found" in filtered_res


def test_truncate_observation():
    from maulness.core.tools import truncate_observation

    # Under limit
    short_out = "Short output\n"
    assert truncate_observation(short_out) == short_out

    # Over 300 lines
    huge_out = "\n".join(f"Line {i}" for i in range(500))
    truncated = truncate_observation(huge_out, max_lines=300, head_lines=50, tail_lines=50)
    assert "lines omitted for brevity" in truncated
    assert "Line 0" in truncated
    assert "Line 499" in truncated


def test_tombstone_tool_output():
    from maulness.core.tools import tombstone_tool_output

    # Tiny output remains unchanged
    tiny = "clean"
    assert tombstone_tool_output("git_status", {}, tiny) == tiny

    # Bulky output gets compacted
    bulky = "\n".join(f"Item {i}: result details here" for i in range(20))
    tomb = tombstone_tool_output("run_command", {"command": "pytest -v"}, bulky)
    assert "[Tool result for 'run_command' (`pytest -v`) compacted:" in tomb
    assert "20 lines" in tomb


def test_action_loop_detector():
    from maulness.core.tools import ActionLoopDetector

    detector = ActionLoopDetector(window_size=6, repetition_threshold=3)
    session = "test_session_loop"

    # Step 1: run pytest (fails)
    loop, msg = detector.record_and_check(session, "run_command", {"command": "pytest"})
    assert not loop

    # Step 2: run pytest again (fails)
    loop, msg = detector.record_and_check(session, "run_command", {"command": "pytest"})
    assert not loop

    # Step 3: run pytest 3rd time without any intervening modifying action -> TRIPPED!
    loop, msg = detector.record_and_check(session, "run_command", {"command": "pytest"})
    assert loop
    assert "[LOOP INTERVENTION]" in msg

    # Now verify modifying action in between resets the consecutive thrashing counter
    session2 = "test_session_recovery"
    detector.record_and_check(session2, "run_command", {"command": "pytest"})
    detector.record_and_check(session2, "replace_file_content", {"path": "a.py", "target_content": "1", "replacement_content": "2"})
    detector.record_and_check(session2, "run_command", {"command": "pytest"})
    loop, _ = detector.record_and_check(session2, "replace_file_content", {"path": "a.py", "target_content": "2", "replacement_content": "3"})
    assert not loop



