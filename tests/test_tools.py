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
    assert "load_skill" in names


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

    # Command execution
    b1 = format_lean_tool_breadcrumb(
        "run_command", {"command": "uname -srm"}, "Linux 6.18.48-1-cachyos-lts x86_64"
    )
    assert b1 == "> ⚡ `run_command: uname -srm`\n\n"

    # Git status
    b2 = format_lean_tool_breadcrumb(
        "git_status", {"repo_path": "/tmp"}, "M file1.py\n?? file2.py"
    )
    assert b2 == "> 🌿 Git status\n\n"

    # Reading file with explicit lines
    b3 = format_lean_tool_breadcrumb(
        "read_file", {"path": "src/main.py", "start_line": 1, "end_line": 45}, "code"
    )
    assert b3 == "> 📖 Reading file `src/main.py` 1:45\n\n"

    # Editing file with additions and deletions
    b4 = format_lean_tool_breadcrumb(
        "replace_file_content",
        {
            "path": "src/main.py",
            "target_content": "line1\nline2\n",
            "replacement_content": "new1\nnew2\nnew3\n",
        },
        "success",
    )
    assert b4 == "> ✏️ Editing file `src/main.py` (+3 -2)\n\n"


def test_detect_simulated_tool_call():
    from maulness.core.tools import detect_simulated_tool_call

    # Detected cases
    sim1 = detect_simulated_tool_call(
        "> ⚡ **`list_dir: yodu`** ➔\n```\n[dir] app\n```"
    )
    assert sim1 == ("list_dir", {"path": "yodu"})

    sim2 = detect_simulated_tool_call(
        "> **`run_command: ls -la /home/zlnew/www/yodu`**:\n```\ntotal 44\n```"
    )
    assert sim2 == ("run_command", {"command": "ls -la /home/zlnew/www/yodu"})

    sim3 = detect_simulated_tool_call("> **`git_status`** -> clean")
    assert sim3 == ("git_status", {"repo_path": ""})

    # Non-simulated regular text
    assert (
        detect_simulated_tool_call("Here is what you need to know about git_status.")
        is None
    )
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
    text3 = "[STATUS: IN_PROGRESS]\n- Next Step: Deploy service and check logs"
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
    assert (
        extract_checkpoint_info("Everything is finished! [STATUS: COMPLETE]")[0]
        is False
    )
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
    cmd_disabled, is_shell = build_sandboxed_command(
        "echo hello", tmp_path, sandbox_mode="none"
    )
    assert cmd_disabled == "echo hello"
    assert is_shell is True

    # 2. bwrap / auto mode
    if shutil.which("bwrap"):
        args, is_shell_bwrap = build_sandboxed_command(
            "echo hello", tmp_path, sandbox_mode="bwrap"
        )
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
    truncated = truncate_observation(
        huge_out, max_lines=300, head_lines=50, tail_lines=50
    )
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
    detector.record_and_check(
        session2,
        "replace_file_content",
        {"path": "a.py", "target_content": "1", "replacement_content": "2"},
    )
    detector.record_and_check(session2, "run_command", {"command": "pytest"})
    loop, _ = detector.record_and_check(
        session2,
        "replace_file_content",
        {"path": "a.py", "target_content": "2", "replacement_content": "3"},
    )
    assert not loop


def test_truncate_observation_chars_and_empty():
    from maulness.core.tools import truncate_observation

    assert truncate_observation("") == ""

    # Character truncation
    huge_str = "x" * 20000
    res = truncate_observation(huge_str, max_chars=1000)
    assert "characters omitted for brevity" in res
    assert len(res) < 2000


def test_tombstone_tool_output_all_tools():
    from maulness.core.tools import tombstone_tool_output

    bulk = "line\n" * 10
    assert "read_file" in tombstone_tool_output("read_file", {"path": "test.py"}, bulk)
    assert "write_file" in tombstone_tool_output(
        "write_file", {"path": "test.py"}, bulk
    )
    assert "replace_file_content" in tombstone_tool_output(
        "replace_file_content", {"path": "test.py"}, bulk
    )
    assert "search_files" in tombstone_tool_output(
        "search_files", {"pattern": "needle"}, bulk
    )
    assert "list_dir" in tombstone_tool_output("list_dir", {"path": "."}, bulk)
    assert "git_status" in tombstone_tool_output(
        "git_status", {"repo_path": "repo"}, bulk
    )


def test_action_loop_detector_edge_cases():
    from maulness.core.tools import ActionLoopDetector, get_loop_detector

    detector = ActionLoopDetector(window_size=4, repetition_threshold=2)
    # Empty session
    loop, msg = detector.record_and_check("", "run_command", {"command": "ls"})
    assert not loop
    assert msg == ""

    # Modifying action repeated
    detector.record_and_check("s1", "write_file", {"path": "a.txt", "content": "1"})
    loop, msg = detector.record_and_check(
        "s1", "write_file", {"path": "a.txt", "content": "1"}
    )
    assert loop
    assert "[LOOP INTERVENTION]" in msg

    # Clear specific session
    detector.clear("s1")
    assert "s1" not in detector._history

    # Clear all
    detector.record_and_check("s2", "list_dir", {})
    detector.clear()
    assert len(detector._history) == 0

    assert get_loop_detector() is not None


def test_active_turn_cache_and_history():
    from maulness.core.tools import (
        _ACTIVE_TURN_HISTORY,
        _ACTIVE_TURN_CACHE,
        get_turn_executed_tools,
        clear_turn_tools,
    )

    session_id = "session_cache_test"
    _ACTIVE_TURN_HISTORY[session_id] = [{"tool": "read_file"}]
    _ACTIVE_TURN_CACHE[session_id] = {"k": "v"}

    tools = get_turn_executed_tools(session_id)
    assert len(tools) == 1
    assert tools[0]["tool"] == "read_file"

    clear_turn_tools(session_id)
    assert len(get_turn_executed_tools(session_id)) == 0


def test_build_sandboxed_command_options(tmp_path: Path, monkeypatch):
    from maulness.core.tools import build_sandboxed_command

    # Disabled mode
    cmd, is_shell = build_sandboxed_command("echo 1", tmp_path, sandbox_mode="disabled")
    assert cmd == "echo 1"
    assert is_shell is True

    # None mode
    cmd, is_shell = build_sandboxed_command("echo 2", tmp_path, sandbox_mode="none")
    assert cmd == "echo 2"
    assert is_shell is True

    # Missing bwrap
    monkeypatch.setattr("shutil.which", lambda prog: None)
    cmd, is_shell = build_sandboxed_command("echo 3", tmp_path, sandbox_mode="bwrap")
    assert cmd == "echo 3"
    assert is_shell is True


@pytest.mark.asyncio
async def test_read_file_edge_cases(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("L1\nL2\nL3\nL4\nL5")

    # start_line > total_lines
    res = await execute_tool_call(
        name="read_file",
        args={"path": str(f), "start_line": 10},
        workspace_path=tmp_path,
    )
    assert "exceeds total lines" in res

    # start_line > end_line
    res = await execute_tool_call(
        name="read_file",
        args={"path": str(f), "start_line": 4, "end_line": 2},
        workspace_path=tmp_path,
    )
    assert "is greater than end_line" in res

    # target is a directory
    sub = tmp_path / "dir"
    sub.mkdir()
    res = await execute_tool_call(
        name="read_file",
        args={"path": str(sub)},
        workspace_path=tmp_path,
    )
    assert "is a directory" in res

    # File > 50,000 characters
    big_file = tmp_path / "big.txt"
    big_file.write_text("A" * 60000)
    res = await execute_tool_call(
        name="read_file",
        args={"path": str(big_file)},
        workspace_path=tmp_path,
    )
    assert "truncated 50,000 chars" in res


@pytest.mark.asyncio
async def test_write_file_rejection_and_error(tmp_path: Path, monkeypatch):
    async def reject(req):
        return False

    res = await execute_tool_call(
        name="write_file",
        args={"path": "rejected.txt", "content": "data"},
        workspace_path=tmp_path,
        on_approval=reject,
        yolo=False,
    )
    assert "Write cancelled: User rejected" in res

    # Write error
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("Disk full")),
    )
    res = await execute_tool_call(
        name="write_file",
        args={"path": "err.txt", "content": "data"},
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Error writing file" in res


@pytest.mark.asyncio
async def test_replace_file_content_rejection(tmp_path: Path):
    f = tmp_path / "target.txt"
    f.write_text("orig")

    async def reject(req):
        return False

    res = await execute_tool_call(
        name="replace_file_content",
        args={"path": str(f), "target_content": "orig", "replacement_content": "new"},
        workspace_path=tmp_path,
        on_approval=reject,
        yolo=False,
    )
    assert "Execution cancelled: User rejected" in res


@pytest.mark.asyncio
async def test_search_files_edge_cases(tmp_path: Path, monkeypatch):
    # Empty pattern
    res = await execute_tool_call(
        name="search_files",
        args={"pattern": ""},
        workspace_path=tmp_path,
    )
    assert "No pattern provided" in res

    # Invalid regex pattern (fallback path)
    from maulness.core.tools import _execute_search_files

    monkeypatch.setattr("shutil.which", lambda x: None)  # force python fallback
    res = await _execute_search_files(pattern="[unclosed", target_dir=tmp_path)
    assert "Invalid regex pattern" in res

    # Fallback search with match
    (tmp_path / "file1.py").write_text("def hello_world(): pass")
    (tmp_path / "file2.txt").write_text("other stuff")
    res = await _execute_search_files(
        pattern="hello_world", target_dir=tmp_path, glob_pattern="*.py"
    )
    assert "hello_world" in res
    assert "file1.py" in res

    # Fallback search no match
    res = await _execute_search_files(pattern="nonexistent_needle", target_dir=tmp_path)
    assert "No matches found" in res


@pytest.mark.asyncio
async def test_list_dir_edge_cases(tmp_path: Path):
    # Nonexistent dir
    res = await execute_tool_call(
        name="list_dir",
        args={"path": str(tmp_path / "missing_dir")},
        workspace_path=tmp_path,
    )
    assert "Directory" in res and "not found" in res

    # Empty dir
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    res = await execute_tool_call(
        name="list_dir",
        args={"path": str(empty)},
        workspace_path=tmp_path,
    )
    assert "(Directory is empty)" in res


@pytest.mark.asyncio
async def test_git_status_edge_cases(tmp_path: Path):
    import subprocess

    # Run git status in empty non-git dir
    res = await execute_tool_call(
        name="git_status",
        args={"repo_path": str(tmp_path)},
        workspace_path=tmp_path,
    )
    assert (
        "Error running git status" in res
        or "fatal" in res
        or "Working tree clean" in res
    )

    # Init git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
    res = await execute_tool_call(
        name="git_status",
        args={"repo_path": str(tmp_path)},
        workspace_path=tmp_path,
    )
    assert "Working tree clean" in res


def test_detect_simulated_tool_call_all_variants():
    from maulness.core.tools import detect_simulated_tool_call

    assert detect_simulated_tool_call("") is None
    assert detect_simulated_tool_call("Just talking here") is None
    assert detect_simulated_tool_call("> **fake_tool: test**") is None

    t, a = detect_simulated_tool_call("> **run_command: pytest -v**")
    assert t == "run_command"
    assert a["command"] == "pytest -v"

    t, a = detect_simulated_tool_call("> **read_file: src/main.py**")
    assert t == "read_file"
    assert a["path"] == "src/main.py"

    t, a = detect_simulated_tool_call("> **write_file: src/out.txt**")
    assert t == "write_file"
    assert a["path"] == "src/out.txt"

    t, a = detect_simulated_tool_call("> **replace_file_content: src/main.py**")
    assert t == "replace_file_content"

    t, a = detect_simulated_tool_call("> **search_files: def run**")
    assert t == "search_files"

    t, a = detect_simulated_tool_call("> **git_status**")
    assert t == "git_status"


def test_clean_history_message_all_variants():
    from maulness.core.tools import clean_history_message

    assert clean_history_message("") == ""

    # Multiline
    multiline = (
        "> **run_command: ls**:\n```\nfile1.txt\nfile2.txt\n```\nNow I will proceed."
    )
    assert clean_history_message(multiline) == "Now I will proceed."

    # Single line
    single = "> **`read_file: a.txt`** -> `content`\n\nI have read the file."
    assert clean_history_message(single) == "I have read the file."

    # All breadcrumbs stripped leading to fallback
    only_crumb = "> **run_command: ls** -> `ok`"
    cleaned = clean_history_message(only_crumb)
    assert "[" in cleaned and "]" in cleaned


def test_extract_checkpoint_info_all_variants():
    from maulness.core.tools import extract_checkpoint_info

    assert extract_checkpoint_info("") == (False, "", "")

    # Finished
    is_in, _, _ = extract_checkpoint_info("[STATUS: COMPLETE] All done.")
    assert not is_in

    # In progress with next step
    text = "[STATUS: IN_PROGRESS]\n- [ ] Task item\nNext step: Run the tests"
    is_in, body, next_step = extract_checkpoint_info(text)
    assert is_in
    assert next_step == "Run the tests"

    # In progress with Next:
    text2 = "Moving forward.\nNext: Deploy container"
    is_in, _, next_step = extract_checkpoint_info(text2)
    assert is_in
    assert next_step == "Deploy container"

    # Pending item marked -> next
    text3 = "Here is the checklist:\n1. Update db -> next\n2. Run migration"
    is_in, _, next_step = extract_checkpoint_info(text3)
    assert is_in
    assert "Update db -> next" in next_step


def test_clean_relay_completion_tags():
    from maulness.core.tools import clean_relay_completion_tags

    assert clean_relay_completion_tags("") == ""
    assert (
        clean_relay_completion_tags("Hello world [STATUS: COMPLETE]") == "Hello world"
    )
    assert clean_relay_completion_tags("Task done [STATUS: FINISHED]") == "Task done"


def test_tools_resolve_path_parent_and_home(tmp_path: Path):
    from unittest.mock import patch

    # 1. Base parent candidate exists (line 196)
    sub = tmp_path / "child"
    sub.mkdir()
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("parent data")
    resolved_parent = resolve_path(Path("sibling.txt"), workspace_path=sub)
    assert resolved_parent == sibling.resolve()

    # 2. Home candidate exists (line 201)
    home_dir = tmp_path / "fake_home"
    home_dir.mkdir()
    home_file = home_dir / "user_home.txt"
    home_file.write_text("home data")
    with patch("pathlib.Path.home", return_value=home_dir):
        resolved_home = resolve_path(Path("user_home.txt"), workspace_path=sub)
        assert resolved_home == home_file.resolve()


def test_loop_detector_unserializable_args():
    from maulness.core.tools import ActionLoopDetector

    ld = ActionLoopDetector()
    # Dict with mixed unorderable keys will cause sort_keys=True in json.dumps to fail
    bad_args = {1: "a", "b": "c"}
    h = ld._hash_args(bad_args)
    assert isinstance(h, str) and len(h) == 16


@pytest.mark.asyncio
async def test_tools_loop_intervention_and_generic_hitl_ask(tmp_path: Path):
    from unittest.mock import MagicMock
    from maulness.core.rules import PolicyAction
    from maulness.core.tools import _LOOP_DETECTOR, execute_tool_call

    session_id = "test_loop_sess"
    _LOOP_DETECTOR._history.clear()

    # 1. Loop detection intervention (line 435)
    for _ in range(_LOOP_DETECTOR.repetition_threshold):
        _LOOP_DETECTOR.record_and_check(
            session_id, "write_file", {"path": "loop.txt", "content": "same"}
        )

    res_loop = await execute_tool_call(
        name="write_file",
        args={"path": "loop.txt", "content": "same"},
        workspace_path=tmp_path,
        session_id=session_id,
        yolo=True,
    )
    assert "[LOOP INTERVENTION]" in res_loop

    # 2. Policy DENY gate (lines 428-430)
    mock_engine_deny = MagicMock()
    mock_engine_deny.evaluate.return_value = (
        PolicyAction.DENY,
        "blocked by test policy",
    )
    res_deny = await execute_tool_call(
        name="custom_tool",
        args={"foo": "bar"},
        workspace_path=tmp_path,
        rule_engine=mock_engine_deny,
    )
    assert "Execution blocked by security policy: blocked by test policy" in res_deny

    # 3. Generic HITL Gate for custom tool with ASK (lines 440-449)
    mock_engine_ask = MagicMock()
    mock_engine_ask.evaluate.return_value = (PolicyAction.ASK, "requires review")

    async def reject_cb(req: ApprovalRequestEvent) -> bool:
        assert req.tool_name == "custom_tool"
        return False

    async def approve_cb(req: ApprovalRequestEvent) -> bool:
        return True

    res_rejected = await execute_tool_call(
        name="custom_tool",
        args={"foo": "bar"},
        workspace_path=tmp_path,
        rule_engine=mock_engine_ask,
        on_approval=reject_cb,
        yolo=False,
    )
    assert "Execution cancelled: User rejected tool 'custom_tool'" in res_rejected

    res_approved = await execute_tool_call(
        name="custom_tool",
        args={"foo": "bar"},
        workspace_path=tmp_path,
        rule_engine=mock_engine_ask,
        on_approval=approve_cb,
        yolo=False,
    )
    assert "Error: Unknown tool 'custom_tool'" in res_approved


def test_execute_replace_file_content_edge_cases(tmp_path: Path):
    from unittest.mock import patch
    from maulness.core.tools import _execute_replace_file_content

    # Line 484: not exists
    res_not_exists = _execute_replace_file_content(tmp_path / "ghost.txt", "a", "b")
    assert "does not exist" in res_not_exists

    # Line 486: is_dir
    sub = tmp_path / "sub"
    sub.mkdir()
    res_dir = _execute_replace_file_content(sub, "a", "b")
    assert "is a directory" in res_dir

    # Line 488: empty target_content
    f = tmp_path / "file.txt"
    f.write_text("data")
    res_empty = _execute_replace_file_content(f, "", "b")
    assert "must not be empty" in res_empty

    # Line 492-493: read error
    with patch.object(
        Path, "read_text", side_effect=PermissionError("Permission denied")
    ):
        res_read_err = _execute_replace_file_content(f, "data", "new")
        assert "Error reading file" in res_read_err

    # Line 519-520: write error
    with patch.object(Path, "write_text", side_effect=OSError("Disk full")):
        res_write_err = _execute_replace_file_content(f, "data", "new")
        assert "Error writing file" in res_write_err


@pytest.mark.asyncio
async def test_execute_search_files_edge_cases(tmp_path: Path):
    from unittest.mock import patch
    import subprocess
    from maulness.core.tools import _execute_search_files

    # Line 532: directory does not exist
    res_no_dir = await _execute_search_files("foo", tmp_path / "missing_dir")
    assert "does not exist" in res_no_dir

    # Line 559: ripgrep >= max_results capped
    mock_rg_out = "\n".join([f"file.py:{i}: match {i}" for i in range(1, 60)])
    mock_completed = subprocess.CompletedProcess(
        args=["rg"], returncode=0, stdout=mock_rg_out, stderr=""
    )
    with patch("shutil.which", return_value="/usr/bin/rg"):
        with patch("subprocess.run", return_value=mock_completed):
            res_rg_capped = await _execute_search_files(
                "match", tmp_path, max_results=50
            )
            assert "Results capped at 50 matches" in res_rg_capped

    # Line 561-562: ripgrep exception falls back to python search
    # And python search skips unreadable files (lines 584-586, 588, 590, 596)
    search_dir = tmp_path / "search_test"
    search_dir.mkdir()
    unreadable = search_dir / "00_unreadable.txt"
    unreadable.write_text("special_token")
    for i in range(1, 55):
        (search_dir / f"f_{i}.txt").write_text(f"special_token in line {i}\n")

    orig_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self.name == "00_unreadable.txt":
            raise PermissionError("Access denied")
        return orig_read_text(self, *args, **kwargs)

    with patch("shutil.which", return_value="/usr/bin/rg"):
        with patch("subprocess.run", side_effect=RuntimeError("Ripgrep crash")):
            with patch.object(Path, "read_text", fake_read_text):
                res_py_fallback = await _execute_search_files(
                    "special_token", search_dir, max_results=50
                )
                assert "Results capped at 50 matches" in res_py_fallback


@pytest.mark.asyncio
async def test_execute_tool_action_all_exceptions_and_empty_cmd(tmp_path: Path):
    from unittest.mock import patch
    import subprocess
    from maulness.core.rules import PolicyAction
    from maulness.core.tools import _execute_tool_action

    # Line 616: run_command empty command
    res_empty_cmd = await _execute_tool_action(
        name="run_command",
        args={"command": "   "},
        cwd=tmp_path,
        session_id="sess",
        policy=PolicyAction.ALLOW,
        on_approval=None,
        yolo=True,
    )
    assert res_empty_cmd == "Error: No command provided."

    # Lines 650-651: run_command timeout
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = mock_popen.return_value
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="sleep 100", timeout=60
        )
        mock_proc.pid = 12345
        with patch("os.killpg"):
            res_timeout = await _execute_tool_action(
                name="run_command",
                args={"command": "sleep 100"},
                cwd=tmp_path,
                session_id="sess",
                policy=PolicyAction.ALLOW,
                on_approval=None,
                yolo=True,
            )
            assert "timed out after 60s" in res_timeout

    # Lines 652-653: run_command general exception
    with patch("subprocess.Popen", side_effect=RuntimeError("Subprocess failed")):
        res_cmd_err = await _execute_tool_action(
            name="run_command",
            args={"command": "echo test"},
            cwd=tmp_path,
            session_id="sess",
            policy=PolicyAction.ALLOW,
            on_approval=None,
            yolo=True,
        )
        assert "Error executing command: Subprocess failed" in res_cmd_err

    # Lines 684-685: read_file general exception
    f = tmp_path / "read_err.txt"
    f.write_text("data")
    with patch.object(Path, "read_text", side_effect=PermissionError("Locked")):
        res_read_err = await _execute_tool_action(
            name="read_file",
            args={"path": str(f)},
            cwd=tmp_path,
            session_id="sess",
            policy=PolicyAction.ALLOW,
            on_approval=None,
            yolo=True,
        )
        assert "Error reading file" in res_read_err

    # Lines 760-761: list_dir general exception
    with patch.object(Path, "iterdir", side_effect=PermissionError("Cannot list")):
        res_list_err = await _execute_tool_action(
            name="list_dir",
            args={"path": str(tmp_path)},
            cwd=tmp_path,
            session_id="sess",
            policy=PolicyAction.ALLOW,
            on_approval=None,
            yolo=True,
        )
        assert "Error listing directory" in res_list_err

    # Lines 776-777: git_status general exception
    with patch("subprocess.run", side_effect=RuntimeError("Git failed")):
        res_git_err = await _execute_tool_action(
            name="git_status",
            args={"repo_path": str(tmp_path)},
            cwd=tmp_path,
            session_id="sess",
            policy=PolicyAction.ALLOW,
            on_approval=None,
            yolo=True,
        )
        assert "Error running git status" in res_git_err


def test_format_lean_tool_breadcrumb_all_branches():
    from maulness.core.tools import format_lean_tool_breadcrumb

    # write_file
    b1 = format_lean_tool_breadcrumb(
        "write_file", {"path": "test.txt", "content": "hello\nworld"}, "done"
    )
    assert b1 == "> 📝 Writing file `test.txt` (+2 lines)\n\n"

    # replace_file_content with old and new
    b2 = format_lean_tool_breadcrumb(
        "replace_file_content",
        {"path": "test.py", "old_string": "a\nb", "new_string": "c"},
        "replaced",
    )
    assert b2 == "> ✏️ Editing file `test.py` (+1 -2)\n\n"

    # patch_file
    b_patch = format_lean_tool_breadcrumb(
        "patch_file",
        {"path": "app.py", "patch": "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new"},
        "success",
    )
    assert b_patch == "> ✏️ Patching file `app.py` (+1 -1)\n\n"

    # search_files with long pattern
    long_pat = "a" * 50
    b3 = format_lean_tool_breadcrumb(
        "search_files", {"pattern": long_pat}, "line 1\nline 2"
    )
    assert "Search 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...'" in b3
    assert "2 matches" in b3

    # list_dir
    b4 = format_lean_tool_breadcrumb("list_dir", {}, "file1\nfile2\n")
    assert b4 == "> 📁 List `.` -> 2 items\n\n"

    # run_command failure
    b5 = format_lean_tool_breadcrumb(
        "run_command", {"command": "cat missing.txt"}, "Error: failed with returncode 1"
    )
    assert b5 == "> ⚠️ ⚡ `run_command: cat missing.txt` (failed)\n\n"

    # read_file with error
    b_read_err = format_lean_tool_breadcrumb(
        "read_file", {"path": "notfound.txt"}, "Error: File not found"
    )
    assert b_read_err == "> ⚠️ Reading file `notfound.txt` (Error: File not found)\n\n"

    # read_file from header lines
    b_read_hdr = format_lean_tool_breadcrumb(
        "read_file", {"path": "app.py"}, "[app.py lines 10-25 of 100]\nL10: x\nL25: y"
    )
    assert b_read_hdr == "> 📖 Reading file `app.py` 10:25\n\n"

    # generic fallback tool
    b_custom = format_lean_tool_breadcrumb("custom_tool", {"name": "action"}, "ok")
    assert b_custom == "> 🔧 `custom_tool: action`\n\n"


@pytest.mark.asyncio
async def test_execute_load_skill(tmp_path: Path):
    from maulness.core.tools import execute_tool_call

    # Missing skill name argument
    err1 = await execute_tool_call(
        name="load_skill",
        args={},
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Error: Skill name is required" in err1

    # Load custom skill in workspace
    sk_dir = tmp_path / ".maulness" / "skills" / "custom"
    sk_dir.mkdir(parents=True)
    (sk_dir / "SKILL.md").write_text(
        "---\nname: custom\ndescription: Custom playbook\n---\nStep 1: Do testing.",
        encoding="utf-8",
    )

    res = await execute_tool_call(
        name="load_skill",
        args={"name": "custom"},
        workspace_path=tmp_path,
        yolo=True,
    )
    assert "Playbook Instruction: custom" in res
    assert "Step 1: Do testing." in res


@pytest.mark.asyncio
async def test_fetch_doc_markdown_ssrf_blocked():
    from maulness.core.tools import execute_tool_call

    # Localhost / loopback
    res = await execute_tool_call(
        name="fetch_doc_markdown",
        args={"url": "http://localhost:8000/secret"},
        yolo=True,
    )
    assert "blocked for security" in res

    # 127.0.0.1
    res2 = await execute_tool_call(
        name="fetch_doc_markdown",
        args={"url": "http://127.0.0.1:5432/"},
        yolo=True,
    )
    assert "blocked for security" in res2

    # Cloud metadata endpoint
    res3 = await execute_tool_call(
        name="fetch_doc_markdown",
        args={"url": "http://169.254.169.254/latest/meta-data/"},
        yolo=True,
    )
    assert "blocked for security" in res3


@pytest.mark.asyncio
async def test_patch_file_single_approval(tmp_path: Path):
    from maulness.core.tools import execute_tool_call
    from unittest.mock import AsyncMock

    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\n", encoding="utf-8")

    approval_mock = AsyncMock(return_value=True)
    patch_content = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,2 +1,2 @@\n line1\n-line2\n+line_patched\n"

    await execute_tool_call(
        name="patch_file",
        args={"path": "hello.txt", "patch": patch_content},
        workspace_path=tmp_path,
        on_approval=approval_mock,
        yolo=False,
    )
    # Ensure on_approval was invoked EXACTLY once, not twice!
    assert approval_mock.call_count == 1

