import asyncio
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from maulness.core.kernel.models import KernelEventType
from maulness.core.tools import build_sandboxed_command
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.kernel.gates")


class GateFailureItem(BaseModel):
    """Details of an individual test or validation failure."""

    identifier: str
    message: str
    details: Optional[str] = None


class GateResult(BaseModel):
    """Normalized result of a deterministic verification gate execution."""

    passed: bool
    command: str
    exit_code: int
    summary: str
    failures: list[GateFailureItem] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "command": self.command,
            "exit_code": self.exit_code,
            "summary": self.summary,
            "failures": [f.model_dump() for f in self.failures],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }

    def to_feedback_prompt(self) -> str:
        """Format normalized failure details into an actionable feedback prompt for the generator."""
        if self.passed:
            return f"Deterministic verification gate passed for `{self.command}`: {self.summary}"

        lines = [
            "## Deterministic Verification Gate FAILED",
            f"Command: `{self.command}` (Exit Code: {self.exit_code})",
            f"Summary: {self.summary}",
            "",
        ]

        if self.failures:
            lines.append("### Specific Failures:")
            for i, f in enumerate(self.failures, 1):
                lines.append(f"{i}. **{f.identifier}**")
                lines.append(f"   Error: {f.message}")
                if f.details:
                    indented_details = "\n   ".join(f.details.strip().splitlines())
                    lines.append(f"   Details:\n   {indented_details}")
                lines.append("")
        else:
            combined_output = (self.stderr or self.stdout).strip()
            tail_lines = combined_output.splitlines()[-25:]
            lines.append("### Error Output:")
            lines.append("```")
            lines.append("\n".join(tail_lines))
            lines.append("```")
            lines.append("")

        lines.append(
            "Please resolve the deterministic failures shown above before completing the stage."
        )
        return "\n".join(lines)


class SemanticErrorNormalizer:
    """Parses raw stdout/stderr from test runners, linters, and compilers into structured failures."""

    @classmethod
    def normalize(
        cls,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> GateResult:
        """Analyze command execution output and extract normalized failures."""
        if exit_code == 0:
            summary = (
                cls._extract_success_summary(stdout, stderr)
                or "All checks passed successfully."
            )
            return GateResult(
                passed=True,
                command=command,
                exit_code=exit_code,
                summary=summary,
                failures=[],
                stdout=stdout,
                stderr=stderr,
            )

        cmd_lower = command.lower()
        combined_lower = (stdout + "\n" + stderr).lower()

        # 1. Pytest
        if (
            "pytest" in cmd_lower
            or ("failed" in combined_lower and "failed in" in combined_lower)
            or "short test summary info" in combined_lower
        ):
            return cls._normalize_pytest(command, exit_code, stdout, stderr)

        # 2. Python unittest
        if "unittest" in cmd_lower:
            return cls._normalize_unittest(command, exit_code, stdout, stderr)

        # 3. Cargo test
        if "cargo test" in cmd_lower:
            return cls._normalize_cargo(command, exit_code, stdout, stderr)

        # 4. Go test
        if "go test" in cmd_lower:
            return cls._normalize_go(command, exit_code, stdout, stderr)

        # 5. Jest / Vitest / npm test
        if any(
            tool in cmd_lower
            for tool in ("jest", "vitest", "npm test", "yarn test", "pnpm test")
        ):
            return cls._normalize_javascript_tests(command, exit_code, stdout, stderr)

        # 6. Static Analysis / Linters (ruff, mypy, flake8, tsc)
        if any(
            tool in cmd_lower for tool in ("ruff", "mypy", "flake8", "tsc", "pylint")
        ):
            return cls._normalize_linter(command, exit_code, stdout, stderr)

        # 7. Generic Fallback
        return cls._normalize_generic(command, exit_code, stdout, stderr)

    @classmethod
    def _extract_success_summary(cls, stdout: str, stderr: str) -> Optional[str]:
        combined = stdout + "\n" + stderr
        # Pytest summary line like '=== 142 passed in 8.35s ==='
        m = re.search(r"=+ ([\d]+ passed[^\n=]*) =+", combined)
        if m:
            return m.group(1).strip()

        # Cargo summary like 'test result: ok. 12 passed;'
        m = re.search(r"test result:\s*ok\.\s*([\d]+ passed[^\n;]*)", combined)
        if m:
            return m.group(1).strip()

        # Go test summary like 'PASS' or 'ok  pkg  0.123s'
        m = re.search(r"^ok\s+([^\s]+)\s+([\d\.]+s)", combined, re.MULTILINE)
        if m:
            return f"ok {m.group(1)} in {m.group(2)}"

        return None

    @classmethod
    def _normalize_pytest(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        # Look for short summary info: 'FAILED tests/test_foo.py::test_bar - AssertionError: ...'
        short_summary_matches = re.findall(
            r"^FAILED\s+([^\s\-]+)\s*-\s*(.*)$", combined, re.MULTILINE
        )
        for identifier, msg in short_summary_matches:
            failures.append(
                GateFailureItem(
                    identifier=identifier.strip(),
                    message=msg.strip(),
                )
            )

        # Also search for '___ test_name ___' blocks if short summary wasn't present
        if not failures:
            block_matches = re.findall(
                r"_{3,}\s+(.*?)\s+_{3,}\n(.*?)(?=\n_{3,}|\n={3,}|\Z)",
                combined,
                re.DOTALL,
            )
            for identifier, block in block_matches:
                err_line = ""
                for line in reversed(block.strip().splitlines()):
                    if any(
                        line.startswith(p)
                        for p in (
                            "E   ",
                            "AssertionError",
                            "TypeError",
                            "ValueError",
                            "KeyError",
                        )
                    ):
                        err_line = line.strip()
                        break
                failures.append(
                    GateFailureItem(
                        identifier=identifier.strip(),
                        message=err_line or "Test failed",
                        details=block.strip()[-500:],
                    )
                )

        # Pytest summary footer e.g. '= 1 failed, 141 passed in 8.35s ='
        summary = ""
        m = re.search(r"=+ ([\d]+ failed[^\n=]*) =+", combined)
        if m:
            summary = m.group(1).strip()
        elif failures:
            summary = f"{len(failures)} tests failed"
        else:
            summary = f"Pytest failed with exit code {exit_code}"

        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_unittest(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        matches = re.findall(
            r"^FAIL:\s+([^\s]+)\s+\((.*?)\)\n(.*?)(?=\n={3,}|\n-{3,}|\Z)",
            combined,
            re.MULTILINE | re.DOTALL,
        )
        for test_fn, test_cls, trace in matches:
            last_line = (
                trace.strip().splitlines()[-1] if trace.strip() else "AssertionError"
            )
            failures.append(
                GateFailureItem(
                    identifier=f"{test_cls}.{test_fn}",
                    message=last_line,
                    details=trace.strip()[-400:],
                )
            )

        summary_match = re.search(r"^FAILED \((.*?)\)$", combined, re.MULTILINE)
        summary = (
            summary_match.group(1).strip()
            if summary_match
            else f"{len(failures)} tests failed"
        )

        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_cargo(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        matches = re.findall(
            r"---- ([^\s]+) stdout ----\n(.*?)(?=\n----|\ntest result:|\Z)",
            combined,
            re.DOTALL,
        )
        for identifier, details in matches:
            panic_line = ""
            for line in details.strip().splitlines():
                if "panicked at" in line:
                    panic_line = line.strip()
                    break
            if not panic_line:
                non_empty = [
                    item.strip()
                    for item in details.strip().splitlines()
                    if item.strip()
                    and not item.strip().startswith("failures:")
                    and item.strip() != identifier.strip()
                ]
                panic_line = non_empty[-1] if non_empty else "Test failure"

            failures.append(
                GateFailureItem(
                    identifier=identifier.strip(),
                    message=panic_line,
                    details=details.strip()[-400:],
                )
            )

        summary_match = re.search(r"test result:\s*FAILED\.\s*([^\n]+)", combined)
        summary = (
            summary_match.group(1).strip() if summary_match else "Cargo test failed"
        )

        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_go(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        matches = re.findall(
            r"^--- FAIL:\s+([^\s]+)\s+\(([^\)]+)\)", combined, re.MULTILINE
        )
        for test_name, duration in matches:
            failures.append(
                GateFailureItem(
                    identifier=test_name,
                    message=f"Failed in {duration}",
                )
            )

        summary = (
            f"{len(failures)} Go tests failed" if failures else "Go test suite failed"
        )
        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_javascript_tests(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        fail_files = re.findall(r"FAIL\s+([^\s\n]+)", combined)
        for ff in fail_files:
            failures.append(
                GateFailureItem(
                    identifier=ff,
                    message="Test suite failed",
                )
            )

        summary_match = re.search(r"Tests:\s*([^\n]+)", combined)
        summary = (
            summary_match.group(1).strip()
            if summary_match
            else "JavaScript tests failed"
        )

        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_linter(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stdout + "\n" + stderr
        failures: list[GateFailureItem] = []

        # Line format: filepath:line:col: message or filepath:line: message
        line_matches = re.findall(
            r"^([a-zA-Z0-9_\-\.\/]+):(\d+):(?:\d+:)?\s*(.*)$",
            combined,
            re.MULTILINE,
        )
        for filepath, line_no, msg in line_matches[:20]:
            failures.append(
                GateFailureItem(
                    identifier=f"{filepath}:{line_no}",
                    message=msg.strip(),
                )
            )

        summary = (
            f"{len(failures)} lint/type diagnostics found"
            if failures
            else "Linter check failed"
        )
        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _normalize_generic(
        cls, command: str, exit_code: int, stdout: str, stderr: str
    ) -> GateResult:
        combined = stderr.strip() or stdout.strip()
        lines = [line.strip() for line in combined.splitlines() if line.strip()]

        # Identify explicit error lines
        error_lines = [
            line
            for line in lines
            if any(
                term in line.lower()
                for term in ("error", "exception", "failed", "fatal", "panic")
            )
        ]

        failures = [
            GateFailureItem(
                identifier=f"error_{idx + 1}",
                message=line,
            )
            for idx, line in enumerate(error_lines[:5])
        ]

        summary = (
            error_lines[0]
            if error_lines
            else f"Command failed with exit code {exit_code}"
        )
        return GateResult(
            passed=False,
            command=command,
            exit_code=exit_code,
            summary=summary[:120],
            failures=failures,
            stdout=stdout,
            stderr=stderr,
        )


class DeterministicGateRunner:
    """Executes verification commands inside a sandbox/workspace and journals gate evaluation events."""

    def __init__(self, storage: Optional[StorageManager] = None):
        self.storage = storage or StorageManager()

    async def run_gate(
        self,
        task_id: str,
        stage: str,
        step_index: int,
        command: str,
        workspace_path: Path,
        timeout_seconds: int = 120,
        sandbox_mode: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> GateResult:
        """Run a verification gate, evaluate pass/fail, and persist immutable event record."""
        start_time = time.monotonic()
        workspace = workspace_path.resolve()

        # Wrap in bwrap sandbox if configured/available
        cmd_or_args, is_shell = build_sandboxed_command(
            cmd=command,
            cwd=workspace,
            sandbox_mode=sandbox_mode,
        )

        try:
            if is_shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd_or_args,
                    cwd=str(workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout_seconds),
            )
            exit_code = proc.returncode if proc.returncode is not None else 1
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            gate_result = GateResult(
                passed=False,
                command=command,
                exit_code=-1,
                summary=f"Gate timed out after {timeout_seconds} seconds",
                failures=[
                    GateFailureItem(
                        identifier="timeout",
                        message=f"Command execution exceeded timeout limit of {timeout_seconds}s",
                    )
                ],
                stdout="",
                stderr=f"Timeout: process killed after {timeout_seconds} seconds",
                duration_ms=duration_ms,
            )
            await self._record_event(task_id, stage, step_index, gate_result)
            return gate_result

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            gate_result = GateResult(
                passed=False,
                command=command,
                exit_code=-1,
                summary=f"Execution error: {e}",
                failures=[
                    GateFailureItem(
                        identifier="execution_error",
                        message=str(e),
                    )
                ],
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms,
            )
            await self._record_event(task_id, stage, step_index, gate_result)
            return gate_result

        duration_ms = (time.monotonic() - start_time) * 1000
        gate_result = SemanticErrorNormalizer.normalize(
            command=command,
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
        )
        gate_result.duration_ms = duration_ms

        await self._record_event(task_id, stage, step_index, gate_result)
        return gate_result

    async def _record_event(
        self,
        task_id: str,
        stage: str,
        step_index: int,
        gate_result: GateResult,
    ):
        """Record immutable gate_eval event in agent_events table."""
        try:
            payload = gate_result.to_dict()
            payload["stdout"] = payload["stdout"][:4000]
            payload["stderr"] = payload["stderr"][:4000]
            await self.storage.record_agent_event(
                task_id=task_id,
                stage=stage,
                step_index=step_index,
                event_type=KernelEventType.GATE_EVAL.value,
                payload=payload,
            )
        except Exception as e:
            logger.warning(
                "[gate] Failed to record gate_eval event for task %s: %s", task_id, e
            )


class TestTamperingDetectedError(Exception):
    """Raised when the agent deletes or modifies baseline tests without authorization."""

    __test__ = False

    def __init__(self, modified_or_deleted: list[str]):
        super().__init__(
            f"TEST_TAMPERING_DETECTED: Pre-existing tests were modified or deleted: {', '.join(modified_or_deleted)}"
        )
        self.modified_or_deleted = modified_or_deleted


class TestFreezeGate:
    """Computes and enforces immutable SHA256 checksums on baseline test files."""

    __test__ = False

    TEST_PATTERNS = (
        "test_*.py",
        "*_test.py",
        "*_test.go",
        "*.test.ts",
        "*.test.js",
        "*.spec.ts",
        "*.spec.js",
    )

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path.resolve()
        self.baseline_hashes: dict[str, str] = self._snapshot_test_hashes()

    def _snapshot_test_hashes(self) -> dict[str, str]:
        """Scan workspace for test files and record their SHA-256 hashes."""
        hashes: dict[str, str] = {}
        if not self.workspace_path.exists():
            return hashes

        # Scan tests directory or whole workspace if tests/ exists
        candidate_dirs = [self.workspace_path / "tests", self.workspace_path / "test"]
        scan_roots = [d for d in candidate_dirs if d.exists() and d.is_dir()]
        if not scan_roots:
            scan_roots = [self.workspace_path]

        for root_dir in scan_roots:
            for root, dirs, files in os.walk(root_dir):
                dirs[:] = [
                    d
                    for d in dirs
                    if d
                    not in (
                        ".git",
                        ".venv",
                        "__pycache__",
                        "node_modules",
                        ".worktrees",
                    )
                ]
                for f in files:
                    if any(
                        re.search(pat.replace("*", ".*"), f)
                        for pat in self.TEST_PATTERNS
                    ):
                        fpath = Path(root) / f
                        try:
                            rel_p = str(fpath.relative_to(self.workspace_path))
                            hashes[rel_p] = hashlib.sha256(
                                fpath.read_bytes()
                            ).hexdigest()
                        except Exception as e:
                            logger.debug(
                                "[test_freeze] Unable to hash %s: %e", fpath, e
                            )

        logger.info(
            "[test_freeze] Baseline test snapshot recorded with %d test files",
            len(hashes),
        )
        return hashes

    def verify_no_tampering(self) -> list[str]:
        """Check current workspace against baseline test hashes.

        Returns list of violated test paths (modified or deleted).
        """
        violations: list[str] = []
        for rel_p, orig_hash in self.baseline_hashes.items():
            fpath = self.workspace_path / rel_p
            if not fpath.exists():
                violations.append(f"{rel_p} (DELETED)")
                continue
            try:
                curr_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
                if curr_hash != orig_hash:
                    violations.append(f"{rel_p} (MODIFIED)")
            except Exception as e:
                violations.append(f"{rel_p} (UNREADABLE: {e})")

        if violations:
            logger.warning("[test_freeze] Test tampering detected: %s", violations)
            raise TestTamperingDetectedError(violations)

        return violations
