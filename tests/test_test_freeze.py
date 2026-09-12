import pytest
from pathlib import Path
from maulness.core.kernel.gates import TestFreezeGate, TestTamperingDetectedError


def test_test_freeze_gate_clean(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_example.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")

    gate = TestFreezeGate(tmp_path)
    assert len(gate.baseline_hashes) == 1

    # Clean verification passes
    violations = gate.verify_no_tampering()
    assert violations == []


def test_test_freeze_gate_detects_modification(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_math.py"
    test_file.write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

    gate = TestFreezeGate(tmp_path)

    # Tamper with test file
    test_file.write_text("def test_add(): assert True # bypassed\n", encoding="utf-8")

    with pytest.raises(TestTamperingDetectedError) as exc_info:
        gate.verify_no_tampering()

    assert "tests/test_math.py (MODIFIED)" in str(exc_info.value)


def test_test_freeze_gate_detects_deletion(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_del.py"
    test_file.write_text("def test_must_fail(): assert False\n", encoding="utf-8")

    gate = TestFreezeGate(tmp_path)

    # Delete test file
    test_file.unlink()

    with pytest.raises(TestTamperingDetectedError) as exc_info:
        gate.verify_no_tampering()

    assert "tests/test_del.py (DELETED)" in str(exc_info.value)
