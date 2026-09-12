import time
import pytest
from maulness.core.kernel.budgets import BudgetGuard, BudgetLimits, BudgetExceededError


def test_budget_guard_turn_limit():
    limits = BudgetLimits(max_turns=3, max_duration_seconds=100.0, max_cost_usd=10.0)
    guard = BudgetGuard(limits=limits)

    guard.record_turn()
    guard.record_turn()
    assert guard.total_turns == 2

    with pytest.raises(BudgetExceededError) as exc_info:
        guard.record_turn()

    assert "Turn limit reached" in str(exc_info.value)
    assert exc_info.value.metric == "turns"
    assert exc_info.value.current == 3.0


def test_budget_guard_cost_limit():
    limits = BudgetLimits(max_turns=10, max_duration_seconds=100.0, max_cost_usd=0.05)
    guard = BudgetGuard(limits=limits)

    guard.record_turn(cost_usd=0.02)
    guard.record_turn(cost_usd=0.02)
    assert round(guard.accumulated_cost_usd, 4) == 0.04

    with pytest.raises(BudgetExceededError) as exc_info:
        guard.record_turn(cost_usd=0.02)

    assert "Cost budget reached" in str(exc_info.value)
    assert exc_info.value.metric == "cost_usd"


def test_budget_guard_duration_limit(monkeypatch):
    limits = BudgetLimits(max_turns=10, max_duration_seconds=5.0, max_cost_usd=10.0)
    guard = BudgetGuard(limits=limits)

    # Fast-forward time
    monkeypatch.setattr(time, "monotonic", lambda: guard.start_time + 6.0)

    with pytest.raises(BudgetExceededError) as exc_info:
        guard.check_limits()

    assert "Wall-clock timeout reached" in str(exc_info.value)
    assert exc_info.value.metric == "seconds"


def test_budget_summary():
    guard = BudgetGuard()
    guard.record_turn(prompt_tokens=100, completion_tokens=50, cost_usd=0.001)
    summary = guard.summary()

    assert summary["turns"] == 1
    assert summary["prompt_tokens"] == 100
    assert summary["completion_tokens"] == 50
    assert summary["cost_usd"] == 0.001
