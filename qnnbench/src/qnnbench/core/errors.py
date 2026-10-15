"""
AGENT INSTRUCTIONS — core/errors.py
=====================================
M01 — Error taxonomy. Define these exact exception classes.
ALL of these must be importable from qnnbench.core.errors.

BudgetExhausted: clean stop, NOT a failure. Budget ran out as expected.
BudgetPreflightError: raised BEFORE execution — op would exceed budget.
NonFiniteError: loss/grad/solve produced nan or inf.
LeakageError: scaler fitted on non-train data.
ConfigFrozenError: frozen config hash mismatch on load.
AccountingMismatchError: ledger total != formula total.

TESTS: tests/unit/test_errors.py
"""

from __future__ import annotations


class QNNBenchError(Exception):
    """Base exception for all qnnbench errors."""


class BudgetExhausted(QNNBenchError):
    """Budget cleanly exhausted — training stops, status = completed_budget.

    This is NOT a failure. The run completed its budget allowance normally.
    """


class BudgetPreflightError(QNNBenchError):
    """An operation would exceed the remaining budget.

    Raised BEFORE any device/circuit execution. The ledger must be unchanged.

    Attributes:
        category: budget category that would be exceeded
        requested: shots/circuits requested
        remaining: shots/circuits remaining
    """

    def __init__(self, category: str, requested: int, remaining: int) -> None:
        self.category = category
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Preflight failed: category={category!r}, "
            f"requested={requested}, remaining={remaining}"
        )


class NonFiniteError(QNNBenchError):
    """A non-finite value (nan/inf) appeared in loss, gradient, or linear solve.

    The run is marked failed. It is NEVER silently reseeded or dropped.
    """

    def __init__(self, location: str, value: object = None) -> None:
        self.location = location
        self.value = value
        super().__init__(f"NonFiniteError in {location!r}: value={value}")


class LeakageError(QNNBenchError):
    """fit_scaler() called on non-train data — data leakage prevention.

    The Scaler must only see train-split rows. This error fires when a caller
    passes val or test data to fit_scaler.
    """


class ConfigFrozenError(QNNBenchError):
    """A frozen config file's hash does not match the stored hash.

    Raised by load_frozen() when the YAML was tampered with after freezing.
    """

    def __init__(self, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Frozen config hash mismatch at {path!r}: "
            f"expected {expected}, got {actual}"
        )


class AccountingMismatchError(QNNBenchError):
    """Ledger total != formula total for a non-adaptive plan.

    Raised during reconciliation step. Indicates a counting bug.
    """

    def __init__(self, ledger_total: int, formula_total: int, context: str = "") -> None:
        self.ledger_total = ledger_total
        self.formula_total = formula_total
        super().__init__(
            f"Accounting mismatch: ledger={ledger_total}, "
            f"formula={formula_total}. {context}"
        )
