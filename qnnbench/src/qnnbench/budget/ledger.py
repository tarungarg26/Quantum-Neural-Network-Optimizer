"""
AGENT INSTRUCTIONS — budget/ledger.py + budget/manager.py
===========================================================
M08 — Budget ledger & manager.

Ledger: append-only event log. Events are COMMITTED only after successful execution.
BudgetManager enforces preflight before every op.

LedgerEvent fields:
  seq, run_id, category, tag, circuits, shots_per_circuit, shots_total, update, wall_s

Categories:
  training    — main training shots (charged)
  monitoring  — validation evaluation shots (charged)
  pilot       — allocation pilot shots (charged)
  search      — hyper-param tuning shots (charged)
  reference   — analytic-only, FREE (never charged)

Invariants:
  - Preflight before EVERY op: raises BudgetPreflightError if it would exceed budget
  - Never overshoot, never silently shrink an operation to fit
  - Run ends when next WHOLE update is unaffordable (status=completed_budget)
  - Reference events NEVER reduce any remaining budget
  - Categories NEVER leak across each other

TESTS: tests/unit/test_budget.py (T-M08-01..08)
  T-M08-01: property: any random sequence ≤ budget per category, always
  T-M08-02: refused op leaves ledger unchanged
  T-M08-03: categories never leak
  T-M08-04: reference events never reduce remaining budget
  T-M08-05: arithmetic sanity checks
  T-M08-06: pilot shots counted in matched-budget total
  T-M08-07: ledger serialises/deserialises losslessly; seq strictly increasing
  T-M08-08: final totals == sum of events.jsonl
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from qnnbench.core.errors import BudgetPreflightError

Category = Literal["training", "monitoring", "pilot", "reference", "search"]
_FREE_CATEGORIES: frozenset[str] = frozenset({"reference"})


@dataclass
class LedgerEvent:
    """A single budget accounting event (immutable after creation)."""

    run_id: str
    category: str
    tag: str
    circuits: int
    shots_per_circuit: int
    shots_total: int
    wall_s: float
    update: int = 0         # which training update this belongs to
    seq: int = field(default=-1, compare=False)  # assigned by Ledger on commit

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    """Append-only event log for budget accounting.

    This is the authoritative source for shot/circuit counts.
    Every circuit execution emits exactly one LedgerEvent.
    """

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []
        self._spent: dict[str, int] = {}  # category → total shots spent
        self._budget_manager: BudgetManager | None = None

    def attach_budget_manager(self, manager: BudgetManager) -> None:
        """Attach a BudgetManager for preflight and spend tracking."""
        self._budget_manager = manager

    def preflight(self, category: str, circuits: int, shots_per_circuit: int) -> None:
        """Check if operation can proceed. Raises BudgetPreflightError if not.

        Called BEFORE any device execution. If this raises, the device is NOT called.
        This is the enforcement point for never-overshoot invariant.
        """
        # Reference is free — always passes
        if category in _FREE_CATEGORIES:
            return
        if self._budget_manager is not None:
            self._budget_manager.preflight(category, circuits, shots_per_circuit)

    def commit(self, event: LedgerEvent) -> None:
        """Record an event. Only called AFTER successful execution."""
        event.seq = len(self._events)
        self._events.append(event)
        if event.category not in _FREE_CATEGORIES:
            self._spent[event.category] = (
                self._spent.get(event.category, 0) + event.shots_total
            )
            if self._budget_manager is not None:
                self._budget_manager.commit(event.category, event.shots_total)

    @property
    def events(self) -> list[LedgerEvent]:
        return list(self._events)

    def total_shots(self, category: str) -> int:
        return self._spent.get(category, 0)

    def total_circuits(self, category: str | None = None) -> int:
        if category is None:
            return sum(e.circuits for e in self._events if e.category not in _FREE_CATEGORIES)
        return sum(e.circuits for e in self._events if e.category == category)

    def to_jsonl(self) -> str:
        lines = [json.dumps(e.to_dict()) for e in self._events]
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, text: str) -> Ledger:
        ledger = cls()
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            event = LedgerEvent(**d)
            ledger._events.append(event)
            if event.category not in _FREE_CATEGORIES:
                ledger._spent[event.category] = (
                    ledger._spent.get(event.category, 0) + event.shots_total
                )
        return ledger

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_jsonl())


class BudgetManager:
    """Enforce shot budgets across categories.

    Never overshoots. If an op would exceed the budget, raises BudgetPreflightError.
    Operations are ALL-OR-NOTHING: they either fully proceed or don't proceed at all.
    """

    def __init__(
        self,
        train_shots: int,
        monitoring_shots: int,
        search_shots: int | None = None,
    ) -> None:
        self._limits: dict[str, int] = {
            "training": train_shots,
            "monitoring": monitoring_shots,
            "search": search_shots or 0,
            "pilot": 0,  # pilot shares the training budget; see can_afford_update
        }
        self._spent: dict[str, int] = {k: 0 for k in self._limits}
        self._ledger: Ledger | None = None

    def attach_ledger(self, ledger: Ledger) -> None:
        self._ledger = ledger
        ledger.attach_budget_manager(self)

    def set_pilot_budget(self, shots: int) -> None:
        """Set pilot budget (portion of training budget reserved for allocation pilots)."""
        self._limits["pilot"] = shots

    def preflight(self, category: str, circuits: int, shots_per_circuit: int) -> None:
        """Raise BudgetPreflightError BEFORE execution if budget would be exceeded.

        Called by Backend.execute() before any device call.
        """
        if category in _FREE_CATEGORIES:
            return
        if category not in self._limits:
            raise ValueError(f"Unknown budget category: {category!r}")

        requested = circuits * shots_per_circuit
        limit = self._limits[category]
        spent = self._spent[category]
        remaining = limit - spent

        if requested > remaining:
            raise BudgetPreflightError(
                category=category,
                requested=requested,
                remaining=remaining,
            )

    def commit(self, category: str, shots: int) -> None:
        """Record spent shots after successful execution."""
        if category in _FREE_CATEGORIES:
            return
        self._spent[category] = self._spent.get(category, 0) + shots

    def remaining(self, category: str) -> int:
        return max(0, self._limits.get(category, 0) - self._spent.get(category, 0))

    def can_afford_update(self, circuits: int, shots_per_circuit: int, category: str = "training") -> bool:
        """Check if a FULL update step is affordable (all-or-nothing)."""
        requested = circuits * shots_per_circuit
        return self.remaining(category) >= requested

    def can_afford(self, category: str, circuits: int, shots_per_circuit: int) -> bool:
        """Check if op is affordable."""
        if category in _FREE_CATEGORIES:
            return True
        return self.can_afford_update(circuits, shots_per_circuit, category)

    def spent(self, category: str) -> int:
        """Return spent shots for category."""
        return self._spent.get(category, 0)

    def total_spent(self, category: str | None = None) -> dict[str, int] | int:
        if category is None:
            return dict(self._spent)
        return self._spent.get(category, 0)
