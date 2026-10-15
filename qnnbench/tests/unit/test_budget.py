import numpy as np
import pytest

from qnnbench.budget.ledger import BudgetManager, Ledger, LedgerEvent
from qnnbench.core.errors import BudgetPreflightError


def test_t_m08_01_property_random_ops_never_overshoot():
    """T-M08-01: Any random sequence of ops spent <= budget per category, always."""
    mgr = BudgetManager(train_shots=1000, monitoring_shots=500)
    ledger = Ledger()
    mgr.attach_ledger(ledger)

    rng = np.random.default_rng(42)
    categories = ["training", "monitoring"]

    for _ in range(100):
        cat = rng.choice(categories)
        shots_per_circ = int(rng.choice([16, 32, 64]))
        circuits = int(rng.integers(1, 10))

        try:
            ledger.preflight(cat, circuits, shots_per_circ)
            # Commit event if preflight passes
            ledger.commit(LedgerEvent(
                run_id="test",
                category=cat,
                tag="step",
                circuits=circuits,
                shots_per_circuit=shots_per_circ,
                shots_total=circuits * shots_per_circ,
                wall_s=0.01,
            ))
        except BudgetPreflightError:
            pass

    assert mgr.total_spent("training") <= 1000
    assert mgr.total_spent("monitoring") <= 500


def test_t_m08_02_refused_op_leaves_ledger_unchanged():
    """T-M08-02: Refused op leaves ledger unchanged (no partial commit)."""
    mgr = BudgetManager(train_shots=100, monitoring_shots=50)
    ledger = Ledger()
    mgr.attach_ledger(ledger)

    # Spend 80
    ledger.preflight("training", circuits=8, shots_per_circuit=10)
    ledger.commit(LedgerEvent("test", "training", "ok", 8, 10, 80, 0.01))

    assert len(ledger.events) == 1
    assert mgr.total_spent("training") == 80

    # Attempt to spend 30 (exceeds remaining 20)
    with pytest.raises(BudgetPreflightError):
        ledger.preflight("training", circuits=3, shots_per_circuit=10)

    # Verify ledger untouched
    assert len(ledger.events) == 1
    assert mgr.total_spent("training") == 80


def test_t_m08_03_categories_never_leak():
    """T-M08-03: Categories never leak across each other."""
    mgr = BudgetManager(train_shots=100, monitoring_shots=50)
    ledger = Ledger()
    mgr.attach_ledger(ledger)

    # Exhaust monitoring
    ledger.preflight("monitoring", circuits=5, shots_per_circuit=10)
    ledger.commit(LedgerEvent("test", "monitoring", "m", 5, 10, 50, 0.01))

    assert mgr.remaining("monitoring") == 0
    assert mgr.remaining("training") == 100

    # Training can still proceed
    ledger.preflight("training", circuits=10, shots_per_circuit=10)
    ledger.commit(LedgerEvent("test", "training", "t", 10, 10, 100, 0.01))
    assert mgr.remaining("training") == 0


def test_t_m08_04_reference_events_free():
    """T-M08-04: Reference events never reduce any remaining budget."""
    mgr = BudgetManager(train_shots=100, monitoring_shots=50)
    ledger = Ledger()
    mgr.attach_ledger(ledger)

    # Emit reference event
    ledger.preflight("reference", circuits=500, shots_per_circuit=1000)
    ledger.commit(LedgerEvent("test", "reference", "ref", 500, 1000, 500000, 0.05))

    assert mgr.remaining("training") == 100
    assert mgr.remaining("monitoring") == 50
    assert ledger.total_shots("reference") == 0


def test_t_m08_05_arithmetic_sanity():
    """T-M08-05: Arithmetic sanity on PSR vs SPSA affordance."""
    # PSR: p=16 gate params, m=10 minibatch => (2*16 + 1)*10 = 330 circuits per update
    # With N=32 shots => 330 * 32 = 10,560 shots per update
    # In 100,000 budget => floor(100,000 / 10,560) = 9 updates
    train_budget = 100_000
    circuits_psr = 330
    shots_psr = 32
    shots_per_update_psr = circuits_psr * shots_psr
    max_psr_updates = train_budget // shots_per_update_psr
    assert max_psr_updates == 9

    # SPSA: 2 loss evals per update => 2 * 10 = 20 circuits
    # With N=32 shots => 20 * 32 = 640 shots per update
    # In 100,000 budget => floor(100,000 / 640) = 156 updates
    circuits_spsa = 20
    shots_spsa = 32
    shots_per_update_spsa = circuits_spsa * shots_spsa
    max_spsa_updates = train_budget // shots_per_update_spsa
    assert max_spsa_updates == 156


def test_t_m08_06_pilot_budget_tracking():
    """T-M08-06: Pilot shots counted in budget tracking."""
    mgr = BudgetManager(train_shots=1000, monitoring_shots=200)
    ledger = Ledger()
    mgr.attach_ledger(ledger)
    mgr.set_pilot_budget(300)

    ledger.preflight("pilot", circuits=10, shots_per_circuit=16)
    ledger.commit(LedgerEvent("test", "pilot", "pilot_step", 10, 16, 160, 0.02))

    assert mgr.total_spent("pilot") == 160
    assert mgr.remaining("pilot") == 140


def test_t_m08_07_serialization_lossless_and_monotonic():
    """T-M08-07: Ledger serialises/deserialises losslessly; seq strictly increasing."""
    ledger = Ledger()
    for i in range(5):
        ledger.commit(LedgerEvent("run1", "training", f"tag_{i}", i + 1, 10, (i + 1) * 10, 0.01))

    # Verify monotonic seq
    seqs = [e.seq for e in ledger.events]
    assert seqs == [0, 1, 2, 3, 4]

    jsonl_str = ledger.to_jsonl()
    loaded = Ledger.from_jsonl(jsonl_str)
    assert len(loaded.events) == 5
    assert [e.seq for e in loaded.events] == seqs
    assert loaded.total_shots("training") == ledger.total_shots("training")


def test_t_m08_08_final_totals_match_events():
    """T-M08-08: Final totals equal sum of events."""
    ledger = Ledger()
    ledger.commit(LedgerEvent("run1", "training", "t1", 10, 32, 320, 0.01))
    ledger.commit(LedgerEvent("run1", "training", "t2", 15, 32, 480, 0.02))
    ledger.commit(LedgerEvent("run1", "monitoring", "m1", 5, 32, 160, 0.01))

    total_training = sum(e.shots_total for e in ledger.events if e.category == "training")
    assert total_training == 800
    assert ledger.total_shots("training") == 800
