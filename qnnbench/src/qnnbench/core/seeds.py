"""
AGENT INSTRUCTIONS — core/seeds.py
=====================================
M02 — RNG policy. THIS IS CRITICAL FOR REPRODUCIBILITY.

Rules:
- Use np.random.SeedSequence([split_seed, init_seed, PURPOSE_ID]) ALWAYS.
- PURPOSE_ID must be from the PurposeID enum below.
- `init` and `minibatch` streams depend ONLY on (split_seed, init_seed).
  They are IDENTICAL across optimisers → enables paired comparison.
- `shot_sampling` stream is also keyed by optimiser name → different shot noise per optimiser.
- Device seeds derived from shot_sampling stream only — NEVER from global state.

BANNED (enforced by T-M02-05 lint):
  - np.random.seed(...)
  - np.random.rand(...)  / np.random.randint(...)  / any global-state call
  - random.seed(...)

TESTS: tests/unit/test_seeds.py (T-M02-01..05)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class PurposeID(IntEnum):
    """Integer IDs for seeding purposes (deterministic across runs)."""
    INIT = 0
    MINIBATCH = 1
    SPSA_DELTA = 2
    SHOT_SAMPLING = 3
    PILOT_SHOTS = 4
    BOOTSTRAP = 5
    TUNING = 6


def _optimiser_salt(optimiser: str | None) -> int:
    """Deterministic integer hash of optimiser name for use in SeedSequence."""
    if optimiser is None:
        return 0
    # Use first 4 bytes of sha1 as an int — stable across Python sessions
    import hashlib
    return int.from_bytes(hashlib.sha1(optimiser.encode()).digest()[:4], "big")


def stream(
    split_seed: int,
    init_seed: int,
    purpose: PurposeID,
    optimiser: str | None = None,
) -> np.random.Generator:
    """Return a fresh, independent Generator for the given purpose.

    Parameters
    ----------
    split_seed:  int — which data split (from {7,17,27,37,47})
    init_seed:   int — which initialisation (0,1,2)
    purpose:     PurposeID — what the RNG is used for
    optimiser:   str | None — required for shot_sampling/pilot_shots/spsa_delta;
                 None means shared across optimisers (init, minibatch, bootstrap, tuning)

    Notes
    -----
    - INIT, MINIBATCH, BOOTSTRAP, TUNING: optimiser=None (shared).
    - SHOT_SAMPLING, PILOT_SHOTS, SPSA_DELTA: pass optimiser name → different per optimiser.
    """
    if purpose in (PurposeID.SHOT_SAMPLING, PurposeID.PILOT_SHOTS, PurposeID.SPSA_DELTA):
        salt = _optimiser_salt(optimiser)
    else:
        salt = 0  # shared across optimisers

    seed_seq = np.random.SeedSequence([split_seed, init_seed, int(purpose), salt])
    return np.random.default_rng(seed_seq)
