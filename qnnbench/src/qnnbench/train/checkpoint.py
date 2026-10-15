"""
AGENT INSTRUCTIONS — train/checkpoint.py
========================================
M11 — Checkpointing and model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qnnbench.core.types import Theta


@dataclass
class CheckpointRecord:
    update: int
    val_mse: float
    val_acc: float
    theta: Theta
    optimiser_state: dict[str, Any]


class CheckpointTracker:
    """Tracks checkpoints and selects best model by minimum val MSE."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.checkpoints: list[CheckpointRecord] = []
        self.best_checkpoint: CheckpointRecord | None = None
        self.patience_counter: int = 0

    def step(
        self,
        update: int,
        val_mse: float,
        val_acc: float,
        theta: Theta,
        optimiser_state: dict[str, Any],
    ) -> bool:
        """Record checkpoint. Returns True if early stopping triggered."""
        rec = CheckpointRecord(
            update=update,
            val_mse=val_mse,
            val_acc=val_acc,
            theta=theta,
            optimiser_state=optimiser_state,
        )
        self.checkpoints.append(rec)

        if self.best_checkpoint is None or val_mse < self.best_checkpoint.val_mse:
            self.best_checkpoint = rec
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience
