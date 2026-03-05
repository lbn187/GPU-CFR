"""Main training orchestrator for the 6-Max NLH TurboReBeL agent.

The Trainer class ties together the TurboReBeL training loop, checkpointing,
and logging.  It is the top-level entry point for long training runs.

Usage::

    from src.model import ValueNetwork
    from src.training import Trainer

    network = ValueNetwork()
    trainer = Trainer(network, checkpoint_dir="checkpoints/")
    trainer.train(num_iterations=10_000)
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from ..model.value_network import ValueNetwork
from .turborebel import TurboReBeL


class Trainer:
    """Orchestrates long TurboReBeL training runs with checkpointing.

    Args:
        value_network: Value network V_φ to train.
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_every: Save a checkpoint every N iterations.
        depth_limit: CFR tree depth limit.
        cfr_iterations: RL-CFR iterations per subgame.
        num_sampled_worlds: Hand-profile samples per subgame.
        replay_buffer_size: Replay buffer capacity.
        min_buffer_size: Min entries before training begins.
        train_batch_size: SGD mini-batch size.
        train_epochs: Gradient update epochs per iteration.
    """

    def __init__(
        self,
        value_network: ValueNetwork,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 500,
        depth_limit: int = 4,
        cfr_iterations: int = 100,
        num_sampled_worlds: int = 50,
        replay_buffer_size: int = 10_000,
        min_buffer_size: int = 200,
        train_batch_size: int = 128,
        train_epochs: int = 1,
    ) -> None:
        self.value_network = value_network
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

        self._algo = TurboReBeL(
            value_network=value_network,
            depth_limit=depth_limit,
            cfr_iterations=cfr_iterations,
            num_sampled_worlds=num_sampled_worlds,
            replay_buffer_size=replay_buffer_size,
            min_buffer_size=min_buffer_size,
            train_batch_size=train_batch_size,
            train_epochs=train_epochs,
        )

        self._total_iterations: int = 0
        self._loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self, num_iterations: int, verbose: bool = True) -> List[float]:
        """Train for *num_iterations* TurboReBeL steps.

        Args:
            num_iterations: Total number of iterations to run.
            verbose: Print progress to stdout.

        Returns:
            List of per-iteration training losses.
        """
        start_time = time.time()
        losses: List[float] = []

        remaining = num_iterations
        chunk = min(self.checkpoint_every, num_iterations)

        while remaining > 0:
            n = min(chunk, remaining)
            batch_losses = self._algo.run(n, verbose=verbose)
            losses.extend(batch_losses)
            self._loss_history.extend(batch_losses)
            self._total_iterations += n
            remaining -= n

            # Checkpoint
            self._save_checkpoint(self._total_iterations)

        elapsed = time.time() - start_time
        if verbose:
            print(
                f"[Trainer] Training complete: {num_iterations} iterations "
                f"in {elapsed:.1f}s ({elapsed/num_iterations*1000:.1f} ms/iter)"
            )
        return losses

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:07d}.pt")
        self.value_network.save(path)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a saved checkpoint."""
        self.value_network.load(path)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def loss_history(self) -> List[float]:
        """Return the full loss history."""
        return list(self._loss_history)

    def smoothed_loss(self, window: int = 100) -> float:
        """Return the mean loss over the last *window* iterations."""
        recent = self._loss_history[-window:]
        if not recent:
            return 0.0
        return float(np.mean(recent))
