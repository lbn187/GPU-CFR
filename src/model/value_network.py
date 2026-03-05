"""Value network V_φ for PBS-conditioned hand-value estimation.

The network takes an encoded PublicBeliefState as input and outputs a value
estimate for each (player, hand) pair – i.e., E[payoff | PBS, player holds hand].

Architecture: Multi-layer perceptron with residual blocks.

References:
  Brown et al., "Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games" (ReBeL, NeurIPS 2020).
  Brown et al., "TurboReBeL", 2021.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..belief.pbs import PublicBeliefState
from ..game.card import NUM_HANDS
from ..game.state import NUM_PLAYERS


class ResidualBlock(nn.Module):
    """A simple residual block: Linear → LayerNorm → ReLU → Linear + skip."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.relu(out + residual)


class ValueNetworkModel(nn.Module):
    """PyTorch module: PBS encoding → (NUM_PLAYERS × NUM_HANDS) values."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_residual_blocks: int = 4,
        output_size: int = NUM_PLAYERS * NUM_HANDS,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_residual_blocks)]
        )
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_size).
        Returns:
            Tensor of shape (batch_size, NUM_PLAYERS, NUM_HANDS).
        """
        h = self.input_proj(x)
        h = self.residual_blocks(h)
        out = self.output_head(h)
        return out.view(-1, NUM_PLAYERS, NUM_HANDS)


class ValueNetwork:
    """Wrapper around ValueNetworkModel providing predict / fit / save / load.

    Args:
        hidden_size: Width of each hidden layer.
        num_residual_blocks: Number of residual blocks in the MLP trunk.
        lr: Learning rate for Adam optimiser.
        device: 'cpu' or 'cuda' (auto-detected if None).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_residual_blocks: int = 4,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        input_size = PublicBeliefState.encoding_size()
        self.model = ValueNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_residual_blocks=num_residual_blocks,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, pbs: PublicBeliefState) -> np.ndarray:
        """Return predicted values for a single PBS.

        Returns:
            ndarray of shape (NUM_PLAYERS, NUM_HANDS) with expected payoffs.
        """
        self.model.eval()
        with torch.no_grad():
            enc = pbs.encode()
            x = torch.tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.model(x)  # (1, NUM_PLAYERS, NUM_HANDS)
        return out.squeeze(0).cpu().numpy()

    def predict_batch(self, pbs_list: List[PublicBeliefState]) -> np.ndarray:
        """Batch inference for a list of PBS.

        Returns:
            ndarray of shape (batch, NUM_PLAYERS, NUM_HANDS).
        """
        self.model.eval()
        with torch.no_grad():
            encodings = np.stack([p.encode() for p in pbs_list], axis=0)
            x = torch.tensor(encodings, dtype=torch.float32, device=self.device)
            out = self.model(x)
        return out.cpu().numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        pbs_list: List[PublicBeliefState],
        targets: np.ndarray,
        epochs: int = 1,
        batch_size: int = 128,
    ) -> float:
        """Update the network on a batch of (PBS, target_value) pairs.

        Args:
            pbs_list: List of PublicBeliefState objects.
            targets: ndarray of shape (N, NUM_PLAYERS, NUM_HANDS) with target
                values produced by the subgame solver.
            epochs: Number of passes over the training data.
            batch_size: Mini-batch size.

        Returns:
            Mean training loss over the last epoch.
        """
        self.model.train()
        encodings = np.stack([p.encode() for p in pbs_list], axis=0)
        x_all = torch.tensor(encodings, dtype=torch.float32)
        y_all = torch.tensor(targets, dtype=torch.float32)

        n = len(pbs_list)
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x = x_all[idx].to(self.device)
                y = y_all[idx].to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and optimiser state to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
