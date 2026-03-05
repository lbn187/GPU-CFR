"""EVPA – Expected Value with Perfect Approximation (partial implementation).

EVPA computes per-hand expected values by treating opponent hands as fixed and
computing best-response values against them.  It provides an alternative value
target to RL-CFR for the training data pipeline.

Note: This module contains the structural scaffolding for EVPA integration.
Full EVPA with multi-player best-response iteration will be completed in a
subsequent version.

References:
  Waugh, K., et al. "Abstraction Pathologies in Extensive Games." (2009).
  Johanson, M., et al. "Finding Optimal Abstract Strategies in Extensive-Form
  Games." AAAI 2012.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..belief.pbs import PublicBeliefState
from ..game.card import NUM_HANDS, index_to_hand
from ..game.state import GameState, NUM_PLAYERS


class EVPA:
    """Partial EVPA implementation.

    Computes an approximate expected-value vector for each player hand by
    evaluating a single-step best response against the opponent's current range.

    Args:
        num_br_iterations: Number of best-response computation passes.
            (Currently used as a placeholder for future multi-pass BR.)
    """

    def __init__(self, num_br_iterations: int = 1) -> None:
        self.num_br_iterations = num_br_iterations

    def compute_values(
        self,
        pbs: PublicBeliefState,
        terminal_value_fn,
    ) -> np.ndarray:
        """Compute approximate EVPA values for each (player, hand) pair.

        The current implementation uses a simplified one-step look-ahead:
        for each player p and each hand h held by p, it estimates the value
        as the expected terminal payoff against opponents' ranges, assuming
        all players play according to their current (range-weighted) strategy.

        Args:
            pbs: The current Public Belief State.
            terminal_value_fn: Callable (pbs) -> ndarray of shape
                (NUM_PLAYERS, NUM_HANDS) giving terminal/leaf values.

        Returns:
            ndarray of shape (NUM_PLAYERS, NUM_HANDS) with EVPA value estimates.
        """
        # Phase 1: Base value estimate from terminal_value_fn
        base_values = terminal_value_fn(pbs)

        # Phase 2 (stub): Best-response correction passes.
        # TODO: Implement full multi-player BR iteration with range updates.
        values = base_values.copy()
        for _ in range(self.num_br_iterations):
            values = self._br_pass(pbs, values)

        return values

    def _br_pass(
        self,
        pbs: PublicBeliefState,
        values: np.ndarray,
    ) -> np.ndarray:
        """One best-response update pass (simplified).

        Adjusts each player's per-hand value by computing the weighted
        average of opponent ranges, approximating a single BR iteration.

        This is a simplified approximation; a complete implementation would
        solve the full best-response problem over the game tree.
        """
        updated = values.copy()
        active = pbs.active_players()

        for p in active:
            # Compute the weighted sum of opponents' range distributions
            opp_range_sum = np.zeros(NUM_HANDS, dtype=float)
            for opp in active:
                if opp != p:
                    opp_range_sum += pbs.ranges[opp]

            # Normalise
            total = opp_range_sum.sum()
            if total > 1e-12:
                opp_range_norm = opp_range_sum / total
            else:
                opp_range_norm = np.ones(NUM_HANDS) / NUM_HANDS

            # Adjust values: hands that block opponent strong holdings are
            # slightly more valuable (blocking effect).
            board_set = set(pbs.board)
            for h_idx in range(NUM_HANDS):
                c1, c2 = index_to_hand(h_idx)
                if c1 in board_set or c2 in board_set:
                    continue
                # Blocked hands lower opponent's probability of holding
                # strong combos; we capture this as a small positive adjustment.
                blocker_weight = float(opp_range_norm[h_idx])
                updated[p, h_idx] += 0.01 * blocker_weight  # small correction term

        return updated
