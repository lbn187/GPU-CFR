"""Depth-limited subgame solver for 6-max NLH.

The SubgameSolver wraps RLCFR and coordinates:
  1. Constructing a GameState from a PublicBeliefState.
  2. Sampling concrete hand profiles for the CFR traversal.
  3. Running RL-CFR to obtain a strategy profile.
  4. Collecting (PBS, per-hand-value) training pairs.

References:
  Brown et al., "TurboReBeL: 250× Accelerated Belief Learning for large
  Imperfect-Information Extensive-Form Games", 2021.
  Jackson, "Targeted CFR", 2017 (depth-limited solving).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..belief.pbs import PublicBeliefState
from ..game.card import hand_index, index_to_hand, NUM_HANDS, Deck
from ..game.state import GameState, NUM_PLAYERS, Street
from .cfr import RLCFR


TrainingDatum = Tuple[PublicBeliefState, np.ndarray]
# (pbs, values) where values has shape (NUM_PLAYERS, NUM_HANDS):
#   values[p, h] = expected payoff for player p if they hold hand h at this PBS.


class SubgameSolver:
    """Solves a depth-limited subgame rooted at a PBS using RL-CFR.

    Args:
        value_network: An object with a ``predict(pbs)`` method returning an
            ndarray of shape (NUM_PLAYERS, NUM_HANDS) with value estimates.
        depth_limit: Maximum CFR tree depth before querying the value network.
        cfr_iterations: Number of CFR iterations per subgame.
        num_sampled_worlds: Number of hand-profile samples used to estimate
            expected values for the training target.
    """

    def __init__(
        self,
        value_network,
        depth_limit: int = 4,
        cfr_iterations: int = 100,
        num_sampled_worlds: int = 50,
    ) -> None:
        self.value_network = value_network
        self.depth_limit = depth_limit
        self.cfr_iterations = cfr_iterations
        self.num_sampled_worlds = num_sampled_worlds

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, pbs: PublicBeliefState) -> TrainingDatum:
        """Solve the subgame at *pbs* and return a training datum.

        The subgame is solved by:
          1. Sampling ``num_sampled_worlds`` consistent hand profiles.
          2. For each profile, constructing a concrete GameState and running
             RL-CFR to obtain a strategy.
          3. Averaging the per-hand CFR values across sampled worlds to get
             an estimate of V_φ(PBS).

        Returns:
            (pbs, values) where values[p, h] = E[payoff for player p with hand h].
        """
        accumulated_values = np.zeros((NUM_PLAYERS, NUM_HANDS), dtype=float)
        accumulated_counts = np.zeros((NUM_PLAYERS, NUM_HANDS), dtype=float)

        for _ in range(self.num_sampled_worlds):
            profile = pbs.sample_hand_profile()
            state = self._pbs_to_state(pbs, profile)
            if state is None:
                continue

            solver = RLCFR(
                value_fn=self._make_value_fn(pbs),
                depth_limit=self.depth_limit,
                num_iterations=self.cfr_iterations,
            )
            solver.solve(state)

            # Collect per-hand values from the average strategy
            world_values = self._compute_world_values(state, solver, pbs)
            for p in range(NUM_PLAYERS):
                if not pbs.folded[p]:
                    h_idx = profile[p]
                    if h_idx >= 0:
                        accumulated_values[p, h_idx] += world_values[p]
                        accumulated_counts[p, h_idx] += 1.0

        # Average across sampled worlds
        mask = accumulated_counts > 0
        avg_values = np.where(mask, accumulated_values / np.maximum(accumulated_counts, 1), 0.0)

        return pbs, avg_values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pbs_to_state(
        self, pbs: PublicBeliefState, profile: List[int]
    ) -> Optional[GameState]:
        """Construct a concrete GameState from PBS and a hand profile."""
        try:
            state = GameState(
                starting_stack=float(pbs.stacks.mean()) if pbs.stacks.any() else 100.0,
            )
            # Override game attributes from PBS
            state.pot = pbs.pot
            state.bets = pbs.bets.copy()
            state.stacks = pbs.stacks.copy()
            state.street = pbs.street
            state.folded = pbs.folded.copy()
            state.board = list(pbs.board)

            # Assign hole cards from profile
            used: set = set(pbs.board)
            for p in range(NUM_PLAYERS):
                if pbs.folded[p]:
                    state.hole_cards[p] = []
                    continue
                h_idx = profile[p]
                if h_idx < 0:
                    state.hole_cards[p] = []
                    continue
                c1, c2 = index_to_hand(h_idx)
                if c1 in used or c2 in used:
                    return None  # conflicting cards – skip this world
                used.add(c1)
                used.add(c2)
                state.hole_cards[p] = [c1, c2]

            return state
        except Exception:
            return None

    def _make_value_fn(self, pbs: PublicBeliefState):
        """Return a value function that queries the value network at leaves."""
        network = self.value_network
        _pbs = pbs

        def value_fn(state: GameState, player: int) -> np.ndarray:
            # Build a PBS from the current leaf state to query the network
            leaf_pbs = PublicBeliefState.from_game_state(state)
            predicted = network.predict(leaf_pbs)  # (NUM_PLAYERS, NUM_HANDS)
            return predicted[player]

        return value_fn

    def _compute_world_values(
        self,
        root: GameState,
        solver: RLCFR,
        pbs: PublicBeliefState,
    ) -> np.ndarray:
        """Estimate per-player values for one sampled world.

        Uses the terminal values if the subgame is already terminal, otherwise
        uses the average strategy to compute expected value via a quick rollout.
        """
        if root.is_terminal():
            return root.terminal_values()

        # Single-sample rollout following the average strategy
        state = root
        max_steps = 50
        for _ in range(max_steps):
            if state.is_terminal():
                break
            actions = state.legal_actions()
            key = state.info_set_key(state.current_player)
            if key in solver._nodes:
                avg_strat = solver._nodes[key].get_average_strategy()
                action_idx = int(np.random.choice(len(actions), p=avg_strat))
            else:
                action_idx = np.random.randint(len(actions))
            state = state.apply_action(actions[action_idx])

        if state.is_terminal():
            return state.terminal_values()
        # Fallback: query value network
        leaf_pbs = PublicBeliefState.from_game_state(state)
        predicted = self.value_network.predict(leaf_pbs)
        return predicted.mean(axis=1)  # shape (NUM_PLAYERS,)
