"""RL-CFR: Counterfactual Regret Minimisation with learned value functions.

RL-CFR extends standard CFR by replacing hand-crafted terminal (or depth-limit)
value estimates with a neural-network value function V_φ.  At each leaf reached
by the depth limit, V_φ(PBS) is queried to get expected values for every hand.

The implementation follows:
  Brown et al., "Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games" (ReBeL, NeurIPS 2020), with adaptations from
  TurboReBeL for efficient PBS handling.

Key data-structures
-------------------
CFRNode
  Stores per-action cumulative regrets and cumulative strategy (for averaging).
  Indexed by information-set key (a string).

RLCFR
  Runs vanilla CFR+ iterations over a depth-limited game tree whose leaf
  values come from a callable value function (typically the value network).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..game.action import Action, ActionType
from ..game.card import hand_index
from ..game.state import GameState, NUM_PLAYERS

# Type aliases
ValueFn = Callable[[GameState, int], np.ndarray]
# ValueFn(state, player) -> array of shape (NUM_HANDS,) giving expected value
#   for each possible holding of *player* at *state*.


class CFRNode:
    """Stores regrets and average strategy for one information set.

    Args:
        num_actions: Number of legal actions at this information set.
    """

    def __init__(self, num_actions: int) -> None:
        self.num_actions: int = num_actions
        self.regret_sum: np.ndarray = np.zeros(num_actions, dtype=float)
        self.strategy_sum: np.ndarray = np.zeros(num_actions, dtype=float)
        self._iteration: int = 0

    # ------------------------------------------------------------------
    # Strategy computation (regret matching)
    # ------------------------------------------------------------------

    def get_strategy(self, reach_prob: float = 1.0) -> np.ndarray:
        """Return current strategy via regret matching.

        Positive regrets are used as action weights; if all regrets are ≤ 0 a
        uniform strategy is returned.
        """
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        if total > 1e-12:
            strategy = positive / total
        else:
            strategy = np.ones(self.num_actions) / self.num_actions

        # Accumulate weighted strategy for average computation
        self.strategy_sum += reach_prob * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Return the average strategy (Nash approximation)."""
        total = self.strategy_sum.sum()
        if total > 1e-12:
            return self.strategy_sum / total
        return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, action_values: np.ndarray, node_value: float) -> None:
        """Add instantaneous regrets: regret[a] += v(a) - v(node).

        Args:
            action_values: Expected value of taking each action (shape: num_actions).
            node_value: Expected value of the current strategy mix.
        """
        self.regret_sum += action_values - node_value
        # CFR+: clip regrets at zero after update
        self.regret_sum = np.maximum(self.regret_sum, 0.0)
        self._iteration += 1


class RLCFR:
    """Runs RL-CFR (CFR with neural-network leaf values) on a subgame.

    Args:
        value_fn: Callable that returns per-hand values at depth-limit leaves.
            Signature: value_fn(state, player) -> ndarray of shape (NUM_HANDS,).
        depth_limit: Maximum plies to explore before querying value_fn.
        num_iterations: Number of CFR traversals per call to ``solve``.
        discount_alpha: Linear CFR discounting factor (1.0 = no discount).
    """

    def __init__(
        self,
        value_fn: ValueFn,
        depth_limit: int = 4,
        num_iterations: int = 100,
        discount_alpha: float = 1.5,
    ) -> None:
        self.value_fn = value_fn
        self.depth_limit = depth_limit
        self.num_iterations = num_iterations
        self.discount_alpha = discount_alpha
        self._nodes: Dict[str, CFRNode] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, root: GameState) -> Dict[str, np.ndarray]:
        """Run CFR for ``num_iterations`` starting at *root*.

        Returns:
            A dict mapping information-set key → average strategy (ndarray).
        """
        self._nodes.clear()
        reach_probs = np.ones(NUM_PLAYERS, dtype=float)

        for t in range(1, self.num_iterations + 1):
            # Linear CFR discounting: discount regrets from previous iterations
            discount = (t / (t + 1)) ** self.discount_alpha if t > 1 else 1.0
            for p in range(NUM_PLAYERS):
                if not root.folded[p]:
                    self._cfr(root, reach_probs.copy(), p, depth=0, discount=discount)

        return {key: node.get_average_strategy() for key, node in self._nodes.items()}

    # ------------------------------------------------------------------
    # CFR traversal
    # ------------------------------------------------------------------

    def _cfr(
        self,
        state: GameState,
        reach_probs: np.ndarray,
        traverser: int,
        depth: int,
        discount: float,
    ) -> np.ndarray:
        """Recursive CFR traversal.

        Returns an array of shape (NUM_PLAYERS,) with the expected value for
        each player under the current joint strategy.
        """
        # --- Terminal state --------------------------------------------------
        if state.is_terminal():
            return state.terminal_values()

        p = state.current_player
        if p == -1:
            return state.terminal_values()

        # --- Depth-limit leaf: query value function --------------------------
        if depth >= self.depth_limit:
            # Query the value function for the traverser
            return self._leaf_values(state)

        actions = state.legal_actions()
        num_actions = len(actions)
        key = state.info_set_key(p)

        if key not in self._nodes:
            self._nodes[key] = CFRNode(num_actions)
        node = self._nodes[key]

        strategy = node.get_strategy(reach_prob=reach_probs[p])

        # --- Compute value of each action ------------------------------------
        action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=float)
        for a_idx, action in enumerate(actions):
            new_reach = reach_probs.copy()
            new_reach[p] *= strategy[a_idx]
            next_state = state.apply_action(action)
            action_values[a_idx] = self._cfr(
                next_state, new_reach, traverser, depth + 1, discount
            )

        # --- Node value (weighted mix of action values) ----------------------
        node_value = strategy @ action_values  # shape (NUM_PLAYERS,)

        # --- Update regrets for the traverser's player -----------------------
        if p == traverser:
            counterfactual_reach = np.prod(
                [reach_probs[i] for i in range(NUM_PLAYERS) if i != p]
            )
            instant_regrets = counterfactual_reach * (
                action_values[:, p] - node_value[p]
            )
            # Apply discount then add (Linear CFR+)
            node.regret_sum *= discount
            node.regret_sum += instant_regrets
            node.regret_sum = np.maximum(node.regret_sum, 0.0)

        return node_value

    def _leaf_values(self, state: GameState) -> np.ndarray:
        """Estimate per-player values at a depth-limit leaf.

        Queries the value function for each active player and aggregates into
        a single value array of shape (NUM_PLAYERS,).
        """
        values = np.zeros(NUM_PLAYERS, dtype=float)
        for p in range(NUM_PLAYERS):
            if state.folded[p]:
                continue
            v_per_hand = self.value_fn(state, p)  # shape (NUM_HANDS,)
            # Expected value for player p = expectation over their hole cards
            if state.hole_cards[p]:
                c1, c2 = sorted(state.hole_cards[p])
                h_idx = hand_index(c1, c2)
                values[p] = float(v_per_hand[h_idx])
            else:
                # Uniform average if cards not set (should not happen in solve)
                values[p] = float(v_per_hand.mean())
        return values
