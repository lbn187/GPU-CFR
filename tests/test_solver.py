"""Tests for RL-CFR and the depth-limited subgame solver."""

import numpy as np
import pytest

from src.game.card import Deck, NUM_HANDS, hand_index
from src.game.state import GameState, NUM_PLAYERS, Street
from src.game.action import FOLD
from src.belief.pbs import PublicBeliefState
from src.solver.cfr import CFRNode, RLCFR
from src.solver.subgame_solver import SubgameSolver


# ---------------------------------------------------------------------------
# Stub value network for testing
# ---------------------------------------------------------------------------


class ZeroValueNetwork:
    """Always predicts zero values – useful for unit testing solver structure."""

    def predict(self, pbs: PublicBeliefState) -> np.ndarray:
        return np.zeros((NUM_PLAYERS, NUM_HANDS), dtype=float)


class UniformValueNetwork:
    """Predicts a small positive constant value."""

    def predict(self, pbs: PublicBeliefState) -> np.ndarray:
        return np.full((NUM_PLAYERS, NUM_HANDS), 0.5, dtype=float)


# ---------------------------------------------------------------------------
# CFRNode tests
# ---------------------------------------------------------------------------


class TestCFRNode:
    def test_uniform_strategy_initially(self):
        node = CFRNode(4)
        strategy = node.get_strategy()
        assert strategy.shape == (4,)
        assert abs(strategy.sum() - 1.0) < 1e-9
        np.testing.assert_allclose(strategy, np.full(4, 0.25), atol=1e-9)

    def test_strategy_after_positive_regret(self):
        node = CFRNode(2)
        # Push regret towards action 0
        node.regret_sum[0] = 10.0
        node.regret_sum[1] = 0.0
        strategy = node.get_strategy()
        assert strategy[0] > strategy[1]

    def test_regret_update_cfr_plus(self):
        node = CFRNode(3)
        action_values = np.array([1.0, 0.5, -1.0])
        node_value = 0.0
        node.update_regrets(action_values, node_value)
        # CFR+: regrets clipped at zero
        assert node.regret_sum[2] == 0.0
        assert node.regret_sum[0] > 0.0

    def test_average_strategy_uniform_when_no_history(self):
        node = CFRNode(3)
        avg = node.get_average_strategy()
        np.testing.assert_allclose(avg, np.full(3, 1 / 3), atol=1e-9)


# ---------------------------------------------------------------------------
# RLCFR tests
# ---------------------------------------------------------------------------


class TestRLCFR:
    def _make_simple_state(self):
        """Return a GameState with hole cards dealt."""
        deck = Deck()
        deck.shuffle()
        state = GameState()
        state.deal_hole_cards(deck)
        return state

    def test_solve_returns_strategies(self):
        state = self._make_simple_state()
        net = ZeroValueNetwork()
        solver = RLCFR(
            value_fn=lambda s, p: net.predict(PublicBeliefState.from_game_state(s))[p],
            depth_limit=2,
            num_iterations=5,
        )
        strategies = solver.solve(state)
        assert isinstance(strategies, dict)
        for key, strat in strategies.items():
            assert isinstance(strat, np.ndarray)
            assert abs(strat.sum() - 1.0) < 1e-9

    def test_solve_builds_nodes(self):
        state = self._make_simple_state()
        net = ZeroValueNetwork()
        solver = RLCFR(
            value_fn=lambda s, p: net.predict(PublicBeliefState.from_game_state(s))[p],
            depth_limit=2,
            num_iterations=3,
        )
        solver.solve(state)
        # Should have created at least one CFR node
        assert len(solver._nodes) > 0

    def test_cfr_converges_to_uniform_on_symmetric_game(self):
        """In a symmetric game with zero terminal values, CFR should converge
        to a near-uniform strategy.  We verify the average strategy is not
        degenerate (all mass on one action)."""
        state = self._make_simple_state()
        net = ZeroValueNetwork()
        solver = RLCFR(
            value_fn=lambda s, p: net.predict(PublicBeliefState.from_game_state(s))[p],
            depth_limit=1,
            num_iterations=20,
        )
        strategies = solver.solve(state)
        for strat in strategies.values():
            assert strat.max() < 1.0 - 1e-6  # not fully degenerate


# ---------------------------------------------------------------------------
# SubgameSolver tests
# ---------------------------------------------------------------------------


class TestSubgameSolver:
    def test_solve_returns_training_datum(self):
        pbs = PublicBeliefState.initial()
        net = ZeroValueNetwork()
        solver = SubgameSolver(
            value_network=net,
            depth_limit=2,
            cfr_iterations=5,
            num_sampled_worlds=3,
        )
        pbs_out, values = solver.solve(pbs)
        assert isinstance(pbs_out, PublicBeliefState)
        assert values.shape == (NUM_PLAYERS, NUM_HANDS)

    def test_values_are_finite(self):
        pbs = PublicBeliefState.initial()
        net = ZeroValueNetwork()
        solver = SubgameSolver(
            value_network=net,
            depth_limit=2,
            cfr_iterations=5,
            num_sampled_worlds=3,
        )
        _, values = solver.solve(pbs)
        assert np.all(np.isfinite(values))
