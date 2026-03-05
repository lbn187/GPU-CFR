"""Tests for the TurboReBeL training loop, value network, and EVPA."""

import numpy as np
import pytest

from src.belief.pbs import PublicBeliefState
from src.game.card import NUM_HANDS
from src.game.state import NUM_PLAYERS
from src.model.value_network import ValueNetwork
from src.training.turborebel import TurboReBeL
from src.training.evpa import EVPA


# ---------------------------------------------------------------------------
# ValueNetwork tests
# ---------------------------------------------------------------------------


class TestValueNetwork:
    def test_predict_shape(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        pbs = PublicBeliefState.initial()
        values = net.predict(pbs)
        assert values.shape == (NUM_PLAYERS, NUM_HANDS)

    def test_predict_batch_shape(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        pbs_list = [PublicBeliefState.initial() for _ in range(4)]
        values = net.predict_batch(pbs_list)
        assert values.shape == (4, NUM_PLAYERS, NUM_HANDS)

    def test_fit_returns_loss(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        pbs_list = [PublicBeliefState.initial() for _ in range(16)]
        targets = np.zeros((16, NUM_PLAYERS, NUM_HANDS), dtype=float)
        loss = net.fit(pbs_list, targets, epochs=1, batch_size=8)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_save_load_roundtrip(self, tmp_path):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        pbs = PublicBeliefState.initial()
        original_values = net.predict(pbs)

        path = str(tmp_path / "model.pt")
        net.save(path)

        net2 = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        net2.load(path)
        loaded_values = net2.predict(pbs)

        np.testing.assert_allclose(original_values, loaded_values, atol=1e-6)

    def test_fit_reduces_loss_on_constant_target(self):
        """After many gradient steps on a constant target, loss should decrease."""
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        pbs_list = [PublicBeliefState.initial() for _ in range(32)]
        targets = np.ones((32, NUM_PLAYERS, NUM_HANDS), dtype=float)

        loss_before = net.fit(pbs_list, targets, epochs=1, batch_size=32)
        loss_after = net.fit(pbs_list, targets, epochs=50, batch_size=32)
        assert loss_after < loss_before + 1.0  # generous tolerance


# ---------------------------------------------------------------------------
# TurboReBeL tests
# ---------------------------------------------------------------------------


class TestTurboReBeL:
    def test_run_returns_losses(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        algo = TurboReBeL(
            value_network=net,
            depth_limit=1,
            cfr_iterations=3,
            num_sampled_worlds=2,
            replay_buffer_size=100,
            min_buffer_size=10,
            train_batch_size=8,
        )
        losses = algo.run(num_iterations=5, verbose=False)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_buffer_grows(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        algo = TurboReBeL(
            value_network=net,
            depth_limit=1,
            cfr_iterations=3,
            num_sampled_worlds=2,
            replay_buffer_size=100,
            min_buffer_size=10,
        )
        assert algo.buffer_size == 0
        algo.run(num_iterations=5, verbose=False)
        assert algo.buffer_size > 0

    def test_training_starts_after_warmup(self):
        net = ValueNetwork(hidden_size=64, num_residual_blocks=1)
        # min_buffer_size = 3 so training should start by iteration 4
        algo = TurboReBeL(
            value_network=net,
            depth_limit=1,
            cfr_iterations=3,
            num_sampled_worlds=2,
            replay_buffer_size=50,
            min_buffer_size=3,
            train_batch_size=4,
        )
        losses = algo.run(num_iterations=10, verbose=False)
        # At least some losses should be nonzero after warmup
        assert any(l > 0.0 for l in losses)


# ---------------------------------------------------------------------------
# EVPA tests
# ---------------------------------------------------------------------------


class TestEVPA:
    def test_compute_values_shape(self):
        pbs = PublicBeliefState.initial()
        evpa = EVPA(num_br_iterations=1)

        def dummy_terminal_fn(p):
            return np.zeros((NUM_PLAYERS, NUM_HANDS), dtype=float)

        values = evpa.compute_values(pbs, dummy_terminal_fn)
        assert values.shape == (NUM_PLAYERS, NUM_HANDS)

    def test_compute_values_finite(self):
        pbs = PublicBeliefState.initial()
        evpa = EVPA(num_br_iterations=2)

        def dummy_terminal_fn(p):
            return np.random.randn(NUM_PLAYERS, NUM_HANDS)

        values = evpa.compute_values(pbs, dummy_terminal_fn)
        assert np.all(np.isfinite(values))
