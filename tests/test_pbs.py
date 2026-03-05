"""Tests for the Public Belief State module."""

import numpy as np
import pytest

from src.belief.pbs import PublicBeliefState, _uniform_range
from src.game.card import NUM_HANDS, hand_index, index_to_hand, str_to_card
from src.game.state import GameState, Street, NUM_PLAYERS
from src.game.card import Deck


class TestUniformRange:
    def test_sums_to_one(self):
        r = _uniform_range([])
        assert abs(r.sum() - 1.0) < 1e-9

    def test_blocked_hands_zero(self):
        board = [str_to_card("Ah"), str_to_card("Kd")]
        r = _uniform_range(board)
        # Any hand containing Ah or Kd should have zero weight
        for idx in range(NUM_HANDS):
            c1, c2 = index_to_hand(idx)
            if c1 in board or c2 in board:
                assert r[idx] == pytest.approx(0.0)

    def test_with_full_board(self):
        board = [str_to_card(s) for s in ["Ah", "Kd", "Qc", "Js", "Th"]]
        r = _uniform_range(board)
        assert r.sum() > 0
        assert abs(r.sum() - 1.0) < 1e-9


class TestPublicBeliefState:
    def test_initial_factory(self):
        pbs = PublicBeliefState.initial()
        assert pbs.ranges.shape == (NUM_PLAYERS, NUM_HANDS)
        # All player ranges should sum to 1
        for p in range(NUM_PLAYERS):
            assert abs(pbs.ranges[p].sum() - 1.0) < 1e-9

    def test_initial_with_board(self):
        board = [str_to_card("Ah"), str_to_card("Kd"), str_to_card("Qc")]
        pbs = PublicBeliefState.initial(board=board)
        # Hands containing board cards should have zero probability
        for p in range(NUM_PLAYERS):
            for idx in range(NUM_HANDS):
                c1, c2 = index_to_hand(idx)
                if c1 in board or c2 in board:
                    assert pbs.ranges[p, idx] == pytest.approx(0.0)

    def test_from_game_state(self):
        deck = Deck()
        deck.shuffle()
        state = GameState()
        state.deal_hole_cards(deck)
        pbs = PublicBeliefState.from_game_state(state)
        assert pbs.ranges.shape == (NUM_PLAYERS, NUM_HANDS)
        # Player 0's range should be a point mass on their actual hand
        c1, c2 = sorted(state.hole_cards[0])
        h_idx = hand_index(c1, c2)
        assert pbs.ranges[0, h_idx] == pytest.approx(1.0)

    def test_update_range(self):
        pbs = PublicBeliefState.initial()
        # Multiply all hands by equal likelihood – should not change range
        likelihood = np.ones(NUM_HANDS)
        pbs.update_range(0, likelihood)
        assert abs(pbs.ranges[0].sum() - 1.0) < 1e-9

    def test_update_range_focuses_distribution(self):
        pbs = PublicBeliefState.initial()
        # Give one hand high likelihood
        likelihood = np.zeros(NUM_HANDS)
        likelihood[0] = 1.0
        pbs.update_range(0, likelihood)
        # After update, all weight should be on hand 0
        assert pbs.ranges[0, 0] == pytest.approx(1.0)

    def test_sample_hand_profile_consistent(self):
        pbs = PublicBeliefState.initial()
        profile = pbs.sample_hand_profile()
        assert len(profile) == NUM_PLAYERS
        # Each hand index should be in valid range or -1 (folded)
        cards_used = set()
        for p, h_idx in enumerate(profile):
            assert h_idx == -1 or 0 <= h_idx < NUM_HANDS
            if h_idx >= 0:
                c1, c2 = index_to_hand(h_idx)
                # No card conflicts between players
                assert c1 not in cards_used
                assert c2 not in cards_used
                cards_used.add(c1)
                cards_used.add(c2)

    def test_encode_shape(self):
        pbs = PublicBeliefState.initial()
        enc = pbs.encode()
        assert enc.shape == (PublicBeliefState.encoding_size(),)

    def test_encoding_size_matches_encode(self):
        pbs = PublicBeliefState.initial()
        enc = pbs.encode()
        assert len(enc) == PublicBeliefState.encoding_size()

    def test_copy_is_independent(self):
        pbs = PublicBeliefState.initial()
        pbs_copy = pbs.copy()
        pbs_copy.ranges[0, 0] += 999.0
        assert pbs.ranges[0, 0] != pbs_copy.ranges[0, 0]

    def test_active_players_all_active(self):
        pbs = PublicBeliefState.initial()
        assert set(pbs.active_players()) == set(range(NUM_PLAYERS))

    def test_active_players_with_folds(self):
        pbs = PublicBeliefState.initial()
        pbs.folded[2] = True
        pbs.folded[4] = True
        active = pbs.active_players()
        assert 2 not in active
        assert 4 not in active
        assert len(active) == NUM_PLAYERS - 2
