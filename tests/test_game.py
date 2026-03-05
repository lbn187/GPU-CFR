"""Tests for the game engine components."""

import pytest

from src.game.card import (
    Card,
    Deck,
    NUM_CARDS,
    NUM_HANDS,
    card_rank,
    card_suit,
    card_to_str,
    hand_index,
    index_to_hand,
    str_to_card,
)
from src.game.action import Action, ActionType, FOLD, CHECK, CALL, raise_action
from src.game.hand_evaluator import (
    HandCategory,
    evaluate_5cards,
    evaluate_hand,
    showdown_winner,
)
from src.game.state import GameState, Street, NUM_PLAYERS


# ---------------------------------------------------------------------------
# Card tests
# ---------------------------------------------------------------------------


class TestCard:
    def test_from_str_round_trip(self):
        for raw in ["2c", "As", "Kh", "Td", "9s", "Jc"]:
            c = Card.from_str(raw)
            assert str(c) == raw

    def test_rank_suit(self):
        ace_spades = Card.from_str("As")
        assert ace_spades.rank() == 12
        assert ace_spades.suit() == 3

        two_clubs = Card.from_str("2c")
        assert two_clubs.rank() == 0
        assert two_clubs.suit() == 0

    def test_card_range(self):
        for v in range(NUM_CARDS):
            c = Card(v)
            assert 0 <= c.rank() <= 12
            assert 0 <= c.suit() <= 3

    def test_invalid_card_raises(self):
        with pytest.raises(ValueError):
            Card(-1)
        with pytest.raises(ValueError):
            Card(52)


class TestHandIndex:
    def test_index_to_hand_round_trip(self):
        for idx in range(NUM_HANDS):
            c1, c2 = index_to_hand(idx)
            assert c1 < c2
            assert hand_index(c1, c2) == idx

    def test_hand_index_commutative(self):
        assert hand_index(0, 1) == hand_index(1, 0)
        assert hand_index(5, 10) == hand_index(10, 5)

    def test_total_hands(self):
        seen = set()
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                seen.add(hand_index(c1, c2))
        assert len(seen) == NUM_HANDS


class TestDeck:
    def test_deal_unique(self):
        deck = Deck()
        deck.shuffle()
        cards = deck.deal(52)
        assert len(set(cards)) == 52

    def test_deal_respects_removed(self):
        deck = Deck()
        deck.shuffle()
        deck.remove([0, 1, 2])
        cards = deck.deal(5)
        for c in [0, 1, 2]:
            assert c not in cards

    def test_deal_too_many_raises(self):
        deck = Deck()
        deck.shuffle()
        with pytest.raises(RuntimeError):
            deck.deal(53)


# ---------------------------------------------------------------------------
# Hand evaluator tests
# ---------------------------------------------------------------------------


class TestHandEvaluator:
    def _cards(self, *names):
        return [str_to_card(n) for n in names]

    def test_straight_flush_beats_quads(self):
        sf = evaluate_5cards(self._cards("As", "Ks", "Qs", "Js", "Ts"))
        quads = evaluate_5cards(self._cards("Ah", "Ad", "Ac", "As", "Kh"))
        # re-use quads without ace of spades conflict
        quads = evaluate_5cards(self._cards("Ah", "Ad", "Ac", "2s", "Kh"))
        assert sf > quads

    def test_royal_flush(self):
        score = evaluate_5cards(self._cards("As", "Ks", "Qs", "Js", "Ts"))
        assert score[0] == HandCategory.STRAIGHT_FLUSH

    def test_wheel_straight(self):
        score = evaluate_5cards(self._cards("Ac", "2d", "3h", "4s", "5c"))
        assert score[0] == HandCategory.STRAIGHT
        assert score[1] == [3]  # 5-high

    def test_full_house(self):
        score = evaluate_5cards(self._cards("Ah", "Ad", "Ac", "Kh", "Kd"))
        assert score[0] == HandCategory.FULL_HOUSE

    def test_two_pair(self):
        score = evaluate_5cards(self._cards("Ah", "Ad", "Kh", "Kd", "Qc"))
        assert score[0] == HandCategory.TWO_PAIR

    def test_evaluate_7cards(self):
        # Best of 7 should pick the flush
        cards = self._cards("Ah", "Kh", "Qh", "Jh", "2h", "3c", "7d")
        score = evaluate_hand(cards)
        assert score[0] == HandCategory.FLUSH

    def test_showdown_winner_basic(self):
        # Ace-high vs king-high (no board)
        # We need 5 board cards for a valid showdown
        board = self._cards("2c", "3d", "5s", "7h", "9c")
        p1 = self._cards("Ah", "Kd")
        p2 = self._cards("Qh", "Jd")
        result = showdown_winner([p1, p2], board)
        assert result[0] > result[1]

    def test_showdown_split_pot(self):
        board = self._cards("Ah", "Kh", "Qh", "Jh", "Th")
        p1 = self._cards("2c", "3c")
        p2 = self._cards("4d", "5d")
        result = showdown_winner([p1, p2], board)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------


class TestAction:
    def test_singletons(self):
        assert FOLD.action_type == ActionType.FOLD
        assert CHECK.action_type == ActionType.CHECK
        assert CALL.action_type == ActionType.CALL

    def test_raise_action(self):
        r = raise_action(10.0)
        assert r.action_type == ActionType.RAISE
        assert r.amount == 10.0

    def test_equality(self):
        assert raise_action(5.0) == raise_action(5.0)
        assert raise_action(5.0) != raise_action(10.0)


# ---------------------------------------------------------------------------
# GameState tests
# ---------------------------------------------------------------------------


class TestGameState:
    def test_initial_state(self):
        state = GameState()
        assert state.street == Street.PREFLOP
        assert not state.is_terminal()
        assert state.current_player in range(NUM_PLAYERS)

    def test_legal_actions_nonempty(self):
        state = GameState()
        actions = state.legal_actions()
        assert len(actions) > 0
        # Should always include fold
        assert FOLD in actions

    def test_deal_and_fold_to_one(self):
        deck = Deck()
        deck.shuffle()
        state = GameState()
        state.deal_hole_cards(deck)

        # Fold everyone except one player
        for _ in range(NUM_PLAYERS - 1):
            if state.is_terminal():
                break
            state = state.apply_action(FOLD)

        assert state.is_terminal()

    def test_terminal_values_sum_to_zero(self):
        """In a zero-sum game, net values should sum to approximately zero."""
        deck = Deck()
        deck.shuffle()
        state = GameState()
        state.deal_hole_cards(deck)
        deck.deal(5)  # deal board
        state.board = deck.available()[:5]  # force a board

        # Fold everyone except two players
        steps = 0
        while not state.is_terminal() and steps < 20:
            active_count = sum(1 for p in range(NUM_PLAYERS) if not state.folded[p])
            if active_count <= 2:
                break
            state = state.apply_action(FOLD)
            steps += 1

        if state.is_terminal():
            vals = state.terminal_values()
            # Net gains should sum to ~0 (ignoring blind posting edge cases)
            assert abs(vals.sum()) < 1.0  # within 1 chip of zero-sum

    def test_info_set_key_includes_hole_cards(self):
        deck = Deck()
        deck.shuffle()
        state = GameState()
        state.deal_hole_cards(deck)
        key = state.info_set_key(0)
        # Key should be a non-empty string
        assert isinstance(key, str)
        assert len(key) > 0

    def test_public_info_dict(self):
        state = GameState()
        info = state.public_info()
        assert "street" in info
        assert "board" in info
        assert "pot" in info
        assert "bets" in info
