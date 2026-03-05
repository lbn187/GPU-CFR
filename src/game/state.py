"""6-Max No-Limit Texas Hold'em game state.

This module models the complete state of a 6-max NLH cash game, including:
  - Player stacks, bets, and pot
  - Hole cards and community cards
  - Current street and acting player
  - Legal action generation
  - Terminal value computation
"""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .action import Action, ActionType, FOLD, CHECK, CALL, raise_action
from .card import Deck, NUM_CARDS, hand_index
from .hand_evaluator import showdown_winner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLAYERS = 6
SMALL_BLIND = 0.5
BIG_BLIND = 1.0
STARTING_STACK = 100.0

# Allowed raise sizes as multiples of the current pot (rounded to 2 d.p.)
POT_FRACTIONS = [0.5, 1.0, 2.0]


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


# Number of board cards dealt at the start of each street
_BOARD_CARDS_PER_STREET = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 1,
    Street.RIVER: 1,
}


class GameState:
    """Complete state of a 6-max NLH hand.

    Attributes:
        num_players: Number of seats (always 6).
        stacks: Current stack for each player (array of floats).
        bets: Amount each player has committed in the *current* street.
        pot: Chips already committed in previous streets.
        hole_cards: List of 2 ints per player (empty = not dealt / folded).
        board: Community cards dealt so far.
        street: Current street (preflop / flop / turn / river).
        current_player: Index of the player whose turn it is to act.
        folded: Boolean array – True if the player has folded.
        all_in: Boolean array – True if the player is all-in.
        button: Seat index of the dealer button.
        action_history: Ordered list of (player_idx, Action) tuples this hand.
    """

    def __init__(
        self,
        button: int = 0,
        starting_stack: float = STARTING_STACK,
        small_blind: float = SMALL_BLIND,
        big_blind: float = BIG_BLIND,
    ) -> None:
        self.num_players: int = NUM_PLAYERS
        self.button: int = button % NUM_PLAYERS
        self.small_blind: float = small_blind
        self.big_blind: float = big_blind
        self._starting_stack: float = starting_stack

        self.stacks: np.ndarray = np.full(NUM_PLAYERS, starting_stack, dtype=float)
        self.bets: np.ndarray = np.zeros(NUM_PLAYERS, dtype=float)
        self.pot: float = 0.0
        self.hole_cards: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.board: List[int] = []
        self.street: Street = Street.PREFLOP
        self.folded: np.ndarray = np.zeros(NUM_PLAYERS, dtype=bool)
        self.all_in: np.ndarray = np.zeros(NUM_PLAYERS, dtype=bool)
        self.action_history: List[Tuple[int, Action]] = []

        # Betting round tracking: number of active players who still need to act.
        # A player "needs to act" if they haven't acted since the last raise (or
        # at all this street).  The round ends when _players_left reaches 0.
        self._current_bet: float = 0.0
        self._players_left: int = 0

        # Post blinds and set acting player
        self._post_blinds()
        self._deck: Optional[Deck] = None
        self.current_player: int = self._first_to_act_preflop()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _sb_seat(self) -> int:
        return (self.button + 1) % self.num_players

    def _bb_seat(self) -> int:
        return (self.button + 2) % self.num_players

    def _post_blinds(self) -> None:
        sb = self._sb_seat()
        bb = self._bb_seat()
        self._place_bet(sb, self.small_blind)
        self._place_bet(bb, self.big_blind)
        self._current_bet = self.big_blind
        # Preflop: ALL players must act (SB and BB posting does not count as action).
        self._players_left = self._num_active_non_allin()

    def _first_to_act_preflop(self) -> int:
        """UTG is the first to act preflop (seat after BB)."""
        return self._next_active((self.button + 3) % self.num_players, skip_current=False)

    def _place_bet(self, player: int, amount: float) -> None:
        """Move *amount* from player's stack to their current-street bet."""
        actual = min(amount, self.stacks[player])
        self.stacks[player] -= actual
        self.bets[player] += actual
        if self.stacks[player] == 0:
            self.all_in[player] = True
        self._current_bet = max(self._current_bet, self.bets[player])

    def _num_active_non_allin(self) -> int:
        """Return count of non-folded, non-all-in players."""
        return int(np.sum(~self.folded & ~self.all_in))

    # ------------------------------------------------------------------
    # Dealing
    # ------------------------------------------------------------------

    def deal_hole_cards(self, deck: Deck) -> None:
        """Deal 2 hole cards to every active player using *deck*."""
        self._deck = deck
        for p in range(self.num_players):
            self.hole_cards[p] = deck.deal(2)

    def deal_board(self, deck: Deck) -> None:
        """Deal the next street's board cards from *deck*."""
        n = _BOARD_CARDS_PER_STREET[self.street]
        self.board.extend(deck.deal(n))

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def legal_actions(self) -> List[Action]:
        """Return the list of legal actions for the current player."""
        p = self.current_player
        actions: List[Action] = [FOLD]

        call_amount = self._current_bet - self.bets[p]
        if call_amount <= 0:
            actions.append(CHECK)
        else:
            # Can call (possibly all-in)
            actions.append(CALL)

        # Raise options
        for frac in POT_FRACTIONS:
            pot_total = self.pot + self.bets.sum()
            raise_size = round(frac * pot_total, 2)
            total_bet = self._current_bet + raise_size
            total_bet = max(total_bet, self._current_bet + self.big_blind)
            total_bet = min(total_bet, self.stacks[p] + self.bets[p])  # cap at stack
            if total_bet > self._current_bet:
                actions.append(raise_action(round(total_bet, 2)))

        # All-in shove
        shove = self.stacks[p] + self.bets[p]
        if shove > self._current_bet:
            # avoid duplicates with raise actions already added
            shove_action = raise_action(round(shove, 2))
            if shove_action not in actions:
                actions.append(shove_action)

        return actions

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> "GameState":
        """Return a *new* GameState after the current player applies *action*."""
        state = copy.deepcopy(self)
        p = state.current_player

        if action.action_type == ActionType.FOLD:
            state.folded[p] = True
            state._players_left -= 1

        elif action.action_type == ActionType.CHECK:
            state._players_left -= 1

        elif action.action_type == ActionType.CALL:
            call_amount = state._current_bet - state.bets[p]
            state._place_bet(p, call_amount)
            state._players_left -= 1

        elif action.action_type == ActionType.RAISE:
            raise_total = action.amount
            additional = raise_total - state.bets[p]
            state._place_bet(p, additional)
            # After a raise all active non-allin players except the raiser must
            # act again.  If the raiser went all-in they are no longer in the
            # active_non_allin count, so we do not subtract them.
            active_non_allin = state._num_active_non_allin()
            if not state.all_in[p]:
                state._players_left = active_non_allin - 1
            else:
                state._players_left = active_non_allin

        state.action_history.append((p, action))
        state._advance()
        return state

    def _advance(self) -> None:
        """Advance the game: move to next player or next street."""
        # One player remaining – uncontested pot
        active = [i for i in range(self.num_players) if not self.folded[i]]
        if len(active) == 1:
            self.current_player = -1
            return

        # Betting round over when no active non-allin player still needs to act
        if self._players_left <= 0 or self._num_active_non_allin() == 0:
            self._end_street()
        else:
            # Advance to the next non-folded, non-allin player
            last_actor = self.current_player
            self.current_player = self._next_active(
                (last_actor + 1) % self.num_players
            )

    def _end_street(self) -> None:
        """Move chips to pot and advance to the next street."""
        self.pot += float(self.bets.sum())
        self.bets[:] = 0.0
        self._current_bet = 0.0

        if self.street == Street.RIVER:
            self.current_player = -1  # showdown
            return

        self.street = Street(self.street + 1)
        # Reset players_left for the new street
        self._players_left = self._num_active_non_allin()
        if self._players_left == 0:
            # Everyone remaining is all-in: run out board and showdown
            self.current_player = -1
            return
        # First to act post-flop: first active player left of button
        self.current_player = self._next_active(
            (self.button + 1) % self.num_players, skip_current=False
        )

    def _next_active(self, start: int, skip_current: bool = True) -> int:
        """Find the next non-folded, non-all-in player starting at *start*."""
        checked = 0
        p = start
        while checked < self.num_players:
            if not self.folded[p] and not self.all_in[p]:
                return p
            p = (p + 1) % self.num_players
            checked += 1
        return -1  # all-in or folded (shouldn't happen in normal play)

    # ------------------------------------------------------------------
    # Terminal state
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        """Return True if the hand is over."""
        active = [i for i in range(self.num_players) if not self.folded[i]]
        return len(active) == 1 or self.current_player == -1

    def terminal_values(self) -> np.ndarray:
        """Compute each player's net chip gain/loss at terminal state.

        Returns an array of shape (num_players,) representing each player's
        profit relative to their starting stack.
        """
        assert self.is_terminal(), "terminal_values called on non-terminal state"
        total_pot = float(self.pot + self.bets.sum())
        active = [i for i in range(self.num_players) if not self.folded[i]]

        # Amount each player has committed = starting_stack - current_stack
        amount_in: np.ndarray = self._starting_stack - self.stacks

        if len(active) == 1:
            # Uncontested: winner gets total pot
            values = -amount_in.copy()
            values[active[0]] += total_pot
        else:
            # Showdown
            active_hole_cards = [self.hole_cards[i] for i in active]
            shares = showdown_winner(active_hole_cards, self.board)
            values = -amount_in.copy()
            for idx, player in enumerate(active):
                values[player] += shares[idx] * total_pot

        return values

    # ------------------------------------------------------------------
    # Information set key
    # ------------------------------------------------------------------

    def info_set_key(self, player: int) -> str:
        """Return a string key identifying the information set for *player*.

        The key captures everything the player can observe:
        their hole cards, the community board, and the action history.
        """
        hole = tuple(sorted(self.hole_cards[player])) if self.hole_cards[player] else ()
        board = tuple(self.board)
        history = tuple(
            (p, str(a)) for p, a in self.action_history
        )
        return str((hole, board, history))

    # ------------------------------------------------------------------
    # Public information
    # ------------------------------------------------------------------

    def public_info(self) -> Dict:
        """Return a dict of information visible to all players."""
        return {
            "street": int(self.street),
            "board": list(self.board),
            "pot": self.pot,
            "bets": self.bets.tolist(),
            "stacks": self.stacks.tolist(),
            "folded": self.folded.tolist(),
            "all_in": self.all_in.tolist(),
            "button": self.button,
            "action_history": [(p, str(a)) for p, a in self.action_history],
        }

    def __repr__(self) -> str:
        board_str = " ".join(str(c) for c in self.board)
        return (
            f"GameState(street={self.street.name}, board=[{board_str}], "
            f"pot={self.pot:.2f}, current_player={self.current_player})"
        )
