"""Card and deck representations for 6-max NLH poker."""

import random
from typing import List, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"
NUM_CARDS = 52
NUM_HANDS = 1326  # C(52, 2)


class Card:
    """Represents a single playing card as an integer in [0, 51].

    Encoding: card = rank * 4 + suit
      rank: 0=2, 1=3, ..., 12=Ace
      suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    """

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        if not (0 <= value < NUM_CARDS):
            raise ValueError(f"Card value must be in [0, 51], got {value}")
        self.value = value

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """Parse a card from a string such as 'As', 'Kh', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank_str, suit_str = s[0].upper(), s[1].lower()
        rank_str = rank_str if rank_str != "1" else "A"
        rank = RANKS.index(rank_str)
        suit = SUITS.index(suit_str)
        return cls(rank * 4 + suit)

    def rank(self) -> int:
        """Return rank index (0=2, …, 12=Ace)."""
        return self.value >> 2

    def suit(self) -> int:
        """Return suit index (0=clubs, …, 3=spades)."""
        return self.value & 3

    def __str__(self) -> str:
        return RANKS[self.rank()] + SUITS[self.suit()]

    def __repr__(self) -> str:
        return f"Card({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __lt__(self, other: "Card") -> bool:
        return self.value < other.value

    def __int__(self) -> int:
        return self.value


def card_rank(card: int) -> int:
    """Return rank of a card integer (0=2, …, 12=Ace)."""
    return card >> 2


def card_suit(card: int) -> int:
    """Return suit of a card integer (0=clubs, …, 3=spades)."""
    return card & 3


def card_to_str(card: int) -> str:
    """Convert card integer to string (e.g., 'As', 'Kh')."""
    return RANKS[card_rank(card)] + SUITS[card_suit(card)]


def str_to_card(s: str) -> int:
    """Convert string to card integer."""
    rank = RANKS.index(s[0].upper())
    suit = SUITS.index(s[1].lower())
    return rank * 4 + suit


def hand_index(c1: int, c2: int) -> int:
    """Return canonical index in [0, 1325] for a 2-card hand.

    The index is defined so that for c1 < c2:
      index = 51*c1 - c1*(c1-1)//2 + (c2 - c1 - 1)
    """
    if c1 > c2:
        c1, c2 = c2, c1
    return 51 * c1 - c1 * (c1 - 1) // 2 + (c2 - c1 - 1)


def index_to_hand(idx: int) -> Tuple[int, int]:
    """Convert a hand index to a (c1, c2) pair with c1 < c2."""
    c1 = 0
    while idx >= 51 - c1:
        idx -= 51 - c1
        c1 += 1
    c2 = c1 + 1 + idx
    return c1, c2


class Deck:
    """A standard 52-card deck that tracks which cards have been dealt."""

    def __init__(self) -> None:
        self._cards = list(range(NUM_CARDS))
        self._dealt: set = set()

    def shuffle(self) -> None:
        """Shuffle the deck and reset dealt set."""
        random.shuffle(self._cards)
        self._dealt = set()

    def deal(self, n: int = 1) -> List[int]:
        """Deal *n* cards and return their integer values."""
        cards: List[int] = []
        for card in self._cards:
            if card not in self._dealt:
                cards.append(card)
                self._dealt.add(card)
                if len(cards) == n:
                    break
        if len(cards) < n:
            raise RuntimeError("Not enough cards remaining in deck")
        return cards

    def remove(self, cards: List[int]) -> None:
        """Mark cards as already dealt (e.g., board cards fixed in advance)."""
        self._dealt.update(cards)

    def available(self) -> List[int]:
        """Return list of cards not yet dealt."""
        return [c for c in self._cards if c not in self._dealt]

    def __len__(self) -> int:
        return NUM_CARDS - len(self._dealt)
