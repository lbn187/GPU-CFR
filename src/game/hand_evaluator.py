"""Hand strength evaluator for Texas Hold'em poker.

Uses a pure-Python implementation that evaluates the best 5-card hand
out of up to 7 cards (2 hole cards + up to 5 board cards).
"""

from collections import Counter
from enum import IntEnum
from itertools import combinations
from typing import List, Tuple

from .card import card_rank, card_suit


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


# A hand score is (HandCategory, [tiebreaker ranks, ...]).
HandScore = Tuple[int, List[int]]


def evaluate_5cards(cards: List[int]) -> HandScore:
    """Return a comparable score for exactly 5 cards.

    Higher score beats lower score (tuple comparison works directly).
    """
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Standard straight
    is_straight = False
    straight_high = 0
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
        straight_high = ranks[0]
    # Wheel: A-2-3-4-5 (ranks sorted desc: [12, 3, 2, 1, 0])
    elif sorted(ranks) == [0, 1, 2, 3, 12]:
        is_straight = True
        straight_high = 3  # 5-high straight

    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.values(), reverse=True)

    if is_straight and is_flush:
        return (HandCategory.STRAIGHT_FLUSH, [straight_high])

    if counts_sorted[0] == 4:
        quad_rank = max(r for r, c in rank_counts.items() if c == 4)
        kicker = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandCategory.FOUR_OF_A_KIND, [quad_rank] + kicker)

    if counts_sorted[0] == 3 and counts_sorted[1] == 2:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        return (HandCategory.FULL_HOUSE, [trip_rank, pair_rank])

    if is_flush:
        return (HandCategory.FLUSH, ranks)

    if is_straight:
        return (HandCategory.STRAIGHT, [straight_high])

    if counts_sorted[0] == 3:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandCategory.THREE_OF_A_KIND, [trip_rank] + kickers)

    if counts_sorted[0] == 2 and counts_sorted[1] == 2:
        pair_ranks = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandCategory.TWO_PAIR, pair_ranks + kicker)

    if counts_sorted[0] == 2:
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandCategory.ONE_PAIR, [pair_rank] + kickers)

    return (HandCategory.HIGH_CARD, ranks)


def evaluate_hand(cards: List[int]) -> HandScore:
    """Return the best 5-card score from *cards* (5–7 cards supported)."""
    if len(cards) == 5:
        return evaluate_5cards(cards)
    best: HandScore = (HandCategory.HIGH_CARD, [-1])
    for combo in combinations(cards, 5):
        score = evaluate_5cards(list(combo))
        if score > best:
            best = score
    return best


def showdown_winner(hole_cards_list: List[List[int]], board: List[int]) -> List[float]:
    """Compute each player's share of the pot at showdown.

    Args:
        hole_cards_list: list of [c1, c2] for each active player.
        board: list of community cards (3–5 cards).

    Returns:
        List of floats in [0, 1] summing to 1 (equal share on tie).
    """
    n = len(hole_cards_list)
    scores = [evaluate_hand(hc + board) for hc in hole_cards_list]
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    share = 1.0 / len(winners)
    return [share if i in winners else 0.0 for i in range(n)]
