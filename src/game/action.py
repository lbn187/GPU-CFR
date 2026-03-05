"""Action representations for 6-max NLH poker."""

from enum import IntEnum
from typing import Optional


class ActionType(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3


class Action:
    """A player action in a betting round.

    For RAISE actions, *amount* is the total bet size (not the raise increment).
    """

    __slots__ = ("action_type", "amount")

    def __init__(self, action_type: ActionType, amount: float = 0.0) -> None:
        self.action_type = action_type
        self.amount = amount

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Action):
            return self.action_type == other.action_type and self.amount == other.amount
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.action_type, self.amount))

    def __repr__(self) -> str:
        if self.action_type == ActionType.RAISE:
            return f"Action(RAISE, {self.amount})"
        return f"Action({self.action_type.name})"

    def __str__(self) -> str:
        if self.action_type == ActionType.RAISE:
            return f"RAISE({self.amount:.2f})"
        return self.action_type.name


# Convenience singletons for non-parameterised actions.
FOLD = Action(ActionType.FOLD)
CHECK = Action(ActionType.CHECK)
CALL = Action(ActionType.CALL)


def raise_action(amount: float) -> Action:
    """Create a RAISE action for the given total bet size."""
    return Action(ActionType.RAISE, amount)
