from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List


class GameTemplate(ABC):
    VARIANT_DRAWN = 0
    VARIANT_WHITE_WON = 1
    VARIANT_BLACK_WON = -1

    VARIANT_DRAWN_STR = "drawn"
    VARIANT_WHITE_WON_STR = "white"
    VARIANT_BLACK_WON_STR = "black"

    BLACK = 0
    WHITE = 1

    @abstractproperty
    def legal_moves(self) -> List[Any]:
        pass

    @abstractmethod
    def make_move(self, move: Any) -> None:
        pass

    @abstractmethod
    def result(self) -> int:
        pass

    def is_game_over(self) -> bool:
        return self.result() is not None

    @abstractmethod
    def print_board(self) -> str:
        pass
