from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Union


class GameTemplate(ABC):
    VARIANT_DRAWN = 0
    VARIANT_WHITE_WON = 1
    VARIANT_BLACK_WON = -1

    VARIANT_DRAWN_STR = "drawn"
    VARIANT_WHITE_WON_STR = "white"
    VARIANT_BLACK_WON_STR = "black"

    BLACK = 0
    WHITE = 1

    ROWS: int
    COLUMNS: int

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def turn(self) -> int:
        pass

    @abstractproperty
    def legal_moves(self) -> List[Any]:
        pass

    @abstractmethod
    def make_move(self, move: Any) -> None:
        pass

    @abstractmethod
    def result(self) -> Union[int, None]:
        pass

    def is_game_over(self) -> bool:
        return self.result() is not None

    def get_board_copy(self) -> List:
        return [[player_row[:] for player_row in player_board[:]] for player_board in self._board]

    @abstractmethod
    def get_copy(self) -> Any:
        pass

    @abstractmethod
    def print_board(self) -> None:
        pass
