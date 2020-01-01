from abc import ABC, abstractmethod
from typing import Any, List


class GameTemplate(ABC):
    @abstractmethod
    def legal_moves(self) -> List[Any]:
        pass

    @abstractmethod
    def make_move(self, move: Any) -> None:
        pass

    @abstractmethod
    def result(self) -> str:
        pass

    def is_game_over(self) -> bool:
        return self.result() is not None
