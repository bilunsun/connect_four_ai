import numpy as np
import random
import time
from typing import List, Tuple

from game_template import GameTemplate


class Gomoku(GameTemplate):
    ROWS = 6
    COLUMNS = 6

    WINNING_CONDITIONS = (
        (  # Vertical
            ((0, -4), (0, -3), (0, -2), (0, -1)),
            ((0, 1), (0, -3), (0, -2), (0, -1)),
            ((0, 3), (0, 2), (0, -1), (0, 1)),
            ((0, 4), (0, 3), (0, 2), (0, 1))
        ),

        (  # Horizontal
            ((-4, 0), (-3, 0), (-2, 0), (-1, 0)),
            ((-3, 0), (1, 0), (-2, 0), (-1, 0)),
            ((3, 0), (2, 0), (-1, 0), (1, 0)),
            ((4, 0), (3, 0), (2, 0), (1, 0))
        ),

        (  # Forward Diagonal
            ((1, -1), (2, -2), (3, -3), (4, -4)),
            ((-1, 1), (1, -1), (2, -2), (3, -3)),
            ((-3, 3), (-2, 2), (-1, 1), (1, -1)),
            ((-4, 4), (-3, 3), (-2, 2), (-1, 1))
        ),

        (  # Backward Diagonal
            ((4, 4), (3, 3), (2, 2), (1, 1)),
            ((3, 3), (2, 2), (-1, -1), (1, 1)),
            ((-3, -3), (-2, -2), (1, 1), (-1, -1)),
            ((-4, -4), (-3, -3), (-2, -2), (-1, -1))
        ),
    )

    def __init__(self) -> None:
        self._board = np.zeros((2, self.ROWS, self.COLUMNS), dtype=int)

        self._turn = self.BLACK
        self._result = None
        self._pieces_count = 0

    def turn(self) -> int:
        return int(self._turn)

    @property
    def legal_moves(self) -> List:
        legal_moves_list = []

        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if not self._board[self.BLACK, i, j] and not self._board[self.WHITE, i, j]:
                    legal_moves_list.append((i, j))

        return legal_moves_list

    def make_move(self, move: Tuple[int, int]) -> None:
        row, col = move
        self._board[self.turn(), row, col] = 1
        self._pieces_count += 1

        if not self.game_ending_move(move):
            self._turn = not self._turn

    def result(self) -> int:
        return self._result

    def game_ending_move(self, move: Tuple[int, int]) -> bool:
        if self._pieces_count == self.ROWS * self.COLUMNS:
            self._result = self.VARIANT_DRAWN
            return True

        row, column = move

        for win_conditions in self.WINNING_CONDITIONS:
            for win_condition in win_conditions:
                win = True
                checked_all = True

                for win_delta in win_condition:
                    delta_column, delta_row = win_delta
                    shifted_row, shifted_column = row + delta_row, column + delta_column

                    # Make sure (shifted_x, shifted_y) is in the board
                    if (shifted_column < 0 or shifted_column > self.COLUMNS - 1
                            or shifted_row < 0 or shifted_row > self.ROWS - 1):
                        checked_all = False
                        break
                    else:
                        if not self.current_board[shifted_row, shifted_column]:
                            win = False
                            break

                if checked_all and win:
                    if self._turn == self.BLACK:
                        self._result = self.VARIANT_BLACK_WON
                    else:
                        self._result = self.VARIANT_WHITE_WON

                    self._turn = not self._turn

                    return True

        return False

    @property
    def current_board(self) -> np.ndarray:
        return self._board[int(self._turn)]

    def make_random_move(self) -> None:
        random_move = random.choice(self.legal_moves)
        self.make_move(random_move)

    def print_board(self) -> None:
        player = "White" if self._turn else "Black"
        output_repr = f"Turn: {player}\n"

        output_repr += "   " + "  ".join(map(chr, np.arange(self.COLUMNS) + ord("A"))) + "\n"

        for i in range(self.ROWS):
            row = chr(i + ord("A")) + "  "
            for j in range(self.COLUMNS):
                if self._board[self.BLACK, i, j]:
                    row += "X  "
                elif self._board[self.WHITE, i, j]:
                    row += "O  "
                else:
                    row += ".  "
            output_repr += row + "\n"

        print(output_repr)


def play_sample_game(verbose: bool = False) -> None:
    game = Gomoku()

    while not game.is_game_over():
        game.make_random_move()

        if verbose:
            game.print_board()

    print(f"Final Result: {game.result()}")
    game.print_board()


def benchmark(iterations: int = 100) -> None:
    times = []

    for _ in range(iterations):
        start_time_s = time.time()

        game = Gomoku()

        while not game.is_game_over():
            game.make_random_move()

        game.print_board()

        times.append(time.time() - start_time_s)

    print(f"Average time (s): {sum(times) / len(times)}")
    print(f"Min time(s): {min(times)}")
    print(f"Max time (s): {max(times)}")


def main() -> None:
    benchmark()
    # play_sample_game(verbose=True)


if __name__ == "__main__":
    main()