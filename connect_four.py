import random
import time
from typing import List, Union

from game_template import GameTemplate


class ConnectFour(GameTemplate):
    ROWS = 6
    COLUMNS = 7

    WINNING_CONDITIONS = (
        (  # Vertical
            ((0, -3), (0, -2), (0, -1)),
            ((0, 1), (0, -2), (0, -1)),
            ((0, 2), (0, -1), (0, 1)),
            ((0, 3), (0, 2), (0, 1))
        ),

        (  # Horizontal
            ((-3, 0), (-2, 0), (-1, 0)),
            ((1, 0), (-2, 0), (-1, 0)),
            ((2, 0), (-1, 0), (1, 0)),
            ((3, 0), (2, 0), (1, 0))
        ),

        (  # Forward Diagonal
            ((1, -1), (2, -2), (3, -3)),
            ((-1, 1), (1, -1), (2, -2)),
            ((-2, 2), (-1, 1), (1, -1)),
            ((-3, 3), (-2, 2), (-1, 1))
        ),

        (  # Backward Diagonal
            ((3, 3), (2, 2), (1, 1)),
            ((2, 2), (-1, -1), (1, 1)),
            ((-2, -2), (1, 1), (-1, -1)),
            ((-3, -3), (-2, -2), (-1, -1))
        ),
    )

    def __init__(self) -> None:
        self._board = [[[0 for j in range(self.COLUMNS)] for i in range(self.ROWS)] for _ in range(2)]
        self._free_row_indices = [self.ROWS - 1 for i in range(self.COLUMNS)]

        self._turn = self.BLACK
        self._result: int = None
        self._pieces_count = 0

    def turn(self) -> int:
        return self._turn

    @property
    def legal_moves(self) -> List:
        return [i for i, free_row_index in enumerate(self._free_row_indices) if free_row_index >= 0]

    def make_move(self, column: int) -> None:
        free_row_index = self._free_row_indices[column]
        self._current_board[free_row_index][column] = 1
        self._pieces_count += 1

        if not self.game_ending_move(column):
            self._turn = not self._turn
            self._free_row_indices[column] -= 1

    def result(self) -> Union[int, None]:
        return self._result

    def game_ending_move(self, column: int) -> bool:
        if self._pieces_count == self.ROWS * self.COLUMNS:
            self._result = self.VARIANT_DRAWN
            return True

        row = self._free_row_indices[column]

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
                        if not self._current_board[shifted_row][shifted_column]:
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
    def _current_board(self) -> List:
        return self._board[int(self._turn)]

    def make_random_move(self) -> None:
        random_move = random.choice(self.legal_moves)
        self.make_move(random_move)

    def get_copy(self) -> "ConnectFour":
        copied_self = ConnectFour()
        copied_self._board = [[player_row[:] for player_row in player_board[:]] for player_board in self._board]
        copied_self._free_row_indices = self._free_row_indices[:]
        copied_self._turn = self._turn
        copied_self._result = self._result
        copied_self._pieces_count = self._pieces_count

        return copied_self

    def print_board(self) -> None:
        player = "White" if self._turn else "Black"
        output_repr = f"Turn: {player}\n"

        output_repr += "0  1  2  3  4  5  6\n"
        for i in range(self.ROWS):
            row = ""
            for j in range(self.COLUMNS):
                if self._board[self.BLACK][i][j]:
                    row += "X  "
                elif self._board[self.WHITE][i][j]:
                    row += "O  "
                else:
                    row += ".  "
            output_repr += row + "\n"

        print(output_repr)


def play_sample_game(verbose: bool = False) -> None:
    game = ConnectFour()

    while not game.is_game_over():
        game.make_random_move()

        if verbose:
            game.print_board()

    print(f"Final Result: {game.result()}")
    game.print_board()


def benchmark(iterations: int = 1000) -> None:
    times = []

    for _ in range(iterations):
        start_time_s = time.time()

        game = ConnectFour()

        while not game.is_game_over():
            game.make_random_move()

        times.append(time.time() - start_time_s)

    print(f"Average time (s): {sum(times) / len(times)}")
    print(f"Min time(s): {min(times)}")
    print(f"Max time (s): {max(times)}")


def main() -> None:
    # benchmark()
    play_sample_game(verbose=True)


if __name__ == "__main__":
    main()
