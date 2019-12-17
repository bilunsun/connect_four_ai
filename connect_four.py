import numpy as np
import random
import time
from typing import List

class ConnectFour:
    ROWS = 6
    COLUMNS = 7

    BLACK = 0
    WHITE = 1

    WINNING_CONDITIONS = [
        [  # Vertical
            [[0, -3], [0, -2], [0, -1]],
            [[0, 1], [0, -2], [0, -1]],
            [[0, 2], [0, -1], [0, 1]],
            [[0, 3], [0, 2], [0, 1]]
        ],

        [  # Horizontal
            [[-3, 0], [-2, 0], [-1, 0]],
            [[1, 0], [-2, 0], [-1, 0]],
            [[2, 0], [-1, 0], [1, 0]],
            [[3, 0], [2, 0], [1, 0]]
        ],

        [  # Forward Diagonal
            [[1, -1], [2, -2], [3, -3]],
            [[-1, 1], [1, -1], [2, -2]],
            [[-2, 2], [-1, 1], [1, -1]],
            [[-3, 3], [-2, 2], [-1, 1]]
        ],

        [  # Backward Diagonal
            [[3, 3], [2, 2], [1, 1]],
            [[2, 2], [-1, -1], [1, 1]],
            [[-2, -2], [1, 1], [-1, -1]],
            [[-3, -3], [-2, -2], [-1, -1]]
        ],
    ]

    def __init__(self) -> None:
        self.board = np.zeros((2, self.ROWS, self.COLUMNS), dtype=int)
        self.free_row_indices = [self.ROWS - 1 for i in range(self.COLUMNS)]
        self.legal_moves = [i for i in range(self.COLUMNS)]

        self.turn = self.BLACK
        self.game_over = False
        self.winner = None
        self.pieces_count = 0

    def make_move(self, column: int) -> bool:
        free_row_index = self.free_row_indices[column]
        assert free_row_index != -1
        self.current_board[free_row_index, column] = 1

        if self.is_winning_move(column):
            self.game_over = True

            if self.turn == self.WHITE:
                self.winner = "white"
            elif self.turn == self.BLACK:
                self.winner = "black"
            else:
                self.winner = "drawn"
            return

        self.free_row_indices[column] -= 1
        if self.free_row_indices[column] < 0:
            self.legal_moves.remove(column)

        self.turn = not self.turn
        self.pieces_count += 1

        if self.pieces_count == self.ROWS * self.COLUMNS:
            self.game_over = True
            self.winner = "drawn"

    def is_winning_move(self, column: int) -> bool:
        row = self.free_row_indices[column]

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
                    return True

        return False

    @property
    def current_board(self) -> np.ndarray:
        return self.board[int(self.turn)]

    # @property
    # def legal_moves(self) -> List:
    #     return [i for i in range(self.COLUMNS) if self.free_row_indices[i] >= 0]


    def make_random_move(self) -> None:
        # random_move = random.choice(self.legal_moves)
        random_move = self.legal_moves[0]
        self.make_move(random_move)

    def print_board(self) -> None:
        player = "White" if self.turn else "Black"
        output_repr = f"Turn: {player}\n"
        output_repr += f"free_row_indices: {self.free_row_indices}\n"
        output_repr += f"legal_moves: {self.legal_moves}\n"

        output_repr += "0  1  2  3  4  5  6\n"
        for i in range(self.ROWS):
            row = ""
            for j in range(self.COLUMNS):
                if self.board[self.BLACK, i, j]:
                    row += "X  "
                elif self.board[self.WHITE, i, j]:
                    row += "O  "
                else:
                    row += ".  "
            output_repr += row + "\n"

        print(output_repr)


def play_sample_game(verbose: bool = False) -> None:
    game = ConnectFour()

    while not game.game_over:
        game.make_random_move()

        if verbose:
            game.print_board()

    game.print_board()


def benchmark(iterations: int = 1000) -> None:
    times = []

    for i in range(iterations):
        start_time_s = time.time()

        game = ConnectFour()

        while not game.game_over:
            game.make_random_move()
            # game.print_board()

        times.append(time.time() - start_time_s)

    print(f"Average time (s): {sum(times) / len(times)}")
    print(f"Min time(s): {min(times)}")
    print(f"Max time (s): {max(times)}")


def main() -> None:
    benchmark()
    # play_sample_game(verbose=True)


if __name__ == "__main__":
    main()
