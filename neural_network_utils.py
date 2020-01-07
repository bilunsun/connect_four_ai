import numpy as np
from typing import List, Tuple, Union

from connect_four import ConnectFour
from game_template import GameTemplate
from gomoku import Gomoku
from mcts import MCTS


def generate_state_representation(game_state: GameTemplate) -> np.ndarray:
    """
    Generate a state representation for the neural network with planes
        White pieces location
        Black pieces location
        Player's turn
    """
    board_representation = np.array(game_state.get_board_copy())  # Is the get_board_copy() method needed?
    turn_representation = np.full((game_state.ROWS, game_state.COLUMNS), game_state.turn())
    state_representation = np.stack((board_representation[0], board_representation[1], turn_representation), axis=0)

    return state_representation


def generate_policy_representation(game_state: GameTemplate, move: Union[int, Tuple[int, int]]) -> np.ndarray:
    policy: np.ndarray

    if isinstance(game_state, ConnectFour):
        policy = np.zeros(shape=ConnectFour.COLUMNS)
        policy[move] = 1

    if isinstance(game_state, Gomoku):
        policy = np.zeros(shape=(Gomoku.ROWS, Gomoku.COLUMNS))
        row, column = move
        policy[row, column] = 1

    return policy


def generate_data(Game: GameTemplate) -> np.ndarray:
    """
    Player a game, and generate training data for the neural network
    """
    sample_state: GameTemplate = Game()

    white_mcts = MCTS(root_state=sample_state, itermax=800, timeout_s=5, debug=False)
    black_mcts = MCTS(root_state=sample_state, itermax=800, timeout_s=2, debug=False)

    training_data: List[np.ndarray] = []

    # Play an entire game
    while not sample_state.is_game_over():
        move: Union[int, Tuple[int, int]]

        if sample_state.turn() == Game.WHITE:
            move = white_mcts.get_best_move()
            black_mcts.make_opponent_move(move)
        else:
            move = black_mcts.get_best_move()
            white_mcts.make_opponent_move(move)

        # For each move, save the state
        current_state_representation = generate_state_representation(sample_state)
        current_policy_representation = generate_policy_representation(sample_state, move)

        # Note that None will take the value of the terminal state's result
        training_data.append([current_state_representation, current_policy_representation])

        sample_state.make_move(move)

    # At the end of the game, include the final outcome
    end_result = sample_state.result()
    for data in training_data:
        data.append(np.full((Game.ROWS, Game.COLUMNS), fill_value=end_result))

    return training_data


def main():
    print(generate_data(Game=Gomoku))


if __name__ == "__main__":
    main()
