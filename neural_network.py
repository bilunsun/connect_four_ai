from typing import List, Tuple, Union

from connect_four import ConnectFour
from game_template import GameTemplate
from gomoku import Gomoku
from mcts import MCTS


def generate_state_representation(game_state: GameTemplate) -> List:
    """
    Generate a state representation for the neural network with planes
        White pieces location
        Black pieces location
        Player's turn
    """
    board = game_state.get_board_copy()
    turn = game_state.turn()

    state_representation = []
    state_representation.append(board[GameTemplate.WHITE])
    state_representation.append(board[GameTemplate.BLACK])
    state_representation.append([[turn for column in range(game_state.COLUMNS)] for row in range(game_state.ROWS)])

    return state_representation


def generate_policy_representation(game_state: GameTemplate, move: Union[int, Tuple[int, int]]) -> Union[List[int], List[List[int]]]:
    policy: Union[List[int], List[List[int]]]

    if isinstance(game_state, ConnectFour):
        assert isinstance(move, int)
        policy = [0 for _ in range(ConnectFour.COLUMNS)]
        policy[move] = 1

    if isinstance(game_state, Gomoku):
        assert isinstance(move, Tuple[int, int])
        policy = [[0 for column in range(Gomoku.COLUMNS)] for row in range(Gomoku.ROWS)]
        row, column = move
        policy[row][column] = 1

    return policy


def generate_data(Game: GameTemplate) -> List:
    """
    Player a game, and generate training data for the neural network
    """
    sample_state: GameTemplate = Game()

    white_mcts = MCTS(root_state=sample_state, itermax=800, timeout_s=5, debug=False)
    black_mcts = MCTS(root_state=sample_state, itermax=800, timeout_s=2, debug=False)

    state_reprensetations: List= []

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
        current_policy_reprensentation = generate_policy_representation(sample_state, move)
        state_reprensetations.append(current_state_representation)

        sample_state.make_move(move)

    # At the end of the game, include the final outcome
    pass

    return state_reprensetations