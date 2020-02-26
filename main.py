import numpy as np
import random
import torch

from connect_four import ConnectFour
from neural_network_utils import generate_state_representation, load_npy_data
from torch_model import Model


def play_sample_game_with_model(model) -> None:
    sample_state = ConnectFour()

    while not sample_state.is_game_over():
        if sample_state.turn == sample_state.BLACK:
            sample_state.make_random_move()
        else:
            state_representation = generate_state_representation(sample_state)
            state_representation = torch.from_numpy(state_representation).float().cuda()
            state_representation = state_representation.view(1, 3, 6, 7)

            with torch.no_grad():
                predicted_policy, predicted_value = model(state_representation)

            predicted_policy = predicted_policy[0].cpu()
            predicted_value = predicted_value[0].cpu()

            print("NN Policy:", predicted_policy)
            print("NN Value:", predicted_value)

            best_move = np.argmax(predicted_policy)
            while best_move not in sample_state.legal_moves:
                predicted_policy[best_move] = -999
                best_move = np.argmax(predicted_policy)

            sample_state.make_move(best_move)

        sample_state.print_board()


def main():
    # First, load the data
    states, policies, values = load_npy_data()

    # Then, build the model and train
    model = Model().cuda()

    # Play a sample game
    play_sample_game_with_model(model)


if __name__ == "__main__":
    main()
