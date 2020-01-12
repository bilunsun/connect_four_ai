import numpy as np

from connect_four import ConnectFour
from model import build_model
from neural_network_utils import generate_state_representation, load_npy_data


def play_sample_game_with_model(model) -> None:
    sample_state = ConnectFour()

    while not sample_state.is_game_over():
        state_representation = generate_state_representation(sample_state)

        predicted_policy, predicted_value = model.predict(np.array([state_representation]))

        predicted_policy = predicted_policy[0]
        predicted_value = predicted_value[0]

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

    # Then, build the model
    model = build_model()

    # Train the model
    model.fit(x=states, y=[policies, values], batch_size=32, epochs=50, verbose=2, validation_split=0.2, shuffle=True)

    # Play a sample game
    play_sample_game_with_model(model)


if __name__ == "__main__":
    main()
