import numpy as np

from connect_four import ConnectFour
from model import build_model
from neural_network_utils import generate_neural_network_data


def main():
    x, y = generate_neural_network_data(Game=ConnectFour)

    model = build_model()
    values, policies = model.predict(np.array(x[-3:]))

    for p in policies:
        print(p.shape)


if __name__ == "__main__":
    main()