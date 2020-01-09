from keras import regularizers
from keras.layers import add, Activation, BatchNormalization, Conv2D, ReLU
import tensorflow as tf


RESIDUAL_LAYER_PARAMETERS = {
    "filters": 64,
    "kernel_size": 3,
    "padding": "same",
    "kernel_regularizer": regularizers.l2(0.01)
}


def residual_layer(input_block):
    x = Conv2D(filters=RESIDUAL_LAYER_PARAMETERS["filters"],
               kernel_size=RESIDUAL_LAYER_PARAMETERS["kernel_size"],
               padding=RESIDUAL_LAYER_PARAMETERS["padding"],
               kernel_regularizer=RESIDUAL_LAYER_PARAMETERS["kernel_regularizer"]
    )(input_block)

    x = BatchNormalization(axis=1)(x)

    x = add([input_block, x])

    x = ReLU()(x)

    return x


def main():
    pass


if __name__ == "__main__":
    main()


