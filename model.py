from keras import regularizers
from keras.layers import Add, Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, ReLU
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
import tensorflow as tf


RESIDUAL_LAYER_PARAMETERS = {
    "filters": 16,
    "kernel_size": (4, 4),
    "padding": "same",
    "kernel_regularizer": regularizers.l2(l=0.01)
}

CONVOLUTIONAL_LAYER_PARAMETERS = {
    "filters": 16,
    "kernel_size": (4, 4),
    "padding": "same",
    "kernel_regularizer": regularizers.l2(l=0.01)
}

POLICY_HEAD_CONV_PARAMETERS = {
    "filters": 2,
    "kernel_size": (1, 1),
    "padding": "same",
    "kernel_regularizer": regularizers.l2(l=0.01)
}

POLICY_HEAD_DENSE_PARAMETERS = {
    "units": 42,  # For ConnectFour, 6 rows * 7 columns
    "name": "policy_head"
}

VALUE_HEAD_CONV_PARAMETERS = {
    "filters": 1,
    "kernel_size": (1, 1),
    "padding": "same",
    "kernel_regularizer": regularizers.l2(l=0.01)
}

VALUE_HEAD_DENSE_1_UNITS = 16

VALUE_HEAD_DENSE_PARAMETERS = {
    "units": 1,
    "name": "value_head"
}

RESIDUAL_LAYERS_COUNT = 8
LEARNING_RATE = 0.1
MOMENTUM = 0.9


def residual_layer(input_block):
    x = Conv2D(filters=RESIDUAL_LAYER_PARAMETERS["filters"],
               kernel_size=RESIDUAL_LAYER_PARAMETERS["kernel_size"],
               padding=RESIDUAL_LAYER_PARAMETERS["padding"],
               data_format="channels_first",
               kernel_regularizer=RESIDUAL_LAYER_PARAMETERS["kernel_regularizer"]
    )(input_block)

    x = BatchNormalization(axis=1)(x)

    x = Add()([input_block, x])

    x = ReLU()(x)

    return x  # Are the parentheses needed?


def convolutional_layer(input_block):
    x = Conv2D(filters=CONVOLUTIONAL_LAYER_PARAMETERS["filters"],
               kernel_size=CONVOLUTIONAL_LAYER_PARAMETERS["kernel_size"],
               padding=CONVOLUTIONAL_LAYER_PARAMETERS["padding"],
               data_format="channels_first",
               kernel_regularizer=CONVOLUTIONAL_LAYER_PARAMETERS["kernel_regularizer"]
    )(input_block)

    x = BatchNormalization(axis=1)(x)

    x = ReLU()(x)

    return x


def policy_head(input_block):
    x = Conv2D(filters=POLICY_HEAD_CONV_PARAMETERS["filters"],
               kernel_size=POLICY_HEAD_CONV_PARAMETERS["kernel_size"],
               padding=POLICY_HEAD_CONV_PARAMETERS["padding"],
               data_format="channels_first",
               kernel_regularizer=POLICY_HEAD_CONV_PARAMETERS["kernel_regularizer"]
    )(input_block)

    x = BatchNormalization(axis=1)(x)

    x = ReLU()(x)

    x = Flatten()(x)

    x = Dense(units=POLICY_HEAD_DENSE_PARAMETERS["units"], name=POLICY_HEAD_DENSE_PARAMETERS["name"])(x)

    return x


def value_head(input_block):
    x = Conv2D(filters=VALUE_HEAD_CONV_PARAMETERS["filters"],
               kernel_size=VALUE_HEAD_CONV_PARAMETERS["kernel_size"],
               padding=VALUE_HEAD_CONV_PARAMETERS["padding"],
               data_format="channels_first",
               kernel_regularizer=VALUE_HEAD_CONV_PARAMETERS["kernel_regularizer"]
    )(input_block)

    x = BatchNormalization(axis=1)(x)

    x = ReLU()(x)

    x = Flatten()(x)

    x = Dense(units=VALUE_HEAD_DENSE_1_UNITS)(x)

    x = ReLU()(x)

    x = Dense(units=VALUE_HEAD_DENSE_PARAMETERS["units"], name=VALUE_HEAD_DENSE_PARAMETERS["name"])(x)

    return x


def build_model() -> Model:
    network_input = Input(shape=(3, 6, 7), name="network_input")

    x = convolutional_layer(network_input)

    for _ in range(RESIDUAL_LAYERS_COUNT):
        x = residual_layer(x)

    value = value_head(x)
    policy = policy_head(x)

    model = Model(inputs=[network_input], outputs=[value, policy])
    model.compile(
        loss={
            "value_head": "mean_squared_error",
            "policy_head": "categorical_crossentropy"
        },
        optimizer=SGD(
            lr=LEARNING_RATE,
            momentum=MOMENTUM
        ),
        loss_weights={
            "value_head": 0.5,
            "policy_head": 0.5
        }
    )

    return model


def main():
    model = build_model()

    print(model.summary())

    plot_model(model)


if __name__ == "__main__":
    main()


