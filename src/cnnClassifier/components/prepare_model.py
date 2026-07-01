import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.regularizers import l2
from pathlib import Path






class PrepareModel:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def create_model(self):

        weight_decay = 0.0001

        model = Sequential()

        # Block 1
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                input_shape=tuple(self.config.params_image_size),
            )
        )
        model.add(BatchNormalization())

        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        # Block 2
        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        # Block 3
        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        # Block 4
        model.add(
            Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(
            Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        # Classification Head
        model.add(Flatten())
        model.add(Dense(self.config.params_classes, activation="softmax"))

        model.summary()

        self.save_model(self.config.model_path, model)