import tensorflow as tf
import numpy as np
from pathlib import Path

from src.cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_data(self):

        data_path = Path(self.config.training_data)

        self.X_train = np.load(data_path / "X_train.npy")
        self.X_test = np.load(data_path / "X_test.npy")
        self.y_train = np.load(data_path / "y_train.npy")
        self.y_test = np.load(data_path / "y_test.npy")

    def get_base_model(self):

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def get_callbacks(self):

        return [

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
                verbose=1
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )

        ]

    def train(self):

        self.load_data()

        self.get_base_model()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.get_callbacks(),
            verbose=2
        )

        self.model.save(self.config.trained_model_path)