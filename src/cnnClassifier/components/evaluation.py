import numpy as np
import tensorflow as tf
from pathlib import Path
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.cnnClassifier.entity.config_entity import EvaluationConfig
from src.cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path):
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)

        X_test = np.load(self.config.training_data / "X_test.npy")
        y_test = np.load(self.config.training_data / "y_test.npy")

        X_test = X_test.astype("float32") / 255.0

        print("X_test Shape:", X_test.shape)
        print("y_test Shape:", y_test.shape)

        self.score = self.model.evaluate(
            X_test,
            y_test,
            batch_size=self.config.params_batch_size,
            verbose=1
        )

    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(
            path=Path("scores.json"),
            data=scores
        )