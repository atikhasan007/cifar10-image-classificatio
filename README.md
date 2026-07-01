# cifar10-image-classificatio


import tensorflow as tf
import numpy as np
from pathlib import Path


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
            loss="sparse_categorical_crossentropy",  # ✅ FIX
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.get_callbacks(),
            verbose=2
        )

        self.model.save(self.config.trained_model_path)

        return history



```base
conda create -n abc python=3.8
conda activate abc
pip install -r requiremetns.txt




# WorkFlow
```bash
Update config.yaml
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the app.py   

```










# USER UI












<img width="686" height="728" alt="Screenshot 2026-07-01 193025" src="https://github.com/user-attachments/assets/9483cb75-9fbf-4c67-9e9e-203ffd0b7ef5" />
<img width="967" height="997" alt="Screenshot 2026-07-01 192927" src="https://github.com/user-attachments/assets/18535123-f5cc-4eae-bb20-5d2178cf6abd" />
<img width="915" height="992" alt="Screenshot 2026-07-01 192832" src="https://github.com/user-attachments/assets/7f9cb3dc-2a8f-41a9-bf1f-abbc2076c239" />
<img width="927" height="1002" alt="Screenshot 2026-07-01 192733" src="https://github.com/user-attachments/assets/7c91d444-5076-4953-9132-fe43873507d2" />
<img width="925" height="990" alt="Screenshot 2026-07-01 192613" src="https://github.com/user-attachments/assets/d0c9619d-bb00-44db-a4a9-c7fac470d39d" />
<img width="938" height="1017" alt="Screenshot 2026-07-01 192548" src="https://github.com/user-attachments/assets/29db3893-506e-40ee-aae6-06b64a3ec297" />
<img width="940" height="866" alt="Screenshot 2026-07-01 192300" src="https://github.com/user-attachments/assets/1b16b712-96ee-468b-ae6d-c4ec94530092" />
