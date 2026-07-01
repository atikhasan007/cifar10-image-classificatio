import os
import numpy as np
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import DataIngestionConfig
from tensorflow.keras.datasets import cifar10


class DataIngestion:

    def __init__(self, config):
        self.config = config

    def download_data(self):

        print("Downloading CIFAR-10 dataset...")

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # folder create
        os.makedirs(self.config.root_dir, exist_ok=True)

        # save as .npy files
        np.save(self.config.root_dir / "X_train.npy", X_train)
        np.save(self.config.root_dir / "y_train.npy", y_train)
        np.save(self.config.root_dir / "X_test.npy", X_test)
        np.save(self.config.root_dir / "y_test.npy", y_test)

        print("Dataset saved successfully!")
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)