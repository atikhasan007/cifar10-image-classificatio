from pathlib import Path

from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # ==========================
    # Data Ingestion
    # ==========================
    def get_data_ingestion_config(self):

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir)
        )

    # ==========================
    # Prepare Model
    # ==========================
    def get_prepare_model_config(self):

        config = self.config.prepare_model

        create_directories([config.root_dir])

        return PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_image_size=tuple(self.params.IMAGE_SIZE),
            params_classes=int(self.params.CLASSES),
        )

    # ==========================
    # Training
    # ==========================
    def get_training_config(self):

        training = self.config.training
        prepare_model = self.config.prepare_model
        params = self.params

        create_directories([training.root_dir])

        training_data = Path("artifacts/data_ingestion")

        return TrainingConfig(

            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_model.model_path),
            training_data=training_data,

            epochs=int(params.EPOCHS),
            batch_size=int(params.BATCH_SIZE),
            image_size=tuple(params.IMAGE_SIZE),

            is_augmentation=bool(params.AUGMENTATION),

            learning_rate=float(params.LEARNING_RATE),
            weight_decay=float(params.WEIGHT_DECAY),

            rotation_range=int(params.ROTATION_RANGE),
            width_shift_range=float(params.WIDTH_SHIFT_RANGE),
            height_shift_range=float(params.HEIGHT_SHIFT_RANGE),

            horizontal_flip=bool(params.HORIZONTAL_FLIP),
            zoom_range=float(params.ZOOM_RANGE),
            brightness_range=tuple(params.BRIGHTNESS_RANGE),
            shear_range=float(params.SHEAR_RANGE),
            channel_shift_range=float(params.CHANNEL_SHIFT_RANGE),

            reduce_lr_factor=float(params.REDUCE_LR_FACTOR),
            reduce_lr_patience=int(params.REDUCE_LR_PATIENCE),
            min_learning_rate=float(params.MIN_LEARNING_RATE),
            early_stopping_patience=int(params.EARLY_STOPPING_PATIENCE),
        )

    # ==========================
    # Evaluation
    # ==========================
    def get_validation_config(self):

        return EvaluationConfig(

            path_of_model=Path(
                "artifacts/training/model.h5"
            ),

            training_data=Path(
                "artifacts/data_ingestion"
            ),

            all_params=self.params,

            params_image_size=tuple(self.params.IMAGE_SIZE),

            params_batch_size=int(self.params.BATCH_SIZE),
        )