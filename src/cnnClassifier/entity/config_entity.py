from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path



@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_classes: int




@dataclass
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path

    epochs: int
    batch_size: int
    is_augmentation: bool
    image_size: Tuple[int, int, int]

    learning_rate: float
    weight_decay: float

    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    horizontal_flip: bool
    zoom_range: float
    brightness_range: tuple
    shear_range: float
    channel_shift_range: float

    reduce_lr_factor: float
    reduce_lr_patience: int
    min_learning_rate: float
    early_stopping_patience: int