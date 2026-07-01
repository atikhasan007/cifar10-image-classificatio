from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path



@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_classes: int