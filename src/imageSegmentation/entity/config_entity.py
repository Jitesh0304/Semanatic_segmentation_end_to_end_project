from dataclasses import dataclass
from pathlib import Path

                                
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    aws_access_id: str
    aws_secrete_key: str
    local_data_file: Path
    unzip_dir: Path
    bucket_name: str
    data_filename: str



@dataclass(frozen=True)
class DataProcessingConfig:
    training_org_img_folder: Path
    training_segment_img_folder: Path
    testing_org_img_folder: Path
    testing_segment_img_folder: Path
    number_of_classes: int



@dataclass(frozen=True)
class ModelCallbackConfig:
    root_dir: Path
    model_save_path: Path
    monitor: str
    patience: int
    mode: str
    restore_best_weights: bool
    min_delta: float
    factor: float
    min_learningrate: float



@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    base_model: Path
    number_of_classes: int
    image_size: list
    batch_size: int
    epochs: int
    learning_rate: float
    backbone: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    training_org_img_folder: Path
    training_segment_img_folder: Path
    testing_org_img_folder: Path
    testing_segment_img_folder: Path
    number_of_classes: int