from src.imageSegmentation.constants import *                   ## CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.imageSegmentation.utils.common import read_yaml, create_directories
from src.imageSegmentation.entity.config_entity import (DataIngestionConfig,
                                                         DataProcessingConfig,
                                                           ModelTrainingConfig,
                                                            ModelEvaluationConfig,
                                                             ModelCallbackConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])           


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            aws_access_id = config.aws_access_id,
            aws_secrete_key = config.aws_secrete_key,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir,
            bucket_name = config.bucket_name,
            data_filename = config.data_filename 
        )
        
        return data_ingestion_config
    

    def get_data_processing_config(self) -> DataProcessingConfig:
        config = self.config.data_preprocessing
        
        data_processing_config = DataProcessingConfig(
            training_org_img_folder = Path(config.training_org_img_folder),
            training_segment_img_folder = Path(config.training_segment_img_folder),
            testing_org_img_folder = Path(config.testing_org_img_folder),
            testing_segment_img_folder = Path(config.testing_segment_img_folder),
            number_of_classes = self.params.CLASSES
        )
        return data_processing_config


    def get_callback_model(self) -> ModelCallbackConfig:

        config = self.config.model_callback
        params = self.params.for_callback_only
        create_directories([config.root_dir])

        model_callback_config = ModelCallbackConfig(
            root_dir = Path(config.root_dir),
            model_save_path= Path(config.model_save_path),
            monitor = params.monitor,
            patience = params.patience,
            mode = params.mode,
            restore_best_weights = params.restore_best_weights,
            min_delta = params.min_delta,
            factor = params.factor,
            min_learningrate = params.min_learningrate
        )
        return model_callback_config


    def get_training_config(self) -> ModelTrainingConfig:
        
        config = self.config.model_training

        create_directories([config.root_dir])

        training_config = ModelTrainingConfig(
            root_dir = Path(config.root_dir),
            base_model= Path(config.base_model),
            number_of_classes = self.params.CLASSES,
            image_size = self.params.IMAGE_SIZE,
            batch_size = self.params.BATCH_SIZE,
            epochs = self.params.EPOCHS,
            learning_rate = self.params.LEARNING_RATE,
            backbone = self.params.BACKBONE
        )
        return training_config
    

    def get_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        evaluation_config = ModelEvaluationConfig(
            model_path = Path(config.model_path),
            training_org_img_folder= Path(config.training_org_img_folder),
            training_segment_img_folder = Path(config.training_segment_img_folder),
            testing_org_img_folder = Path(config.testing_org_img_folder),
            testing_segment_img_folder = Path(config.testing_segment_img_folder),
            number_of_classes = self.params.CLASSES
        )
        return evaluation_config