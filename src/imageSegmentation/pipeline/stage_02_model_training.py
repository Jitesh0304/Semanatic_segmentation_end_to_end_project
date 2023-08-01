from src.imageSegmentation import logger
from src.imageSegmentation.config.configuration import ConfigurationManager
from src.imageSegmentation.components.data_processing import DataPreprocess
from src.imageSegmentation.components.training_process import ModelTraining
from src.imageSegmentation.components.model_callback import ModelCallback

STAGE_NAME = 'Data preprocessing and model Training'

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_process_config = config.get_data_processing_config()
        data_process = DataPreprocess(config = data_process_config)
        org_img_arr = data_process.convert_org_img_to_array()
        seg_img_arr = data_process.convert_seg_img_to_array()
        y_cls = data_process.rgb_to_class_num()
        y_cat = data_process.convert_to_categorical()

        callback_config = config.get_callback_model()
        callback = ModelCallback(config= callback_config)
        callback_details = callback.get_call_back()

        training_config = config.get_training_config()
        training = ModelTraining(config= training_config)
        X_train, X_valid , y_train, y_valid = training.spliting_the_dataset(org_img_arr, y_cat)
        fitting = training.fit_model(X_train, X_valid , y_train, y_valid, callback_details)



if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
    except Exception as e:
        logger.exception(e)
        raise e
