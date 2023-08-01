from src.imageSegmentation import logger
from src.imageSegmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.imageSegmentation.pipeline.stage_02_model_training import ModelTrainingPipeline
from src.imageSegmentation.pipeline.stage_03_model_evaluation import EvaluationPipeline



STAGE_NAME = 'Data Ingestion stage'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    dataIngesion = DataIngestionTrainingPipeline()
    dataIngesion.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = 'Data preprocessing and model Training'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    training = ModelTrainingPipeline()
    training.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Evaluation'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    evaluation = EvaluationPipeline()
    evaluation.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e