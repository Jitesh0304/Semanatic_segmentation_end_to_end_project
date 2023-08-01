from src.imageSegmentation import logger
from src.imageSegmentation.config.configuration import ConfigurationManager
from src.imageSegmentation.components.model_evaluation import ModelEvaluation



STAGE_NAME = 'Model evaluation'

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        eval_process = ModelEvaluation(config = evaluation_config)
        eval_process.evaluation()
        eval_process.save_score()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
    except Exception as e:
        logger.exception(e)
        raise e
