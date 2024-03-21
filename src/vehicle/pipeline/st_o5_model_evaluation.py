from src.vehicle.conponents.data_transformation import DataTransformation
from src.vehicle.entity import DataIngestionConfig
from src.vehicle.conponents.data_ingestion import DataIngestion
from src.vehicle.conponents.data_validation import DataValiadtion
from vehicle.logger import logger
from src.vehicle.config.configuration import ConfigurationManager
from pathlib import Path
import os
from src.vehicle.conponents.model_evaluation import ModelEvaluation
from src.vehicle.logger import logger

STAGE_NAME = "Data evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()

if __name__ == '__main__':   
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = ModelEvaluationTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e