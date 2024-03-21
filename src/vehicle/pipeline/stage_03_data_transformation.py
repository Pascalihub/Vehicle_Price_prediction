from src.vehicle.conponents.data_transformation import DataTransformation
from src.vehicle.entity import DataIngestionConfig
from src.vehicle.conponents.data_ingestion import DataIngestion
from src.vehicle.conponents.data_validation import DataValiadtion
from vehicle.logger import logger
from src.vehicle.config.configuration import ConfigurationManager
from pathlib import Path
import os

STAGE_NAME = "Data transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = DataTransformationTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e