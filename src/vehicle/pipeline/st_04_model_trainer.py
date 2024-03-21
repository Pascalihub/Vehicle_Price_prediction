from src.vehicle.entity import DataIngestionConfig
from src.vehicle.conponents.data_ingestion import DataIngestion
from vehicle.logger import logger
from src.vehicle.config.configuration import ConfigurationManager
from src.vehicle.conponents.model_trainer import ModelTrainer

STAGE_NAME = "Data transformation stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_training()

if __name__ == '__main__':   
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = ModelTrainerTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e