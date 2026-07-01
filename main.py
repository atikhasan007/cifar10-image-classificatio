from src.cnnClassifier import logger

from src.cnnClassifier.pipeline.stage_03_model_trainer import (
    ModelTrainingPipeline,
    
)
from src.cnnClassifier import logger
# from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_model import PrepareModelTrainingPipeline
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage{STAGE_NAME} completed <<<<<\n\nx=======")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare  model"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = PrepareModelTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage{STAGE_NAME} completed <<<<<\n\nx=======")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

    training = ModelTrainingPipeline()
    training.main()

    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e