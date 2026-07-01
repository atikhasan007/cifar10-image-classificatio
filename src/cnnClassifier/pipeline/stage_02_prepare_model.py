from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prepare_model import PrepareModel
from src.cnnClassifier import logger


STAGE_NAME = "Prepare Model"


class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        prepare_model_config = config.get_prepare_model_config()

        prepare_model = PrepareModel(config=prepare_model_config)

        prepare_model.create_model()