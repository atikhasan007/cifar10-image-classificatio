from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.evaluation import Evaluation
from src.cnnClassifier import logger

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        evaluation_config = config.get_validation_config()

        evaluation = Evaluation(config=evaluation_config)

        evaluation.evaluation()

        evaluation.save_score()

