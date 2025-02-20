from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        '''config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()'''
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        # Instantiate ModelTrainer
        model_trainer = ModelTrainer(config=model_trainer_config)
        # Start Training (with checkpointing)
        model_trainer.train()


        