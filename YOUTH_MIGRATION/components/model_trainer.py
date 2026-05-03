import sys
from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model training started (dummy)")

            # ⚠️ dummy model path (temporary)
            trained_model_file_path = "artifact/model.pkl"

            metric_artifact = ClassificationMetricArtifact(
                f1_score=0.0,
                precision_score=0.0,
                recall_score=0.0
            )

            return ModelTrainerArtifact(
                trained_model_file_path=trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise CustomException(e, sys)