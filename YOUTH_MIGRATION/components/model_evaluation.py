import sys
from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.entity.artifact_entity import ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(self, model_eval_config, data_ingestion_artifact, model_trainer_artifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Model evaluation started (dummy)")

            return ModelEvaluationArtifact(
                is_model_accepted=True,
                changed_accuracy=0.0,
                s3_model_path="",
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)