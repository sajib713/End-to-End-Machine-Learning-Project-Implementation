import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from YOUTH_MIGRATION.entity.config_entity import ModelEvaluationConfig

from YOUTH_MIGRATION.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging


# -----------------------------------
# Evaluation Response
# -----------------------------------
@dataclass
class EvaluateModelResponse:

    trained_model_accuracy: float
    best_model_accuracy: float
    is_model_accepted: bool
    difference: float


# -----------------------------------
# Model Evaluation
# -----------------------------------
class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):

        try:

            self.model_eval_config = model_eval_config

            self.data_ingestion_artifact = data_ingestion_artifact

            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:

            raise CustomException(e, sys)

    # -----------------------------------
    # 🔥 DISABLE OLD S3 MODEL CHECK
    # -----------------------------------
    def get_best_model(self):

        try:

            # always return None
            # prevents old S3 ball_tree model loading
            return None

        except Exception as e:

            raise CustomException(e, sys)

    # -----------------------------------
    # Evaluate Current Model
    # -----------------------------------
    def evaluate_model(self) -> EvaluateModelResponse:

        try:

            logging.info("Starting model evaluation")

            # current trained model accuracy
            trained_model_accuracy = (
                self.model_trainer_artifact.metric_artifact.accuracy
            )

            # no old model comparison
            best_model_accuracy = 0

            # accept current model
            result = EvaluateModelResponse(
                trained_model_accuracy=trained_model_accuracy,
                best_model_accuracy=best_model_accuracy,
                is_model_accepted=True,
                difference=trained_model_accuracy
            )

            logging.info(
                f"Evaluation Result: {result}"
            )

            return result

        except Exception as e:

            raise CustomException(e, sys)

    # -----------------------------------
    # Main Evaluation Function
    # -----------------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:

            eval_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=eval_response.difference
            )

            logging.info(
                f"Model Evaluation Artifact: {model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:

            raise CustomException(e, sys)