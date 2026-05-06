import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score

from youth_migration.entity.config_entity import ModelEvaluationConfig
from youth_migration.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.entity.s3_estimator import MigrationEstimator


@dataclass
class EvaluateModelResponse:
    trained_model_accuracy: float
    best_model_accuracy: float
    is_model_accepted: bool
    difference: float


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

    # -------------------------------
    # Get Best Model from S3
    # -------------------------------
    def get_best_model(self) -> Optional[MigrationEstimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            estimator = MigrationEstimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            if estimator.is_model_present(model_path=model_path):
                return estimator

            return None

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Evaluate Model
    # -------------------------------
    def evaluate_model(self) -> EvaluateModelResponse:

        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # normalize columns
            test_df.columns = test_df.columns.str.strip().str.lower()

            target_column = "have you decided to move abroad?"

            X = test_df.drop(columns=[target_column])
            y = test_df[target_column]

            # 🔥 target cleaning (same as transformation)
            y = y.astype(str).str.strip().str.lower().replace({"yes": 1, "no": 0})
            y = pd.to_numeric(y, errors="coerce")
            y = y.fillna(y.mode()[0])

            # trained model score
            trained_model_accuracy = self.model_trainer_artifact.metric_artifact.accuracy

            best_model_accuracy = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_pred_best = best_model.predict(X)
                best_model_accuracy = accuracy_score(y, y_pred_best)

            best_score = 0 if best_model_accuracy is None else best_model_accuracy

            result = EvaluateModelResponse(
                trained_model_accuracy=trained_model_accuracy,
                best_model_accuracy=best_model_accuracy,
                is_model_accepted=trained_model_accuracy > best_score,
                difference=trained_model_accuracy - best_score
            )

            logging.info(f"Evaluation Result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Initiate Evaluation
    # -------------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            eval_response = self.evaluate_model()

            artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=eval_response.difference
            )

            logging.info(f"Model Evaluation Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)