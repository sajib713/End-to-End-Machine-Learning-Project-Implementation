import sys

from YOUTH_MIGRATION.cloud_storage.aws_storage import SimpleStorageService
from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging
from YOUTH_MIGRATION.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact
)
from YOUTH_MIGRATION.entity.config_entity import ModelPusherConfig
from YOUTH_MIGRATION.entity.s3_estimator import MigrationEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        """
        :param model_evaluation_artifact: Output of model evaluation stage
        :param model_pusher_config: Configuration for model pusher
        """
        try:
            self.s3 = SimpleStorageService()
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

            self.estimator = MigrationEstimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Upload trained model to S3 bucket
        """

        logging.info("Starting Model Pusher...")

        try:
            # 🔥 Only push if model is accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Skipping push.")
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=None
                )

            logging.info("Uploading model to S3...")

            self.estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path,
                remove=False
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Model successfully uploaded to S3")
            logging.info(f"ModelPusherArtifact: {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)