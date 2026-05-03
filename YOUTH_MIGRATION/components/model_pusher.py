import sys
from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.entity.artifact_entity import ModelPusherArtifact


class ModelPusher:
    def __init__(self, model_evaluation_artifact, model_pusher_config):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Model pusher started (dummy)")

            return ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

        except Exception as e:
            raise CustomException(e, sys)