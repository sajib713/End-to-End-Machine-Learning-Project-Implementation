import sys

from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging

from YOUTH_MIGRATION.components.data_ingestion import DataIngestion
from YOUTH_MIGRATION.components.data_validation import DataValidation
from YOUTH_MIGRATION.components.data_transformation import DataTransformation
from YOUTH_MIGRATION.components.model_trainer import ModelTrainer
from YOUTH_MIGRATION.components.model_evaluation import ModelEvaluation
from YOUTH_MIGRATION.components.model_pusher import ModelPusher

from YOUTH_MIGRATION.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from YOUTH_MIGRATION.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()
            self.model_pusher_config = ModelPusherConfig()

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Data Ingestion
    # -------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("📥 Starting Data Ingestion")

            data_ingestion = DataIngestion(self.data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"✅ Data Ingestion Completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"❌ Data Ingestion Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Data Validation
    # -------------------------------
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("🔍 Starting Data Validation")

            data_validation = DataValidation(
                data_ingestion_artifact,
                self.data_validation_config
            )

            artifact = data_validation.initiate_data_validation()

            # 🔥 DEBUG LOG
            logging.info(f"Validation Status: {artifact.validation_status}")
            logging.info(f"Validation Message: {artifact.message}")

            print("Validation Status:", artifact.validation_status)
            print("Validation Message:", artifact.message)

            logging.info("✅ Data Validation Completed")
            return artifact

        except Exception as e:
            logging.error(f"❌ Data Validation Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Data Transformation
    # -------------------------------
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:
            logging.info("🔄 Starting Data Transformation")

            data_transformation = DataTransformation(
                data_ingestion_artifact,
                self.data_transformation_config,
                data_validation_artifact
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info(f"✅ Data Transformation Completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"❌ Data Transformation Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Model Trainer
    # -------------------------------
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("🤖 Starting Model Training")

            model_trainer = ModelTrainer(
                data_transformation_artifact,
                self.model_trainer_config
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info(f"✅ Model Training Completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"❌ Model Training Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Model Evaluation
    # -------------------------------
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:

        try:
            logging.info("📊 Starting Model Evaluation")

            model_evaluation = ModelEvaluation(
                self.model_evaluation_config,
                data_ingestion_artifact,
                model_trainer_artifact
            )

            artifact = model_evaluation.initiate_model_evaluation()

            logging.info(f"✅ Model Evaluation Completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"❌ Model Evaluation Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Model Pusher
    # -------------------------------
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logging.info("🚀 Starting Model Pusher")

            model_pusher = ModelPusher(
                model_evaluation_artifact,
                self.model_pusher_config
            )

            artifact = model_pusher.initiate_model_pusher()

            logging.info(f"✅ Model Pusher Completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"❌ Model Pusher Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Run Pipeline
    # -------------------------------
    def run_pipeline(self) -> None:
        try:
            logging.info("🚀 Training Pipeline Started")

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(ingestion_artifact)

            # 🔥 FIX 1: show real error
            if not validation_artifact.validation_status:
                raise Exception(f"❌ Data validation failed: {validation_artifact.message}")

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact,
                validation_artifact
            )

            trainer_artifact = self.start_model_trainer(transformation_artifact)

            evaluation_artifact = self.start_model_evaluation(
                ingestion_artifact,
                trainer_artifact
            )

            # 🔥 FIX 2: better message
            if not evaluation_artifact.is_model_accepted:
                raise Exception(
                    f"❌ Model not accepted. Accuracy change: {evaluation_artifact.changed_accuracy}"
                )

            self.start_model_pusher(evaluation_artifact)

            logging.info("🎉 Pipeline Completed Successfully")

        except Exception as e:
            logging.error(f"❌ Pipeline Failed: {e}")
            raise CustomException(e, sys)