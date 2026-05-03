import sys

from youth_migration.exception import CustomException
from youth_migration.logger import logging

from youth_migration.components.data_ingestion import DataIngestion
from youth_migration.components.data_validation import DataValidation
from youth_migration.components.data_transformation import DataTransformation
from youth_migration.components.model_trainer import ModelTrainer
from youth_migration.components.model_evaluation import ModelEvaluation
from youth_migration.components.model_pusher import ModelPusher

from youth_migration.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from youth_migration.entity.artifact_entity import (
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
            logging.info(" Starting Data Ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            artifact = data_ingestion.initiate_data_ingestion()

            logging.info(" Data Ingestion Completed")
            return artifact

        except Exception as e:
            logging.error(f"Data Ingestion Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Data Validation
    # -------------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:
            logging.info(" Starting Data Validation")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            artifact = data_validation.initiate_data_validation()

            logging.info(" Data Validation Completed")
            return artifact

        except Exception as e:
            logging.error(f"Data Validation Failed: {e}")
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
            logging.info(" Starting Data Transformation")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info(" Data Transformation Completed")
            return artifact

        except Exception as e:
            logging.error(f"Data Transformation Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Model Trainer
    # -------------------------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:
            logging.info("🤖 Starting Model Training")

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info(" Model Training Completed")
            return artifact

        except Exception as e:
            logging.error(f"Model Training Failed: {e}")
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
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            artifact = model_evaluation.initiate_model_evaluation()

            logging.info("✅ Model Evaluation Completed")
            return artifact

        except Exception as e:
            logging.error(f"Model Evaluation Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Model Pusher
    # -------------------------------
    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:

        try:
            logging.info(" Starting Model Pusher")

            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )

            artifact = model_pusher.initiate_model_pusher()

            logging.info(" Model Pusher Completed")
            return artifact

        except Exception as e:
            logging.error(f"Model Pusher Failed: {e}")
            raise CustomException(e, sys)

    # -------------------------------
    # Run Full Pipeline
    # -------------------------------
    def run_pipeline(self) -> None:
        try:
            logging.info(" Training Pipeline Started")

            # Step 1: Ingestion
            ingestion_artifact = self.start_data_ingestion()

            # Step 2: Validation
            validation_artifact = self.start_data_validation(
                data_ingestion_artifact=ingestion_artifact
            )

            #  Important check
            if not validation_artifact.validation_status:
                logging.error(" Data validation failed. Pipeline stopped.")
                return

            # Step 3: Transformation
            transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_artifact=validation_artifact
            )

            # Step 4: Model Training
            trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=transformation_artifact
            )

            # Step 5: Model Evaluation
            evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=ingestion_artifact,
                model_trainer_artifact=trainer_artifact
            )

            #  Model acceptance check
            if not evaluation_artifact.is_model_accepted:
                raise Exception(" Model not accepted. Pipeline stopped.")

            # Step 6: Model Push
            self.start_model_pusher(
                model_evaluation_artifact=evaluation_artifact
            )

            logging.info(" Training Pipeline Completed Successfully")

        except Exception as e:
            logging.error(f" Pipeline Failed: {e}")
            raise CustomException(e, sys)