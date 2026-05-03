import sys
from youth_migration.exception import CustomException
from youth_migration.logger import logging


class DataValidation:
    def __init__(self, data_ingestion_artifact, data_validation_config):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        try:
            logging.info("Data Validation started")

            # 👉 Temporary dummy artifact
            class DataValidationArtifact:
                pass

            logging.info("Data Validation completed")
            return DataValidationArtifact()

        except Exception as e:
            raise CustomException(e, sys)