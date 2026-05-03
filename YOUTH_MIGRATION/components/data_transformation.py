import sys
from youth_migration.exception import CustomException
from youth_migration.logger import logging


class DataTransformation:
    def __init__(self, data_ingestion_artifact, data_validation_artifact, data_transformation_config):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            logging.info("Data Transformation started")

            # Tporary dummy artifact
            class DataTransformationArtifact:
                pass

            logging.info("Data Transformation completed")
            return DataTransformationArtifact()

        except Exception as e:
           raise CustomException(e, sys)

