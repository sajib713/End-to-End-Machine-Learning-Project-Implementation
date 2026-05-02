import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from youth_migration.entity.config_entity import DataIngestionConfig
from youth_migration.entity.artifact_entity import DataIngestionArtifact
from youth_migration.exception import YouthMigrationException
from youth_migration.logger import logging
from youth_migration.data_access.usvisa_data import YouthMigrationData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise YouthMigrationException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Starting data export from MongoDB")

            data_obj = YouthMigrationData()

            dataframe = data_obj.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(f"Data fetched. Shape: {dataframe.shape}")
            logging.info(f"Columns: {list(dataframe.columns)}")

            #  Safety Check 1: Empty dataset
            if dataframe.empty:
                raise ValueError("No data found in MongoDB collection")

            #  Safety Check 2: Target column exists
            if "case_status" not in dataframe.columns:
                raise ValueError("Target column 'case_status' not found in dataset")

            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

            dataframe.to_csv(feature_store_path, index=False)

            logging.info(f"Feature store file saved at: {feature_store_path}")

            return dataframe

        except Exception as e:
            logging.error("Error in export_data_into_feature_store", exc_info=True)
            raise YouthMigrationException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Starting train-test split")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42  # reproducibility
            )

            logging.info(f"Train shape: {train_set.shape}")
            logging.info(f"Test shape: {test_set.shape}")

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)

            logging.info("Train-test files saved successfully")

        except Exception as e:
            logging.error("Error in split_data_as_train_test", exc_info=True)
            raise YouthMigrationException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Initiating data ingestion pipeline")

        try:
            dataframe = self.export_data_into_feature_store()

            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data Ingestion Artifact: {artifact}")

            return artifact

        except Exception as e:
            logging.error("Error in initiate_data_ingestion", exc_info=True)
            raise YouthMigrationException(e, sys) from e