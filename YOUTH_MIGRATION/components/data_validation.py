import os
import sys
import json
import pandas as pd
from pandas import DataFrame

from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.utils.main_utils import read_yaml_file, write_yaml_file
from youth_migration.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from youth_migration.entity.config_entity import DataValidationConfig
from youth_migration.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Read CSV
    # -------------------------------
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Validate number of columns
    # -------------------------------
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected_cols = len(self._schema_config["columns"].keys())
            actual_cols = len(dataframe.columns)

            status = expected_cols == actual_cols
            logging.info(f"Expected cols: {expected_cols}, Actual cols: {actual_cols}")
            logging.info(f"Column count validation status: {status}")

            return status
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Validate column existence
    # -------------------------------
    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns

            missing_numerical = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]

            missing_categorical = [
                col for col in self._schema_config["categorical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical:
                logging.info(f"Missing numerical columns: {missing_numerical}")

            if missing_categorical:
                logging.info(f"Missing categorical columns: {missing_categorical}")

            return not (missing_numerical or missing_categorical)

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Detect data drift (simple version)
    # -------------------------------
    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            drift_report = {}

            for col in reference_df.columns:
                if col in current_df.columns and reference_df[col].dtype != "object":
                    ref_mean = reference_df[col].mean()
                    curr_mean = current_df[col].mean()

                    drift = abs(ref_mean - curr_mean)

                    drift_report[col] = {
                        "reference_mean": float(ref_mean),
                        "current_mean": float(curr_mean),
                        "drift": float(drift)
                    }

            # save report
            os.makedirs(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                exist_ok=True
            )

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report
            )

            logging.info("Drift report saved")

            # simple rule: drift detected if any column diff > threshold
            drift_detected = any(v["drift"] > 0.1 for v in drift_report.values())

            return drift_detected

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Main function
    # -------------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(
                file_path=self.data_ingestion_artifact.train_file_path
            )

            test_df = self.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            error_message = ""

            # 1. Column count check
            if not self.validate_number_of_columns(train_df):
                error_message += "Train dataframe column mismatch. "

            if not self.validate_number_of_columns(test_df):
                error_message += "Test dataframe column mismatch. "

            # 2. Column existence check
            if not self.is_column_exist(train_df):
                error_message += "Missing columns in train data. "

            if not self.is_column_exist(test_df):
                error_message += "Missing columns in test data. "

            validation_status = len(error_message) == 0

            # 3. Drift check (only if validation ok)
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)

                if drift_status:
                    error_message = "Data drift detected"
                else:
                    error_message = "No data drift detected"

            logging.info(f"Validation status: {validation_status}")
            logging.info(f"Message: {error_message}")

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=error_message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys)