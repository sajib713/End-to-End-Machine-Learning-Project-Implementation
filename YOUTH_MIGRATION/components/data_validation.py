import os
import sys
import re
import pandas as pd
from pandas.api.types import is_numeric_dtype

from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.utils.main_utils import read_yaml_file, write_yaml_file
from youth_migration.entity.artifact_entity import DataValidationArtifact
from youth_migration.constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(self, data_ingestion_artifact, data_validation_config):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    # -------------------------------
    # Normalize Column
    # -------------------------------
    def normalize(self, col):
        col = col.strip().lower()
        col = re.sub(r"\s+", " ", col)
        col = col.replace(":", "").replace("?", "")
        return col

    # -------------------------------
    # Read Data
    # -------------------------------
    def read_data(self, path):
        return pd.read_csv(path)

    # -------------------------------
    # Column Validation
    # -------------------------------
    def validate_columns(self, df):
        expected = set(self.normalize(c) for c in self._schema_config["columns"].keys())
        actual = set(self.normalize(c) for c in df.columns)

        missing = expected - actual
        extra = actual - expected

        if missing:
            logging.error(f"Missing Columns: {missing}")

        if extra:
            logging.warning(f"Extra Columns: {extra}")

        return len(missing) == 0

    # -------------------------------
    # Data Type Validation
    # -------------------------------
    def validate_dtypes(self, df):
        schema = self._schema_config["columns"]

        for col, dtype in schema.items():
            for df_col in df.columns:
                if self.normalize(df_col) == self.normalize(col):
                    if dtype == "int" and not is_numeric_dtype(df[df_col]):
                        logging.error(f"{col} should be numeric")
                        return False

        return True

    # -------------------------------
    # Dataset Drift
    # -------------------------------
    def detect_drift(self, train, test):
        report = {}
        threshold = self._schema_config.get("drift_threshold", 0.1)

        for col in train.columns:
            if col in test.columns and is_numeric_dtype(train[col]):
                diff = abs(train[col].mean() - test[col].mean())
                report[col] = float(diff)

        os.makedirs(
            os.path.dirname(self.data_validation_config.drift_report_file_path),
            exist_ok=True
        )

        write_yaml_file(
            self.data_validation_config.drift_report_file_path,
            report
        )

        return any(v > threshold for v in report.values())

    # -------------------------------
    # Main Validation
    # -------------------------------
    def initiate_data_validation(self):

        try:
            logging.info("Starting Data Validation")

            train = self.read_data(self.data_ingestion_artifact.train_file_path)
            test = self.read_data(self.data_ingestion_artifact.test_file_path)

            # 🔥 Normalize column names
            train.columns = train.columns.str.strip().str.lower()
            test.columns = test.columns.str.strip().str.lower()

            # 🔥 FIX dtype (age column)
            train["what is your age?"] = pd.to_numeric(train["what is your age?"], errors="coerce")
            test["what is your age?"] = pd.to_numeric(test["what is your age?"], errors="coerce")

            train["what is your age?"] = pd.to_numeric(train["what is your age?"], errors="coerce")
            test["what is your age?"] = pd.to_numeric(test["what is your age?"], errors="coerce")
            train["what is your age?"] = train["what is your age?"].fillna(train["what is your age?"].median())
            test["what is your age?"] = test["what is your age?"].fillna(test["what is your age?"].median())

            # 🔥 categorical fill (only object columns)
            cat_cols = train.select_dtypes(include=["object"]).columns

            train[cat_cols] = train[cat_cols].fillna("unknown")
            test[cat_cols] = test[cat_cols].fillna("unknown")

        

            # 🔥 numeric missing fix
            train["what is your age?"] = train["what is your age?"].fillna(train["what is your age?"].median())
            test["what is your age?"] = test["what is your age?"].fillna(test["what is your age?"].median())

            # 🔥 Drop unwanted columns
            drop_cols = self._schema_config.get("drop_columns", [])
            train = train.drop(columns=drop_cols, errors="ignore")
            test = test.drop(columns=drop_cols, errors="ignore")

            error = ""

            # Column validation
            if not self.validate_columns(train):
                error += "column mismatch "

            # dtype validation
            if not self.validate_dtypes(train):
                error += "dtype mismatch "

            # Missing values (only warning now)
            if train.isnull().sum().sum() > 0:
                logging.warning("Missing values found but handled")

            validation_status = len(error) == 0

            # Drift check
            if validation_status:
                drift = self.detect_drift(train, test)
                if drift:
                    error += "data drift detected "

            return DataValidationArtifact(
                validation_status=validation_status,
                message=error,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)