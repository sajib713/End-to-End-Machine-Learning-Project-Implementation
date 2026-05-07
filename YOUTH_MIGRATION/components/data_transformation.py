import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns
)

from youth_migration.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from youth_migration.constants import SCHEMA_FILE_PATH


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config,
        data_validation_artifact: DataValidationArtifact
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    @staticmethod
    def read_data(file_path):
        return pd.read_csv(file_path)

    # -------------------------------
    # Preprocessor
    # -------------------------------
    def get_data_transformer_object(self, X_train):
        num_features = [col.strip().lower() for col in self._schema_config["numerical_columns"]]
        cat_features = [col.strip().lower() for col in self._schema_config["categorical_columns"]]

        num_features = [col for col in num_features if col in X_train.columns]
        cat_features = [col for col in cat_features if col in X_train.columns]

        num_features = [col for col in num_features if col not in cat_features]

        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features)
        ])

        return preprocessor

    # -------------------------------
    # Main function
    # -------------------------------
    def initiate_data_transformation(self):

        try:
            logging.info("Starting data transformation")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # normalize columns
            train_df.columns = train_df.columns.str.strip().str.lower()
            test_df.columns = test_df.columns.str.strip().str.lower()

            target_column = self._schema_config["target_column"]

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # drop unwanted
            drop_cols = self._schema_config["drop_columns"]
            X_train = drop_columns(X_train, drop_cols)
            X_test = drop_columns(X_test, drop_cols)

            # 🔥 ALIGN columns
            X_test = X_test.reindex(columns=X_train.columns, fill_value="unknown")

            # 🔥 CLEAN TARGET (IMPORTANT FIX)
            y_train = y_train.astype(str).str.strip().str.lower()
            y_test = y_test.astype(str).str.strip().str.lower()

            # replace invalid values safely
            y_train = y_train.replace({"yes": 1, "no": 0})
            y_test = y_test.replace({"yes": 1, "no": 0})

            # 🔥 fill invalid with most frequent
            y_train = pd.to_numeric(y_train, errors="coerce")
            y_test = pd.to_numeric(y_test, errors="coerce")

            y_train = y_train.fillna(y_train.mode()[0])
            y_test = y_test.fillna(y_test.mode()[0])

            # preprocessing
            preprocessor = self.get_data_transformer_object(X_train)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # convert sparse to dense
            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
            if hasattr(X_test_arr, "toarray"):
                X_test_arr = X_test_arr.toarray()

            # reshape
            y_train_arr = np.array(y_train).reshape(-1, 1)
            y_test_arr = np.array(y_test).reshape(-1, 1)

            # combine
            train_arr = np.hstack((X_train_arr, y_train_arr))
            test_arr = np.hstack((X_test_arr, y_test_arr))

            # save
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)