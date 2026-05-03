import sys
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from imblearn.combine import SMOTEENN

from youth_migration.exception import CustomException
from youth_migration.logger import logging
from youth_migration.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns
)

from youth_migration.entity.config_entity import DataTransformationConfig
from youth_migration.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from youth_migration.constants import SCHEMA_FILE_PATH, CURRENT_YEAR


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Preprocessor
    # -------------------------------
    def get_data_transformer_object(self):

        try:
            logging.info("Creating preprocessing pipeline")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            num_features = self._schema_config['num_features']
            transform_columns = self._schema_config['transform_columns']

            # pipelines
            numeric_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            oh_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            ordinal_pipeline = Pipeline([
                ("ordinal", OrdinalEncoder())
            ])

            transform_pipeline = Pipeline([
                ("power", PowerTransformer(method="yeo-johnson"))
            ])

            preprocessor = ColumnTransformer([
                ("onehot", oh_pipeline, oh_columns),
                ("ordinal", ordinal_pipeline, or_columns),
                ("transform", transform_pipeline, transform_columns),
                ("numeric", numeric_pipeline, num_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Main function
    # -------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            target_column = "case_status"

            # split
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            #  feature engineering
            X_train["company_age"] = CURRENT_YEAR - X_train["yr_of_estab"]
            X_test["company_age"] = CURRENT_YEAR - X_test["yr_of_estab"]

            # drop columns
            drop_cols = self._schema_config["drop_columns"]
            X_train = drop_columns(X_train, drop_cols)
            X_test = drop_columns(X_test, drop_cols)

            #  target encoding (simple)
            y_train = y_train.map({"Approved": 1, "Denied": 0})
            y_test = y_test.map({"Approved": 1, "Denied": 0})

            # preprocessing
            preprocessor = self.get_data_transformer_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            #  SMOTE only on TRAIN (IMPORTANT FIX)
            smote = SMOTEENN(sampling_strategy="minority")

            X_train_final, y_train_final = smote.fit_resample(X_train_arr, y_train)

            #  test set-এ SMOTE দিবে না
            X_test_final, y_test_final = X_test_arr, y_test

            # combine
            train_arr = np.c_[X_train_final, np.array(y_train_final)]
            test_arr = np.c_[X_test_final, np.array(y_test_final)]

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