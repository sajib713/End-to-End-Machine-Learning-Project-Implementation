import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from youth_migration.exception import CustomException
from youth_migration.logger import logging


# -------------------------------
# Target Mapping
# -------------------------------
class TargetValueMapping:
    def __init__(self):
        self.Approved: int = 1
        self.Denied: int = 0

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping = self._asdict()
        return dict(zip(mapping.values(), mapping.keys()))


# -------------------------------
# Model Wrapper
# -------------------------------
class MigrationModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        preprocessing_object → saved preprocessor (ColumnTransformer)
        trained_model_object → trained ML model (RF, XGB etc.)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        try:
            logging.info("Starting prediction")

            # Step 1: preprocessing
            transformed_data = self.preprocessing_object.transform(dataframe)

            # Step 2: prediction
            prediction = self.trained_model_object.predict(transformed_data)

            logging.info("Prediction completed")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, dataframe: DataFrame):
        """
        optional → probability prediction (useful for API)
        """
        try:
            transformed_data = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict_proba(transformed_data)

        except Exception as e:
            raise CustomException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"