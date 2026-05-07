import sys
from pandas import DataFrame

from youth_migration.cloud_storage.aws_storage import SimpleStorageService
from youth_migration.exception import CustomException
from youth_migration.entity.estimator import MigrationModel


class MigrationEstimator:
    """
    This class is used to save and retrieve migration model from S3 bucket and perform prediction
    """

    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MigrationModel = None

    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self) -> MigrationModel:
        try:
            return self.s3.load_model(
                model_name=self.model_path,
                bucket_name=self.bucket_name
            )
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            return self.loaded_model.predict(dataframe=dataframe)

        except Exception as e:
            raise CustomException(e, sys)