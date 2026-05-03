import sys
import pandas as pd
from pandas import DataFrame

from youth_migration.entity.config_entity import MigrationPredictorConfig
from youth_migration.entity.s3_estimator import MigrationEstimator
from youth_migration.exception import CustomException
from youth_migration.logger import logging


# -------------------------------
# Input Data Class
# -------------------------------
class MigrationData:
    def __init__(
        self,
        age,
        gender,
        occupation,
        family_abroad,
        aware_programs,
        migration_trend,
        return_plan,
        family_impact,
        govt_support,
        responsibility,
        recommend_score,
        stress_score,
        country_count
    ):
        try:
            self.age = age
            self.gender = gender
            self.occupation = occupation
            self.family_abroad = family_abroad
            self.aware_programs = aware_programs
            self.migration_trend = migration_trend
            self.return_plan = return_plan
            self.family_impact = family_impact
            self.govt_support = govt_support
            self.responsibility = responsibility
            self.recommend_score = recommend_score
            self.stress_score = stress_score
            self.country_count = country_count

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_as_dict(self):
        try:
            return {
                "What is your age?": [self.age],
                "What is your gender?": [self.gender],
                "What is your occupation?": [self.occupation],
                "Does any member of your family or a close acquaintance live abroad?": [self.family_abroad],
                "Are you aware of any specific migration programs or scholarships available for students or workers abroad?": [self.aware_programs],
                "Do you think migration trends have increased among Bangladeshi youth in the last 5 years?": [self.migration_trend],
                "Do you think you will return to Bangladesh in the future after living abroad?": [self.return_plan],
                "Do you think your decision to move abroad will impact your family or community in Bangladesh?": [self.family_impact],
                "Do you feel that the government of Bangladesh offers enough support for youth planning to migrate abroad?": [self.govt_support],
                "Do you feel a sense of responsibility towards your family while planning to move abroad?": [self.responsibility],
                "recommend_score": [self.recommend_score],
                "stress_score": [self.stress_score],
                "country_count": [self.country_count]
            }

        except Exception as e:
            raise CustomException(e, sys)

    def get_input_dataframe(self) -> DataFrame:
        try:
            return pd.DataFrame(self.get_data_as_dict())
        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------
# Prediction Class
# -------------------------------
class MigrationClassifier:
    def __init__(
        self,
        prediction_pipeline_config: MigrationPredictorConfig = MigrationPredictorConfig(),
    ) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            logging.info("Starting prediction")

            model = MigrationEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)

            return result

        except Exception as e:
            raise CustomException(e, sys)