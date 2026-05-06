import sys
import pandas as pd
from pandas import DataFrame

from youth_migration.entity.config_entity import MigrationPredictorConfig
from youth_migration.entity.s3_estimator import MigrationEstimator
from youth_migration.exception import CustomException
from youth_migration.logger import logging


# -----------------------------------
# Input Data Class
# -----------------------------------
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

    # -----------------------------------
    # Convert Input To Dictionary
    # -----------------------------------
    def get_data_as_dict(self):

        try:

            input_data = {

                # -----------------------------------
                # Real Input Fields
                # -----------------------------------
                "what is your age?": [self.age],

                "what is your gender?": [self.gender],

                "what is your occupation?": [self.occupation],

                "does any member of your family or a close acquaintance live abroad?": [
                    self.family_abroad
                ],

                "how long do you plan to stay abroad?": [
                    "2-5 years"
                ],

                "are you aware of any specific migration programs or scholarships available for students or workers abroad?": [
                    self.aware_programs
                ],

                "do you think migration trends have increased among bangladeshi youth in the last 5 years?": [
                    self.migration_trend
                ],

                "do you think you will return to bangladesh in the future after living abroad?": [
                    self.return_plan
                ],

                "do you think your decision to move abroad will impact your family or community in bangladesh?": [
                    self.family_impact
                ],

                "do you feel that the government of bangladesh offers enough support for youth planning to migrate abroad?": [
                    self.govt_support
                ],

                "do you feel a sense of responsibility towards your family while planning to move abroad?": [
                    self.responsibility
                ],

                "how likely are you to recommend going abroad to others in your age group? (rate on a scale of 1 to 5, where 1 is very unlikely and 5 is very likely).": [
                    self.recommend_score
                ],

                # -----------------------------------
                # Default Values For Missing Columns
                # -----------------------------------
                "what role do social media and online success stories play in influencing your decision to move abroad?": [
                    "medium"
                ],

                "who or what influences your decision to move abroad? (multiple options can be selected)": [
                    "family"
                ],

                "which countries are you most interested in moving to?": [
                    "canada"
                ],

                "how much psychological stress or anxiety have you experienced while considering moving abroad?": [
                    "medium"
                ],

                "what is your primary goal for considering moving abroad? (multiple options can be selected)": [
                    "education"
                ],

                "what kind of psychological stress have you experienced while planning to move abroad? (multiple options can be selected)": [
                    "financial"
                ],

                "how do you cope with the stress of planning to move abroad?": [
                    "family"
                ],

                "what do you think is the biggest barrier to migrating abroad? (multiple options can be selected)": [
                    "financial issues"
                ]
            }

            return input_data

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------
    # Convert To DataFrame
    # -----------------------------------
    def get_input_dataframe(self) -> DataFrame:

        try:

            df = pd.DataFrame(self.get_data_as_dict())

            # normalize columns
            df.columns = df.columns.str.strip().str.lower()

            return df

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------------
# Prediction Class
# -----------------------------------
class MigrationClassifier:

    def __init__(
        self,
        prediction_pipeline_config: MigrationPredictorConfig = MigrationPredictorConfig(),
    ):

        try:

            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------
    # Prediction Function
    # -----------------------------------
    def predict(self, dataframe: DataFrame):

        try:

            logging.info("Starting Prediction")

            model = MigrationEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)

            logging.info("Prediction Completed")

            return result

        except Exception as e:
            raise CustomException(e, sys)