import sys
import numpy as np
import importlib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from youth_migration.exception import CustomException
from youth_migration.logger import logging

from youth_migration.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    read_yaml_file
)

from youth_migration.entity.config_entity import ModelTrainerConfig
from youth_migration.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from youth_migration.entity.estimator import MigrationModel


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

        # 🔥 load model.yaml
        self.model_config = read_yaml_file(
            self.model_trainer_config.model_config_file_path
        )

    # -------------------------------
    # Create model from yaml
    # -------------------------------
    def get_model(self, module_name, class_name, params):
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class(**params)

    # -------------------------------
    # Train using GridSearch
    # -------------------------------
    def train_with_grid_search(self, X_train, y_train, X_test, y_test):

        best_model = None
        best_score = 0

        for module_name, model_info in self.model_config["model_selection"].items():

            logging.info(f"Training: {model_info['class']}")

            model = self.get_model(
                model_info["module"],
                model_info["class"],
                model_info["params"]
            )

            grid = GridSearchCV(
                estimator=model,
                param_grid=model_info["search_param_grid"],
                cv=self.model_config["grid_search"]["params"]["cv"],
                verbose=self.model_config["grid_search"]["params"]["verbose"]
            )

            grid.fit(X_train, y_train)

            y_pred = grid.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            logging.info(f"{model_info['class']} Accuracy: {acc}")

            if acc > best_score:
                best_score = acc
                best_model = grid.best_estimator_

        return best_model, best_score

    # -------------------------------
    # Main
    # -------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            logging.info("Loading transformed data")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # 🔥 Grid Search training
            best_model, best_score = self.train_with_grid_search(
                X_train, y_train, X_test, y_test
            )

            logging.info(f"Best Model Score: {best_score}")

            if best_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No model met expected accuracy")

            # metrics
            y_pred = best_model.predict(X_test)

            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred),
                precision_score=precision_score(y_test, y_pred),
                recall_score=recall_score(y_test, y_pred)
            )

            # load preprocessor
            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            final_model = MigrationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )

            # save
            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model
            )

            logging.info("Model saved successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise CustomException(e, sys)