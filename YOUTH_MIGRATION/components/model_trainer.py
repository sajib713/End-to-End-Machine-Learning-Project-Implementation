import sys
import numpy as np
import importlib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import issparse

from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging

from YOUTH_MIGRATION.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    read_yaml_file
)

from YOUTH_MIGRATION.entity.config_entity import ModelTrainerConfig
from YOUTH_MIGRATION.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from YOUTH_MIGRATION.entity.estimator import MigrationModel


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

        # load model.yaml
        self.model_config = read_yaml_file(
            self.model_trainer_config.model_config_file_path
        )

    # -----------------------------------
    # Load Model Dynamically
    # -----------------------------------
    def get_model(self, module_name, class_name, params):

        module = importlib.import_module(module_name)

        model_class = getattr(module, class_name)

        return model_class(**params)

    # -----------------------------------
    # Train Models
    # -----------------------------------
    def train_with_grid_search(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        best_model = None
        best_score = 0

        for module_name, model_info in self.model_config["model_selection"].items():

            logging.info(f"Training Started: {model_info['class']}")

            model = self.get_model(
                model_info["module"],
                model_info["class"],
                model_info["params"]
            )

            # -----------------------------------
            # 🔥 FIX FOR KNN SPARSE MATRIX ERROR
            # -----------------------------------
            if isinstance(model, KNeighborsClassifier):

                model.set_params(algorithm="brute")

            grid = GridSearchCV(
                estimator=model,
                param_grid=model_info["search_param_grid"],
                cv=self.model_config["grid_search"]["params"]["cv"],
                verbose=self.model_config["grid_search"]["params"]["verbose"],
                n_jobs=-1
            )

            # -----------------------------------
            # Sparse matrix handling
            # -----------------------------------
            X_train_fit = X_train
            X_test_fit = X_test

            if isinstance(model, KNeighborsClassifier):

                if issparse(X_train):
                    X_train_fit = X_train.toarray()

                if issparse(X_test):
                    X_test_fit = X_test.toarray()

            # train
            grid.fit(X_train_fit, y_train)

            # predict
            y_pred = grid.best_estimator_.predict(X_test_fit)

            # accuracy
            acc = accuracy_score(y_test, y_pred)

            logging.info(
                f"{model_info['class']} Accuracy: {acc}"
            )

            # save best
            if acc > best_score:

                best_score = acc

                best_model = grid.best_estimator_

        return best_model, best_score

    # -----------------------------------
    # Main Training Function
    # -----------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            logging.info("Loading transformed train and test arrays")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            # split X/y
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]

            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(
                f"Train Shape: {X_train.shape}"
            )

            logging.info(
                f"Test Shape: {X_test.shape}"
            )

            # -----------------------------------
            # Train models
            # -----------------------------------
            best_model, best_score = self.train_with_grid_search(
                X_train,
                y_train,
                X_test,
                y_test
            )

            logging.info(
                f"Best Model Score: {best_score}"
            )

            # expected accuracy check
            if best_score < self.model_trainer_config.expected_accuracy:

                raise Exception(
                    "No model met expected accuracy"
                )

            # -----------------------------------
            # Prediction
            # -----------------------------------
            X_test_eval = X_test

            if isinstance(best_model, KNeighborsClassifier):

                if issparse(X_test):
                    X_test_eval = X_test.toarray()

            y_pred = best_model.predict(X_test_eval)

            # -----------------------------------
            # Metrics
            # -----------------------------------
            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred),
                precision_score=precision_score(y_test, y_pred),
                recall_score=recall_score(y_test, y_pred)
            )

            # -----------------------------------
            # Load Preprocessor
            # -----------------------------------
            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # -----------------------------------
            # Final Model
            # -----------------------------------
            final_model = MigrationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )

            # -----------------------------------
            # Save Model
            # -----------------------------------
            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model
            )

            logging.info(
                "Final model saved successfully"
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:

            raise CustomException(e, sys)