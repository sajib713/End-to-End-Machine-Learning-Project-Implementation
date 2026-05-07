import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging


# -------------------------------
# YAML READ
# -------------------------------
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# YAML WRITE
# -------------------------------
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# LOAD OBJECT
# -------------------------------
def load_object(file_path: str) -> object:
    logging.info("Entered load_object method")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited load_object method")
        return obj

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# SAVE NUMPY ARRAY
# -------------------------------
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# LOAD NUMPY ARRAY
# -------------------------------
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# SAVE OBJECT
# -------------------------------
def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered save_object method")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited save_object method")

    except Exception as e:
        raise CustomException(e, sys) from e


# -------------------------------
# DROP COLUMNS
# -------------------------------
def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    logging.info("Entered drop_columns method")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited drop_columns method")
        return df

    except Exception as e:
        raise CustomException(e, sys) from e