import sys
import pandas as pd
import numpy as np
from typing import Optional

from youth_migration.configuration.mongo_db_connection import MongoDBClient
from youth_migration.constants import DATABASE_NAME
from youth_migration.exception import YouthMigrationException
from youth_migration.logger import logging


class YouthMigrationData:
    """
    This class helps to export MongoDB collection as pandas DataFrame
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise YouthMigrationException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export entire collection as DataFrame

        Args:
            collection_name (str): MongoDB collection name
            database_name (Optional[str]): Optional DB override

        Returns:
            pd.DataFrame
        """
        try:
            # Select database
            if database_name is None:
                db = self.mongo_client.database
            else:
                db = self.mongo_client.client[database_name]

            collection = db[collection_name]

            logging.info(f"Exporting collection: {collection_name}")

            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                logging.warning(f"No data found in collection: {collection_name}")

            # Drop MongoDB internal id
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Replace "na" with np.nan
            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Data exported successfully. Shape: {df.shape}")

            return df

        except Exception as e:
            logging.error("Error while exporting data", exc_info=True)
            raise YouthMigrationException(e, sys)