from YOUTH_MIGRATION.configuration.mongo_db_connection import MongoDBClient
from YOUTH_MIGRATION.constants import DATABASE_NAME
from YOUTH_MIGRATION.exception import CustomException

import pandas as pd
import sys
import numpy as np
from typing import Optional


class YouthMigrationData:
    """
    MongoDB collection → Pandas DataFrame convert করার জন্য class
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            # -------------------------------
            # Collection select
            # -------------------------------
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            # -------------------------------
            # Convert to DataFrame
            # -------------------------------
            df = pd.DataFrame(list(collection.find()))

            # -------------------------------
            # Drop MongoDB _id
            # -------------------------------
            if "_id" in df.columns:
                df.drop(columns=["_id"], axis=1, inplace=True)

            # -------------------------------
            # Replace 'na' → np.nan
            # -------------------------------
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)