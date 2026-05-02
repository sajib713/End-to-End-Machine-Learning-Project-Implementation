import sys
import os
import pymongo
import certifi

from youth_migration.exception import YouthMigrationException
from youth_migration.logger import logging
from youth_migration.constants import DATABASE_NAME, MONGODB_URL_KEY

ca = certifi.where()


class MongoDBClient:
    """
    MongoDB Client for establishing connection

    Returns:
        MongoDB database connection
    """

    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if mongo_db_url is None:
                    raise ValueError(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set."
                    )

                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca
                )

                logging.info("MongoDB client created successfully")

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

            logging.info(f"Connected to database: {database_name}")

        except Exception as e:
            raise YouthMigrationException(e, sys)