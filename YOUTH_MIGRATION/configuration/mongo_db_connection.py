import sys
import os
import pymongo
import certifi
from dotenv import load_dotenv

load_dotenv()

from YOUTH_MIGRATION.exception import CustomException
from YOUTH_MIGRATION.logger import logging
from YOUTH_MIGRATION.constants import DATABASE_NAME, MONGODB_URL_KEY


class MongoDBClient:

    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if not mongo_db_url:
                    raise Exception(f"{MONGODB_URL_KEY} not found in environment")

                ca = certifi.where()

                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tls=True,
                    tlsCAFile=ca,
                    serverSelectionTimeoutMS=10000
                )

                # 🔥 test connection
                MongoDBClient.client.admin.command("ping")

            self.client = MongoDBClient.client
            self.database = self.client[database_name]

            logging.info("MongoDB connected successfully")

        except Exception as e:
            logging.error(f"MongoDB error: {e}")
            raise CustomException(e, sys)