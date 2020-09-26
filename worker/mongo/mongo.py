from pymongo import MongoClient
import os


class MongoWrapper:

    def __init__(self):

        self.client = MongoClient(
            'mongodb://' + os.getenv('MONGO_USERNAME', 'francisco') + ':' + os.getenv('MONGO_PASSWORD', 'francisco') + \
                '@' + os.getenv('MONGO', 'localhost:60222')
        )
        self.db = self.client[os.getenv('MONGO_BBDD', 'mlflow')]

    def set_one(self, collection, document):

        return self.db[collection].insert_one(document).acknowledged

    def get_one(self, collection, query):

        return self.db[collection].find_one(query)

    def get_many(self, collection, query):

        return self.db[collection].find(query)

    def __upsert(self, collection, query, document, options):

        return self.db[collection].update(query, document, **options).get('ok') == 1.0

    def upsert_one(self, collection, query, document):

        return self.__upsert(collection, query, document, {"upsert": True, "multi": False})

    def upsert_many(self, collection, query, document):

        return self.__upsert(collection, query, document, {"upsert": True, "multi": True})

    def delete_one(self, collection, query):

        self.db[collection].delete_one(query)

    def delete_many(self, collection, query):

        self.db[collection].delete_many(query)