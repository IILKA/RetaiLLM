import pymongo
import pandas as pd

class MongoDBClient:
    def __init__(self, user, password, server, db_name):
        self.client = pymongo.MongoClient(f"mongodb://{user}:{password}@{server}:27017/")
        self.db = self.client[db_name]

    def list_collections(self):
        collections = self.db.list_collection_names()
        print(collections)
        return collections

    def search_text(self, searchkey):    
        collections = self.db.list_collection_names()
        for col in collections:
            for line in self.db[col].find({"line": {"$regex": searchkey}}):
                print(line)
            return {}
                #_name: name string, description: des
    def get_df(self, name):
        collections = self.db.list_collection_names()
        data = []

        for col in collections:
            for line in self.db[col].find({"line": {"$regex": name}}):
                id = line["_id"]
                content = line["line"]
                data.append({"collection": col, "id": id, "content": content})

        df = pd.DataFrame(data, columns=["collection", "id", "content"])
        return df
                


    

# Example usage
user = "mongo_nX8tFs"
password = "mongo_tCE4Tr"
server = "154.208.10.3"
db_name = "4011"

mongo_client = MongoDBClient(user, password, server, db_name)
#mongo_client.list_collections()
#mongo_client.search_text("inventory")
df = mongo_client.get_df("inventory")
print(mongo_client.list_collections())
print(type(mongo_client.list_collections()))
# print(df)